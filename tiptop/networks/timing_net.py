"""A small network that learns the *timing* of a manipulation stroke from DROID human teleop.

Context
-------
``trajectory_blending`` turns a cuTAMP plan's stop-and-go segments into one continuous stroke per
operation by *path-velocity decomposition*: a smoothing spline fixes the GEOMETRY (the collision-checked
path, unchanged) and a hand-tuned quintic / asymmetric TIME LAW sweeps along it. This module replaces
only that time law with a learned one -- a network that predicts the speed-vs-progress profile a human
would use, trained on real DROID strokes. The geometry, the robot vel/accel caps, the endpoint pinning
and the non-idle boundary speed are all still enforced analytically in ``neural_blending`` (we learn the
*style*, not the constraints).

What the network predicts
-------------------------
For one stroke it outputs ``p(tau)`` on a fixed grid of ``N_KNOTS`` points -- the (strictly positive,
unit-mean) joint-space arc-length SPEED as a function of normalized time ``tau in [0, 1]``. This is
exactly the ``p(tau)`` the analytic ``_asymmetric_stroke`` builds by hand, so a learned ``p`` is a
drop-in: integrate it to the time law ``h(tau)`` and sample. Being unit-mean it carries only the SHAPE
(where along the reach the arm speeds up / slows down); the absolute pace still comes from the plan's
wall-clock, and the boundary levels the openpi non-idle filter needs are enforced on top in
``neural_blending._neural_stroke``.

Conditioning (the inputs)
-------------------------
1. ``(lead_kind, trail_kind)`` -- what the stroke starts from / decelerates into, each in {rest, grasp,
   release}. Present identically in DROID (gripper-command edges) and a cuTAMP plan (adjacent gripper
   steps), so it transfers directly.
2. GEOMETRY + PACE features (:func:`stroke_features`) -- transferable, sampling-invariant descriptors of
   the path being timed: total arc length, straightness (chord/arc), average speed (= pace = arc /
   duration), mean/max turning, and a short turning-vs-progress curvature profile. These let the profile
   depend on WHICH reach it is (a 5 cm nudge vs a 60 cm curved cross-table reach are timed differently) --
   the bulk of stroke-to-stroke variance that ``(lead, trail)`` alone cannot see. They are standardized
   with training-set stats baked into the checkpoint. Note: the network only *reads* geometry; it never
   moves waypoints (the collision-checked path is untouched), so the path-velocity split holds.
3. Optional TASK-LANGUAGE embedding (``use_language``) -- a frozen sentence embedding of the task prompt,
   so e.g. a careful placement and a quick toss can carry different timing. Learned from DROID's task
   variety; applied by prompt at plan time.

This file holds both the ``TimingNet`` module (shared by training and inference) and the ``TimingModel``
plan-time wrapper that ``neural_blending`` calls. Training lives in ``train_timing.py`` and the DROID
extraction in ``extract_stroke_timing.py``; checkpoints default to ``tiptop/tiptop/checkpoints``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_log = logging.getLogger(__name__)

# Number of grid points the speed profile p(tau) is represented on (tau uniform in [0, 1]).
N_KNOTS = 64

# Geometry/pace feature layout. The path is resampled to FEAT_RESAMPLE arc-even points before computing
# turning so the features are sampling-invariant (a DROID 15 Hz stroke and a cuTAMP waypoint list give the
# same features for the same shape). N_FEAT = 5 scalars + N_CURV curvature-profile knots.
N_CURV = 12
FEAT_RESAMPLE = 64
FEAT_SCALARS = ("arc_length", "straightness", "avg_speed", "mean_turn", "max_turn")
N_FEAT = len(FEAT_SCALARS) + N_CURV

# The stroke-boundary kinds. A stroke starts from one and decelerates into another; these are exactly
# recoverable in both DROID (gripper-command edges) and a cuTAMP plan (adjacent gripper steps).
#   rest    -- the arm is genuinely stationary here (episode start or end)
#   grasp   -- a gripper CLOSE abuts this end (a pick)
#   release -- a gripper OPEN abuts this end (a place)
EVENT_KINDS: tuple[str, ...] = ("rest", "grasp", "release")
KIND_TO_IDX = {k: i for i, k in enumerate(EVENT_KINDS)}

# Default frozen sentence encoder for the optional task-language conditioning (384-D, small + local).
DEFAULT_LANG_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Default checkpoint location (per repo layout: tiptop/tiptop/checkpoints/timing_net.pt).
_PKG_DIR = Path(__file__).resolve().parents[1]  # .../tiptop/tiptop
DEFAULT_CKPT = _PKG_DIR / "checkpoints" / "timing_net.pt"


def tau_grid(n_knots: int = N_KNOTS) -> np.ndarray:
    """The normalized-time knot grid the profile is defined on."""
    return np.linspace(0.0, 1.0, n_knots)


def _clean_path(q: np.ndarray) -> np.ndarray | None:
    """Drop zero-length steps; return the cleaned path or None if it has no meaningful motion."""
    q = np.asarray(q, dtype=np.float64)
    if len(q) < 3:
        return None
    step = np.linalg.norm(np.diff(q, axis=0), axis=1)
    keep = np.concatenate([[True], step > 1e-9])
    q = q[keep]
    return q if len(q) >= 3 else None


def stroke_speed_profile(q: np.ndarray, dt: float, n_knots: int = N_KNOTS) -> np.ndarray | None:
    """Unit-mean joint-space arc-length speed vs normalized time for ONE stroke's joint positions.

    ``q`` is ``[T, dof]`` joint angles at spacing ``dt``. Returns the ``[n_knots]`` profile ``p(tau)``
    (mean 1, strictly positive), the training target and the exact quantity the analytic blend's ``p(tau)``
    represents, or ``None`` for a degenerate stroke. Speed is ``|dq|/dt`` at step midpoints (``|dq|`` is
    the arc-length increment since ``u`` is joint-space arc length), resampled onto the knot grid and
    normalized to unit mean so only the SHAPE survives.
    """
    q = np.asarray(q, dtype=np.float64)
    if len(q) < 3:
        return None
    step = np.linalg.norm(np.diff(q, axis=0), axis=1)
    if step.sum() < 1e-4:
        return None
    speed = step / dt
    tau_mid = (np.arange(len(speed)) + 0.5) / len(speed)
    p = np.interp(tau_grid(n_knots), tau_mid, speed)
    m = float(p.mean())
    if m <= 1e-9:
        return None
    return (p / m).astype(np.float32)


def stroke_features(positions: np.ndarray, duration: float, n_curv: int = N_CURV) -> np.ndarray | None:
    """Transferable, sampling-invariant GEOMETRY + PACE features of one stroke's path.

    ``positions`` is ``[N, dof]`` joint waypoints and ``duration`` the stroke's wall-clock (seconds). The
    path is first resampled to ``FEAT_RESAMPLE`` arc-even points so the features do not depend on the
    source sampling (DROID 15 Hz frames vs cuTAMP waypoints). Returns a length-``N_FEAT`` RAW (un-standardized)
    vector ``[arc_length, straightness, avg_speed, mean_turn, max_turn, curv_0..curv_{n_curv-1}]`` where
    ``curv`` is the turning angle vs normalized progress, or ``None`` for a degenerate path. Standardization
    (with training-set stats) happens in :meth:`TimingModel.standardize` / the trainer.
    """
    q = _clean_path(positions)
    if q is None or duration <= 0:
        return None
    step = np.linalg.norm(np.diff(q, axis=0), axis=1)
    arc = float(step.sum())
    if arc < 1e-4:
        return None
    u = np.concatenate([[0.0], np.cumsum(step)])
    us = np.linspace(0.0, arc, FEAT_RESAMPLE)
    qs = np.stack([np.interp(us, u, q[:, j]) for j in range(q.shape[1])], axis=1)  # arc-even resample
    ds = np.diff(qs, axis=0)
    ls = np.linalg.norm(ds, axis=1)
    tang = ds / np.clip(ls[:, None], 1e-9, None)
    dots = np.clip(np.sum(tang[1:] * tang[:-1], axis=1), -1.0, 1.0)
    turn = np.arccos(dots)  # turning angle at each interior point (discrete curvature proxy)

    chord = float(np.linalg.norm(q[-1] - q[0]))
    straightness = chord / arc
    avg_speed = arc / duration
    mean_turn = float(turn.mean()) if len(turn) else 0.0
    max_turn = float(turn.max()) if len(turn) else 0.0
    if len(turn):
        prog = (np.arange(len(turn)) + 0.5) / len(turn)
        curv = np.interp(np.linspace(0.0, 1.0, n_curv), prog, turn)
    else:
        curv = np.zeros(n_curv)
    return np.concatenate([[arc, straightness, avg_speed, mean_turn, max_turn], curv]).astype(np.float32)


class TimingNet(nn.Module):
    """Maps ``(lead_kind, trail_kind[, geom feats][, language])`` to a positive speed profile ``p(tau)``.

    The raw head output is passed through ``softplus`` (plus a small floor) so the profile is strictly
    positive -- a valid, monotone-integrable time law. Normalization to unit mean is done by the caller
    (:func:`normalize_profile`). ``n_feat``/``lang_dim`` of 0 disable those inputs.
    """

    def __init__(
        self,
        n_knots: int = N_KNOTS,
        n_kinds: int = len(EVENT_KINDS),
        kind_emb_dim: int = 8,
        n_feat: int = 0,
        lang_dim: int = 0,
        hidden: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_knots = n_knots
        self.n_feat = n_feat
        self.lang_dim = lang_dim
        self.lead_emb = nn.Embedding(n_kinds, kind_emb_dim)
        self.trail_emb = nn.Embedding(n_kinds, kind_emb_dim)
        in_dim = 2 * kind_emb_dim + n_feat + lang_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_knots),
        )

    def forward(
        self,
        lead_idx: torch.Tensor,
        trail_idx: torch.Tensor,
        feats: torch.Tensor | None = None,
        lang: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b = lead_idx.shape[0]
        parts = [self.lead_emb(lead_idx), self.trail_emb(trail_idx)]
        if self.n_feat > 0:
            parts.append(feats if feats is not None else torch.zeros(b, self.n_feat, device=lead_idx.device))
        if self.lang_dim > 0:
            parts.append(lang if lang is not None else torch.zeros(b, self.lang_dim, device=lead_idx.device))
        x = torch.cat(parts, dim=-1)
        return F.softplus(self.mlp(x)) + 1e-3  # strictly-positive speed profile [B, n_knots]


def normalize_profile(p: torch.Tensor) -> torch.Tensor:
    """Scale each profile to unit mean along the last axis (shape-only; the pace is set elsewhere)."""
    return p / p.mean(dim=-1, keepdim=True).clamp_min(1e-6)


class TimingModel:
    """Plan-time inference wrapper: load a checkpoint and return speed profiles for ``neural_blending``.

    A checkpoint is ``{"state_dict": ..., "meta": {...}}`` (written by ``train_timing``). ``meta`` records
    the architecture (``n_knots``, ``event_kinds``, ``kind_emb_dim``, ``hidden``), the geometry-feature
    config (``n_feat``, ``n_curv``, ``feat_mean``, ``feat_std``), and the language config (``use_language``,
    ``lang_dim``, ``lang_model``) so inference reconstructs the exact model and standardizes features the
    same way training did. The blender calls :meth:`features` (raw -> standardized geometry vector) and
    :meth:`profile_on` (profile resampled onto a dense ``tau`` grid).
    """

    def __init__(self, ckpt_path: str | Path | None = None, device: str = "cpu"):
        path = _resolve_ckpt(ckpt_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Timing checkpoint not found: {path}. Train one with "
                f"`python -m tiptop.networks.train_timing` or set blend_model_path in the config."
            )
        blob = torch.load(path, map_location=device, weights_only=False)
        meta = blob["meta"]
        self.meta = meta
        self.path = path
        self.device = device
        self.n_knots = int(meta["n_knots"])
        self.event_kinds = tuple(meta["event_kinds"])
        self.kind_to_idx = {k: i for i, k in enumerate(self.event_kinds)}
        self.n_feat = int(meta.get("n_feat", 0))
        self.n_curv = int(meta.get("n_curv", N_CURV))
        self.use_language = bool(meta.get("use_language", False))
        self.lang_dim = int(meta.get("lang_dim", 0))
        self.lang_model = meta.get("lang_model", DEFAULT_LANG_MODEL)
        self._feat_mean = np.asarray(meta.get("feat_mean", np.zeros(self.n_feat)), np.float64)
        self._feat_std = np.asarray(meta.get("feat_std", np.ones(self.n_feat)), np.float64)
        self.net = TimingNet(
            n_knots=self.n_knots,
            n_kinds=len(self.event_kinds),
            kind_emb_dim=int(meta.get("kind_emb_dim", 8)),
            n_feat=self.n_feat,
            lang_dim=self.lang_dim if self.use_language else 0,
            hidden=int(meta.get("hidden", 128)),
            dropout=0.0,
        )
        self.net.load_state_dict(blob["state_dict"])
        self.net.to(device).eval()
        self._tau_knots = tau_grid(self.n_knots)
        self._lang_encoder = None
        _log.info(
            "Loaded timing model %s (knots=%d, kinds=%s, geom_feats=%d, language=%s)",
            path.name, self.n_knots, self.event_kinds, self.n_feat, self.use_language,
        )

    def _kind_idx(self, kind: str) -> int:
        if kind not in self.kind_to_idx:
            _log.warning("Unknown stroke kind %r; falling back to 'rest'", kind)
            return self.kind_to_idx.get("rest", 0)
        return self.kind_to_idx[kind]

    def standardize(self, raw_feats: np.ndarray) -> np.ndarray:
        """Standardize a RAW geometry-feature vector with the checkpoint's training stats."""
        return ((np.asarray(raw_feats, np.float64) - self._feat_mean) / self._feat_std).astype(np.float32)

    def features(self, positions: np.ndarray, duration: float) -> np.ndarray | None:
        """Standardized geometry/pace features for a stroke's path (or None if the model uses none)."""
        if self.n_feat == 0:
            return None
        raw = stroke_features(positions, duration, self.n_curv)
        return None if raw is None else self.standardize(raw)

    @torch.no_grad()
    def predict_profile(
        self,
        lead_kind: str,
        trail_kind: str,
        feats: np.ndarray | None = None,
        lang_emb: np.ndarray | None = None,
    ) -> np.ndarray:
        """Unit-mean positive profile ``p`` on the knot grid for one stroke's conditioning."""
        lead = torch.tensor([self._kind_idx(lead_kind)], dtype=torch.long, device=self.device)
        trail = torch.tensor([self._kind_idx(trail_kind)], dtype=torch.long, device=self.device)
        ft = None
        if self.n_feat > 0 and feats is not None:
            ft = torch.as_tensor(np.asarray(feats, np.float32)[None], device=self.device)
        lg = None
        if self.use_language and lang_emb is not None:
            lg = torch.as_tensor(np.asarray(lang_emb, np.float32)[None], device=self.device)
        p = normalize_profile(self.net(lead, trail, ft, lg))[0]
        return p.cpu().numpy().astype(np.float64)

    def profile_on(
        self,
        tau_dense: np.ndarray,
        lead_kind: str,
        trail_kind: str,
        feats: np.ndarray | None = None,
        lang_emb: np.ndarray | None = None,
    ) -> np.ndarray:
        """The profile resampled onto an arbitrary (dense) ``tau`` grid -- what the blender consumes."""
        p = self.predict_profile(lead_kind, trail_kind, feats, lang_emb)
        return np.interp(np.asarray(tau_dense, dtype=np.float64), self._tau_knots, p)

    def embed_language(self, prompt: str | None) -> np.ndarray | None:
        """Frozen sentence embedding of a task prompt, or ``None`` if language conditioning is off."""
        if not self.use_language or not prompt:
            return None
        try:
            enc = self._get_lang_encoder()
            return np.asarray(enc(prompt), dtype=np.float32)
        except Exception:
            _log.exception("Language embedding failed; the timing model will run un-conditioned on task")
            return None

    def _get_lang_encoder(self):
        if self._lang_encoder is None:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(self.lang_model, device=self.device)
            self._lang_encoder = lambda s: model.encode([s], normalize_embeddings=True)[0]
        return self._lang_encoder


def _resolve_ckpt(ckpt_path: str | Path | None) -> Path:
    """Resolve a checkpoint path: default, absolute, or relative to the tiptop package dir."""
    if ckpt_path is None:
        return DEFAULT_CKPT
    p = Path(ckpt_path)
    if p.is_absolute():
        return p
    # Relative paths (e.g. "checkpoints/timing_net.pt" from a config) resolve under tiptop/tiptop.
    return (_PKG_DIR / p).resolve()
