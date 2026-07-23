"""A small network that learns the *timing* of a manipulation stroke from DROID human teleop.

Context
-------
``trajectory_blending`` turns a cuTAMP plan's stop-and-go segments into one continuous stroke per
operation by *path-velocity decomposition*: a smoothing spline fixes the GEOMETRY (the collision-checked
path, unchanged) and a hand-tuned quintic / asymmetric TIME LAW sweeps along it. This module replaces
only that time law with a learned one -- a network that predicts the speed-vs-time profile a human would
use, trained on real DROID strokes. The geometry, the robot vel/accel caps, the endpoint pinning and the
non-idle boundary speed are all still enforced analytically in ``neural_blending`` (we learn the *style*,
not the constraints).

What the network predicts
-------------------------
For one stroke it outputs ``p(tau)`` on a fixed grid of ``N_KNOTS`` points -- the (strictly positive,
unit-mean) joint-space arc-length SPEED as a function of normalized time ``tau in [0, 1]``. This is
exactly the ``p(tau)`` the analytic ``_asymmetric_stroke`` builds by hand, so a learned ``p`` is a
drop-in: integrate it to the time law ``h(tau)`` and sample. Being unit-mean it carries only the SHAPE
(where along the reach the arm speeds up / slows down); the absolute pace still comes from the plan's
wall-clock, and the boundary levels the openpi non-idle filter needs are enforced on top in
``neural_blending._neural_stroke``.

Conditioning (the input)
------------------------
The profile is conditioned ONLY on transferable, sampling-invariant GEOMETRY + PACE features of the path
being timed (:func:`stroke_features`): total arc length, straightness (chord/arc), average speed (= pace =
arc / duration), mean/max turning, and a short turning-vs-progress curvature profile. These let the profile
depend on WHICH reach it is (a 5 cm nudge vs a 60 cm curved cross-table reach are timed differently) -- the
bulk of stroke-to-stroke variance. They are standardized with training-set stats baked into the checkpoint.
The network only *reads* geometry; it never moves waypoints (the collision-checked path is untouched), so
the path-velocity split holds. (Boundary kinds and task language were ablated out: on held-out DROID they
add only ~4% MSE over geometry alone, not worth the extra inputs / the plan-time language dependency.)

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
    """Maps a stroke's GEOMETRY/PACE feature vector to a positive speed profile ``p(tau)`` over N_KNOTS.

    The raw head output is passed through ``softplus`` (plus a small floor) so the profile is strictly
    positive -- a valid, monotone-integrable time law. Normalization to unit mean is done by the caller
    (:func:`normalize_profile`).
    """

    def __init__(self, n_knots: int = N_KNOTS, n_feat: int = N_FEAT, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.n_knots = n_knots
        self.n_feat = n_feat
        self.mlp = nn.Sequential(
            nn.Linear(n_feat, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_knots),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.mlp(feats)) + 1e-3  # strictly-positive speed profile [B, n_knots]


def normalize_profile(p: torch.Tensor) -> torch.Tensor:
    """Scale each profile to unit mean along the last axis (shape-only; the pace is set elsewhere)."""
    return p / p.mean(dim=-1, keepdim=True).clamp_min(1e-6)


class TimingModel:
    """Plan-time inference wrapper: load a checkpoint and return speed profiles for ``neural_blending``.

    A checkpoint is ``{"state_dict": ..., "meta": {...}}`` (written by ``train_timing``). ``meta`` records
    the architecture (``n_knots``, ``hidden``) and the geometry-feature config (``n_feat``, ``n_curv``,
    ``feat_mean``, ``feat_std``) so inference reconstructs the exact model and standardizes features the
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
        self.n_feat = int(meta["n_feat"])
        self.n_curv = int(meta.get("n_curv", N_CURV))
        self._feat_mean = np.asarray(meta["feat_mean"], np.float64)
        self._feat_std = np.asarray(meta["feat_std"], np.float64)
        self.net = TimingNet(n_knots=self.n_knots, n_feat=self.n_feat, hidden=int(meta.get("hidden", 128)), dropout=0.0)
        self.net.load_state_dict(blob["state_dict"])
        self.net.to(device).eval()
        self._tau_knots = tau_grid(self.n_knots)
        _log.info("Loaded timing model %s (knots=%d, geom_feats=%d)", path.name, self.n_knots, self.n_feat)

    def standardize(self, raw_feats: np.ndarray) -> np.ndarray:
        """Standardize a RAW geometry-feature vector (or [M, n_feat] batch) with the training stats."""
        return ((np.asarray(raw_feats, np.float64) - self._feat_mean) / self._feat_std).astype(np.float32)

    def features(self, positions: np.ndarray, duration: float) -> np.ndarray | None:
        """Standardized geometry/pace features for a stroke's path (None if the path is degenerate)."""
        raw = stroke_features(positions, duration, self.n_curv)
        return None if raw is None else self.standardize(raw)

    @torch.no_grad()
    def predict_profile(self, feats: np.ndarray | None) -> np.ndarray:
        """Unit-mean positive profile ``p`` on the knot grid for one stroke's standardized features.

        ``feats`` None (a degenerate path) falls back to the feature mean (a standardized zero vector).
        """
        f = np.zeros(self.n_feat, np.float32) if feats is None else np.asarray(feats, np.float32)
        ft = torch.as_tensor(f[None], device=self.device)
        p = normalize_profile(self.net(ft))[0]
        return p.cpu().numpy().astype(np.float64)

    def profile_on(self, tau_dense: np.ndarray, feats: np.ndarray | None) -> np.ndarray:
        """The profile resampled onto an arbitrary (dense) ``tau`` grid -- what the blender consumes."""
        p = self.predict_profile(feats)
        return np.interp(np.asarray(tau_dense, dtype=np.float64), self._tau_knots, p)


def _resolve_ckpt(ckpt_path: str | Path | None) -> Path:
    """Resolve a checkpoint path: default, absolute, or relative to the tiptop package dir."""
    if ckpt_path is None:
        return DEFAULT_CKPT
    p = Path(ckpt_path)
    if p.is_absolute():
        return p
    # Relative paths (e.g. "checkpoints/timing_net.pt" from a config) resolve under tiptop/tiptop.
    return (_PKG_DIR / p).resolve()
