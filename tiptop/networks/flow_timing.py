"""Conditional flow-matching blender: generate a whole human stroke, no analytic time law or spline.

Where ``timing_net`` learns only a deterministic speed profile (and, being MSE-trained, collapses the
different teleoperator styles to their mean), this module learns the ENTIRE blend generatively -- so
sampling different noise yields different human-like realizations of the same path (decelerate-into-grasp
vs constant-velocity), expressing the multimodality a regressor averages away.

Representation (endpoint-anchored deviation)
--------------------------------------------
A stroke is ``x(tau)`` = start-centered joint positions over normalized TIME, ``x[0]=0`` and ``x[-1]=e``
(the net displacement / endpoint). We generate the DEVIATION from the straight-line motion between the
two endpoints:

    base(tau) = tau * e            (constant-velocity straight line, 0 -> e)
    delta     = x - base           (delta[0] = delta[-1] = 0 by construction)

The flow models ``delta``; at sampling we add ``base`` back and pin ``x[0]=0, x[-1]=e`` EXACTLY. This
hard-anchors both endpoints (no grasp overshoot), makes the target smaller/smoother, and -- crucially --
gives the model a fixed endpoint to decelerate into (the earlier position-space model captured accel from
the anchored start but not decel into the un-anchored end). The conditioning is the timing-stripped PATH
``c`` (positions over ARC LENGTH) plus the endpoint ``e`` -- conditioning on an arc-length path cannot leak
the target timing.

Architecture
------------
``TrajFlow`` is a small 1-D temporal U-Net over the time axis (T=32 -> 16 -> 8 -> 16 -> 32 with skip
connections), FiLM-conditioned on ``(encode(c), e, timestep(t))``. The temporal convolutions give the
smoothness inductive bias the earlier MLP-over-flattened-positions lacked (its samples were ~5x jerkier
than human).

Training uses conditional (rectified) flow matching on ``delta``: draw ``d0 ~ N(0,I)``, ``d1 = data``,
``t ~ U(0,1)``, ``dt = (1-t) d0 + t d1``, regress ``v(dt, t, c, e)`` onto ``d1 - d0``. Sampling integrates
``dx/dt = v`` from Gaussian noise over ``t: 0->1``. Training: ``train_flow.py``; extraction:
``extract_stroke_flow.py``. NOTE: this is the generative model; wiring its samples into a cuTAMP plan
(vel/accel caps, non-idle) is a separate integration step.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_log = logging.getLogger(__name__)

T_TIME = 32
S_ARC = 32
DOF = 7

_PKG_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CKPT = _PKG_DIR / "checkpoints" / "flow_net.pt"


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding of the flow time t in [0,1] -> [B, dim]."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / max(half - 1, 1))
    args = t[:, None] * freqs[None] * 100.0
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class _FiLM(nn.Module):
    """Per-channel scale/shift of a [B,C,L] feature map from a conditioning vector [B,cond_dim]."""

    def __init__(self, cond_dim: int, ch: int):
        super().__init__()
        self.lin = nn.Linear(cond_dim, 2 * ch)

    def forward(self, h: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        s, b = self.lin(g).chunk(2, dim=-1)
        return h * (1.0 + s[..., None]) + b[..., None]


class _ResBlock(nn.Module):
    """Two 1-D convs with a FiLM in between and a residual connection."""

    def __init__(self, cin: int, cout: int, cond_dim: int, k: int = 5):
        super().__init__()
        self.conv1 = nn.Conv1d(cin, cout, k, padding=k // 2)
        self.film = _FiLM(cond_dim, cout)
        self.conv2 = nn.Conv1d(cout, cout, k, padding=k // 2)
        self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.film(self.conv1(x), g))
        h = F.silu(self.conv2(h))
        return h + self.skip(x)


class TrajFlow(nn.Module):
    """1-D temporal U-Net velocity field v(delta_t, t, c, e) for conditional flow matching."""

    def __init__(self, dof: int = DOF, cond_dim: int = 128, t_dim: int = 64, base_ch: int = 64):
        super().__init__()
        self.t_dim = t_dim
        ch0, ch1 = base_ch, base_ch * 2
        # Condition encoder over the path c [B,dof,S] -> global vector.
        self.c_enc = nn.Sequential(
            nn.Conv1d(dof, 64, 5, padding=2), nn.SiLU(),
            nn.Conv1d(64, 64, 5, padding=2), nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.g_mlp = nn.Sequential(nn.Linear(64 + dof + t_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        # U-Net over the time axis.
        self.in_conv = nn.Conv1d(dof, ch0, 5, padding=2)
        self.d1 = _ResBlock(ch0, ch0, cond_dim)
        self.down1 = nn.Conv1d(ch0, ch0, 4, stride=2, padding=1)
        self.d2 = _ResBlock(ch0, ch1, cond_dim)
        self.down2 = nn.Conv1d(ch1, ch1, 4, stride=2, padding=1)
        self.mid = _ResBlock(ch1, ch1, cond_dim)
        self.up2 = nn.ConvTranspose1d(ch1, ch1, 4, stride=2, padding=1)
        self.u2 = _ResBlock(ch1 + ch1, ch1, cond_dim)
        self.up1 = nn.ConvTranspose1d(ch1, ch0, 4, stride=2, padding=1)
        self.u1 = _ResBlock(ch0 + ch0, ch0, cond_dim)
        self.out_conv = nn.Conv1d(ch0, dof, 5, padding=2)

    def forward(self, delta_t: torch.Tensor, t: torch.Tensor, c: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        xt = delta_t.transpose(1, 2)          # [B,dof,T]
        cg = self.c_enc(c.transpose(1, 2)).squeeze(-1)  # [B,64]
        g = self.g_mlp(torch.cat([cg, e, timestep_embedding(t, self.t_dim)], dim=-1))
        h = self.in_conv(xt)
        h1 = self.d1(h, g)
        h2 = self.d2(self.down1(h1), g)
        h = self.mid(self.down2(h2), g)
        h = self.u2(torch.cat([self.up2(h), h2], dim=1), g)
        h = self.u1(torch.cat([self.up1(h), h1], dim=1), g)
        return self.out_conv(h).transpose(1, 2)  # [B,T,dof]


def cfm_loss(net: TrajFlow, d1: torch.Tensor, c: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    """Conditional flow-matching loss on the endpoint-anchored deviation delta."""
    d0 = torch.randn_like(d1)
    t = torch.rand(d1.shape[0], device=d1.device)
    dt = (1.0 - t)[:, None, None] * d0 + t[:, None, None] * d1
    return F.mse_loss(net(dt, t, c, e), d1 - d0)


def _resample_arc(q: np.ndarray, n: int) -> np.ndarray | None:
    step = np.linalg.norm(np.diff(np.asarray(q, np.float64), axis=0), axis=1)
    arc = step.sum()
    if arc < 1e-6:
        return None
    u = np.concatenate([[0.0], np.cumsum(step)])
    dst = np.linspace(0.0, arc, n)
    return np.stack([np.interp(dst, u, q[:, j]) for j in range(q.shape[1])], axis=1)


class FlowModel:
    """Plan-time wrapper: load a checkpoint and SAMPLE endpoint-anchored human-like strokes given a path."""

    def __init__(self, ckpt_path: str | Path | None = None, device: str = "cpu"):
        path = Path(ckpt_path) if ckpt_path else DEFAULT_CKPT
        if not path.is_absolute():
            path = (_PKG_DIR / path).resolve()
        blob = torch.load(path, map_location=device, weights_only=False)
        meta = blob["meta"]
        self.meta = meta
        self.device = device
        self.T, self.S, self.dof = int(meta["t_time"]), int(meta["s_arc"]), int(meta["dof"])
        self.c_std, self.delta_std, self.e_std = float(meta["c_std"]), float(meta["delta_std"]), float(meta["e_std"])
        self.net = TrajFlow(self.dof, meta["cond_dim"], meta["t_dim"], meta["base_ch"])
        self.net.load_state_dict(blob["state_dict"])
        self.net.to(device).eval()
        self._tau = np.linspace(0.0, 1.0, self.T)[:, None]
        _log.info("Loaded flow model %s (T=%d S=%d dof=%d)", path.name, self.T, self.S, self.dof)

    def _cond(self, path_positions: np.ndarray):
        """Raw stroke path [N,7] -> (c_norm [S,dof], e [dof]) or None if degenerate."""
        c = _resample_arc(np.asarray(path_positions, np.float64), self.S)
        if c is None:
            return None
        c = c - c[0:1]
        return (c / self.c_std).astype(np.float32), c[-1].astype(np.float32)  # e = net displacement

    @torch.no_grad()
    def sample(self, path_positions: np.ndarray, n_samples: int = 1, n_steps: int = 60) -> np.ndarray | None:
        """Sample ``n_samples`` strokes for a raw path -> [n_samples, T, dof] (start-centered, endpoint-pinned)."""
        cond = self._cond(path_positions)
        if cond is None:
            return None
        c_norm, e = cond
        c = torch.as_tensor(c_norm[None], device=self.device).repeat(n_samples, 1, 1)
        e_t = torch.as_tensor((e / self.e_std)[None], device=self.device).repeat(n_samples, 1)
        d = torch.randn(n_samples, self.T, self.dof, device=self.device)
        ts = torch.linspace(0.0, 1.0, n_steps + 1, device=self.device)
        for i in range(n_steps):
            d = d + self.net(d, ts[i].repeat(n_samples), c, e_t) * (ts[i + 1] - ts[i])
        delta = d.cpu().numpy() * self.delta_std
        x = self._tau[None] * e[None, None, :] + delta       # base + delta
        x[:, 0] = 0.0
        x[:, -1] = e                                          # pin both endpoints exactly
        return x

    @torch.no_grad()
    def sample_batch(self, arc_paths: np.ndarray, n_steps: int = 60) -> np.ndarray:
        """One trajectory for each of B distinct arc-length paths [B,S,7] -> [B,T,dof] (endpoint-pinned)."""
        c = np.asarray(arc_paths, np.float64)
        c = c - c[:, 0:1]
        e = c[:, -1].astype(np.float32)                      # [B,7] endpoints
        b = len(c)
        ct = torch.as_tensor((c / self.c_std).astype(np.float32), device=self.device)
        et = torch.as_tensor((e / self.e_std).astype(np.float32), device=self.device)
        d = torch.randn(b, self.T, self.dof, device=self.device)
        ts = torch.linspace(0.0, 1.0, n_steps + 1, device=self.device)
        for i in range(n_steps):
            d = d + self.net(d, ts[i].repeat(b), ct, et) * (ts[i + 1] - ts[i])
        delta = d.cpu().numpy() * self.delta_std
        x = self._tau[None] * e[:, None, :] + delta
        x[:, 0] = 0.0
        x[:, -1] = e
        return x
