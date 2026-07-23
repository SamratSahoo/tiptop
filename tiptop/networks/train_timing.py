"""Train the (geometry-only) stroke-timing network on DROID-extracted profiles.

Consumes the output of ``extract_stroke_timing`` (``checkpoints/timing_data/stroke_timing.npz``) and fits
:class:`~tiptop.networks.timing_net.TimingNet` to predict a stroke's unit-mean speed profile ``p(tau)``
from its GEOMETRY + PACE features (arc length, straightness, average speed, turning/curvature),
standardized with training-set stats baked into the checkpoint. Writes a checkpoint (``state_dict`` +
``meta``) that ``neural_blending`` / :class:`TimingModel` load at plan time; the default output is
``tiptop/tiptop/checkpoints/timing_net.pt``.

Loss: MSE between the (unit-mean) predicted and target profiles, plus a small second-difference
smoothness penalty. The absolute pace is NOT an output (profiles are unit-mean); pace only enters as an
INPUT feature (average speed), so the shape can adapt to fast-vs-slow strokes.

Run in the openpi venv (has torch)::

    openpi/.venv/bin/python -m tiptop.networks.train_timing
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from timing_net import N_CURV, N_KNOTS, TimingNet, _resolve_ckpt, normalize_profile  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "checkpoints" / "timing_data"
MERGED = DATA_DIR / "stroke_timing.npz"


def _smoothness(p: torch.Tensor) -> torch.Tensor:
    """Mean squared second difference along the knots -- penalizes a jagged profile."""
    d2 = p[:, 2:] - 2.0 * p[:, 1:-1] + p[:, :-2]
    return (d2 * d2).mean()


def train(args):
    if not MERGED.exists():
        raise FileNotFoundError(
            f"{MERGED} not found. Extract the DROID timing data first:\n"
            f"    python -m tiptop.networks.extract_stroke_timing --files 2   # smoke\n"
        )
    blob = np.load(MERGED)
    profiles = torch.as_tensor(blob["profiles"], dtype=torch.float32)  # [M, N_KNOTS], unit-mean targets
    if profiles.shape[1] != N_KNOTS:
        raise ValueError(f"profiles have {profiles.shape[1]} knots, expected {N_KNOTS}")
    if "feats" not in blob.files:
        raise KeyError("stroke_timing.npz has no 'feats' -- re-run extract_stroke_timing.")
    raw = np.asarray(blob["feats"], np.float64)
    feat_mean = raw.mean(0)
    feat_std = raw.std(0).clip(min=1e-6)
    feats = torch.as_tensor((raw - feat_mean) / feat_std, dtype=torch.float32)
    m, n_feat = feats.shape
    print(f"Loaded {m} strokes ({profiles.shape[1]} knots, {n_feat} standardized geometry features).")

    device = args.device
    net = TimingNet(n_knots=N_KNOTS, n_feat=n_feat, hidden=args.hidden, dropout=args.dropout).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(m, generator=g)
    n_val = max(1, int(0.1 * m))
    val_i, tr_i = perm[:n_val], perm[n_val:]

    def batches(idx, bs, shuffle):
        order = idx[torch.randperm(len(idx), generator=g)] if shuffle else idx
        for k in range(0, len(order), bs):
            yield order[k : k + bs]

    def run_batch(bi, train_mode):
        pred = normalize_profile(net(feats[bi].to(device)))
        loss = nn.functional.mse_loss(pred, profiles[bi].to(device)) + args.smooth_w * _smoothness(pred)
        if train_mode:
            opt.zero_grad()
            loss.backward()
            opt.step()
        return float(loss)

    best = float("inf")
    out_path = _resolve_ckpt(args.out)  # relative paths land under tiptop/tiptop (like TimingModel)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        net.train()
        tr_loss = np.mean([run_batch(bi, True) for bi in batches(tr_i, args.batch_size, True)])
        net.eval()
        with torch.no_grad():
            va_loss = np.mean([run_batch(bi, False) for bi in batches(val_i, args.batch_size, False)])
        tag = ""
        if va_loss < best:
            best = va_loss
            meta = {
                "n_knots": N_KNOTS,
                "hidden": args.hidden,
                "n_feat": n_feat,
                "n_curv": N_CURV,
                "feat_mean": feat_mean.tolist(),
                "feat_std": feat_std.tolist(),
            }
            torch.save({"state_dict": net.state_dict(), "meta": meta}, out_path)
            tag = "  *saved"
        if epoch % max(1, args.epochs // 20) == 0 or tag:
            print(f"epoch {epoch:4d}  train {tr_loss:.5f}  val {va_loss:.5f}{tag}", flush=True)

    print(f"\nBest val {best:.5f}. Checkpoint -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--smooth-w", type=float, default=1e-2, help="second-difference smoothness weight")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--out", type=str, default=None, help="checkpoint path (default: checkpoints/timing_net.pt)")
    train(ap.parse_args())
