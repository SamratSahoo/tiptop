"""Train the stroke-timing network on DROID-extracted profiles.

Consumes the output of ``extract_stroke_timing`` (``checkpoints/timing_data/stroke_timing.npz`` +
``tasks.json``) and fits :class:`~tiptop.networks.timing_net.TimingNet` to predict a stroke's unit-mean
speed profile ``p(tau)`` from:
  * its ``(lead_kind, trail_kind)`` boundary kinds (always),
  * GEOMETRY + PACE features (arc length, straightness, average speed, turning/curvature) unless
    ``--no-geometry``; standardized with training-set stats baked into the checkpoint, and
  * a frozen TASK-LANGUAGE embedding with ``--language``.
Writes a checkpoint (``state_dict`` + ``meta``) that ``neural_blending`` / :class:`TimingModel` load at
plan time; the default output is ``tiptop/tiptop/checkpoints/timing_net.pt``.

Loss: MSE between the (unit-mean) predicted and target profiles, plus a small second-difference
smoothness penalty. The absolute pace is NOT an output (profiles are unit-mean); pace only enters as an
INPUT feature (average speed), so the shape can adapt to fast-vs-slow strokes.

Run in the openpi venv (has torch; sentence-transformers for ``--language``)::

    openpi/.venv/bin/python -m tiptop.networks.train_timing                        # lead/trail + geometry
    openpi/.venv/bin/python -m tiptop.networks.train_timing --language             # + task language
    openpi/.venv/bin/python -m tiptop.networks.train_timing --no-geometry          # boundary-only baseline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from timing_net import (  # noqa: E402
    DEFAULT_LANG_MODEL,
    EVENT_KINDS,
    N_CURV,
    N_KNOTS,
    TimingNet,
    _resolve_ckpt,
    normalize_profile,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "checkpoints" / "timing_data"
MERGED = DATA_DIR / "stroke_timing.npz"
TASKS_JSON = DATA_DIR / "tasks.json"


def _smoothness(p: torch.Tensor) -> torch.Tensor:
    """Mean squared second difference along the knots -- penalizes a jagged profile."""
    d2 = p[:, 2:] - 2.0 * p[:, 1:-1] + p[:, :-2]
    return (d2 * d2).mean()


def _embed_tasks(task_idx: np.ndarray, lang_model: str) -> tuple[np.ndarray, int]:
    """Per-sample frozen sentence embedding of each stroke's task instruction (via tasks.json)."""
    if not TASKS_JSON.exists():
        raise FileNotFoundError(
            f"{TASKS_JSON} not found; re-run extract_stroke_timing (it writes the task text table) or "
            f"train without --language."
        )
    from sentence_transformers import SentenceTransformer

    tasks = {int(k): v for k, v in json.loads(TASKS_JSON.read_text()).items()}
    uniq = sorted(set(int(i) for i in task_idx))
    texts = [tasks.get(i, "") for i in uniq]
    model = SentenceTransformer(lang_model, device="cpu")  # GPU kernels mismatch this torch build
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
    lang_dim = emb.shape[1]
    lut = {i: emb[k] for k, i in enumerate(uniq)}
    per_sample = np.stack([lut[int(i)] for i in task_idx]).astype(np.float32)
    return per_sample, lang_dim


def train(args):
    if not MERGED.exists():
        raise FileNotFoundError(
            f"{MERGED} not found. Extract the DROID timing data first:\n"
            f"    python -m tiptop.networks.extract_stroke_timing --files 2   # smoke\n"
        )
    blob = np.load(MERGED)
    profiles = torch.as_tensor(blob["profiles"], dtype=torch.float32)  # [M, N_KNOTS], unit-mean
    lead = torch.as_tensor(blob["lead_idx"], dtype=torch.long)
    trail = torch.as_tensor(blob["trail_idx"], dtype=torch.long)
    m = profiles.shape[0]
    if profiles.shape[1] != N_KNOTS:
        raise ValueError(f"profiles have {profiles.shape[1]} knots, expected {N_KNOTS}")
    print(f"Loaded {m} strokes ({profiles.shape[1]} knots).")

    # Geometry / pace features (standardized), unless disabled.
    feats = None
    n_feat = 0
    feat_mean = feat_std = None
    if not args.no_geometry and "feats" in blob.files:
        raw = np.asarray(blob["feats"], np.float64)
        feat_mean = raw.mean(0)
        feat_std = raw.std(0).clip(min=1e-6)
        feats = torch.as_tensor((raw - feat_mean) / feat_std, dtype=torch.float32)
        n_feat = feats.shape[1]
        print(f"Geometry conditioning ON: {n_feat} features standardized.")
    elif not args.no_geometry:
        print("WARNING: --geometry requested but the data has no 'feats' (re-extract); training without it.")

    # Task-language features, if requested.
    lang = None
    lang_dim = 0
    if args.language:
        per_sample, lang_dim = _embed_tasks(blob["task_idx"], args.lang_model)
        lang = torch.as_tensor(per_sample, dtype=torch.float32)
        print(f"Language conditioning ON: {args.lang_model} ({lang_dim}-D).")

    device = args.device
    net = TimingNet(
        n_knots=N_KNOTS, n_kinds=len(EVENT_KINDS), kind_emb_dim=args.kind_emb_dim,
        n_feat=n_feat, lang_dim=lang_dim, hidden=args.hidden, dropout=args.dropout,
    ).to(device)
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
        ft = feats[bi].to(device) if feats is not None else None
        lg = lang[bi].to(device) if lang is not None else None
        pred = normalize_profile(net(lead[bi].to(device), trail[bi].to(device), ft, lg))
        tgt = profiles[bi].to(device)
        loss = nn.functional.mse_loss(pred, tgt) + args.smooth_w * _smoothness(pred)
        if train_mode:
            opt.zero_grad()
            loss.backward()
            opt.step()
        return float(loss)

    best = float("inf")
    # Resolve the same way TimingModel does: relative paths (e.g. "checkpoints/timing_net.pt", as in the
    # config's blend_model_path) land under the tiptop package dir, not the current working directory.
    out_path = _resolve_ckpt(args.out)
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
                "event_kinds": list(EVENT_KINDS),
                "kind_emb_dim": args.kind_emb_dim,
                "hidden": args.hidden,
                "n_feat": n_feat,
                "n_curv": N_CURV,
                "feat_mean": (feat_mean.tolist() if feat_mean is not None else []),
                "feat_std": (feat_std.tolist() if feat_std is not None else []),
                "use_language": bool(args.language),
                "lang_dim": lang_dim,
                "lang_model": args.lang_model if args.language else None,
            }
            torch.save({"state_dict": net.state_dict(), "meta": meta}, out_path)
            tag = "  *saved"
        if epoch % max(1, args.epochs // 20) == 0 or tag:
            print(f"epoch {epoch:4d}  train {tr_loss:.5f}  val {va_loss:.5f}{tag}", flush=True)

    print(f"\nBest val {best:.5f}. Checkpoint -> {out_path}  (n_feat={n_feat}, language={bool(args.language)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--kind-emb-dim", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--smooth-w", type=float, default=1e-2, help="second-difference smoothness weight")
    ap.add_argument("--no-geometry", action="store_true", help="disable geometry/pace conditioning")
    ap.add_argument("--language", action="store_true", help="condition on a frozen task-language embedding")
    ap.add_argument("--lang-model", type=str, default=DEFAULT_LANG_MODEL)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--out", type=str, default=None, help="checkpoint path (default: checkpoints/timing_net.pt)")
    train(ap.parse_args())
