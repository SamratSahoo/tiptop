"""Evaluate the trained (geometry-only) stroke-timing model on HELD-OUT DROID data.

The trainer's own 90/10 split is stroke-level (strokes from one episode can land on both sides), so its
val number is optimistic. This script streams a DISJOINT set of DROID files (never used in training) and
reports the honest skill:

  * held-out profile MSE of the model vs (a) a flat profile and (b) a single unconditional mean profile,
    as a skill score ``1 - MSE_model / MSE_baseline`` (>0 = better) plus a per-stroke win rate;
  * the mean predicted profile vs the held-out mean (a sanity check that the shape generalizes).

Also carries the exact openpi NON-IDLE predicate (:func:`nonidle_idle_runs`) as a reusable constraint
check for real blended episodes.

Run in the openpi venv (streams DROID; no cuRobo needed)::

    openpi/.venv/bin/python -m tiptop.networks.eval_timing --start 40 --files 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from timing_net import N_KNOTS, TimingModel  # noqa: E402
from extract_stroke_timing import COLS, DROID_REPO, MERGED, reduce_file  # noqa: E402


def _unit_mean(p: np.ndarray) -> np.ndarray:
    return p / p.mean(axis=-1, keepdims=True).clip(1e-6)


def _load_heldout(start: int, n_files: int) -> dict:
    """Stream files [start, start+n_files) and return their strokes (profiles, feats)."""
    import os

    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    fs = HfFileSystem()
    out = {k: [] for k in ("profiles", "feats")}
    for fi in range(start, start + n_files):
        path = f"datasets/{DROID_REPO}/data/chunk-000/file-{fi:03d}.parquet"
        with fs.open(path, "rb") as f:
            tbl = pq.read_table(f, columns=COLS)
        agg = reduce_file(tbl)
        del tbl
        for k in out:
            out[k].append(agg[k])
        print(f"  held-out file-{fi:03d}: {len(agg['profiles'])} strokes", flush=True)
    return {k: np.concatenate(v, 0) for k, v in out.items()}


def nonidle_idle_runs(action_jv: np.ndarray, thresh: float = 1e-3, min_idle_len: int = 7) -> int:
    """Number of openpi-filterable idle runs in an action.joint_velocity sequence [T, 7].

    Mirrors compute_droid_nonidle_ranges exactly: a frame is idle iff ALL 7 joints' commanded velocity
    changed by < ``thresh`` from the previous frame; a run of >= ``min_idle_len`` idle frames is dropped by
    the training filter. Returns how many such runs exist -- 0 means every frame survives. Point this at a
    real (15 Hz) exported episode's action.joint_velocity to confirm neural blending keeps the
    gripper-adjacent frames non-idle.
    """
    jv = np.asarray(action_jv, np.float64)
    is_idle = np.zeros(len(jv), bool)
    is_idle[1:] = np.all(np.abs(np.diff(jv, axis=0)) < thresh, axis=1)
    padded = np.concatenate([[False], is_idle, [False]])
    edges = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return int(np.sum((ends - starts) >= min_idle_len))


def evaluate(args):
    if not MERGED.exists():
        raise FileNotFoundError(f"{MERGED} not found (train first).")
    train = np.load(MERGED)
    global_mean = _unit_mean(train["profiles"].mean(0))  # unconditional baseline

    model = TimingModel(args.model)
    ho = _load_heldout(args.start, args.files)
    P = ho["profiles"]
    m = len(P)
    print(f"\nHeld-out: {m} strokes from files [{args.start}, {args.start + args.files}). "
          f"Model: geometry-only ({model.n_feat} feats).")

    import torch

    ft = torch.as_tensor(model.standardize(ho["feats"]), dtype=torch.float32)
    with torch.no_grad():
        from timing_net import normalize_profile

        pred = normalize_profile(model.net(ft)).cpu().numpy()

    def mse(a):
        return float(((a - P) ** 2).mean())

    mse_model = mse(pred)
    mse_global = mse(global_mean[None])
    mse_flat = mse(np.ones((1, N_KNOTS)))
    win = float((((pred - P) ** 2).mean(1) < ((global_mean[None] - P) ** 2).mean(1)).mean())

    print("\n=== Skill (held-out profile MSE; lower is better) ===")
    print(f"  model (geometry-conditioned) : {mse_model:.4f}")
    print(f"  baseline flat profile        : {mse_flat:.4f}   skill = {1 - mse_model / mse_flat:+.3f}")
    print(f"  baseline unconditional mean  : {mse_global:.4f}   skill = {1 - mse_model / mse_global:+.3f}")
    print(f"  per-stroke win rate vs mean  : {win:.1%}")

    print("\nNote: constraint eval (non-idle / vel-accel / overshoot) belongs on REAL exported episodes.")
    print("      nonidle_idle_runs(episode_action_joint_velocity) must return 0 on a neural-blended,")
    print("      15 Hz-exported episode's gripper-adjacent frames.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="checkpoints/timing_net.pt")
    ap.add_argument("--start", type=int, default=40, help="first held-out file index (disjoint from training)")
    ap.add_argument("--files", type=int, default=4, help="number of held-out files to stream")
    evaluate(ap.parse_args())
