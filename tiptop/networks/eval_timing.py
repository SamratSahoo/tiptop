"""Evaluate the trained stroke-timing model on HELD-OUT DROID data.

The trainer's own 90/10 split is stroke-level (strokes from one episode can land on both sides), so its
val number is optimistic. This script streams a DISJOINT set of DROID files (never used in training) and
asks the honest questions:

  1. Skill vs baselines -- does the model's conditioning reduce held-out profile error relative to
     (a) a flat profile, (b) a single unconditional mean profile, and (c) the ``(lead, trail)`` conditional
     mean? Baseline (c) is the key one: beating it is exactly the value the GEOMETRY + LANGUAGE inputs add
     ON TOP of the boundary kinds. Reported as skill ``1 - MSE_model / MSE_baseline`` (>0 = better) and a
     per-stroke win rate.
  2. Generalization of the mean shapes -- does the model's mean predicted profile per (lead, trail) still
     match the held-out conditional mean (correlation)?

Whatever inputs the checkpoint was trained with (geometry, language) are provided at eval time from the
held-out strokes' own features / task text, so the comparison is fair. Also carries the exact openpi
NON-IDLE predicate (:func:`nonidle_idle_runs`) as a reusable constraint check for real blended episodes.

Run in the openpi venv (streams DROID; sentence-transformers only if the model uses language)::

    openpi/.venv/bin/python -m tiptop.networks.eval_timing --start 40 --files 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from timing_net import KIND_TO_IDX, N_KNOTS, TimingModel, normalize_profile  # noqa: E402
from extract_stroke_timing import COLS, DROID_REPO, MERGED, fetch_tasks, reduce_file  # noqa: E402

KINDS = list(KIND_TO_IDX)


def _unit_mean(p: np.ndarray) -> np.ndarray:
    return p / p.mean(axis=-1, keepdims=True).clip(1e-6)


def _load_heldout(start: int, n_files: int) -> dict:
    """Stream files [start, start+n_files) and return their strokes (profiles, feats, lead, trail, task)."""
    import os

    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    fs = HfFileSystem()
    out = {k: [] for k in ("profiles", "feats", "lead_idx", "trail_idx", "task_idx")}
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


def _heldout_lang(task_idx: np.ndarray, lang_model: str) -> np.ndarray:
    """Per-stroke frozen sentence embedding for held-out task indices (fetches DROID's task table)."""
    from sentence_transformers import SentenceTransformer

    tasks = fetch_tasks()
    uniq = sorted(set(int(i) for i in task_idx))
    texts = [tasks.get(i, "") for i in uniq]
    enc = SentenceTransformer(lang_model, device="cpu")
    emb = enc.encode(texts, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
    lut = {i: emb[k] for k, i in enumerate(uniq)}
    return np.stack([lut[int(i)] for i in task_idx]).astype(np.float32)


def evaluate(args):
    if not MERGED.exists():
        raise FileNotFoundError(f"{MERGED} not found (train first).")
    train = np.load(MERGED)
    P_tr, lead_tr, trail_tr = train["profiles"], train["lead_idx"], train["trail_idx"]
    global_mean = _unit_mean(P_tr.mean(0))  # unconditional baseline
    # (lead, trail) conditional-mean baseline (what a boundary-only model would predict).
    cond_mean = {}
    for li in range(3):
        for ti in range(3):
            mm = (lead_tr == li) & (trail_tr == ti)
            if mm.sum():
                cond_mean[(li, ti)] = _unit_mean(P_tr[mm].mean(0))

    model = TimingModel(args.model)
    ho = _load_heldout(args.start, args.files)
    P, lead, trail = ho["profiles"], ho["lead_idx"], ho["trail_idx"]
    m = len(P)
    print(f"\nHeld-out: {m} strokes from files [{args.start}, {args.start + args.files}).")
    print(f"Model inputs: lead/trail always; geometry={model.n_feat > 0} ({model.n_feat} feats); "
          f"language={model.use_language}")

    # Batched model prediction with whatever inputs the checkpoint uses.
    lead_t = torch.as_tensor(lead, dtype=torch.long)
    trail_t = torch.as_tensor(trail, dtype=torch.long)
    ft = torch.as_tensor(model.standardize(ho["feats"]), dtype=torch.float32) if model.n_feat > 0 else None
    lg = None
    if model.use_language:
        lg = torch.as_tensor(_heldout_lang(ho["task_idx"], model.lang_model), dtype=torch.float32)
    with torch.no_grad():
        pred = normalize_profile(model.net(lead_t, trail_t, ft, lg)).cpu().numpy()

    def mse(a):
        return float(((a - P) ** 2).mean())

    condm = np.stack([cond_mean[(int(lead[i]), int(trail[i]))] for i in range(m)])
    mse_model = mse(pred)
    mse_condm = mse(condm)
    mse_global = mse(global_mean[None])
    mse_flat = mse(np.ones((1, N_KNOTS)))
    err_model = ((pred - P) ** 2).mean(1)
    win_vs_global = float((err_model < ((global_mean[None] - P) ** 2).mean(1)).mean())
    win_vs_condm = float((err_model < ((condm - P) ** 2).mean(1)).mean())

    print("\n=== Skill (held-out profile MSE; lower is better) ===")
    print(f"  model                              : {mse_model:.4f}")
    print(f"  baseline flat profile              : {mse_flat:.4f}   skill = {1 - mse_model / mse_flat:+.3f}")
    print(f"  baseline unconditional mean        : {mse_global:.4f}   skill = {1 - mse_model / mse_global:+.3f}")
    print(f"  baseline (lead,trail) cond. mean   : {mse_condm:.4f}   skill = {1 - mse_model / mse_condm:+.3f}"
          f"   <-- value of GEOMETRY+LANGUAGE beyond boundary kinds")
    print(f"  win rate vs uncond. mean           : {win_vs_global:.1%}")
    print(f"  win rate vs (lead,trail) cond.mean : {win_vs_condm:.1%}")

    print("\n=== Mean predicted shape per kind vs held-out conditional mean ===")
    for li in range(3):
        for ti in range(3):
            mask = (lead == li) & (trail == ti)
            n = int(mask.sum())
            if n < 30:
                continue
            emp = _unit_mean(P[mask].mean(0))
            mp = _unit_mean(pred[mask].mean(0))
            corr = float(np.corrcoef(emp, mp)[0, 1])
            print(f"  {KINDS[li]:7}->{KINDS[ti]:7} n={n:5d}  corr={corr:.3f}  end(pred/emp)={mp[-1]:.2f}/{emp[-1]:.2f}")

    print("\nNote: constraint eval (non-idle / vel-accel / overshoot) belongs on REAL exported episodes.")
    print("      nonidle_idle_runs(episode_action_joint_velocity) must return 0 on a neural-blended,")
    print("      15 Hz-exported episode's gripper-adjacent frames.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="checkpoints/timing_net.pt")
    ap.add_argument("--start", type=int, default=40, help="first held-out file index (disjoint from training)")
    ap.add_argument("--files", type=int, default=4, help="number of held-out files to stream")
    evaluate(ap.parse_args())
