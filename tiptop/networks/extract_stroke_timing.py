"""Extract per-stroke TIMING targets from DROID by STREAMING it (never downloaded).

The timing network learns the human speed-vs-time profile of a manipulation *stroke* -- the motion
between two gripper events. This script produces its training set: it streams DROID 1.0.1 from the Hub
(column-projected parquet over ``HfFileSystem`` -- the video columns, the vast bulk, are never touched),
segments every episode at its gripper-command edges into strokes, and for each stroke saves a compact
record: the unit-mean speed profile ``p(tau)`` (the exact quantity :mod:`neural_blending` consumes),
the boundary kinds ``(lead_kind, trail_kind)`` in {rest, grasp, release}, and the task index/text (for
the optional language conditioning). Each parquet file is reduced to a few thousand small records and
then dropped; the 27.6 M frames never land on disk.

Segmentation mirrors ``analysis2`` exactly (Schmitt trigger + min dwell + a 4-frame actuator lag, scored
F1=0.944 on DROID) so a stroke here is delimited the SAME way a cuTAMP operation is delimited at plan
time (a gripper close = grasp, a gripper open = release) -- that correspondence is what lets the model
transfer. The episode boundaries themselves (start / end) are ``rest`` strokes.

Run in the openpi venv (has ``datasets`` / ``pyarrow`` / ``torch``), like the other DROID streamers::

    openpi/.venv/bin/python -m tiptop.networks.extract_stroke_timing --files 2      # smoke test
    openpi/.venv/bin/python tiptop/tiptop/networks/extract_stroke_timing.py         # full stream

Output: per-file shards under ``checkpoints/timing_data/`` plus a merged ``stroke_timing.npz`` and a
``tasks.json`` (task_index -> instruction). ``train_timing`` consumes those. Resumable: a finished shard
is skipped on re-run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Import the shared profile helper + label vocab straight from the model file (no cuRobo / tiptop-package
# import needed, so this runs in the openpi venv). timing_net.py imports only numpy/torch.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from timing_net import KIND_TO_IDX, N_FEAT, N_KNOTS, stroke_features, stroke_speed_profile  # noqa: E402

# --------------------------------------------------------------------------- #
# DROID streaming config (mirrors analysis2/config.py)                          #
# --------------------------------------------------------------------------- #
DROID_REPO = "lerobot/droid_1.0.1"
DROID_N_FILES = 86  # meta/episodes references file-000..file-085 (all 95,658 episodes)
DROID_FPS = 15.0
DT = 1.0 / DROID_FPS
COLS = ["observation.state.joint_position", "action", "episode_index", "task_index"]
ACT_GRIPPER_COL = 7  # action[:, 7] is the commanded gripper in [0, 1] (0 = open, 1 = closed)

# Gripper-event detection (identical to analysis2/config.py -- the source of truth for these constants).
GRIP_LO, GRIP_HI = 0.20, 0.60
GRIP_MIN_DWELL = 5
GRIP_LAG = 4

# Stroke filter: a usable stroke has enough frames to define a profile and real net motion.
MIN_STROKE_LEN = 6            # frames (>= N_KNOTS is not required; we interpolate)
MIN_STROKE_ARC = 5e-3         # rad, total joint-space path length

OUT_DIR = Path(__file__).resolve().parents[1] / "checkpoints" / "timing_data"
SHARD_DIR = OUT_DIR / "shards"
MERGED = OUT_DIR / "stroke_timing.npz"
TASKS_JSON = OUT_DIR / "tasks.json"


# --------------------------------------------------------------------------- #
# Gripper events (compact copy of analysis2/events.py -- keep in sync with it) #
# --------------------------------------------------------------------------- #
def _schmitt(g: np.ndarray, lo: float, hi: float) -> np.ndarray:
    g = np.asarray(g, np.float64)
    up, dn = g > hi, g < lo
    ev = np.flatnonzero(up | dn)
    if ev.size == 0:
        return np.zeros(g.shape, bool)
    pos = np.searchsorted(ev, np.arange(g.size), side="right") - 1
    out = np.zeros(g.shape, bool)
    hit = pos >= 0
    out[hit] = up[ev[pos[hit]]]
    return out


def _closed_segments(state: np.ndarray, min_dwell: int) -> list[tuple[int, int]]:
    if not state.any():
        return []
    d = np.diff(state.astype(np.int8), prepend=0, append=0)
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1) - 1
    keep = (ends - starts + 1) >= min_dwell
    return list(zip(starts[keep].tolist(), ends[keep].tolist()))


def gripper_events(g: np.ndarray, lo=GRIP_LO, hi=GRIP_HI, min_dwell=GRIP_MIN_DWELL, lag=GRIP_LAG):
    """Grasp / release frame indices from a continuous gripper command (see analysis2/events.py)."""
    g = np.asarray(g, np.float64)
    n = g.size
    if n == 0:
        return np.empty(0, int), np.empty(0, int)
    segs = _closed_segments(_schmitt(g, lo, hi), min_dwell)
    if not segs:
        return np.empty(0, int), np.empty(0, int)
    idx = np.arange(n)
    last_below = np.maximum.accumulate(np.where(g <= 0.5, idx, -1))
    last_above = np.maximum.accumulate(np.where(g >= 0.5, idx, -1))
    grasps, releases = [], []
    for s, e in segs:
        if last_below[s] < 0:
            continue
        grasps.append(min(last_below[s] + 1 + lag, n - 1))
        t50_open = last_above[e] + 1
        if t50_open + lag <= n - 1:
            releases.append(t50_open + lag)
    return np.asarray(grasps, int), np.asarray(releases, int)


# --------------------------------------------------------------------------- #
# pyarrow -> numpy helpers (mirror analysis2/droid_stream.py)                   #
# --------------------------------------------------------------------------- #
def _list2d(col, width: int) -> np.ndarray:
    flat = col.combine_chunks().flatten().to_numpy(zero_copy_only=False)
    return np.asarray(flat, np.float32).reshape(-1, width)


def _scalar(col) -> np.ndarray:
    return col.combine_chunks().to_numpy(zero_copy_only=False)


def _episode_bounds(ep: np.ndarray) -> list[tuple[int, int]]:
    if ep.size == 0:
        return []
    cut = np.flatnonzero(np.diff(ep)) + 1
    starts = np.concatenate([[0], cut])
    stops = np.concatenate([cut, [ep.size]])
    return list(zip(starts.tolist(), stops.tolist()))


# --------------------------------------------------------------------------- #
# Per-episode stroke extraction                                                #
# --------------------------------------------------------------------------- #
def _episode_strokes(q: np.ndarray, g: np.ndarray) -> list[tuple[np.ndarray, str, str, np.ndarray]]:
    """Split one episode's joint series ``q`` [T,7] into (profile, lead_kind, trail_kind, feats) strokes.

    Boundaries are the episode start/end (``rest``) and every gripper event (``grasp`` = close,
    ``release`` = open). Each stroke spans two consecutive boundaries; its profile is the unit-mean
    speed-vs-time curve and ``feats`` the raw geometry/pace features. Degenerate strokes are dropped.
    """
    T = len(q)
    grasps, releases = gripper_events(g)
    # (frame, kind) boundary list, sorted; endpoints are rest.
    bounds = [(0, "rest"), (T - 1, "rest")]
    bounds += [(int(f), "grasp") for f in grasps]
    bounds += [(int(f), "release") for f in releases]
    bounds = sorted(bounds, key=lambda x: x[0])

    strokes = []
    for (a, lead), (b, trail) in zip(bounds[:-1], bounds[1:]):
        if b - a + 1 < MIN_STROKE_LEN:
            continue
        seg = q[a : b + 1]
        if np.linalg.norm(np.diff(seg, axis=0), axis=1).sum() < MIN_STROKE_ARC:
            continue
        prof = stroke_speed_profile(seg, DT)
        feats = stroke_features(seg, (len(seg) - 1) * DT)
        if prof is None or feats is None:
            continue
        strokes.append((prof, lead, trail, feats))
    return strokes


def reduce_file(tbl) -> dict:
    ep = np.asarray(_scalar(tbl["episode_index"]), np.int64)
    J = _list2d(tbl["observation.state.joint_position"], 7)
    A = _list2d(tbl["action"], 8)
    task = np.asarray(_scalar(tbl["task_index"]), np.int64)
    if not np.isfinite(J).all():
        raise ValueError("NaN/inf in DROID joints")

    profiles, lead_idx, trail_idx, task_idx, feats_l = [], [], [], [], []
    n_eps = n_strokes = 0
    for a, b in _episode_bounds(ep):
        q = J[a:b]
        if len(q) < MIN_STROKE_LEN:
            continue
        n_eps += 1
        for prof, lead, trail, feats in _episode_strokes(q, A[a:b, ACT_GRIPPER_COL]):
            profiles.append(prof)
            lead_idx.append(KIND_TO_IDX[lead])
            trail_idx.append(KIND_TO_IDX[trail])
            task_idx.append(int(task[a]))
            feats_l.append(feats)
            n_strokes += 1

    return dict(
        profiles=np.asarray(profiles, np.float32).reshape(-1, N_KNOTS),
        feats=np.asarray(feats_l, np.float32).reshape(-1, N_FEAT),
        lead_idx=np.asarray(lead_idx, np.int64),
        trail_idx=np.asarray(trail_idx, np.int64),
        task_idx=np.asarray(task_idx, np.int64),
        n_episodes=np.int64(n_eps),
        n_strokes=np.int64(n_strokes),
    )


# --------------------------------------------------------------------------- #
# Task-text table (for the optional language conditioning)                     #
# --------------------------------------------------------------------------- #
def fetch_tasks() -> dict[int, str]:
    """Best-effort task_index -> instruction map from DROID's meta/tasks table (small)."""
    try:
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(DROID_REPO, "meta/tasks.parquet", repo_type="dataset")
        tbl = pq.read_table(path)
        cols = {c.lower(): c for c in tbl.column_names}
        text_col = cols.get("task") or cols.get("__index_level_0__") or tbl.column_names[0]
        idx_col = cols.get("task_index")
        texts = tbl[text_col].to_pylist()
        if idx_col is not None:
            idxs = tbl[idx_col].to_pylist()
        else:
            idxs = list(range(len(texts)))
        return {int(i): str(t) for i, t in zip(idxs, texts)}
    except Exception as exc:  # noqa: BLE001
        print(f"  (could not fetch task text table: {type(exc).__name__}: {exc}; language will be off)")
        return {}


# --------------------------------------------------------------------------- #
# Stream + merge (mirror analysis2/droid_stream.py)                            #
# --------------------------------------------------------------------------- #
def stream(n_files: int, sleep: float = 0.5, overwrite: bool = False):
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    fs = HfFileSystem()
    t0 = time.time()
    for fi in range(n_files):
        out = SHARD_DIR / f"file_{fi:03d}.npz"
        if out.exists() and not overwrite:
            print(f"[{fi:3d}/{n_files}] cached", flush=True)
            continue
        path = f"datasets/{DROID_REPO}/data/chunk-000/file-{fi:03d}.parquet"
        tbl = None
        for attempt in range(6):
            try:
                with fs.open(path, "rb") as f:
                    tbl = pq.read_table(f, columns=COLS)
                break
            except Exception as exc:  # noqa: BLE001
                if attempt == 5:
                    raise
                msg = str(exc).lower()
                back = 90.0 if ("429" in msg or "rate limit" in msg) else 3.0 * (attempt + 1)
                print(f"    [{fi}] attempt {attempt + 1} failed ({type(exc).__name__}); backoff {back:.0f}s", flush=True)
                time.sleep(back)
        agg = reduce_file(tbl)
        del tbl
        np.savez_compressed(out, **agg)
        el = time.time() - t0
        print(
            f"[{fi:3d}/{n_files}] {int(agg['n_episodes']):5d} eps  {int(agg['n_strokes']):6d} strokes  "
            f"({el / (fi + 1):.1f}s/file, {el / 60:.1f} min elapsed)",
            flush=True,
        )
        time.sleep(sleep)
    merge()


def merge():
    shards = sorted(SHARD_DIR.glob("file_*.npz"))
    if not shards:
        raise FileNotFoundError(f"no shards under {SHARD_DIR}")
    keys = ("profiles", "feats", "lead_idx", "trail_idx", "task_idx")
    cat = {k: [] for k in keys}
    n_eps = 0
    for s in shards:
        d = np.load(s)
        for k in keys:
            cat[k].append(d[k])
        n_eps += int(d["n_episodes"])
    merged = {k: np.concatenate(v, 0) for k, v in cat.items()}
    merged["n_episodes"] = np.int64(n_eps)
    np.savez_compressed(MERGED, **merged)

    # Task-text table for language conditioning (only the indices we actually saw).
    tasks = fetch_tasks()
    if tasks:
        seen = {int(i): tasks.get(int(i), "") for i in np.unique(merged["task_idx"])}
        TASKS_JSON.write_text(json.dumps(seen))

    m = len(merged["profiles"])
    print(f"\nmerged {len(shards)} shards -> {MERGED}")
    print(f"  episodes {n_eps:,}   strokes {m:,}")
    if m:
        for kind, name in ((merged["lead_idx"], "lead"), (merged["trail_idx"], "trail")):
            counts = np.bincount(kind, minlength=len(KIND_TO_IDX))
            print(f"  {name} kinds: " + ", ".join(f"{k}={counts[i]}" for k, i in KIND_TO_IDX.items()))
    print(f"  tasks.json: {'written' if tasks else 'skipped (no task text)'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", type=int, default=DROID_N_FILES, help="number of parquet files to stream")
    ap.add_argument("--sleep", type=float, default=0.5)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--merge-only", action="store_true")
    a = ap.parse_args()
    if a.merge_only:
        merge()
    else:
        stream(a.files, a.sleep, a.overwrite)
