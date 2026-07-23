"""Extract per-stroke TIMING training data from DROID by STREAMING it (never downloaded).

The timing network learns the human speed-vs-time profile of a manipulation *stroke* -- the motion
between two gripper events. This script produces its training set: it streams DROID 1.0.1 from the Hub
(column-projected parquet over ``HfFileSystem`` -- the video columns, the vast bulk, are never touched),
segments every episode at its gripper-command edges into strokes, and for each stroke saves two things:
the unit-mean speed profile ``p(tau)`` (the target) and the raw GEOMETRY + PACE features (the input).
Each parquet file is reduced to a few thousand small records and then dropped; the 27.6 M frames never
land on disk.

Segmentation mirrors ``analysis2`` exactly (Schmitt trigger + min dwell + a 4-frame actuator lag, scored
F1=0.944 on DROID) so a stroke here is delimited the SAME way a cuTAMP operation is delimited at plan
time (a gripper close and a gripper open both bound strokes); the episode start/end bound the first/last.

Run in the openpi venv (has ``datasets`` / ``pyarrow`` / ``torch``), like the other DROID streamers::

    openpi/.venv/bin/python -m tiptop.networks.extract_stroke_timing --files 2      # smoke test
    openpi/.venv/bin/python tiptop/tiptop/networks/extract_stroke_timing.py         # full stream

Output: per-file shards under ``checkpoints/timing_data/`` plus a merged ``stroke_timing.npz`` (profiles +
feats) that ``train_timing`` consumes. Resumable: a finished shard is skipped on re-run.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

# Import the shared profile/feature helpers straight from the model file (no cuRobo / tiptop-package
# import needed, so this runs in the openpi venv). timing_net.py imports only numpy/torch.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from timing_net import N_FEAT, N_KNOTS, stroke_features, stroke_speed_profile  # noqa: E402

# --------------------------------------------------------------------------- #
# DROID streaming config (mirrors analysis2/config.py)                          #
# --------------------------------------------------------------------------- #
DROID_REPO = "lerobot/droid_1.0.1"
DROID_N_FILES = 86  # meta/episodes references file-000..file-085 (all 95,658 episodes)
DROID_FPS = 15.0
DT = 1.0 / DROID_FPS
COLS = ["observation.state.joint_position", "action", "episode_index"]
ACT_GRIPPER_COL = 7  # action[:, 7] is the commanded gripper in [0, 1] (0 = open, 1 = closed)

# Gripper-event detection (identical to analysis2/config.py -- the source of truth for these constants).
GRIP_LO, GRIP_HI = 0.20, 0.60
GRIP_MIN_DWELL = 5
GRIP_LAG = 4

# Stroke filter: a usable stroke has enough frames to define a profile and real net motion.
MIN_STROKE_LEN = 6            # frames
MIN_STROKE_ARC = 5e-3         # rad, total joint-space path length

OUT_DIR = Path(__file__).resolve().parents[1] / "checkpoints" / "timing_data"
SHARD_DIR = OUT_DIR / "shards"
MERGED = OUT_DIR / "stroke_timing.npz"


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


def gripper_event_frames(g: np.ndarray) -> list[int]:
    """Frames at which the gripper closes (grasp) or opens (release) -- the interior stroke boundaries."""
    g = np.asarray(g, np.float64)
    n = g.size
    if n == 0:
        return []
    segs = _closed_segments(_schmitt(g, GRIP_LO, GRIP_HI), GRIP_MIN_DWELL)
    if not segs:
        return []
    idx = np.arange(n)
    last_below = np.maximum.accumulate(np.where(g <= 0.5, idx, -1))
    last_above = np.maximum.accumulate(np.where(g >= 0.5, idx, -1))
    frames = []
    for s, e in segs:
        if last_below[s] >= 0:
            frames.append(min(last_below[s] + 1 + GRIP_LAG, n - 1))  # grasp (close)
        t50_open = last_above[e] + 1
        if t50_open + GRIP_LAG <= n - 1:
            frames.append(t50_open + GRIP_LAG)  # release (open)
    return frames


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
def _episode_strokes(q: np.ndarray, g: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split one episode's joint series ``q`` [T,7] into (profile, feats) strokes at gripper events.

    Boundaries are the episode start/end and every gripper event; each stroke spans two consecutive
    boundaries. Degenerate strokes (too short / no motion) are dropped.
    """
    T = len(q)
    bounds = sorted({0, T - 1, *gripper_event_frames(g)})
    strokes = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a + 1 < MIN_STROKE_LEN:
            continue
        seg = q[a : b + 1]
        if np.linalg.norm(np.diff(seg, axis=0), axis=1).sum() < MIN_STROKE_ARC:
            continue
        prof = stroke_speed_profile(seg, DT)
        feats = stroke_features(seg, (len(seg) - 1) * DT)
        if prof is None or feats is None:
            continue
        strokes.append((prof, feats))
    return strokes


def reduce_file(tbl) -> dict:
    ep = np.asarray(_scalar(tbl["episode_index"]), np.int64)
    J = _list2d(tbl["observation.state.joint_position"], 7)
    A = _list2d(tbl["action"], 8)
    if not np.isfinite(J).all():
        raise ValueError("NaN/inf in DROID joints")

    profiles, feats_l = [], []
    n_eps = n_strokes = 0
    for a, b in _episode_bounds(ep):
        q = J[a:b]
        if len(q) < MIN_STROKE_LEN:
            continue
        n_eps += 1
        for prof, feats in _episode_strokes(q, A[a:b, ACT_GRIPPER_COL]):
            profiles.append(prof)
            feats_l.append(feats)
            n_strokes += 1

    return dict(
        profiles=np.asarray(profiles, np.float32).reshape(-1, N_KNOTS),
        feats=np.asarray(feats_l, np.float32).reshape(-1, N_FEAT),
        n_episodes=np.int64(n_eps),
        n_strokes=np.int64(n_strokes),
    )


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
    cat = {k: [] for k in ("profiles", "feats")}
    n_eps = 0
    for s in shards:
        d = np.load(s)
        for k in cat:
            cat[k].append(d[k])
        n_eps += int(d["n_episodes"])
    merged = {k: np.concatenate(v, 0) for k, v in cat.items()}
    merged["n_episodes"] = np.int64(n_eps)
    np.savez_compressed(MERGED, **merged)
    print(f"\nmerged {len(shards)} shards -> {MERGED}")
    print(f"  episodes {n_eps:,}   strokes {len(merged['profiles']):,}   features {merged['feats'].shape[1]}")


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
