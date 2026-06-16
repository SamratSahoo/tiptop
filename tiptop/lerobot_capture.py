"""Build raw per-frame data for the LeRobot DROID export from a tiptop plan.

tiptop executes a TAMP plan as a sequence of dense joint-impedance trajectories,
so the ground-truth per-frame stream is the plan itself (``tiptop_plan.json``):
each ``trajectory`` step stores dense 50 Hz joint ``positions`` + ``velocities``,
and ``gripper`` steps mark open/close events.

:func:`dense_frames_from_plan` flattens that plan into a dense, resampled per-frame
trajectory (default 15 Hz, matching DROID): joint positions and the plan's own
instantaneous velocities become the state/action, and the gripper channel is
reconstructed from the open/close events (0 = open, 1 = closed, DROID convention).
:func:`dump_lerobot_raw` writes the result to ``_lerobot_raw.json`` together with
each frame's wall-clock time (from the execution timeline, see
``execute_cutamp_plan``) so ``scripts/lerobot_export.py`` can align camera frames
to the control timeline by ZED hardware timestamp.

This replaces an earlier approach that polled the robot from a background thread
during execution; sharing the busy command socket starved it to ~20 frames per
episode and yielded finite-difference velocities far over the robot's joint limits
and a gripper stuck at a constant value.
"""

import json
import logging
import threading
import time
from pathlib import Path

import numpy as np

_log = logging.getLogger(__name__)

# Robot control / DROID target rate. The plan is stored at 50 Hz; we resample to this.
DEFAULT_TARGET_FPS = 15

# Robotiq 2F-85 fully-open width in metres; used to normalise the measured width to the
# DROID gripper convention (0 = open, 1 = closed).
GRIPPER_MAX_WIDTH = 0.085

# DROID models the gripper ACTION (command) as leading the measured gripper POSITION: the
# fingers lag the command, so the action ramps a couple of frames before the observed state.
# We emulate that by shifting the action's gripper channel ahead of the measured state by this
# many seconds (~2 frames at 15 Hz, matching DROID-100's observed command-to-position lead).
# The observation/proprioception gripper is left as the true measured position (unshifted).
GRIPPER_ACTION_LEAD_S = 0.13


def _read_gripper_width(robot) -> float | None:
    """Best-effort read of the measured gripper opening width in metres. None if unavailable.

    The bamboo client returns ``{"success": ..., "state": {"width": <m>, ...}}``; older
    code read ``["width"]`` directly and always missed, defaulting the gripper to a
    constant. Navigate the real payload, tolerating the flatter shapes too.
    """
    try:
        if not hasattr(robot, "get_gripper_state"):
            return None
        res = robot.get_gripper_state()
        if not isinstance(res, dict):
            return float(res)
        if res.get("success") is False:
            return None
        state = res.get("state", res)
        width = state.get("width") if isinstance(state, dict) else state
        return None if width is None else float(width)
    except Exception:
        return None


class GripperSampler:
    """Background thread sampling the *measured* gripper width during execution.

    Reads go over the gripper server's own ZMQ socket (separate from the joint-impedance
    control socket), so they are not starved by the blocking trajectory execution that
    previously throttled joint-state polling to ~20 samples. A ZMQ REQ socket is not
    thread-safe, so this acquires its OWN client connection rather than sharing the
    command client's; it falls back to the shared client (and logs) if that fails.

    ``samples`` holds ``(wall_clock_seconds, closedness)`` pairs, closedness on the DROID
    convention 0 = open, 1 = closed. Use as a context manager around plan execution::

        with GripperSampler(robot) as g:
            execute_cutamp_plan(plan, client=robot)
        # g.samples now holds the measured gripper trace
    """

    def __init__(self, robot, fps: int = 20):
        from tiptop.utils import new_robot_client

        self._owns_client = False
        try:
            self.robot = new_robot_client()
            self._owns_client = True
        except Exception as e:
            _log.warning(
                f"Could not create a dedicated gripper-state client ({e}); falling back to the shared client."
            )
            self.robot = robot
        self.fps = int(fps)
        self.samples: list[tuple[float, float]] = []  # (wall_seconds, closedness)
        self.width_samples: list[tuple[float, float]] = []  # (wall_seconds, raw width m) for diagnostics
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._unavailable_logged = False

    def _loop(self):
        period = 1.0 / self.fps
        while not self._stop.is_set():
            tick = time.perf_counter()
            width = _read_gripper_width(self.robot)
            if width is not None:
                now = time.time()
                closed = float(np.clip(1.0 - width / GRIPPER_MAX_WIDTH, 0.0, 1.0))
                self.samples.append((now, closed))
                self.width_samples.append((now, width))
            elif not self._unavailable_logged:
                _log.warning("Measured gripper width unavailable; export will fall back to plan gripper events")
                self._unavailable_logged = True
            time.sleep(max(0.0, period - (time.perf_counter() - tick)))

    def __enter__(self) -> "GripperSampler":
        self._thread = threading.Thread(target=self._loop, name="lerobot-gripper-sampler", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self.width_samples:
            w = np.asarray([x[1] for x in self.width_samples])
            # If this range is tiny or not within ~[0, 0.085] m, the width units are off
            # and the [0,1] normalisation will saturate -- worth seeing in the log.
            _log.info(
                "Gripper sampler: %d reads, raw width range [%.4f, %.4f] (expect ~0..0.085 m)",
                len(w), float(w.min()), float(w.max()),
            )
        if self._owns_client:
            try:
                self.robot.close()
            except Exception as e:
                _log.warning(f"Failed to close dedicated gripper-state client: {e}")
        return False


def _matrix_to_xyzrpy(mat: np.ndarray) -> np.ndarray:
    """Convert a 4x4 homogeneous transform to [x, y, z, roll, pitch, yaw]."""
    from scipy.spatial.transform import Rotation

    xyz = mat[:3, 3]
    rpy = Rotation.from_matrix(mat[:3, :3]).as_euler("xyz")
    return np.concatenate([xyz, rpy]).astype(np.float32)


def _load_plan(plan_path: Path) -> dict:
    """Load a serialized tiptop plan, converting trajectory arrays to float32 ndarrays."""
    with open(plan_path) as f:
        plan = json.load(f)
    for step in plan["steps"]:
        if step["type"] == "trajectory":
            step["positions"] = np.asarray(step["positions"], dtype=np.float32)
            step["velocities"] = np.asarray(step["velocities"], dtype=np.float32)
    return plan


def _flatten_plan(plan: dict, timeline: list | None = None) -> dict:
    """Flatten plan steps into dense 50 Hz arrays.

    Returns a dict with, for the M dense rows:
      positions[M,7], velocities[M,7], gripper[M], dt[M] (per-row duration),
      t_plan[M] (start time of each row on the plan clock), and
      t_wall[M] (wall-clock time of each row, NaN where no execution timeline).

    The gripper channel is held across trajectory rows: it starts at 0.0 (open) and
    flips to 1.0 (closed) / 0.0 (open) at each ``gripper`` event, taking effect on the
    rows that follow. Wall-clock times come from ``timeline`` (one entry per plan step,
    in order, each ``{"t_start", "t_end"}``): a trajectory step's rows are spread
    linearly between its measured start and end.

    Gripper steps are instantaneous in the plan but take ~0.5-1 s on the real robot.
    When the timeline gives that pause duration, we insert stationary "hold" rows (arm
    frozen at its last pose, zero velocity) spanning the pause, so the export emits
    frames *during* gripper actuation -- otherwise every frame lands where the gripper
    is already fully open/closed and the measured open->close ramp is never sampled.
    """
    HOLD_DT = 0.02  # 50 Hz, matching the plan's trajectory rate, for inserted hold rows
    pos_chunks, vel_chunks, grip_chunks, dt_chunks, twall_chunks = [], [], [], [], []
    q_init = np.asarray(plan.get("q_init", np.zeros(7)), dtype=np.float32).reshape(-1)
    last_pos = q_init  # arm pose to freeze at during a gripper pause
    g = 0.0  # DROID convention: 0 = open, 1 = closed. Episodes start open.
    for i, step in enumerate(plan["steps"]):
        entry = timeline[i] if (timeline is not None and i < len(timeline)) else None
        has_wall = entry is not None and entry.get("t_start") is not None and entry.get("t_end") is not None

        if step["type"] == "trajectory":
            pos = np.asarray(step["positions"], dtype=np.float32)
            vel = np.asarray(step["velocities"], dtype=np.float32)
            n = len(pos)
            if n == 0:
                continue
            dt = float(step["dt"])
            pos_chunks.append(pos)
            vel_chunks.append(vel)
            grip_chunks.append(np.full(n, g, dtype=np.float32))
            dt_chunks.append(np.full(n, dt, dtype=np.float64))
            if has_wall:
                ts, te = float(entry["t_start"]), float(entry["t_end"])
                twall = np.full(n, ts, dtype=np.float64) if n == 1 else np.linspace(ts, te, n)
            else:
                twall = np.full(n, np.nan, dtype=np.float64)
            twall_chunks.append(twall)
            last_pos = pos[-1]
        elif step["type"] == "gripper":
            g = 1.0 if step["action"] == "close" else 0.0
            # Insert stationary hold rows spanning the measured actuation pause so the
            # gripper ramp is captured. Skipped without a timeline (no known duration).
            if has_wall:
                ts, te = float(entry["t_start"]), float(entry["t_end"])
                n_hold = max(1, round((te - ts) / HOLD_DT))
                pos_chunks.append(np.tile(last_pos, (n_hold, 1)))
                vel_chunks.append(np.zeros((n_hold, 7), dtype=np.float32))
                grip_chunks.append(np.full(n_hold, g, dtype=np.float32))
                dt_chunks.append(np.full(n_hold, HOLD_DT, dtype=np.float64))
                twall_chunks.append(np.linspace(ts, te, n_hold))

    if not pos_chunks:
        return {k: np.empty((0,)) for k in ("positions", "velocities", "gripper", "dt", "t_plan", "t_wall")}

    positions = np.concatenate(pos_chunks, axis=0)
    velocities = np.concatenate(vel_chunks, axis=0)
    gripper = np.concatenate(grip_chunks, axis=0)
    dt = np.concatenate(dt_chunks, axis=0)
    t_wall = np.concatenate(twall_chunks, axis=0)
    # Start time of each row on the plan clock: 0, dt0, dt0+dt1, ...
    t_plan = np.concatenate([[0.0], np.cumsum(dt)[:-1]])
    return {
        "positions": positions,
        "velocities": velocities,
        "gripper": gripper,
        "dt": dt,
        "t_plan": t_plan,
        "t_wall": t_wall,
    }


def _gripper_from_measurements(frame_wall: np.ndarray, gripper_samples: list | None) -> np.ndarray | None:
    """Per-frame gripper closedness from the measured trace, aligned by wall-clock time.

    Returns None (caller falls back to plan events) if there is no usable measured trace
    or the frames have no wall-clock times to align against.
    """
    if not gripper_samples or not np.all(np.isfinite(frame_wall)):
        return None
    gs = np.asarray(gripper_samples, dtype=np.float64)  # [K, 2]: (wall_seconds, closedness)
    if gs.ndim != 2 or len(gs) == 0:
        return None
    gt, gv = gs[:, 0], gs[:, 1]
    nearest = np.abs(gt[None, :] - frame_wall[:, None]).argmin(axis=1)
    return gv[nearest].astype(np.float32)


def dense_frames_from_plan(
    plan: dict,
    *,
    motion_gen,
    tensor_args,
    target_fps: int = DEFAULT_TARGET_FPS,
    timeline: list | None = None,
    gripper_samples: list | None = None,
) -> dict:
    """Resample a tiptop plan to ``target_fps`` dense per-frame DROID-style arrays.

    Returns joint_position[N,7], gripper_position[N,1], cartesian_position[N,6],
    action[N,8] (= 7 plan joint-velocities + gripper), and frame_time[N] (wall-clock
    seconds, or None if no execution timeline was supplied). Velocities come straight
    from the plan -- never a finite difference -- so they stay within the robot's
    joint-velocity limits. The gripper channel is the *measured* width (continuous,
    DROID convention 0 = open, 1 = closed) when ``gripper_samples`` is supplied and the
    frames have wall-clock times; otherwise it is reconstructed from the plan's
    open/close events as a binary step. ``gripper_position`` is the true measured
    position; the action's gripper channel leads it by ``GRIPPER_ACTION_LEAD_S``
    (DROID-style: the command precedes the finger motion).
    """
    dense = _flatten_plan(plan, timeline=timeline)
    positions = dense["positions"]
    M = len(positions)
    if M < 2:
        return {"n_frames": M}

    duration = float(dense["t_plan"][-1] + dense["dt"][-1])
    N = max(2, round(duration * target_fps))
    target_t = np.minimum(np.arange(N) / float(target_fps), dense["t_plan"][-1])
    # Nearest preceding dense row for each uniform target time (t_plan is increasing).
    idx = np.clip(np.searchsorted(dense["t_plan"], target_t, side="right") - 1, 0, M - 1)

    joint = positions[idx].astype(np.float32)  # [N, 7]
    vel = dense["velocities"][idx].astype(np.float32)  # [N, 7]

    frame_wall = dense["t_wall"][idx]  # wall-clock time per output frame (may be NaN)
    grip_meas = _gripper_from_measurements(frame_wall, gripper_samples)
    if grip_meas is not None:
        grip = grip_meas.reshape(N, 1)
        _log.info("Gripper channel from measured width (%d samples)", len(gripper_samples))
    else:
        grip = dense["gripper"][idx].reshape(N, 1).astype(np.float32)  # plan-event fallback (binary)
        _log.info("Gripper channel reconstructed from plan open/close events (no measured trace)")
    # observation gripper = true measured position; action gripper leads it (DROID-style), i.e.
    # action[t] = measured[t + lead], so the command precedes the finger motion. Tail clamps.
    lead = max(1, round(GRIPPER_ACTION_LEAD_S * target_fps))
    action_grip = grip[np.minimum(np.arange(N) + lead, N - 1)]  # [N, 1]
    action = np.concatenate([vel, action_grip], axis=1).astype(np.float32)  # [N, 8]

    frame_time = None if not np.all(np.isfinite(frame_wall)) else frame_wall.tolist()

    # Cartesian pose for every frame via one batched forward-kinematics call.
    try:
        q_pt = tensor_args.to_device(joint)
        mats = motion_gen.kinematics.get_state(q_pt).ee_pose.get_numpy_matrix()  # [N, 4, 4]
        cartesian = np.stack([_matrix_to_xyzrpy(m) for m in mats]).astype(np.float32)  # [N, 6]
    except Exception as e:
        _log.warning("Forward kinematics for cartesian state failed (%s); writing zeros", e)
        cartesian = np.zeros((N, 6), dtype=np.float32)

    return {
        "n_frames": N,
        "joint_position": joint,
        "gripper_position": grip,
        "cartesian_position": cartesian,
        "action": action,
        "frame_time": frame_time,
        "duration": duration,
    }


def dump_lerobot_raw(
    save_dir: Path,
    plan_path: Path,
    *,
    motion_gen,
    tensor_args,
    instruction: str,
    cameras: dict[str, str],
    fps: int = DEFAULT_TARGET_FPS,
    robot_type: str = "franka",
    timeline: list | None = None,
    gripper_samples: list | None = None,
) -> Path | None:
    """Write ``<save_dir>/_lerobot_raw.json`` from a saved ``tiptop_plan.json``.

    ``cameras`` maps LeRobot image keys (e.g. observation.images.exterior_1_left) to
    mp4 filenames relative to ``save_dir``. ``timeline`` is the per-step wall-clock
    record from :func:`execute_cutamp_plan` (used for frame-accurate camera alignment).
    ``gripper_samples`` is the measured gripper trace from a :class:`GripperSampler`
    (used for the gripper channel; falls back to plan events if absent).
    Returns the json path, or None if the plan was too short to export.
    """
    save_dir = Path(save_dir)
    plan_path = Path(plan_path)
    if not plan_path.is_file():
        _log.warning("No plan at %s; skipping LeRobot raw dump", plan_path)
        return None

    plan = _load_plan(plan_path)
    frames = dense_frames_from_plan(
        plan,
        motion_gen=motion_gen,
        tensor_args=tensor_args,
        target_fps=fps,
        timeline=timeline,
        gripper_samples=gripper_samples,
    )
    N = frames.get("n_frames", 0)
    if N < 2:
        _log.warning("Plan flattened to too few frames (%d); skipping LeRobot raw dump", N)
        return None

    raw = {
        "fps": int(fps),
        "instruction": instruction,
        "robot_type": robot_type,
        "cameras": cameras,
        "joint_position": frames["joint_position"].tolist(),
        "gripper_position": frames["gripper_position"].tolist(),
        "cartesian_position": frames["cartesian_position"].tolist(),
        "action": frames["action"].tolist(),
        "frame_time": frames["frame_time"],  # wall-clock seconds per frame, or None
    }
    out_path = save_dir / "_lerobot_raw.json"
    with open(out_path, "w") as f:
        json.dump(raw, f)
    _log.info(
        "Wrote LeRobot raw capture (%d frames, %.1fs, frame_time=%s) to %s",
        N,
        frames["duration"],
        "yes" if raw["frame_time"] is not None else "no",
        out_path,
    )
    return out_path
