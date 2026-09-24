"""Blend + re-time a cuTAMP plan so consecutive arm motions read like one continuous stroke.

A cuTAMP plan is a sequence of independently-planned trajectory segments (retract -> approach ->
grasp, etc.) separated by gripper open/close steps. Every segment is a cuRobo *reaching* motion, so
it accelerates from rest and decelerates back to rest at each of its endpoints. Concatenated, the
arm therefore comes to a full stop at every interior waypoint -- a stop-and-go, per-segment velocity
profile with ~20 zero-velocity dips, quite unlike a human teleop demo, which flows through the
reach and only slows near the grasp/place itself.

This module collapses that: it groups the consecutive trajectory segments *between gripper events*
and replaces each group with a single stroke that slows into each gripper event without stopping and
comes fully to rest only at the episode's start and end. Each stroke is built by path-velocity
decomposition:

* Geometry. The group's joined waypoints are fit with a penalized cubic SMOOTHING spline in joint
  space, parameterized by arc length. The penalty rounds the sharp direction changes at the old
  segment joins (which an interpolating spline would overshoot, producing large phantom
  accelerations) while staying close to the collision-checked path (~1 deg joint deviation at the
  default smoothing).

* Timing. The trajectory encoder's motion-manifold cost picks the stroke's clock
  (:func:`tiptop.encoder_retiming.encoder_retime_group`), scored against the same DROID cluster the
  cuRobo cost uses. At gripper-adjacent boundaries the stroke carries a small NONZERO boundary speed:
  the DROID/openpi non-idle training filter drops idle joint-velocity runs (>=7 frames at 15 Hz), which
  would take the gripper open/close timestep with them, so the only zero-velocity frames at a gripper
  event should be the short (~3-frame) stationary hold the export inserts for the actuation itself.
  The stroke's final commanded POSITION is exactly the grasp, so there is no overshoot.

* Limits. The robot's REAL joint velocity/acceleration limits (from ``motion_gen``, scaled by
  ``retime_vel_slack`` / ``retime_acc_slack``) bound every emitted stroke.

Collision caveat: smoothing rounds the sharp corners at the old segment joins, so the re-timed path is
NOT identical to the one cuRobo collision-checked (largest deviation at those corners, ~1 deg joint at
the default). This is the same rounding a human demo has, but validate in sim / on the viz before
trusting it near tight obstacles.

Re-timing is OFF by default and opt-in per config: set ``retime_trajectory: true`` under
``tamp_overrides`` (see :func:`resolve_blend_config`). This is a pure post-process over the
joint-waypoint arrays (no planner changes); it runs in ``run_planning`` so the saved plan and the
executed plan are the identical re-timed object.
"""

import logging
from dataclasses import dataclass

import numpy as np
import torch
from scipy.interpolate import CubicSpline, make_smoothing_spline

from tiptop.encoder_retiming import encoder_retime_group

# NOTE: ``from curobo.types.state import JointState`` is imported lazily inside
# ``_retime_trajectory_steps`` (its only user) rather than at module top, so this module's pure
# geometry helpers (``_fit_geometry``, ``_finish_stroke`` ...) can be imported -- and unit-tested --
# without a cuRobo install. ``encoder_retiming`` reuses them.

_log = logging.getLogger(__name__)

# Defaults for the tunable knobs (overridable per config -- see resolve_blend_config).
_DEFAULT_SMOOTHING = 1e-4  # smoothing-spline penalty (lam) in the arc-length (radian) parameterization
_DEFAULT_VEL_SLACK = 1.0  # fraction of the robot's real velocity limit the re-timer may use
_DEFAULT_ACC_SLACK = 1.0  # fraction of the robot's real acceleration limit the re-timer may use
# Joint speed (rad/s, L2) each stroke carries at gripper-adjacent boundaries so those frames are not
# idle (see module docstring). ~0.15 clears the ~0.02 non-idle threshold with wide margin while
# staying a small fraction of cruise. Set to 0 for strokes that come to rest at every gripper event.
_DEFAULT_BOUNDARY_SPEED = 0.15
# Divides each group's original wall-clock before it bounds the encoder's duration range, so >1.0 lets
# (and pushes) every stroke run faster. The vel/accel caps still bound it to the robot's real limits.
_DEFAULT_SPEED_SCALE = 1.0

# Waypoints closer than this (Euclidean, joint space, radians) to the previous kept one are dropped
# before fitting so the arc-length parameter is strictly increasing. Consecutive segments share an
# exact endpoint, and each segment's decel tail is a cluster of near-identical points -- this removes
# both.
_MIN_CHORD = 1e-4

# Below this many distinct waypoints the smoothing spline is not well-conditioned; fall back to a
# plain (natural) cubic through the points.
_MIN_SMOOTH_PTS = 6

# Fallback caps when the robot's real limits are unavailable, as multiples of the plan's OWN peak
# per-joint velocity / acceleration. The acceleration multiple is generous because a slowed plan sits
# well below the true acceleration limit, so its own peak is a poor ceiling.
_FALLBACK_VEL_MULT = 1.5
_FALLBACK_ACC_MULT = 4.0


@dataclass
class BlendConfig:
    """Resolved stroke re-timing settings (from a config's ``tamp_overrides``)."""

    enabled: bool = False
    smoothing: float = _DEFAULT_SMOOTHING
    vel_slack: float = _DEFAULT_VEL_SLACK
    acc_slack: float = _DEFAULT_ACC_SLACK
    boundary_speed: float = _DEFAULT_BOUNDARY_SPEED
    speed_scale: float = _DEFAULT_SPEED_SCALE
    # Operation names to restrict re-timing to (e.g. ("Pick", "Place")); None re-times every operation.
    ops: tuple[str, ...] | None = None
    # Upper bound on a stroke's duration, as a multiple of the cuTAMP group's own wall-clock.
    max_duration_mult: float = 2.0
    # Trajectory-encoder checkpoint. Taken from the SAME `encoder_path` override the cuRobo manifold
    # cost uses, so the re-timer scores against the encoder and DROID cluster that shaped the geometry.
    encoder_path: str | None = None
    # False (default) minimizes Mahalanobis distance to the DROID cluster MEAN -- a mode-seeking
    # objective whose optimum is the centroid, which no real motion occupies (the checkpoint bakes
    # maha2_droid_mean = 7.04 over 94,774 real segments). True draws one target latent per stroke from
    # the cluster and aims at THAT, in both the optimizer and the multi-start ranking, which turns the
    # objective from "be maximally typical" into a distribution match. Targets a measured defect:
    # between-stroke residual log-duration sd 0.148 against DROID's 0.414.
    sample_target: bool = False


def resolve_blend_config(overrides: dict | None) -> BlendConfig:
    """Read the stroke re-timing knobs from a config's ``tamp_overrides`` dict (OFF unless opted in).

    Recognized keys (all optional; ``retime_mode`` is checked by the overrides loader, see
    :mod:`tiptop.override_keys`, and needs no reading here since "encoder" is the only mode):
        retime_trajectory: bool         -- master enable (default False)
        retime_smoothing: float         -- geometry smoothing-spline penalty; higher = smoother, less
                                           faithful to the collision-checked path (default 1e-4)
        retime_vel_slack: float         -- fraction of the real velocity limit to use (default 1.0)
        retime_acc_slack: float         -- fraction of the real acceleration limit to use (default 1.0)
        retime_boundary_speed: float    -- joint speed (rad/s) targeted at gripper-adjacent boundaries so
                                           those frames are not idle-filtered (default 0.15; 0 = rest)
        retime_speed_scale: float       -- divides each group's wall-clock before it bounds the encoder's
                                           duration range (default 1.0 = none)
        retime_ops: list[str]           -- restrict re-timing to these operations by name, e.g.
                                           [Pick, Place]; omitted/empty re-times every operation
        retime_max_duration_mult: float -- longest allowed stroke, as a multiple of the group's
                                           wall-clock (default 2.0)
        retime_sample_target: bool      -- aim each stroke at a latent drawn from the DROID cluster
                                           instead of its mean (default False); see
                                           encoder_retiming.target_latent
        encoder_path: str               -- the trajectory-encoder checkpoint, shared with the cuRobo cost
    """
    # local: keeps this module import-light (motion_planning imports cuRobo and cuTAMP)
    from tiptop.motion_planning import resolve_encoder_path

    o = overrides or {}
    raw_ops = o.get("retime_ops")
    ops = tuple(str(x) for x in raw_ops) if raw_ops else None
    speed_scale = float(o.get("retime_speed_scale", _DEFAULT_SPEED_SCALE))
    if speed_scale <= 0.0:
        raise ValueError(f"retime_speed_scale must be > 0 (got {speed_scale})")
    return BlendConfig(
        enabled=bool(o.get("retime_trajectory", False)),
        smoothing=float(o.get("retime_smoothing", _DEFAULT_SMOOTHING)),
        vel_slack=float(o.get("retime_vel_slack", _DEFAULT_VEL_SLACK)),
        acc_slack=float(o.get("retime_acc_slack", _DEFAULT_ACC_SLACK)),
        boundary_speed=float(o.get("retime_boundary_speed", _DEFAULT_BOUNDARY_SPEED)),
        speed_scale=speed_scale,
        ops=ops,
        max_duration_mult=float(o.get("retime_max_duration_mult", 2.0)),
        # Resolved the same way the cuRobo manifold cost resolves it, so both ends of the pipeline
        # score against one checkpoint even when the config gives a repo-relative path.
        encoder_path=resolve_encoder_path(str(o["encoder_path"])) if o.get("encoder_path") else None,
        sample_target=bool(o.get("retime_sample_target", False)),
    )


def _op_name(label: str) -> str:
    """Operation name from a step label, e.g. 'Pick(brown_toy, grasp1, q1)' -> 'Pick'."""
    return (label or "").split("(", 1)[0].strip()


def _dedup_path(positions: np.ndarray) -> np.ndarray:
    """Drop waypoints within ``_MIN_CHORD`` of the previous kept one so chord length is strictly increasing.

    Each segment decelerates to rest, so its final few waypoints are near-identical; the true group
    endpoint must survive, so the final point is forced in and any trailing kept points too close to
    it are dropped first (keeping the endpoint exact and the last interval strictly positive).
    """
    keep = [0]
    for i in range(1, len(positions) - 1):
        if np.linalg.norm(positions[i] - positions[keep[-1]]) > _MIN_CHORD:
            keep.append(i)
    final = len(positions) - 1
    while len(keep) > 1 and np.linalg.norm(positions[final] - positions[keep[-1]]) <= _MIN_CHORD:
        keep.pop()
    keep.append(final)
    return positions[keep]


def _fit_geometry(u: np.ndarray, positions: np.ndarray, smoothing: float) -> list:
    """Per-joint geometry spline q_j(u) over arc length: penalized smoothing spline, cubic fallback."""
    if len(positions) >= _MIN_SMOOTH_PTS:
        try:
            return [make_smoothing_spline(u, positions[:, j], lam=smoothing) for j in range(positions.shape[1])]
        except Exception:
            _log.exception("Smoothing-spline geometry fit failed; falling back to a plain cubic")
    return [CubicSpline(u, positions[:, j], bc_type="natural") for j in range(positions.shape[1])]


def _eval_geometry(geom: list, u: np.ndarray, nu: int) -> np.ndarray:
    """Evaluate the ``nu``-th derivative of every joint's geometry spline -> [len(u), dof]."""
    return np.stack([g(u, nu) for g in geom], axis=1)


def _finish_stroke(out_pos, out_vel, out_acc, duration, lead_speed, trail_speed):
    """Pin rest ends to exactly zero, warn on a shortfall, and return (pos, vel, acc, dt_out)."""
    # Pin a boundary to exact rest only where a zero boundary speed was requested (episode ends);
    # a nonzero boundary is left as computed so the gripper-adjacent frames stay out of the idle band.
    if lead_speed == 0.0:
        out_vel[0] = 0.0
    if trail_speed == 0.0:
        out_vel[-1] = 0.0
    # Surface a boundary that fell short of the request: the encoder's boundary term is a soft target,
    # traded against the manifold score and bounded by the robot's vel/accel limits.
    for end, req in ((0, lead_speed), (-1, trail_speed)):
        realized = float(np.linalg.norm(out_vel[end]))
        if req > 0.0 and realized < 0.7 * req:
            _log.warning(
                f"Re-timed stroke boundary speed {realized:.3f} rad/s is below the requested {req:.3f}; "
                f"these gripper-adjacent frames stay slow."
            )
    dt_out = duration / (len(out_pos) - 1)
    return out_pos, out_vel, out_acc, dt_out


def _resolve_caps(
    orig_velocities: np.ndarray,
    dt: float,
    vel_limit: np.ndarray | None,
    acc_limit: np.ndarray | None,
    vel_slack: float,
    acc_slack: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-joint velocity/acceleration caps: the robot's real limits (scaled by slack) when available,
    else a generous multiple of the plan's own peaks (a slowed plan sits well below the true limits)."""
    if vel_limit is not None and acc_limit is not None:
        return vel_slack * np.abs(vel_limit), acc_slack * np.abs(acc_limit)
    _log.warning(
        f"Stroke re-timing: robot limits unavailable; capping at {_FALLBACK_VEL_MULT:.1f}x/"
        f"{_FALLBACK_ACC_MULT:.1f}x the plan's own peak velocity/acceleration instead."
    )
    ov = np.asarray(orig_velocities, dtype=np.float64)
    vel_cap = _FALLBACK_VEL_MULT * np.abs(ov).max(axis=0)
    orig_acc = np.gradient(ov, dt, axis=0) if len(ov) > 2 else np.zeros_like(ov)
    acc_cap = _FALLBACK_ACC_MULT * np.abs(orig_acc).max(axis=0)
    # Floor tiny-motion joints at a fraction of the largest cap so their near-zero peak does not make
    # the limit check hypersensitive and stretch the whole group.
    for cap in (vel_cap, acc_cap):
        m = float(cap.max())
        if m > 0:
            np.maximum(cap, 0.25 * m, out=cap)
    return vel_cap, acc_cap


def _retime_trajectory_steps(
    steps: list[dict],
    config: BlendConfig,
    vel_limit: np.ndarray | None,
    acc_limit: np.ndarray | None,
    lead_speed: float,
    trail_speed: float,
) -> dict:
    """Blend a run of consecutive ``trajectory`` steps into one re-timed trajectory step."""
    from curobo.types.state import JointState  # lazy: keep the module importable without cuRobo


    template = steps[0]["plan"]
    device = template.position.device
    joint_names = template.joint_names
    dt = float(steps[0]["dt"])

    # Join waypoints, dropping each later segment's first row (an exact copy of the previous
    # segment's last row -- see motion_solver: the next plan starts from the prior end position).
    seg_positions = [s["plan"].position.detach().cpu().numpy().astype(np.float64) for s in steps]
    joined = np.concatenate([seg_positions[0]] + [p[1:] for p in seg_positions[1:]], axis=0)
    orig_velocities = np.concatenate(
        [s["plan"].velocity.detach().cpu().numpy().astype(np.float64) for s in steps], axis=0
    )
    # The group's original wall-clock (divided by speed_scale). The encoder gets it only as a RANGE
    # BOUND, not a target -- letting the manifold cost choose the pace is the point.
    target_duration = sum((len(p) - 1) for p in seg_positions) * dt / config.speed_scale

    vel_cap, acc_cap = _resolve_caps(orig_velocities, dt, vel_limit, acc_limit, config.vel_slack, config.acc_slack)
    pos, vel, acc, dt_out = encoder_retime_group(
        joined,
        dt,
        target_duration,
        vel_cap,
        acc_cap,
        config.smoothing,
        lead_speed,
        trail_speed,
        config.max_duration_mult,
        config.encoder_path,
        config.sample_target,
    )

    plan = JointState(
        position=torch.as_tensor(pos, dtype=torch.float32, device=device),
        velocity=torch.as_tensor(vel, dtype=torch.float32, device=device),
        acceleration=torch.as_tensor(acc, dtype=torch.float32, device=device),
        jerk=torch.zeros(pos.shape, dtype=torch.float32, device=device),
        joint_names=list(joint_names) if joint_names is not None else None,
    )
    # Keep the run's first label; the merged stroke spans the same logical action(s).
    return {"type": "trajectory", "plan": plan, "dt": dt_out, "label": steps[0]["label"]}


def blend_cutamp_plan(
    cutamp_plan: list[dict],
    config: BlendConfig,
    vel_limit: np.ndarray | None = None,
    acc_limit: np.ndarray | None = None,
) -> list[dict]:
    """Return ``cutamp_plan`` with consecutive trajectory segments blended + re-timed, if enabled.

    A no-op returning the plan unchanged when ``config.enabled`` is False (the default -- re-timing is
    opt-in per config via ``retime_trajectory: true``). When enabled, gripper steps pass through
    untouched and delimit the groups; the trajectory steps between two gripper events (or a plan end)
    -- always a single operation, e.g. one Pick or Place -- are merged into a single stroke re-timed by
    the trajectory encoder, which only comes to rest at the plan's ends. If ``config.ops`` is set,
    only operations named in it are re-timed; every other operation's original segments pass through
    untouched. ``vel_limit`` / ``acc_limit`` are the robot's per-arm-joint velocity / acceleration
    limits (see :func:`arm_joint_limits`); when omitted, a fallback derived from the plan's own peaks
    is used.
    """
    if not config.enabled:
        return cutamp_plan

    n_steps = len(cutamp_plan)
    out: list[dict] = []
    run: list[dict] = []
    run_start_idx = 0
    stats = {"blended": 0, "skipped": 0}

    def flush(run_end_idx: int):
        if not run:
            return
        # A run is one operation's trajectory segments (grippers delimit operations). Skip it if an
        # ops filter is set and this operation is not in it -- its original segments pass through.
        if config.ops is not None and _op_name(run[0]["label"]) not in config.ops:
            out.extend(run)
            stats["skipped"] += 1
            run.clear()
            return
        # Rest (zero boundary speed) only where the arm is genuinely stationary: the episode's very
        # start (this run opens the plan) and end (it closes the plan). Every other boundary abuts a
        # gripper event, so carry a nonzero speed there to keep those frames out of the idle filter.
        lead_speed = 0.0 if run_start_idx == 0 else config.boundary_speed
        trail_speed = 0.0 if run_end_idx == n_steps - 1 else config.boundary_speed
        try:
            out.append(_retime_trajectory_steps(run, config, vel_limit, acc_limit, lead_speed, trail_speed))
            stats["blended"] += 1
        except Exception:
            # Best-effort: on any numerical/shape surprise, keep the original segments for this run
            # rather than failing the whole plan.
            _log.exception("Stroke re-timing failed for a segment run; keeping original segments")
            out.extend(run)
        run.clear()

    for idx, step in enumerate(cutamp_plan):
        if step.get("type") == "trajectory":
            if not run:
                run_start_idx = idx
            run.append(step)
        else:
            flush(idx - 1)
            out.append(step)
    flush(n_steps - 1)

    scope = "all operations" if config.ops is None else f"operations {list(config.ops)}"
    n_before = sum(1 for s in cutamp_plan if s.get("type") == "trajectory")
    n_after = sum(1 for s in out if s.get("type") == "trajectory")
    _log.info(
        f"Stroke re-timing ({scope}): {stats['blended']} operation groups re-timed into single strokes, "
        f"{stats['skipped']} left as planned ({n_before} trajectory segments -> {n_after} steps)"
    )
    return out


def arm_joint_limits(motion_gen, dof: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Best-effort (velocity_limit, acceleration_limit) arrays for the first ``dof`` arm joints.

    Reads the robot's joint limits from ``motion_gen`` and returns the upper (positive) velocity and
    acceleration bounds for the arm joints (the plan's dof), or (None, None) if they can't be read --
    in which case the re-timer falls back to plan-derived caps.
    """
    try:
        jl = motion_gen.kinematics.get_joint_limits()
        vel = jl.velocity[1].detach().cpu().numpy().astype(np.float64)  # upper bound per joint
        acc = jl.acceleration[1].detach().cpu().numpy().astype(np.float64)
        # cuRobo cspace lists the arm joints first, then any gripper joints; the plan is the arm dof.
        return vel[:dof], acc[:dof]
    except Exception:
        _log.exception("Could not read robot joint limits for stroke re-timing; using fallback caps")
        return None, None
