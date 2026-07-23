"""Neural trajectory blending: the same continuous-stroke blend as :mod:`trajectory_blending`, but the
hand-tuned TIME LAW is replaced by a network that learned human stroke timing from DROID teleop.

This is the ``blend_mode: neural`` counterpart of the analytic ``blend_mode: spline`` blender. Everything
except the timing is *identical* and reused directly from :mod:`trajectory_blending`:

* Geometry -- unchanged. The group's joined waypoints are de-duplicated and fit with the SAME penalized
  smoothing spline over arc length (``_dedup_path`` / ``_fit_geometry``), so the blended path stays the
  collision-checked one (rounded corners only), exactly as in spline mode.
* Pace / limits / endpoints -- unchanged. The stroke duration still starts at the group's original
  wall-clock and is shortened only within the robot's real vel/accel caps; rest ends are pinned to exact
  zero and gripper-adjacent ends carry a nonzero boundary speed so those frames clear the openpi non-idle
  filter (``_finish_stroke`` / ``_resolve_caps`` / ``arm_joint_limits``).

What changes: instead of a symmetric min-jerk quintic (``_time_law``) or the asymmetric dumbbell
(``_asymmetric_stroke``), the speed-vs-time profile ``p(tau)`` comes from a :class:`~tiptop.networks.
timing_net.TimingModel`, conditioned on the stroke's GEOMETRY + PACE features (arc length, straightness,
average speed, curvature -- see ``stroke_features``). The learned ``p`` is used as the base shape; the
boundary levels the non-idle filter requires are then enforced on top of it (see :func:`_neural_stroke`),
so the model supplies the STYLE (where along the reach the arm speeds up / slows into the grasp) while the
hard constraints remain analytic.

Robustness: any per-stroke failure (a missing model, a numerical surprise) falls back to the analytic
spline ``blend_group`` for that stroke, and a failure to load the model at all falls back to spline mode
for the whole plan (handled by the caller in ``planning.run_planning``). Like spline blending this is a
pure post-process over the joint-waypoint arrays; it runs in ``run_planning`` so the saved and executed
plans are the identical blended object.

Enable per config (``cfg/tamp/*.yml`` ``tamp_overrides``)::

    blend_trajectory: true
    blend_mode: neural
    blend_model_path: "checkpoints/timing_net.pt"   # optional; defaults to the model's own default
"""

from __future__ import annotations

import logging

import numpy as np

from tiptop.networks.timing_net import TimingModel
from tiptop.trajectory_blending import (
    _MAX_BOUNDARY_RATIO,
    _MIN_CHORD,
    BlendConfig,
    _dedup_path,
    _eval_geometry,
    _finish_stroke,
    _fit_geometry,
    _op_name,
    _resolve_caps,
    _smootherstep,
    blend_group,
)

_log = logging.getLogger(__name__)

# Dense grid the learned profile is evaluated + integrated on (matches _asymmetric_stroke's 2048).
_PROFILE_SAMPLES = 2048
# Fraction-of-duration window over which the analytic boundary correction is blended into the learned
# profile at each end (mirrors _asymmetric_stroke's window clamp). Kept small so the learned interior
# shape is preserved and only the endpoint level is coerced to satisfy the non-idle / rest constraint.
_END_WINDOW_FRAC = 0.15


def _neural_stroke(
    geom: list,
    length: float,
    t0: float,
    t1: float,
    dt: float,
    target_duration: float,
    vel_cap: np.ndarray,
    acc_cap: np.ndarray,
    lead_speed: float,
    trail_speed: float,
    profile_fn,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Re-time a stroke using a LEARNED speed profile, then enforce the boundary / limit constraints.

    Mirrors :func:`trajectory_blending._asymmetric_stroke` -- build a strictly-positive speed-vs-time
    profile ``p(tau)``, integrate to the time law ``h(tau)``, sample, and stretch the duration to fit the
    vel/accel caps -- but the base shape of ``p`` is the network's output (``profile_fn(tau)``) instead of
    the hand-built dumbbell. The only thing forced onto the learned shape is the ENDPOINT level, and only
    within a short end window: a rest end (``*_speed == 0``) is ramped to zero, and a gripper-adjacent end
    is lifted to the level that realizes the requested ``*_speed`` **iff the learned profile falls short**
    there -- so the human decel-into-grasp the model learned is preserved unless it would drop the frame
    into the idle band. Returns ``(pos, vel, acc, duration)``.
    """
    v_cruise = length / target_duration  # the plan's average pace = the cruise reference
    t_mid = float(np.median(np.linalg.norm(_eval_geometry(geom, np.linspace(0.0, length, 256), 1), axis=1))) or 1.0

    tau_d = np.linspace(0.0, 1.0, _PROFILE_SAMPLES)
    p_nn = np.asarray(profile_fn(tau_d), dtype=np.float64)
    p_nn = np.clip(p_nn, 1e-3, None)
    p_nn = p_nn / (np.trapezoid(p_nn, tau_d) or 1.0)  # unit-mean over tau: cruise reference = 1

    # Required end level (relative to cruise) to realize the requested boundary joint speed. Dividing by
    # the endpoint tangent compensates a collapsed grasp-reversal tangent, exactly as in _asymmetric_stroke.
    r_lead = 0.0 if lead_speed == 0.0 else min(_MAX_BOUNDARY_RATIO, (lead_speed / v_cruise) * (t_mid / t0))
    r_trail = 0.0 if trail_speed == 0.0 else min(_MAX_BOUNDARY_RATIO, (trail_speed / v_cruise) * (t_mid / t1))

    w = float(np.clip(_END_WINDOW_FRAC, 0.03, 0.35))  # end-window as a fraction of normalized time
    shape_lead = np.where(tau_d < w, 1.0 - _smootherstep(tau_d / w), 0.0)  # 1 at tau=0 -> 0 past the window
    shape_trail = np.where(tau_d > 1.0 - w, 1.0 - _smootherstep((1.0 - tau_d) / w), 0.0)

    p = p_nn.copy()
    # Lead end: ramp to rest, or lift to the boundary level only if the learned profile is below it.
    if lead_speed == 0.0:
        p = p * (1.0 - shape_lead)
    elif p_nn[0] < r_lead:
        p = p + (r_lead - p_nn[0]) * shape_lead
    # Trail end: same.
    if trail_speed == 0.0:
        p = p * (1.0 - shape_trail)
    elif p_nn[-1] < r_trail:
        p = p + (r_trail - p_nn[-1]) * shape_trail
    p = np.maximum(p, 0.0)

    pbar = float(np.trapezoid(p, tau_d)) or 1.0
    # h(tau) in [0,1]: normalized arc length vs normalized time (cumulative p, scaled so h(1)=1).
    h = np.concatenate([[0.0], np.cumsum((p[1:] + p[:-1]) / 2.0 * np.diff(tau_d))]) / pbar

    def sample(duration: float, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ts = np.linspace(0.0, 1.0, n)
        ut = length * np.interp(ts, tau_d, h)
        ps = np.interp(ts, tau_d, p)
        pos = _eval_geometry(geom, ut, 0)
        du_dt = length * (ps / pbar) / duration
        vel = _eval_geometry(geom, ut, 1) * du_dt[:, None]
        acc = np.gradient(vel, duration / (n - 1), axis=0)
        return pos, vel, acc

    # Duration so the plan pace is held through the (unit-mean) profile, then stretch to fit the caps.
    duration = t_mid * length / (pbar * v_cruise)
    _, v0, a0 = sample(duration, 512)
    alpha_v = float((np.abs(v0).max(axis=0) / vel_cap).max())
    alpha_a = float(np.sqrt((np.abs(a0).max(axis=0) / acc_cap).max()))
    duration *= max(1.0, alpha_v, alpha_a)
    n = max(2, int(round(duration / dt)) + 1)
    pos, vel, acc = sample(duration, n)
    return pos, vel, acc, duration


def neural_blend_group(
    positions: np.ndarray,
    dt: float,
    target_duration: float,
    vel_cap: np.ndarray,
    acc_cap: np.ndarray,
    model: TimingModel,
    smoothing: float = None,
    lead_speed: float = 0.0,
    trail_speed: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Re-time one group of joined waypoints into a single stroke using the LEARNED (geometry) time law.

    Geometry is built exactly as in :func:`trajectory_blending.blend_group` (de-dup + arc-length +
    smoothing spline); only the time law differs. The stroke's geometry/pace features select the network's
    speed profile. ``lead_speed`` / ``trail_speed`` are the requested boundary joint speeds (0 = rest),
    used identically to spline mode. On any failure this raises; the caller falls back to ``blend_group``.
    """
    from tiptop.trajectory_blending import _DEFAULT_SMOOTHING

    if smoothing is None:
        smoothing = _DEFAULT_SMOOTHING

    positions = _dedup_path(np.asarray(positions, dtype=np.float64))

    # Degenerate group (no motion): a single held pose -> two stationary, zero-velocity samples.
    if len(positions) < 2:
        pos = np.repeat(positions[:1], 2, axis=0)
        zeros = np.zeros_like(pos)
        return pos, zeros, zeros, dt

    u = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=1))])
    length = float(u[-1])
    if length < _MIN_CHORD:  # no meaningful net motion: hold the pose.
        pos = np.repeat(positions[:1], 2, axis=0)
        zeros = np.zeros_like(pos)
        return pos, zeros, zeros, dt

    geom = _fit_geometry(u, positions, smoothing)
    t0 = float(np.linalg.norm(_eval_geometry(geom, np.array([0.0]), 1)[0])) or 1.0
    t1 = float(np.linalg.norm(_eval_geometry(geom, np.array([length]), 1)[0])) or 1.0

    # Geometry/pace features of THIS stroke (arc length, straightness, avg speed = arc/target_duration,
    # curvature). None for a degenerate path -> profile_on falls back to the feature mean.
    feats = None
    try:
        feats = model.features(positions, target_duration)
    except Exception:
        _log.exception("Geometry feature computation failed; timing runs from the feature mean")

    def profile_fn(tau: np.ndarray) -> np.ndarray:
        return model.profile_on(tau, feats)

    out_pos, out_vel, out_acc, duration = _neural_stroke(
        geom, length, t0, t1, dt, target_duration, vel_cap, acc_cap, lead_speed, trail_speed, profile_fn
    )
    return _finish_stroke(out_pos, out_vel, out_acc, duration, lead_speed, trail_speed)


def _blend_trajectory_steps_neural(
    steps: list[dict],
    config: BlendConfig,
    model: TimingModel,
    vel_limit: np.ndarray | None,
    acc_limit: np.ndarray | None,
    lead_speed: float,
    trail_speed: float,
) -> dict:
    """Blend a run of consecutive ``trajectory`` steps into one re-timed step using the learned time law.

    Parallels :func:`trajectory_blending._blend_trajectory_steps`; only the timing call differs. Any
    numerical failure in the neural path falls back to the analytic ``blend_group`` for this stroke so a
    single bad stroke never loses the whole plan.
    """
    from curobo.types.state import JointState  # lazy: keep the module importable without cuRobo
    import torch

    template = steps[0]["plan"]
    device = template.position.device
    joint_names = template.joint_names
    dt = float(steps[0]["dt"])

    seg_positions = [s["plan"].position.detach().cpu().numpy().astype(np.float64) for s in steps]
    joined = np.concatenate([seg_positions[0]] + [p[1:] for p in seg_positions[1:]], axis=0)
    orig_velocities = np.concatenate(
        [s["plan"].velocity.detach().cpu().numpy().astype(np.float64) for s in steps], axis=0
    )
    target_duration = sum((len(p) - 1) for p in seg_positions) * dt / config.speed_scale

    vel_cap, acc_cap = _resolve_caps(
        orig_velocities, dt, vel_limit, acc_limit, config.vel_slack, config.acc_slack
    )
    try:
        pos, vel, acc, dt_out = neural_blend_group(
            joined, dt, target_duration, vel_cap, acc_cap, model, config.smoothing, lead_speed, trail_speed,
        )
    except Exception:
        _log.exception("Neural timing failed for a stroke; falling back to the analytic spline time law")
        pos, vel, acc, dt_out = blend_group(
            joined, dt, target_duration, vel_cap, acc_cap, config.smoothing, lead_speed, trail_speed,
            config.boundary_window,
        )

    plan = JointState(
        position=torch.as_tensor(pos, dtype=torch.float32, device=device),
        velocity=torch.as_tensor(vel, dtype=torch.float32, device=device),
        acceleration=torch.as_tensor(acc, dtype=torch.float32, device=device),
        jerk=torch.zeros(pos.shape, dtype=torch.float32, device=device),
        joint_names=list(joint_names) if joint_names is not None else None,
    )
    return {"type": "trajectory", "plan": plan, "dt": dt_out, "label": steps[0]["label"]}


def neural_blend_cutamp_plan(
    cutamp_plan: list[dict],
    config: BlendConfig,
    model: TimingModel,
    vel_limit: np.ndarray | None = None,
    acc_limit: np.ndarray | None = None,
) -> list[dict]:
    """Return ``cutamp_plan`` with each operation's trajectory segments blended into one LEARNED-timed stroke.

    Structurally identical to :func:`trajectory_blending.blend_cutamp_plan` (gripper steps pass through and
    delimit the groups; ``config.ops`` restricts which operations are blended), but each group is re-timed
    by the geometry-conditioned network. As in spline mode, a boundary is rest (speed 0) only at the plan's
    very start/end; every gripper-adjacent boundary carries ``config.boundary_speed`` to stay out of the
    non-idle filter. A no-op when ``config.enabled`` is False.
    """
    if not config.enabled:
        return cutamp_plan

    n_steps = len(cutamp_plan)
    out: list[dict] = []
    run: list[dict] = []
    run_start_idx = 0
    stats = {"blended": 0, "skipped": 0, "fallback": 0}

    def flush(run_end_idx: int):
        if not run:
            return
        if config.ops is not None and _op_name(run[0]["label"]) not in config.ops:
            out.extend(run)
            stats["skipped"] += 1
            run.clear()
            return
        # Boundary speed: rest (0) only where the arm is genuinely stationary (plan start/end); a nonzero
        # speed at every gripper-adjacent boundary keeps those frames out of the non-idle filter.
        lead_speed = 0.0 if run_start_idx == 0 else config.boundary_speed
        trail_speed = 0.0 if run_end_idx == n_steps - 1 else config.boundary_speed
        try:
            out.append(
                _blend_trajectory_steps_neural(run, config, model, vel_limit, acc_limit, lead_speed, trail_speed)
            )
            stats["blended"] += 1
        except Exception:
            _log.exception("Neural blending failed for a segment run; keeping original segments")
            out.extend(run)
            stats["fallback"] += 1
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
    _log.info(
        "Neural trajectory blending (%s): %d operation groups re-timed by the learned model, %d left "
        "unblended, %d fell back to originals (%d trajectory segments -> %d steps)",
        scope,
        stats["blended"],
        stats["skipped"],
        stats["fallback"],
        sum(1 for s in cutamp_plan if s.get("type") == "trajectory"),
        sum(1 for s in out if s.get("type") == "trajectory"),
    )
    return out
