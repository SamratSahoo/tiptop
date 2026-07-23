"""Flow-matching trajectory blending: SAMPLE a whole human stroke per operation, then constrain it.

This is the ``blend_mode: flow`` backend. Where ``neural_blending`` applies a single DETERMINISTIC learned
speed profile, this SAMPLES a full stroke from a conditional flow-matching model (:class:`~tiptop.networks.
flow_timing.FlowModel`) for each operation -- so every generated episode draws a different human-like
realization (decelerate-into-grasp vs constant-velocity ...), reproducing the DISTRIBUTION of teleoperator
styles instead of one averaged blend.

Pipeline per operation group (trajectory segments between two gripper events):
  1. Sample the flow model conditioned on the group's joined path -> a full stroke ``x`` (joint positions
     over normalized time, endpoint-pinned to the operation's grasp/place).
  2. Feed that sample's GEOMETRY and its own speed profile into the exact same constraint engine
     ``neural_blending._neural_stroke`` used by the deterministic backend, which enforces the FR3 vel/accel
     caps, pins rest ends to zero, lifts gripper-adjacent ends to the non-idle boundary speed, and resamples
     to the plan ``dt``. So the model supplies the (multimodal, human) geometry+timing while the hard
     constraints remain guaranteed.

The flow sample provides the geometry; if collision-safety near clutter is a concern, pass the cuTAMP path
as the geometry instead (one line in :func:`flow_blend_group`) -- the flow timing is the part that matters.

Robustness mirrors ``neural_blending``: a per-stroke failure falls back to the analytic ``blend_group`` for
that stroke, and a model-load failure falls back to spline mode for the whole plan (handled in
``planning.run_planning``). Enable per config::

    blend_trajectory: true
    blend_mode: flow
    blend_model_path: "checkpoints/flow_net.pt"   # optional; defaults to the model's own default
"""

from __future__ import annotations

import logging

import numpy as np

from tiptop.networks.flow_timing import FlowModel
from tiptop.neural_blending import _neural_stroke
from tiptop.trajectory_blending import (
    _DEFAULT_SMOOTHING,
    _MIN_CHORD,
    BlendConfig,
    _dedup_path,
    _eval_geometry,
    _finish_stroke,
    _fit_geometry,
    _op_name,
    _resolve_caps,
    blend_group,
)

_log = logging.getLogger(__name__)


def _flow_speed_profile(x_flow: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit-mean speed-vs-normalized-time profile of a sampled stroke -> (tau_mid, p)."""
    s = np.linalg.norm(np.diff(x_flow, axis=0), axis=1)  # arc-length speed over the T-1 intervals
    tau_mid = (np.arange(len(s)) + 0.5) / len(s)
    p = np.clip(s / (s.mean() or 1.0), 1e-3, None)
    return tau_mid, p


def flow_blend_group(
    positions: np.ndarray,
    dt: float,
    target_duration: float,
    vel_cap: np.ndarray,
    acc_cap: np.ndarray,
    flow_model: FlowModel,
    n_steps: int = 60,
    smoothing: float = _DEFAULT_SMOOTHING,
    lead_speed: float = 0.0,
    trail_speed: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Sample the flow model conditioned on the group's path, then re-time it under the hard constraints.

    ``positions`` are the group's joined cuTAMP waypoints; ``lead_speed`` / ``trail_speed`` the requested
    boundary joint speeds (0 = rest), as in spline / neural mode. Returns (pos, vel, acc, dt_out). Raises on
    failure so the caller can fall back to the analytic ``blend_group``.
    """
    positions = _dedup_path(np.asarray(positions, dtype=np.float64))
    if len(positions) < 2:
        pos = np.repeat(positions[:1], 2, axis=0)
        zeros = np.zeros_like(pos)
        return pos, zeros, zeros, dt

    # Sample a full human-like stroke conditioned on this path (endpoint-pinned to positions[-1]).
    samp = flow_model.sample(positions, n_samples=1, n_steps=n_steps)
    if samp is None:
        raise ValueError("flow model returned no sample (degenerate path)")
    x_flow = samp[0] + positions[0]  # start-centered sample -> absolute joint positions [T, dof]

    # Geometry from the flow sample (its learned path). Swap `geom_pos = positions` here to instead keep the
    # collision-checked cuTAMP geometry and use only the flow's timing.
    geom_pos = _dedup_path(x_flow)
    if len(geom_pos) < 2:
        raise ValueError("degenerate flow geometry")
    u = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(geom_pos, axis=0), axis=1))])
    length = float(u[-1])
    if length < _MIN_CHORD:
        pos = np.repeat(geom_pos[:1], 2, axis=0)
        zeros = np.zeros_like(pos)
        return pos, zeros, zeros, dt

    geom = _fit_geometry(u, geom_pos, smoothing)
    t0 = float(np.linalg.norm(_eval_geometry(geom, np.array([0.0]), 1)[0])) or 1.0
    t1 = float(np.linalg.norm(_eval_geometry(geom, np.array([length]), 1)[0])) or 1.0

    tau_mid, p = _flow_speed_profile(x_flow)

    def profile_fn(tau: np.ndarray) -> np.ndarray:
        return np.interp(np.asarray(tau, dtype=np.float64), tau_mid, p)

    out_pos, out_vel, out_acc, duration = _neural_stroke(
        geom, length, t0, t1, dt, target_duration, vel_cap, acc_cap, lead_speed, trail_speed, profile_fn
    )
    return _finish_stroke(out_pos, out_vel, out_acc, duration, lead_speed, trail_speed)


def _blend_trajectory_steps_flow(
    steps: list[dict],
    config: BlendConfig,
    flow_model: FlowModel,
    vel_limit: np.ndarray | None,
    acc_limit: np.ndarray | None,
    lead_speed: float,
    trail_speed: float,
) -> dict:
    """Blend a run of consecutive ``trajectory`` steps into one flow-sampled, re-timed trajectory step."""
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
        pos, vel, acc, dt_out = flow_blend_group(
            joined, dt, target_duration, vel_cap, acc_cap, flow_model, config.flow_steps,
            config.smoothing, lead_speed, trail_speed,
        )
    except Exception:
        _log.exception("Flow sampling failed for a stroke; falling back to the analytic spline time law")
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


def flow_blend_cutamp_plan(
    cutamp_plan: list[dict],
    config: BlendConfig,
    flow_model: FlowModel,
    vel_limit: np.ndarray | None = None,
    acc_limit: np.ndarray | None = None,
) -> list[dict]:
    """Return ``cutamp_plan`` with each operation's trajectory segments replaced by a FLOW-SAMPLED stroke.

    Structurally identical to :func:`neural_blending.neural_blend_cutamp_plan`: gripper steps pass through
    and delimit the groups; ``config.ops`` restricts which operations are blended; a boundary is rest only
    at the plan start/end, else it carries ``config.boundary_speed`` for the non-idle filter. A no-op when
    ``config.enabled`` is False.
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
        lead_speed = 0.0 if run_start_idx == 0 else config.boundary_speed
        trail_speed = 0.0 if run_end_idx == n_steps - 1 else config.boundary_speed
        try:
            out.append(
                _blend_trajectory_steps_flow(run, config, flow_model, vel_limit, acc_limit, lead_speed, trail_speed)
            )
            stats["blended"] += 1
        except Exception:
            _log.exception("Flow blending failed for a segment run; keeping original segments")
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
        "Flow trajectory blending (%s): %d operation groups sampled + re-timed, %d left unblended, %d fell "
        "back to originals (%d trajectory segments -> %d steps)",
        scope, stats["blended"], stats["skipped"], stats["fallback"],
        sum(1 for s in cutamp_plan if s.get("type") == "trajectory"),
        sum(1 for s in out if s.get("type") == "trajectory"),
    )
    return out
