"""Drop currently-held object above a labeled target.

This is the fallback path when TiPToP's M2T2-based Place fails or returns no
feasible drop pose. We bypass M2T2 entirely and just:
  1. Read the target's centroid from a previous-run scene_objects.json
  2. Plan a cartesian motion to (cx, cy, cz + clearance) with current EE orientation
  3. Execute the trajectory and open the gripper

Usage:
    pixi run drop-above --target-label green_bowl --centroids-file PATH/scene_objects.json [--z-clearance-m 0.18]
"""
from __future__ import annotations
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import tyro
from curobo.types.math import Pose
from curobo.types.state import JointState
from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig

from tiptop.config import tiptop_cfg
from tiptop.motion_planning import build_curobo_solvers
from tiptop.utils import get_robot_client, setup_logging

_log = logging.getLogger(__name__)


def drop_above(
    target_label: str,
    centroids_file: Path,
    z_clearance_m: float = 0.18,
    dx_m: float = 0.0,
    dy_m: float = 0.0,
    abs_x: float | None = None,
    abs_y: float | None = None,
    abs_z: float | None = None,
) -> None:
    """dx_m / dy_m: drop next to the target instead of on top of it.

    abs_x / abs_y / abs_z: if ALL THREE are provided, skip the centroids lookup
    entirely and drop at absolute world coordinates (cx=abs_x, cy=abs_y,
    cz=abs_z). target_label is then ignored. Use this for memory/restore tasks
    where cortex remembers an object's original position and wants to put it
    back, regardless of any moved-since centroid. dx_m / dy_m / z_clearance_m
    still apply on top of the absolute coords."""
    setup_logging()
    centroids_file = Path(centroids_file)
    if not centroids_file.exists():
        raise FileNotFoundError(f"centroids file not found: {centroids_file}")
    centroids = json.loads(centroids_file.read_text())
    if not centroids:
        raise ValueError(f"centroids file is empty: {centroids_file}")

    # If all three absolute coords are provided, skip label resolution entirely.
    if abs_x is not None and abs_y is not None and abs_z is not None:
        cx, cy, cz = float(abs_x), float(abs_y), float(abs_z)
        half_height = 0.0
        label = f"<abs xyz=({cx:.3f},{cy:.3f},{cz:.3f})>"
        _log.info(f"Using absolute coords {label}, ignoring target_label={target_label!r}")
    else:
        # Resolve label (exact match preferred, fuzzy fallback)
        label = target_label
        if label not in centroids:
            candidates = [k for k in centroids if target_label.lower() in k.lower() or k.lower() in target_label.lower()]
            if not candidates:
                raise ValueError(f"target_label {target_label!r} not in centroids file. Known: {sorted(centroids)}")
            label = candidates[0]
            _log.info(f"Resolved {target_label!r} -> {label!r}")

        # New format: {"centroid": [x,y,z], "extents": [dx,dy,dz]}
        # Back-compat with old format: [x, y, z]
        entry = centroids[label]
        if isinstance(entry, dict):
            cx, cy, cz = entry["centroid"][:3]
            extents = entry.get("extents")
            half_height = (float(extents[2]) / 2.0) if (extents and len(extents) >= 3) else 0.0
        else:
            cx, cy, cz = entry[:3]
            half_height = 0.0
    target_z = float(cz) + half_height + float(z_clearance_m)
    _log.info(
        f"Dropping above {label}: centroid=({cx:.3f}, {cy:.3f}, {cz:.3f}), "
        f"half_height={half_height:.3f}m, clearance={z_clearance_m:.3f}m -> target_z={target_z:.3f}"
    )

    # Robot + curobo
    client = get_robot_client()
    cfg = tiptop_cfg()
    ik_solver, motion_gen, _ = build_curobo_solvers(
        num_particles=100,
        num_spheres=8,
        collision_activation_distance=0.01,
    )

    # PATCH 2026-06-01: move to capture/home pose first so DropAbove always
    # starts from a known, perception-friendly stance. Smoother trajectory and
    # cleaner video framing (the robot returns to the top of frame, then plans
    # to the target).
    from tiptop.motion_planning import go_to_capture
    _log.info("Moving to capture/home pose before DropAbove planning...")
    go_to_capture(time_dilation_factor=cfg.robot.time_dilation_factor, motion_gen=motion_gen)

    # Current EE pose (keep orientation, replace translation) — now at capture pose.
    q_curr = client.get_joint_positions()
    q_curr_pt = torch.tensor(q_curr, dtype=torch.float32, device="cuda")
    world_from_ee = motion_gen.kinematics.get_state(q_curr_pt[None]).ee_pose.get_numpy_matrix()[0]

    target_mat = world_from_ee.copy()
    target_mat[0, 3] = float(cx) + float(dx_m)
    target_mat[1, 3] = float(cy) + float(dy_m)
    target_mat[2, 3] = target_z
    if dx_m or dy_m:
        _log.info(f"Offset drop: dx={dx_m:+.3f}m dy={dy_m:+.3f}m -> drop_xy=({target_mat[0,3]:.3f}, {target_mat[1,3]:.3f})")

    target_pt = torch.tensor(target_mat, dtype=torch.float32, device="cuda")
    target_pose = Pose.from_matrix(target_pt)

    js_curr = JointState.from_position(q_curr_pt[None])
    plan_config = MotionGenPlanConfig(time_dilation_factor=cfg.robot.time_dilation_factor)

    _log.info("Planning motion to above target...")
    t0 = time.perf_counter()
    result = motion_gen.plan_single(js_curr, target_pose, plan_config)
    _log.info(f"Motion planning took {time.perf_counter() - t0:.2f}s")
    if not result.success:
        raise RuntimeError(f"DROP_ABOVE: motion planning failed: {result.status}")

    plan = result.interpolated_plan
    dt = result.interpolation_dt
    timings = [dt] * plan.position.shape[0]

    _log.info(f"Executing trajectory ({plan.position.shape[0]} waypoints)...")
    exec_result = client.execute_joint_impedance_path(
        joint_confs=plan.position.cpu().numpy(),
        joint_vels=plan.velocity.cpu().numpy(),
        durations=timings,
    )
    if not exec_result["success"]:
        raise RuntimeError(f"DROP_ABOVE: trajectory execution failed: {exec_result.get('error')}")
    _log.info("Executed trajectory on the robot")

    _log.info("Opening gripper (release)")
    client.open_gripper(speed=1.0, force=0.1)
    client.close()
    print("DROP_ABOVE: success")


def entrypoint():
    tyro.cli(drop_above)


if __name__ == "__main__":
    entrypoint()
