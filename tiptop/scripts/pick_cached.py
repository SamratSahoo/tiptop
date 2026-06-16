"""Pick from a cached scene using cached M2T2 grasp candidates (no Gemini re-run).

Used by cortex_tamp_server's /pick when there's a fresh /perceive trial dir on
disk: that /perceive already ran Gemini-ER + SAM2 + FoundationStereo + M2T2 and
wrote scene_objects.json with per-object M2T2 grasp candidates (world_from_tcp
poses + confidences). This script reads them and just plans+executes:

  1. Resolve label (exact / substring / token-overlap fuzzy)
  2. Load grasps_world_from_tcp + grasp_confidences for the target
  3. Iterate grasps in descending confidence order, plan with cuRobo; execute
     the first one that plans successfully
  4. Close gripper, lift back to a pre-grasp height (current EE + approach_z_m)

This gives bowls/cups proper RIM grasps (M2T2 found them at /perceive time)
without paying the ~6s Gemini cost on every pick.

If the cached file has no M2T2 grasps for this object (older scene_objects.json
written before the 2026-06-02 patch, or M2T2 produced zero candidates for this
object), we fall back to a simple top-down centroid grasp — the prior pick_cached
behavior.
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
from tiptop.motion_planning import build_curobo_solvers, get_motion_gen
from tiptop.utils import get_robot_client, setup_logging
from curobo.geom.types import Cuboid, WorldConfig
from tiptop.workspace import workspace_cuboids

_log = logging.getLogger(__name__)


def _resolve_label(label: str, centroids: dict) -> str:
    """exact → substring → token-overlap (>=2 shared >=3-char tokens)."""
    if label in centroids:
        return label
    ll = label.lower()
    cands = [k for k in centroids if ll in k.lower() or k.lower() in ll]
    if cands:
        return cands[0]
    import re
    def toks(s):
        return {t for t in re.split(r"[^a-z0-9]+", s.lower()) if len(t) >= 3}
    qtoks = toks(label)
    scored = []
    for k in centroids:
        ktoks = toks(k)
        overlap = len(qtoks & ktoks)
        if overlap >= 2:
            scored.append((overlap, len(ktoks), k))
    if scored:
        scored.sort(key=lambda x: (-x[0], x[1]))
        return scored[0][2]
    raise ValueError(
        f"PICK_CACHED: label {label!r} not in centroids. Known: {sorted(centroids)}"
    )


def pick_cached(
    label: str,
    centroids_file: Path,
    approach_z_m: float = 0.10,
    finger_depth_m: float = 0.015,
    grip_force: float = 80.0,
    max_grasps_to_try: int = 10,
) -> None:
    setup_logging()
    centroids_file = Path(centroids_file)
    if not centroids_file.exists():
        raise FileNotFoundError(f"centroids file not found: {centroids_file}")
    centroids = json.loads(centroids_file.read_text())
    if not centroids:
        raise ValueError(f"centroids file is empty: {centroids_file}")

    resolved = _resolve_label(label, centroids)
    if resolved != label:
        _log.info(f"Resolved {label!r} -> {resolved!r}")

    entry = centroids[resolved]
    if not isinstance(entry, dict):
        # Old format with just [x,y,z]. Fall straight back to centroid path.
        cx, cy, cz = entry[:3]
        return _pick_centroid_fallback(
            cx, cy, cz, 0.0, approach_z_m, finger_depth_m, grip_force, resolved
        )

    cx, cy, cz = entry["centroid"][:3]
    extents = entry.get("extents") or [0, 0, 0]
    half_height = float(extents[2]) / 2.0 if len(extents) >= 3 else 0.0

    grasps = entry.get("grasps_world_from_tcp") or []
    confidences = entry.get("grasp_confidences") or []
    if grasps and confidences and len(grasps) == len(confidences):
        return _pick_m2t2_grasp(
            resolved,
            grasps,
            confidences,
            cx,
            cy,
            cz,
            half_height,
            approach_z_m,
            finger_depth_m,
            grip_force,
            max_grasps_to_try,
        )
    _log.info(
        f"No M2T2 grasps cached for {resolved!r} — falling back to top-down centroid grasp"
    )
    return _pick_centroid_fallback(
        cx, cy, cz, half_height, approach_z_m, finger_depth_m, grip_force, resolved
    )


def _pick_m2t2_grasp(
    label: str,
    grasps,
    confidences,
    cx: float,
    cy: float,
    cz: float,
    half_height: float,
    approach_z_m: float,
    finger_depth_m: float,
    grip_force: float,
    max_grasps_to_try: int,
) -> None:
    client = get_robot_client()
    cfg = tiptop_cfg()
    # Build motion_gen WITHOUT cuda graphs — they cache the goal type+size, which
    # makes plan_goalset with different N counts fail with "changing goal type,
    # cuda graph reset not available". We also need plan_single calls for the
    # descend/lift phases which would conflict with a goalset-cached graph.
    cuboids = [
        *workspace_cuboids(),
        Cuboid(name="table", dims=[0.01, 0.01, 0.01], pose=[99.9, 99.9, 99.9, 1.0, 0.0, 0.0, 0.0]),
    ]
    world_cfg = WorldConfig(cuboid=cuboids)
    motion_gen = get_motion_gen(
        world_cfg,
        collision_activation_distance=0.01,
        num_spheres=8,
        warmup_iters=4,
        use_cuda_graph=False,
    )

    # M2T2 grasps come pre-sorted (descending confidence) by the tiptop_run patch.
    grasps_np = np.asarray(grasps, dtype=np.float32)
    confs_np = np.asarray(confidences, dtype=np.float32)
    n_candidates = min(max_grasps_to_try, len(grasps_np))
    grasps_np = grasps_np[:n_candidates]
    confs_np = confs_np[:n_candidates]
    _log.info(
        f"Pick {label}: planning over {n_candidates} M2T2 grasps via plan_goalset "
        f"(top conf={confs_np[0]:.3f})"
    )

    def js_from_current():
        q_now = client.get_joint_positions()
        q_now_pt = torch.tensor(q_now, dtype=torch.float32, device="cuda")
        return JointState.from_position(q_now_pt[None])

    def pose_single_from_matrix(mat: np.ndarray) -> Pose:
        """Build a single Pose from a (4, 4) array — for plan_single."""
        if mat.ndim == 2:
            mat = mat[None]  # add batch dim → (1, 4, 4)
        t = torch.tensor(mat, dtype=torch.float32, device="cuda")
        return Pose.from_matrix(t)

    def pose_goalset_from_matrices(mats: np.ndarray) -> Pose:
        """Build a goalset Pose from (N, 4, 4). cuRobo plan_goalset requires the
        position/quaternion tensors to have shape (batch, n_goalset, 3 or 4) with
        batch=1 (single start state)."""
        # Build a (1, N, 4, 4) tensor by adding a leading batch dim.
        if mats.ndim == 3:  # (N, 4, 4)
            mats = mats[None]  # → (1, N, 4, 4)
        t = torch.tensor(mats, dtype=torch.float32, device="cuda")
        # Pose.from_matrix flattens 4x4 → position(3) + quat(4), preserving leading dims.
        return Pose.from_matrix(t)

    def plan_set(target_mats: np.ndarray):
        target_pose = pose_goalset_from_matrices(target_mats)
        plan_config = MotionGenPlanConfig(time_dilation_factor=cfg.robot.time_dilation_factor)
        return motion_gen.plan_goalset(js_from_current(), target_pose, plan_config)

    def plan_single_mat(mat: np.ndarray):
        # Use plan_goalset(size=1) instead of plan_single — cuRobo can't switch
        # between plan_single and plan_goalset in the same MotionGen instance
        # ("changing goal type, cuda graph reset not available" error).
        target_pose = pose_goalset_from_matrices(mat[None] if mat.ndim == 2 else mat)
        plan_config = MotionGenPlanConfig(time_dilation_factor=cfg.robot.time_dilation_factor)
        return motion_gen.plan_goalset(js_from_current(), target_pose, plan_config)

    def execute(result):
        plan = result.interpolated_plan
        dt = result.interpolation_dt
        timings = [dt] * plan.position.shape[0]
        exec_result = client.execute_joint_impedance_path(
            joint_confs=plan.position.cpu().numpy(),
            joint_vels=plan.velocity.cpu().numpy(),
            durations=timings,
        )
        if not exec_result["success"]:
            raise RuntimeError(
                f"PICK_CACHED: trajectory execute failed: {exec_result.get('error')}"
            )
        _log.info("Executed trajectory on the robot")

    # Convert M2T2 grasp poses to cuRobo's ee-link frame. cuTAMP does:
    #   world_from_ee = world_from_grasp @ tool_from_ee
    # so motion_gen sees the ee-link target (not the gripper tip). For fr3_robotiq
    # the offset is rotation RPY(π, 0, π/2) + translation (0, 0, 0.015).
    # See cutamp/robots/__init__.py::load_fr3_robotiq_container.
    tool_from_ee = np.eye(4, dtype=np.float32)
    # RPY(π, 0, π/2) via X-Y-Z intrinsic
    cx_, sx_ = np.cos(np.pi), np.sin(np.pi)
    cy_, sy_ = np.cos(0.0), np.sin(0.0)
    cz_, sz_ = np.cos(np.pi / 2.0), np.sin(np.pi / 2.0)
    Rx = np.array([[1, 0, 0], [0, cx_, -sx_], [0, sx_, cx_]], dtype=np.float32)
    Ry = np.array([[cy_, 0, sy_], [0, 1, 0], [-sy_, 0, cy_]], dtype=np.float32)
    Rz = np.array([[cz_, -sz_, 0], [sz_, cz_, 0], [0, 0, 1]], dtype=np.float32)
    tool_from_ee[:3, :3] = Rx @ Ry @ Rz
    tool_from_ee[:3, 3] = [0.0, 0.0, 0.015]
    # world_from_ee = world_from_grasp @ tool_from_ee (per-grasp)
    ee_targets = grasps_np @ tool_from_ee  # (N, 4, 4)

    # Pre-grasps: each ee_target backed off along its local -z (approach direction)
    # by approach_z_m. cuRobo finds whichever pre-grasp is reachable.
    pre_grasps = ee_targets.copy()
    pre_grasps[:, :3, 3] = ee_targets[:, :3, 3] - ee_targets[:, :3, 2] * approach_z_m

    t0 = time.perf_counter()
    result = plan_set(pre_grasps)
    _log.info(
        f"Phase 1/3: plan_goalset over {n_candidates} pre-grasps took "
        f"{time.perf_counter() - t0:.2f}s, success={result.success}"
    )
    if not result.success:
        raise RuntimeError(
            f"PICK_CACHED: plan_goalset motion planning failed on all {n_candidates} "
            f"M2T2 grasps for {label}: {getattr(result, 'status', '?')}"
        )

    # Identify which grasp index cuRobo selected (the goalset_index attr) so we can
    # descend to the matching grasp pose in Phase 2.
    chosen_idx = 0
    for attr in ("goalset_index", "selected_idx", "ik_seed_succ"):
        v = getattr(result, attr, None)
        if v is None:
            continue
        try:
            chosen_idx = int(v.item() if hasattr(v, "item") else (v[0] if hasattr(v, "__getitem__") else v))
            _log.info(f"plan_goalset chose grasp #{chosen_idx} via attr={attr}")
            break
        except Exception:
            continue
    chosen_ee = ee_targets[chosen_idx]
    _log.info(f"Phase 1/3: move to pre-grasp (chosen grasp #{chosen_idx}, conf={confs_np[chosen_idx]:.3f})")
    execute(result)
    client.open_gripper(speed=1.0, force=0.1)

    _log.info("Phase 2/3: descend to grasp pose")
    result = plan_single_mat(chosen_ee)
    if not result.success:
        _log.warning("descend plan_single failed; retrying with plan_goalset on close neighbors")
        result = plan_set(ee_targets[:min(5, n_candidates)])
        if not result.success:
            raise RuntimeError(
                f"PICK_CACHED: motion planning failed at grasp pose: {getattr(result, 'status', '?')}"
            )
    execute(result)

    _log.info("Phase 3a: close gripper")
    grip_result = client.close_gripper(speed=1.0, force=float(grip_force))
    _log.info(f"close_gripper -> {grip_result}")

    _log.info("Phase 3b: lift back to pre-grasp")
    lift = chosen_ee.copy()
    lift[:3, 3] = lift[:3, 3] - lift[:3, 2] * approach_z_m
    result = plan_single_mat(lift)
    if not result.success:
        _log.warning(f"lift plan failed: {getattr(result, 'status', '?')}; trying simple z-lift")
        lift_z = chosen_ee.copy()
        lift_z[2, 3] = lift_z[2, 3] + approach_z_m
        result = plan_single_mat(lift_z)
        if not result.success:
            raise RuntimeError(f"PICK_CACHED: lift failed: {getattr(result, 'status', '?')}")
    execute(result)

    client.close()
    print("PICK_CACHED: success")


def _pick_centroid_fallback(
    cx: float,
    cy: float,
    cz: float,
    half_height: float,
    approach_z_m: float,
    finger_depth_m: float,
    grip_force: float,
    label: str,
) -> None:
    """Original top-down centroid grasp — used when no M2T2 grasps are cached."""
    object_top_z = float(cz) + half_height
    pre_grasp_z = object_top_z + float(approach_z_m)
    grasp_z = object_top_z - float(finger_depth_m)
    _log.info(
        f"Pick {label} (centroid fallback): top_z={object_top_z:.3f}, "
        f"pre={pre_grasp_z:.3f}, grasp={grasp_z:.3f}"
    )

    client = get_robot_client()
    cfg = tiptop_cfg()
    ik_solver, motion_gen, _ = build_curobo_solvers(
        num_particles=100, num_spheres=8, collision_activation_distance=0.01
    )
    q_curr = client.get_joint_positions()
    q_curr_pt = torch.tensor(q_curr, dtype=torch.float32, device="cuda")
    world_from_ee = motion_gen.kinematics.get_state(q_curr_pt[None]).ee_pose.get_numpy_matrix()[0]

    def plan_to(z_target: float):
        target_mat = world_from_ee.copy()
        target_mat[0, 3] = float(cx)
        target_mat[1, 3] = float(cy)
        target_mat[2, 3] = float(z_target)
        target_pt = torch.tensor(target_mat, dtype=torch.float32, device="cuda")
        target_pose = Pose.from_matrix(target_pt)
        q_now = client.get_joint_positions()
        q_now_pt = torch.tensor(q_now, dtype=torch.float32, device="cuda")
        js = JointState.from_position(q_now_pt[None])
        plan_config = MotionGenPlanConfig(time_dilation_factor=cfg.robot.time_dilation_factor)
        result = motion_gen.plan_single(js, target_pose, plan_config)
        if not result.success:
            raise RuntimeError(
                f"PICK_CACHED: motion planning failed at z={z_target:.3f}: {result.status}"
            )
        return result

    def execute(result):
        plan = result.interpolated_plan
        dt = result.interpolation_dt
        timings = [dt] * plan.position.shape[0]
        exec_result = client.execute_joint_impedance_path(
            joint_confs=plan.position.cpu().numpy(),
            joint_vels=plan.velocity.cpu().numpy(),
            durations=timings,
        )
        if not exec_result["success"]:
            raise RuntimeError(
                f"PICK_CACHED: trajectory execute failed: {exec_result.get('error')}"
            )

    execute(plan_to(pre_grasp_z))
    client.open_gripper(speed=1.0, force=0.1)
    execute(plan_to(grasp_z))
    grip_result = client.close_gripper(speed=1.0, force=float(grip_force))
    _log.info(f"close_gripper -> {grip_result}")
    execute(plan_to(pre_grasp_z))
    client.close()
    print("PICK_CACHED: success")


def entrypoint():
    tyro.cli(pick_cached)


if __name__ == "__main__":
    entrypoint()