import logging
import time

import numpy as np
import torch
from curobo.geom.types import Cuboid, WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.state import JointState
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from cutamp.motion_solver import MotionPlanningError
from cutamp.robots import (
    get_panda_robotiq_ik_solver,
    load_fr3_robotiq_container,
    load_panda_container,
    load_panda_robotiq_container,
    load_ur5_container,
    panda_robotiq_curobo_cfg,
)
from cutamp.robots.franka import (
    fr3_franka_curobo_cfg,
    franka_curobo_cfg,
    get_fr3_franka_ik_solver,
    get_franka_ik_solver,
)
from cutamp.robots.franka_robotiq import fr3_robotiq_curobo_cfg, get_fr3_robotiq_ik_solver
from cutamp.robots.ur5 import get_ur5_ik_solver, ur5_curobo_cfg
from cutamp.utils.common import sample_between_bounds
from jaxtyping import Float

from tiptop.config import tiptop_cfg
from tiptop.utils import get_robot_client, patch_log_level
from tiptop.workspace import workspace_cuboids

_log = logging.getLogger(__name__)


def get_ik_solver(world_cfg: WorldConfig, num_particles: int, warmup_iters: int = 8):
    """Get the IKSolver and warm it up."""
    if warmup_iters < 0:
        raise ValueError(f"warmup_iters must be non-negative, got {warmup_iters}")

    cfg = tiptop_cfg()
    with patch_log_level("curobo", logging.ERROR):
        if cfg.robot.type == "fr3_robotiq":
            ik_solver = get_fr3_robotiq_ik_solver(world_cfg)
            container = load_fr3_robotiq_container(TensorDeviceType())
        elif cfg.robot.type == "fr3":
            ik_solver = get_fr3_franka_ik_solver(world_cfg)
            container = load_fr3_robotiq_container(TensorDeviceType())
        elif cfg.robot.type == "panda_robotiq":
            ik_solver = get_panda_robotiq_ik_solver(world_cfg)
            container = load_panda_robotiq_container(TensorDeviceType())
        elif cfg.robot.type == "panda":
            ik_solver = get_franka_ik_solver(world_cfg)
            container = load_panda_container(TensorDeviceType())
        elif cfg.robot.type == "ur5":
            ik_solver = get_ur5_ik_solver(world_cfg)
            container = load_ur5_container(TensorDeviceType())
        else:
            raise ValueError(f"Unknown robot type: {cfg.robot.type}")

    if warmup_iters > 0:
        torch.cuda.synchronize()
        warmup_start = time.perf_counter()
        for _ in range(warmup_iters):
            q = sample_between_bounds(num_particles, bounds=container.joint_limits)
            goal_pose = container.kin_model.get_state(q).ee_pose
            _ = ik_solver.solve_batch(goal_pose)
        torch.cuda.synchronize()
        warmup_dur = time.perf_counter() - warmup_start
        _log.debug(f"Warming up IKSolver took {warmup_dur:.2f}s")

    return ik_solver


def apply_cost_overrides(cost: dict, overrides: dict | None) -> None:
    """Mutate a gradient-trajopt ``cost`` dict in place with UI overrides (if present).

    Single source of truth for how UI knobs map onto gradient_trajopt.yml cost weights,
    used both to build MotionGen (get_motion_gen) and to summarize the config for saving
    (summarize_curobo_config), so the recorded config always matches what was applied.
    """
    if not overrides:
        return
    if overrides.get("uniform_velocity_weight") is not None:
        cost["uniform_velocity_cfg"]["weight"] = float(overrides["uniform_velocity_weight"])
    # VAE motion-manifold cost (see curobo cost/vae_manifold_cost.py): a single weight knob
    # (Mahalanobis distance to the DROID latent cluster). The block may be absent on older
    # configs, so create it on demand when the override is provided.
    if overrides.get("vae_manifold_weight") is not None:
        vm = cost.setdefault("vae_manifold_cfg", {"weight": 0.0, "n_joints": 7, "source_dt": 0.15})
        vm["weight"] = float(overrides["vae_manifold_weight"])
    for idx, val in (overrides.get("smooth_weight") or {}).items():
        cost["bound_cfg"]["smooth_weight"][int(idx)] = float(val)
    if overrides.get("primitive_collision_activation_distance") is not None:
        cost["primitive_collision_cfg"]["activation_distance"] = float(
            overrides["primitive_collision_activation_distance"]
        )
    if overrides.get("self_collision_weight") is not None:
        cost["self_collision_cfg"]["weight"] = float(overrides["self_collision_weight"])
    if overrides.get("cspace_weight") is not None:
        cost["cspace_cfg"]["weight"] = float(overrides["cspace_weight"])


def summarize_curobo_config(overrides: dict | None, time_dilation_factor) -> dict:
    """Resolved cuRobo trajopt config used for a plan, for saving with each run.

    Loads gradient_trajopt.yml (the deciding phase here), applies the same overrides
    used at build time, and returns a compact, JSON-serializable summary.
    """
    import copy

    from curobo.util_file import get_task_configs_path, join_path, load_yaml

    grad = copy.deepcopy(load_yaml(join_path(get_task_configs_path(), "gradient_trajopt.yml")))
    apply_cost_overrides(grad["cost"], overrides or {})
    c = grad["cost"]
    return {
        "source_yaml": "gradient_trajopt.yml",
        "overrides": overrides or {},
        "resolved": {
            "uniform_velocity_weight": c["uniform_velocity_cfg"]["weight"],
            "vae_manifold_weight": c.get("vae_manifold_cfg", {}).get("weight", 0.0),
            "bound_smooth_weight": c["bound_cfg"]["smooth_weight"],
            "self_collision_weight": c["self_collision_cfg"]["weight"],
            "cspace_weight": c["cspace_cfg"]["weight"],
            "primitive_collision_activation_distance": c["primitive_collision_cfg"]["activation_distance"],
        },
        "plan_overrides": {"enable_finetune_trajopt": False, "time_dilation_factor": time_dilation_factor},
    }


def get_motion_gen(
    world_cfg: WorldConfig,
    collision_activation_distance: float,
    num_spheres: int | None = None,
    warmup_iters: int = 16,
    use_cuda_graph: bool = True,
    cost_overrides: dict | None = None,
):
    """Get the motion generator and warm it up.

    Args:
        world_cfg: Collision world configuration (cuboids, meshes, etc.).
        collision_activation_distance: Distance at which collision cost activates (metres).
        num_spheres: Number of collision spheres for attached objects (e.g. grasped items).
            Passed to cuRobo's extra_collision_spheres for the attached_object slot.
        warmup_iters: Number of warmup iterations to run after construction.
        use_cuda_graph: Whether to use CUDA graphs for faster repeated inference.
    """
    if warmup_iters < 0:
        raise ValueError(f"warmup_iters must be non-negative, got {warmup_iters}")

    cfg = tiptop_cfg()
    if cfg.robot.type == "fr3_robotiq":
        robot_cfg = fr3_robotiq_curobo_cfg()
    elif cfg.robot.type == "fr3":
        robot_cfg = fr3_franka_curobo_cfg()
    elif cfg.robot.type == "panda_robotiq":
        robot_cfg = panda_robotiq_curobo_cfg()
    elif cfg.robot.type == "panda":
        robot_cfg = franka_curobo_cfg()
    elif cfg.robot.type == "ur5":
        robot_cfg = ur5_curobo_cfg()
    else:
        raise ValueError(f"Unknown robot type: {cfg.robot.type}")

    if num_spheres is not None:
        robot_cfg["robot_cfg"]["kinematics"]["extra_collision_spheres"]["attached_object"] = num_spheres
        _log.debug(f"Setting number of spheres for attachments to {num_spheres}")

    # Apply UI cuRobo cost overrides by substituting a modified gradient-trajopt config DICT
    # for the gradient_trajopt_file kwarg (a non-str passes straight through cuRobo's
    # load_yaml). This bakes weights in at build time — no runtime cost re-enable bug, no
    # cuda-graph staleness — and targets the GRADIENT phase, which decides here because
    # cuTAMP plans with enable_finetune_trajopt=False.
    grad_file = "gradient_trajopt.yml"
    if cost_overrides:
        import copy

        from curobo.util_file import get_task_configs_path, join_path, load_yaml

        grad_cfg = copy.deepcopy(load_yaml(join_path(get_task_configs_path(), "gradient_trajopt.yml")))
        apply_cost_overrides(grad_cfg["cost"], cost_overrides)
        grad_file = grad_cfg  # dict, not str

    with patch_log_level("curobo", logging.ERROR):
        motion_gen_cfg = MotionGenConfig.load_from_robot_config(
            robot_cfg=robot_cfg,
            world_model=world_cfg,
            use_cuda_graph=use_cuda_graph,
            collision_activation_distance=collision_activation_distance,
            position_threshold=0.01,
            rotation_threshold=0.1,
            gradient_trajopt_file=grad_file,
        )
        motion_gen = MotionGen(motion_gen_cfg)

    if warmup_iters > 0:
        _log.info("Warming up MotionGen... Might take a few seconds")
        torch.cuda.synchronize()
        warmup_start = time.perf_counter()
        for _ in range(warmup_iters):
            motion_gen.warmup()
        torch.cuda.synchronize()
        warmup_dur = time.perf_counter() - warmup_start
        _log.debug(f"Warming up MotionGen took {warmup_dur:.2f}s")

    return motion_gen


def build_curobo_solvers(
    num_particles: int,
    num_spheres: int,
    collision_activation_distance: float = 0.0,
    include_workspace: bool = True,
    cost_overrides: dict | None = None,
) -> tuple:
    """Build and warm up the IK solver and motion generator.

    Args:
        num_particles: number of cuTAMP particles
        num_spheres: number of collision spheres for attached objects
        collision_activation_distance: distance at which collision cost activates (metres)
        include_workspace: if False, skip the real-robot workspace cuboids (e.g. for sim)

    Returns:
        Tuple of (ik_solver, motion_gen, initial_world_cfg). The WorldConfig is returned
        so callers can reset collision state between runs if needed.
    """
    cuboids = [
        *(workspace_cuboids() if include_workspace else []),
        # Placeholder table cuboid placed far away (no collision effect). cuRobo matches obstacles
        # by name when update_world() is called, so "table" must exist at solver-build time for
        # cuTAMP to later swap in the real table geometry detected via RANSAC.
        Cuboid(name="table", dims=[0.01, 0.01, 0.01], pose=[99.9, 99.9, 99.9, 1.0, 0.0, 0.0, 0.0]),
    ]
    world_cfg = WorldConfig(cuboid=cuboids)
    ik_solver = get_ik_solver(world_cfg, num_particles)
    # use_cuda_graph=False: MotionGen is built with a minimal world (1 placeholder cuboid when
    # include_workspace=False), so update_world() must be able to GROW the collision cache when
    # the real scene (table + surfaces + movables) is loaded. CUDA graphs pin the cache size
    # (fix_cache_reference=True), which raises "number of OBB is larger than collision cache".
    # Disabling graphs lets the cache grow, and also avoids a CUDA-graph driver crash (see README).
    motion_gen = get_motion_gen(
        world_cfg, collision_activation_distance=collision_activation_distance, num_spheres=num_spheres,
        use_cuda_graph=False, cost_overrides=cost_overrides,
    )
    return ik_solver, motion_gen, world_cfg


def go_to_q(
    q_target: Float[np.ndarray, "7"] | list[float],
    time_dilation_factor: float,
    dist_tol: float = 0.05,
    motion_gen: MotionGen | None = None,
) -> None:
    """Move the robot to the target joint positions using motion planning against the workspace."""
    dof = tiptop_cfg().robot.dof
    if isinstance(q_target, np.ndarray) and (q_target.ndim != 1 or len(q_target) != dof):
        raise ValueError(f"Expected q_target to be a ({dof},) np.ndarray, but got {q_target.shape}")
    elif isinstance(q_target, list) and len(q_target) != dof:
        raise ValueError(f"Expected q_target to be a list of length {dof} but got {len(q_target)} elements")
    elif not isinstance(q_target, (list, np.ndarray)):
        raise TypeError(f"Unhandled type for q_target: {type(q_target)}")
    if not 0 < time_dilation_factor <= 1:
        raise ValueError(f"time_dilation_factor must be between 0 and 1, but got {time_dilation_factor}")

    client = get_robot_client()
    if motion_gen is None:
        _log.debug(f"Getting MotionGen")
        world_cfg = WorldConfig(cuboid=list(workspace_cuboids()))
        motion_gen = get_motion_gen(world_cfg, collision_activation_distance=0.01, warmup_iters=0)

    tensor_args = TensorDeviceType()
    q_start = tensor_args.to_device(client.get_joint_positions())
    q_target = tensor_args.to_device(q_target)

    # If we're already close to the target, then nothing to do
    dist = torch.norm(q_start - q_target)
    if dist <= dist_tol:
        _log.info(f"Robot already at target joint positions with {dist=:.2f}")
        return

    # Motion plan!
    js_start, js_target = JointState.from_position(q_start), JointState.from_position(q_target)
    plan_config = MotionGenPlanConfig(time_dilation_factor=time_dilation_factor)
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    result = motion_gen.plan_single_js(js_start[None], js_target[None], plan_config)
    torch.cuda.synchronize()
    mp_duration = time.perf_counter() - start_time
    _log.info(f"Motion planning took {mp_duration:.2f}s")
    if not result.success.all():
        raise MotionPlanningError(
            f"Could not motion plan to target joint positions. Reason: {result.status}.\n"
            "You could try moving the arm in 'Programming' mode to more feasible initial joint positions."
        )

    # Execute on the robot
    plan = result.interpolated_plan
    dt = result.interpolation_dt
    timings = [dt] * plan.position.shape[0]
    result = client.execute_joint_impedance_path(
        joint_confs=plan.position.cpu().numpy(), joint_vels=plan.velocity.cpu().numpy(), durations=timings
    )
    client.close()
    if not result["success"]:
        raise RuntimeError(f"Failed to execute trajectory on robot. {result['error']}")
    _log.info("Executed trajectory on the robot")


def go_to_home(time_dilation_factor: float, motion_gen: MotionGen | None = None) -> None:
    """Go to home configuration"""
    cfg = tiptop_cfg()
    go_to_q(q_target=list(cfg.robot.q_home), time_dilation_factor=time_dilation_factor, motion_gen=motion_gen)


def go_to_capture(time_dilation_factor: float, motion_gen: MotionGen | None = None) -> None:
    """Go to capture configuration"""
    cfg = tiptop_cfg()
    go_to_q(q_target=list(cfg.robot.q_capture), time_dilation_factor=time_dilation_factor, motion_gen=motion_gen)
