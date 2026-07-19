import logging
import os
import time
from pathlib import Path

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


# tamp-vla repo root: .../tamp-vla/tiptop/tiptop/motion_planning.py -> parents[2] == tamp-vla.
# Used to resolve repo-relative vae_path overrides (e.g. "vae/checkpoints/vae_full_v2.pt") the same
# way regardless of the process cwd (tiptop-run runs from tiptop/, not the repo root).
_REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_vae_path(vae_path: str) -> str:
    """Resolve a vae_path override to an absolute checkpoint path.

    Absolute (or ~-prefixed) paths are used as-is; relative paths are resolved against the
    tamp-vla repo root so `vae/checkpoints/vae_full_v2.pt` works from any cwd.
    """
    p = Path(os.path.expanduser(vae_path))
    if not p.is_absolute():
        p = _REPO_ROOT / p
    return str(p)


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
    if overrides.get("vae_manifold_weight") is not None or overrides.get("vae_path") is not None:
        vm = cost.setdefault("vae_manifold_cfg", {"weight": 0.0, "n_joints": 7, "source_dt": 0.15})
        if overrides.get("vae_manifold_weight") is not None:
            vm["weight"] = float(overrides["vae_manifold_weight"])
        # vae_path selects WHICH checkpoint the manifold cost loads (encoder + DROID latent stats),
        # overriding the VAE_MANIFOLD_CKPT env default. Resolved to an absolute path so it is cwd-safe.
        if overrides.get("vae_path") is not None:
            vm["checkpoint_path"] = resolve_vae_path(str(overrides["vae_path"]))
    # RND novelty cost (see curobo cost/rnd_novelty_cost.py): a single weight knob that MAXIMIZES how
    # poorly DROID covers the motion (the opposite of vae_manifold_weight). rnd_novelty_log toggles
    # maximizing log(novelty) (default) vs raw novelty. Block may be absent -> create on demand.
    if overrides.get("rnd_novelty_weight") is not None or overrides.get("rnd_novelty_log") is not None:
        rn = cost.setdefault(
            "rnd_novelty_cfg", {"weight": 0.0, "n_joints": 7, "source_dt": 0.15, "use_log": True}
        )
        if overrides.get("rnd_novelty_weight") is not None:
            rn["weight"] = float(overrides["rnd_novelty_weight"])
        if overrides.get("rnd_novelty_log") is not None:
            rn["use_log"] = bool(overrides["rnd_novelty_log"])
    # Joint-position density-matching cost (see curobo cost/joint_density_cost.py): a single weight
    # knob that MINIMIZES the 1-D Wasserstein-1 distance between the trajectory's per-joint position
    # marginal and DROID's. Block may be absent on older configs -> create on demand.
    if overrides.get("joint_density_weight") is not None:
        jd = cost.setdefault("joint_density_cfg", {"weight": 0.0, "n_joints": 7, "huber_delta": 0.05})
        jd["weight"] = float(overrides["joint_density_weight"])
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
    # bound_cfg vector knobs — per-index dicts like smooth_weight (idx -> value), for the
    # [position, velocity, acceleration, jerk] limit-violation weights and activation margins.
    for idx, val in (overrides.get("bound_weight") or {}).items():
        cost["bound_cfg"]["weight"][int(idx)] = float(val)
    for idx, val in (overrides.get("bound_activation_distance") or {}).items():
        cost["bound_cfg"]["activation_distance"][int(idx)] = float(val)
    if overrides.get("run_weight_acceleration") is not None:
        cost["bound_cfg"]["run_weight_acceleration"] = float(overrides["run_weight_acceleration"])
    if overrides.get("run_weight_jerk") is not None:
        cost["bound_cfg"]["run_weight_jerk"] = float(overrides["run_weight_jerk"])
    # pose_cfg knobs — the EE goal-pose cost. weight is [terminal-orient, terminal-pos,
    # run-orient, run-pos]; run_vec_weight is a single scalar applied to all 6 running components.
    for idx, val in (overrides.get("pose_weight") or {}).items():
        cost["pose_cfg"]["weight"][int(idx)] = float(val)
    if overrides.get("run_vec_weight") is not None:
        cost["pose_cfg"]["run_vec_weight"] = [float(overrides["run_vec_weight"])] * 6


def apply_model_overrides(model: dict, overrides: dict | None) -> None:
    """Mutate a gradient-trajopt ``model`` dict in place with UI overrides (if present).

    Companion to apply_cost_overrides for the non-cost trajopt knobs that live under
    gradient_trajopt.yml's ``model`` section (horizon, trajopt timestep). The horizon and dt
    must ALSO be passed to MotionGenConfig.load_from_robot_config (trajopt_tsteps/trajopt_dt),
    since those kwargs have non-None defaults that otherwise win — get_motion_gen does that.
    """
    if not overrides:
        return
    if overrides.get("horizon") is not None:
        model["horizon"] = int(overrides["horizon"])
    if overrides.get("base_dt") is not None:
        # base_dt is the trajopt timestep; keep max_dt equal to it (as in the YAML) so the whole
        # optimization runs at the requested resolution rather than the default 0.15 ceiling.
        dt = float(overrides["base_dt"])
        model["dt_traj_params"]["base_dt"] = dt
        model["dt_traj_params"]["max_dt"] = dt


def _scale_kwargs(overrides: dict | None, n_cspace_joints: int) -> dict:
    """Joint-limit scale kwargs for MotionGenConfig.load_from_robot_config, pulled from overrides.

    cuRobo broadcasts a 1-element list to shape (n, 1) (a latent bug), so we always pass a
    full per-joint list of length ``n_cspace_joints`` — that hits cuRobo's List branch and also
    keeps its feasibility maximum_trajectory_dt handling for scales < 1.0.
    """
    kw = {}
    for key in ("velocity_scale", "acceleration_scale", "jerk_scale"):
        if (overrides or {}).get(key) is not None:
            kw[key] = [float(overrides[key])] * n_cspace_joints
    return kw


def resolve_time_dilation_factor(overrides: dict | None, config_default: float) -> float:
    """Effective time_dilation_factor from UI/sweep overrides.

    ``time_dilation_factor_literal`` bypasses the 1.0 sentinel (used by the parameter sweep) so a
    requested value is applied verbatim. Otherwise a ``time_dilation_factor`` of None or 1.0 means
    "no extra scaling" and we fall back to the config default (tiptop.yml robot.time_dilation_factor).
    """
    overrides = overrides or {}
    if overrides.get("time_dilation_factor_literal") is not None:
        return float(overrides["time_dilation_factor_literal"])
    tdf = overrides.get("time_dilation_factor")
    if tdf is None or abs(float(tdf) - 1.0) < 1e-6:
        return float(config_default)
    return float(tdf)


def summarize_curobo_config(overrides: dict | None, time_dilation_factor) -> dict:
    """Resolved cuRobo trajopt config used for a plan, for saving with each run.

    Loads gradient_trajopt.yml (the deciding phase here), applies the same overrides
    used at build time, and returns a compact, JSON-serializable summary.
    """
    import copy

    from curobo.util_file import get_task_configs_path, join_path, load_yaml

    grad = copy.deepcopy(load_yaml(join_path(get_task_configs_path(), "gradient_trajopt.yml")))
    apply_cost_overrides(grad["cost"], overrides or {})
    apply_model_overrides(grad["model"], overrides or {})
    c, m = grad["cost"], grad["model"]
    ov = overrides or {}
    return {
        "source_yaml": "gradient_trajopt.yml",
        "overrides": ov,
        "resolved": {
            "uniform_velocity_weight": c["uniform_velocity_cfg"]["weight"],
            "vae_manifold_weight": c.get("vae_manifold_cfg", {}).get("weight", 0.0),
            "vae_path": c.get("vae_manifold_cfg", {}).get("checkpoint_path"),
            "rnd_novelty_weight": c.get("rnd_novelty_cfg", {}).get("weight", 0.0),
            "rnd_novelty_log": c.get("rnd_novelty_cfg", {}).get("use_log", True),
            "joint_density_weight": c.get("joint_density_cfg", {}).get("weight", 0.0),
            "bound_smooth_weight": c["bound_cfg"]["smooth_weight"],
            "bound_weight": c["bound_cfg"]["weight"],
            "bound_activation_distance": c["bound_cfg"]["activation_distance"],
            "run_weight_acceleration": c["bound_cfg"]["run_weight_acceleration"],
            "run_weight_jerk": c["bound_cfg"]["run_weight_jerk"],
            "pose_weight": c["pose_cfg"]["weight"],
            "pose_run_vec_weight": c["pose_cfg"]["run_vec_weight"],
            "self_collision_weight": c["self_collision_cfg"]["weight"],
            "cspace_weight": c["cspace_cfg"]["weight"],
            "primitive_collision_activation_distance": c["primitive_collision_cfg"]["activation_distance"],
            "horizon": m["horizon"],
            "base_dt": m["dt_traj_params"]["base_dt"],
            # joint-limit scales aren't in gradient_trajopt.yml — echo the override (default 1.0).
            "velocity_scale": ov.get("velocity_scale", 1.0),
            "acceleration_scale": ov.get("acceleration_scale", 1.0),
            "jerk_scale": ov.get("jerk_scale", 1.0),
            # planning-time knobs (read by tiptop_gt_plan.py), echoed for a self-describing record.
            "num_particles": ov.get("num_particles"),
            "opt_steps_per_skeleton": ov.get("opt_steps_per_skeleton"),
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
    extra_kwargs: dict = {}
    if cost_overrides:
        import copy

        from curobo.util_file import get_task_configs_path, join_path, load_yaml

        grad_cfg = copy.deepcopy(load_yaml(join_path(get_task_configs_path(), "gradient_trajopt.yml")))
        apply_cost_overrides(grad_cfg["cost"], cost_overrides)
        apply_model_overrides(grad_cfg["model"], cost_overrides)
        # Verification: log the RESOLVED cost weights that MotionGen is actually built with, so a
        # data-gen run can confirm the overrides propagated all the way into the cuRobo solver (not
        # just that the CLI arg parsed). Grep tiptop_*.log for "RESOLVED cuRobo cost".
        _gc = grad_cfg["cost"]
        _log.info(
            "RESOLVED cuRobo cost after overrides: vae_manifold_weight=%s vae_path=%s rnd_novelty_weight=%s joint_density_weight=%s | overrides=%s",
            _gc.get("vae_manifold_cfg", {}).get("weight"),
            _gc.get("vae_manifold_cfg", {}).get("checkpoint_path"),
            _gc.get("rnd_novelty_cfg", {}).get("weight"),
            _gc.get("joint_density_cfg", {}).get("weight"),
            cost_overrides,
        )
        grad_file = grad_cfg  # dict, not str

        # horizon and trajopt dt also have to be set as load_from_robot_config kwargs: its
        # trajopt_tsteps default (32) and trajopt_dt fallback (max_trajectory_dt) otherwise win
        # over the gradient_trajopt model dict. Joint-limit scales aren't in that dict at all.
        n_cspace = len(robot_cfg["robot_cfg"]["kinematics"]["cspace"]["joint_names"])
        extra_kwargs.update(_scale_kwargs(cost_overrides, n_cspace))
        if cost_overrides.get("horizon") is not None:
            extra_kwargs["trajopt_tsteps"] = int(cost_overrides["horizon"])
        if cost_overrides.get("base_dt") is not None:
            dt = float(cost_overrides["base_dt"])
            extra_kwargs["trajopt_dt"] = dt
            extra_kwargs["js_trajopt_dt"] = dt

    with patch_log_level("curobo", logging.ERROR):
        motion_gen_cfg = MotionGenConfig.load_from_robot_config(
            robot_cfg=robot_cfg,
            world_model=world_cfg,
            use_cuda_graph=use_cuda_graph,
            collision_activation_distance=collision_activation_distance,
            position_threshold=0.01,
            rotation_threshold=0.1,
            gradient_trajopt_file=grad_file,
            **extra_kwargs,
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
