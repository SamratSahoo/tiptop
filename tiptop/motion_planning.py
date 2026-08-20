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
    load_bimanual_yam_container,
    load_bimanual_yam_dual_container,
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
from cutamp.robots.bimanual_yam import bimanual_yam_curobo_cfg, get_bimanual_yam_ik_solver
from cutamp.robots.franka_robotiq import fr3_robotiq_curobo_cfg, get_fr3_robotiq_ik_solver
from cutamp.robots.ur5 import get_ur5_ik_solver, ur5_curobo_cfg
from cutamp.utils.common import sample_between_bounds
from jaxtyping import Float

from tiptop.config import tiptop_cfg
from tiptop.utils import YAM_ROBOT_TYPES, get_robot_client, patch_log_level
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
        elif cfg.robot.type in YAM_ROBOT_TYPES:
            # The bimanual YAM is registered once per arm; the suffix picks which arm plans and
            # which one is locked as an obstacle. See cutamp/robots/bimanual_yam.py. "dual" is a
            # different container entirely (both arms as one 12-DOF chain, not a per-arm
            # parameterization of the single-arm one) -- get_bimanual_yam_ik_solver already accepts
            # it via _check_arm(allow_dual=True), but the RobotContainer loader does not.
            arm = cfg.robot.type.rsplit("_", 1)[1]
            ik_solver = get_bimanual_yam_ik_solver(world_cfg, arm)
            if arm == "dual":
                container = load_bimanual_yam_dual_container(TensorDeviceType())
            else:
                container = load_bimanual_yam_container(arm, TensorDeviceType())
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


def resolve_trace_cfg(overrides: dict | None) -> dict | None:
    """Build the per-segment cost-trace config for serialize_plan from cfg/tamp cost overrides.

    Records a raw (weight-independent) cost trace for each motion-manifold cost that is ACTIVE in the
    run (nonzero weight), so tiptop_plan.json logs what the optimizer saw for exactly the terms being
    used -- vae_manifold, joint_density, rnd_novelty. Returns None when none are active (nothing to
    trace beyond the always-on speed/uniform_velocity).

    ``source_dt`` is the trajopt base_dt the manifold costs finite-difference at (default 0.15; see
    gradient_trajopt.yml), which an override may change via ``base_dt`` -- NOT the plan's playback dt.
    """
    ov = overrides or {}
    cfg: dict = {"source_dt": float(ov.get("base_dt") or 0.15), "n_joints": 7}
    active = False
    if ov.get("vae_manifold_weight"):
        # checkpoint_path selects the encoder + DROID latent stats; resolve like apply_cost_overrides.
        cfg["vae"] = {"checkpoint_path": resolve_vae_path(str(ov["vae_path"])) if ov.get("vae_path") else None}
        active = True
    if ov.get("joint_density_weight"):
        cfg["joint_density"] = {"huber_delta": 0.05}
        active = True
    if ov.get("rnd_novelty_weight"):
        cfg["rnd_novelty"] = {}
        active = True
    if ov.get("ee_manifold_weight"):
        # EE-pose manifold trace needs the segment's ee poses, which serialize_plan derives from the
        # joint waypoints by forward kinematics -- see planning._per_timestep_cost.
        cfg["ee_manifold"] = {
            "checkpoint_path": resolve_vae_path(str(ov["ee_vae_path"])) if ov.get("ee_vae_path") else None,
            "robot_type": tiptop_cfg().robot.type,
        }
        active = True
    return cfg if active else None


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
        # vae_retiming promotes each waypoint interval's duration to a trajopt decision variable,
        # optimized by the same LBFGS pass as the waypoints. The three guard knobs are optional.
        if resolve_vae_retiming(overrides):
            vm["retiming"] = True
            for key in ("retime_scale", "retime_smooth_weight", "retime_limit_weight"):
                if overrides.get(key) is not None:
                    vm[key] = float(overrides[key])
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
    # End-effector pose manifold cost (see curobo cost/ee_manifold_cost.py): MINIMIZES the
    # Mahalanobis distance between the segment's arc-length EE-pose description and the DROID
    # cluster, pulling Cartesian path shape AND wrist orientation toward human teleop. Timing-blind
    # by construction, so it composes with vae_manifold_weight rather than fighting it.
    #
    # SCALE CHANGED with the invariant manifold (ee_manifold.pt, now the default). Measured on 56
    # real planner segments at weight 1: the legacy VAE charges a median excess of 464 with
    # |dC/dpos| 1.3e4, the invariant manifold 12.4 with 3.1e2 -- it scores 22 whitened quantities
    # rather than a 12-d latent with an inflated precision. Converting the legacy 5000 gives 187k by
    # value parity (what the particle phase sees) and 216k by gradient parity (what the gradient
    # phase sees); they agree, so ~200k is equal authority. Start below it and sweep -- the old
    # weight was over-strong as well as mis-aimed.
    # ee_vae_path selects the checkpoint; ee_manifold_slack turns off the "stop once you are as
    # DROID-like as a typical human stroke" threshold. Block may be absent -> create on demand.
    if any(overrides.get(k) is not None for k in
           ("ee_manifold_weight", "ee_vae_path", "ee_manifold_slack", "ee_manifold_slack_scale")):
        em = cost.setdefault("ee_manifold_cfg", {"weight": 0.0, "slack": True, "slack_scale": 1.0})
        if overrides.get("ee_manifold_weight") is not None:
            em["weight"] = float(overrides["ee_manifold_weight"])
        if overrides.get("ee_manifold_slack") is not None:
            em["slack"] = bool(overrides["ee_manifold_slack"])
        if overrides.get("ee_manifold_slack_scale") is not None:
            em["slack_scale"] = float(overrides["ee_manifold_slack_scale"])
        if overrides.get("ee_vae_path") is not None:
            em["checkpoint_path"] = resolve_vae_path(str(overrides["ee_vae_path"]))
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


def apply_particle_cost_overrides(cost: dict, overrides: dict | None) -> None:
    """Mutate a PARTICLE-trajopt ``cost`` dict in place. Deliberately narrower than its gradient twin.

    Only the EE-pose manifold weight is propagated here, and only because that cost is meant to vote
    in MPPI's population reweighting as well as in the gradient descent -- a cost in the particle
    phase influences WHICH sampled trajectory survives, not just how the winner is bent.

    Everything else is left alone on purpose. ``vae_manifold_weight`` and ``joint_density_weight``
    have historically applied to the gradient phase ONLY (get_motion_gen substitutes just
    gradient_trajopt_file), even though both blocks exist in particle_trajopt.yml. Quietly widening
    them here would change the meaning of every existing sweep and cfg/tamp preset, so it is left as
    an explicit follow-up rather than a side effect of this function.
    """
    if not overrides:
        return
    if not resolve_ee_manifold_particle(overrides):
        return
    if overrides.get("ee_manifold_weight") is None:
        return
    em = cost.setdefault("ee_manifold_cfg", {"weight": 0.0, "slack": True, "slack_scale": 1.0})
    em["weight"] = float(overrides["ee_manifold_weight"])
    if overrides.get("ee_manifold_slack") is not None:
        em["slack"] = bool(overrides["ee_manifold_slack"])
    if overrides.get("ee_manifold_slack_scale") is not None:
        em["slack_scale"] = float(overrides["ee_manifold_slack_scale"])
    if overrides.get("ee_vae_path") is not None:
        em["checkpoint_path"] = resolve_vae_path(str(overrides["ee_vae_path"]))


def resolve_ee_manifold_particle(overrides: dict | None, default: bool = True) -> bool:
    """Whether the EE-pose manifold cost also runs in the particle (MPPI) phase. Default on."""
    val = (overrides or {}).get("ee_manifold_particle")
    return default if val is None else bool(val)


def resolve_seed_kwargs(overrides: dict | None) -> dict:
    """Seed-diversity kwargs for MotionGenConfig.load_from_robot_config.

    cuRobo runs ``num_trajopt_seeds`` candidates in parallel, refines each, and keeps the one with
    the lowest total cost -- machinery for letting a cost CHOOSE a trajectory rather than only
    refine one. It is wasted by default: ``trajopt_seed_ratio`` is {"linear": 1.0}, so every seed is
    the same joint-space straight line and the "choice" is between identical candidates.

    ``trajopt_seed_ratio`` mixes the seed generators (see TrajOptSolver.get_seeds):
      linear -- straight line start->goal; bias -- via the arm's retract configuration;
      start / goal -- held at one endpoint.
    Each generator produces ONE shape repeated across its share of the seeds, so diversity comes
    from mixing TYPES, not from raising the count. ``num_trajopt_noisy_seeds`` multiplies each with
    MPPI's sampling noise on top.
    """
    ov = overrides or {}
    kw: dict = {}
    if ov.get("trajopt_seed_ratio") is not None:
        ratio = {k: float(v) for k, v in dict(ov["trajopt_seed_ratio"]).items()}
        bad = set(ratio) - {"linear", "bias", "start", "goal"}
        if bad:
            raise ValueError(f"unknown trajopt_seed_ratio key(s): {sorted(bad)}")
        kw["trajopt_seed_ratio"] = ratio
    if ov.get("num_trajopt_seeds") is not None:
        kw["num_trajopt_seeds"] = int(ov["num_trajopt_seeds"])
    if ov.get("num_trajopt_noisy_seeds") is not None:
        kw["num_trajopt_noisy_seeds"] = int(ov["num_trajopt_noisy_seeds"])
    return kw


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


def resolve_vae_retiming(overrides: dict | None) -> bool:
    """Whether the VAE manifold cost owns the trajectory clock (the ``vae_retiming`` override).

    When on, the per-interval durations are decision variables of the cuRobo gradient trajopt (see
    curobo cost/vae_manifold_cost.py) and every OTHER retiming stage is suppressed: the
    time_dilation_factor, cuRobo's own time-optimal ``optimize_dt`` rescale, and trajectory
    blending (both the analytic spline and the flow model). Requires a nonzero
    ``vae_manifold_weight`` -- with the cost disabled there is nothing optimizing the clock, and
    silently suppressing every retimer would just emit raw trajopt output at base_dt."""
    ov = overrides or {}
    if not ov.get("vae_retiming"):
        return False
    if not ov.get("vae_manifold_weight"):
        _log.warning(
            "vae_retiming is set but vae_manifold_weight is 0/absent -- nothing would optimize the "
            "trajectory clock, so vae_retiming is IGNORED."
        )
        return False
    return True


def resolve_time_dilation_factor(overrides: dict | None, config_default: float) -> float:
    """Effective time_dilation_factor from UI/sweep overrides.

    ``time_dilation_factor_literal`` bypasses the 1.0 sentinel (used by the parameter sweep) so a
    requested value is applied verbatim. Otherwise a ``time_dilation_factor`` of None or 1.0 means
    "no extra scaling" and we fall back to the config default (tiptop.yml robot.time_dilation_factor).

    ``vae_retiming`` wins over both: the VAE owns the clock, so returning anything but 1.0 would
    rescale the timing it just optimized. 1.0 is cuRobo's no-op sentinel (MotionGenResult.
    retime_trajectory returns immediately), whereas falling back to ``config_default`` would NOT be
    a no-op -- tiptop.yml ships robot.time_dilation_factor: 0.2.
    """
    overrides = overrides or {}
    if resolve_vae_retiming(overrides):
        requested = overrides.get("time_dilation_factor_literal", overrides.get("time_dilation_factor"))
        if requested is not None:
            _log.info(f"vae_retiming active: time_dilation_factor={requested} IGNORED (forced to 1.0)")
        return 1.0
    if overrides.get("time_dilation_factor_literal") is not None:
        return float(overrides["time_dilation_factor_literal"])
    tdf = overrides.get("time_dilation_factor")
    if tdf is None or abs(float(tdf) - 1.0) < 1e-6:
        return float(config_default)
    return float(tdf)


def resolve_traj_length_norm(overrides: dict | None, default: float = 2.0) -> float:
    """Effective norm p for cuTAMP's per-move TrajectoryLength cost, from cfg/tamp tamp_overrides.

    The ``move(q1, tau, q2)`` cost charges the joint-space distance ||q1 - q2||_p; p=2 (default) is
    the Euclidean straight-line distance, p=inf is the max joint displacement (the infinity-norm the
    TiPToP paper minimizes). Both lower-bound the shortest collision-free path length.

    The value is read from the ``traj_length_norm`` override. It is accepted as a string ("inf" /
    "infinity" / "max") or a number (1, 2, ...). A string is required for the infinity-norm because
    the overrides dict round-trips through JSON (Python -> Node -> Python), which cannot represent
    Infinity as a bare number.
    """
    val = (overrides or {}).get("traj_length_norm")
    if val is None:
        return float(default)
    if isinstance(val, str):
        if val.strip().lower() in {"inf", "infinity", "max"}:
            return float("inf")
        return float(val)  # numeric string, e.g. "2"
    return float(val)


def resolve_max_motion_refine_attempts(overrides: dict | None, default: int | None = 32) -> int | None:
    """Effective cap on how many satisfying particles cuTAMP tries motion refinement on.

    cuTAMP works through satisfying particles in cost order until one motion-refines successfully or
    this cap is reached (cutamp/config.py's own default is ``None`` = try all of them). TiPToP bounds
    it at 32 by default to cap planning time when a scene has many satisfying particles. A
    dual/handover scene can need more tries than that: with few grasp candidates on the object (a
    thin M2T2 harvest), most satisfying particles share a similar, hard approach geometry, so a run
    can exhaust 32 attempts -- all failing at the same TRAJOPT_FAIL step -- while satisfying particles
    it never tried might have succeeded. Set ``max_motion_refine_attempts`` in cfg/tamp
    ``tamp_overrides`` to raise the cap, or to ``null`` to try every satisfying particle.
    """
    overrides = overrides or {}
    if "max_motion_refine_attempts" not in overrides:
        return default
    val = overrides["max_motion_refine_attempts"]
    return None if val is None else int(val)


def resolve_grasp_orientation_cost(overrides: dict | None) -> bool:
    """Whether to enable cuTAMP's grasp orientation-change soft cost, from cfg/tamp tamp_overrides.

    Enabled iff a truthy ``grasp_pose_change_weight`` is present (the same key run_planning reads for
    the weight), so a single YAML knob both gates the cost (this bool -> TAMPConfiguration) and sets
    its multiplier. A zero/absent weight leaves it off.
    """
    return bool((overrides or {}).get("grasp_pose_change_weight"))


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
            # With retiming on, the per-interval durations were trajopt decision variables and every
            # other retimer was suppressed -- without this the record would be indistinguishable
            # from a stock run.
            "vae_retiming": c.get("vae_manifold_cfg", {}).get("retiming", False),
            "vae_retime_scale": c.get("vae_manifold_cfg", {}).get("retime_scale"),
            "retiming_suppressed": (
                ["time_dilation_factor", "optimize_dt", "blending"]
                if c.get("vae_manifold_cfg", {}).get("retiming", False)
                else []
            ),
            "rnd_novelty_weight": c.get("rnd_novelty_cfg", {}).get("weight", 0.0),
            "rnd_novelty_log": c.get("rnd_novelty_cfg", {}).get("use_log", True),
            "joint_density_weight": c.get("joint_density_cfg", {}).get("weight", 0.0),
            "ee_manifold_weight": c.get("ee_manifold_cfg", {}).get("weight", 0.0),
            "ee_vae_path": c.get("ee_manifold_cfg", {}).get("checkpoint_path"),
            "ee_manifold_slack": c.get("ee_manifold_cfg", {}).get("slack", True),
            "ee_manifold_particle": resolve_ee_manifold_particle(ov) and bool(
                c.get("ee_manifold_cfg", {}).get("weight", 0.0)
            ),
            "seed_kwargs": resolve_seed_kwargs(ov),
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


def robot_curobo_cfg(robot_type: str) -> dict:
    """cuRobo robot config dict for a tiptop robot type.

    Extracted from get_motion_gen so anything else needing this robot's kinematics -- the EE-pose
    cost trace in planning.py, for one -- selects it the same way instead of re-deriving the mapping.
    """
    if robot_type == "fr3_robotiq":
        return fr3_robotiq_curobo_cfg()
    if robot_type == "fr3":
        return fr3_franka_curobo_cfg()
    if robot_type == "panda_robotiq":
        return panda_robotiq_curobo_cfg()
    if robot_type == "panda":
        return franka_curobo_cfg()
    if robot_type == "ur5":
        return ur5_curobo_cfg()
    if robot_type in YAM_ROBOT_TYPES:
        return bimanual_yam_curobo_cfg(robot_type.rsplit("_", 1)[1])
    raise ValueError(f"Unknown robot type: {robot_type}")


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
    robot_cfg = robot_curobo_cfg(cfg.robot.type)

    if num_spheres is not None:
        extra_spheres = robot_cfg["robot_cfg"]["kinematics"]["extra_collision_spheres"]
        if cfg.robot.type == "bimanual_yam_dual":
            # The dual config has one attachment slot PER HAND (bimanual_yam_dual.yml:
            # left_attached_object / right_attached_object), not the single "attached_object" every
            # other robot (including the single-arm YAM configs) uses -- both need the same budget.
            extra_spheres["left_attached_object"] = num_spheres
            extra_spheres["right_attached_object"] = num_spheres
        else:
            extra_spheres["attached_object"] = num_spheres
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
        # The particle phase gets its own (narrower) substitution so the EE-pose manifold cost can
        # vote in MPPI's reweighting -- see apply_particle_cost_overrides for why only that one.
        part_cfg = copy.deepcopy(load_yaml(join_path(get_task_configs_path(), "particle_trajopt.yml")))
        apply_particle_cost_overrides(part_cfg["cost"], cost_overrides)
        if part_cfg["cost"].get("ee_manifold_cfg", {}).get("weight"):
            extra_kwargs["particle_trajopt_file"] = part_cfg
        # Verification: log the RESOLVED cost weights that MotionGen is actually built with, so a
        # data-gen run can confirm the overrides propagated all the way into the cuRobo solver (not
        # just that the CLI arg parsed). Grep tiptop_*.log for "RESOLVED cuRobo cost".
        _gc = grad_cfg["cost"]
        _log.info(
            "RESOLVED cuRobo cost after overrides: vae_manifold_weight=%s vae_path=%s rnd_novelty_weight=%s joint_density_weight=%s ee_manifold_weight=%s | overrides=%s",
            _gc.get("vae_manifold_cfg", {}).get("weight"),
            _gc.get("vae_manifold_cfg", {}).get("checkpoint_path"),
            _gc.get("rnd_novelty_cfg", {}).get("weight"),
            _gc.get("joint_density_cfg", {}).get("weight"),
            _gc.get("ee_manifold_cfg", {}).get("weight"),
            cost_overrides,
        )
        grad_file = grad_cfg  # dict, not str

        # horizon and trajopt dt also have to be set as load_from_robot_config kwargs: its
        # trajopt_tsteps default (32) and trajopt_dt fallback (max_trajectory_dt) otherwise win
        # over the gradient_trajopt model dict. Joint-limit scales aren't in that dict at all.
        n_cspace = len(robot_cfg["robot_cfg"]["kinematics"]["cspace"]["joint_names"])
        extra_kwargs.update(_scale_kwargs(cost_overrides, n_cspace))
        extra_kwargs.update(resolve_seed_kwargs(cost_overrides))
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
) -> dict | None:
    """Move the robot to the target joint positions using motion planning against the workspace.

    Returns ``{"positions", "velocities", "dt", "t_start", "t_end"}`` describing the trajectory that
    was actually executed, or ``None`` when the arm was already at the target and nothing ran.

    A bimanual YAM episode needs that: parking one arm before the other plans is a real commanded
    motion inside the recorded window, so it has to reach the action channel like any plan segment
    (``tiptop.yam.capture.segment_from_motion``). Callers that only want the motion can keep
    ignoring the return value.
    """
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
        return None

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
    # NOTE: do NOT close the client here. `get_robot_client()` is a process-wide cached singleton
    # (utils.get_bamboo_client is @cache'd) that the warm session holds for the rest of its life --
    # tiptop_run's container, execute_cutamp_plan's gripper commands, the LeRobot samplers. Closing
    # it terminated the shared ZMQ context, and only the CONTROL path recovers from that: a control
    # send hits ZMQError -> _recreate_control_socket -> the terminated context refuses a new socket
    # -> it rebuilds context + gripper socket. The gripper path (_send_robotiq_command) has no such
    # recovery, so a gripper call landing before the next control call raised ENOTSOCK. That is why
    # "return home, then open the gripper" silently failed at the start of every episode.
    # _sync_entrypoint's finally owns the teardown.
    positions = plan.position.cpu().numpy()
    velocities = plan.velocity.cpu().numpy()
    t_start = time.time()
    result = client.execute_joint_impedance_path(
        joint_confs=positions, joint_vels=velocities, durations=timings
    )
    t_end = time.time()
    if not result["success"]:
        raise RuntimeError(f"Failed to execute trajectory on robot. {result['error']}")
    _log.info("Executed trajectory on the robot")
    return {"positions": positions, "velocities": velocities, "dt": dt, "t_start": t_start, "t_end": t_end}


def go_to_home(time_dilation_factor: float, motion_gen: MotionGen | None = None) -> dict | None:
    """Go to home configuration. Returns the executed trajectory (see :func:`go_to_q`)."""
    cfg = tiptop_cfg()
    return go_to_q(q_target=list(cfg.robot.q_home), time_dilation_factor=time_dilation_factor, motion_gen=motion_gen)


def go_to_capture(time_dilation_factor: float, motion_gen: MotionGen | None = None) -> dict | None:
    """Go to capture configuration. Returns the executed trajectory (see :func:`go_to_q`)."""
    cfg = tiptop_cfg()
    return go_to_q(q_target=list(cfg.robot.q_capture), time_dilation_factor=time_dilation_factor, motion_gen=motion_gen)


def go_to_dual_q(
    q_target: Float[np.ndarray, "12"] | list[float],
    time_dilation_factor: float,
    dist_tol: float = 0.05,
    motion_gen: MotionGen | None = None,
) -> dict | None:
    """12-DOF analogue of :func:`go_to_q`, for the ``bimanual_yam_dual`` embodiment only.

    Moves BOTH arms to one shared target configuration at once. This is a separate function, not a
    branch inside ``go_to_q``, because execution genuinely differs: ``YamClient.execute_joint_impedance_path``
    drives whichever ONE arm ``client.arm`` currently names (meaningless in dual mode, where there is
    no single active arm -- see ``YamClient.arm``'s docstring), so this goes through
    ``execute_plan._DualQueuedArm`` instead, the same dual queued-submission path
    ``execute_cutamp_dual_plan`` uses for a cuTAMP plan's trajectory steps. Planning itself
    (``motion_gen.plan_single_js``) is unchanged -- it is DOF-agnostic and already works correctly
    against a 12-DOF ``motion_gen`` built for ``bimanual_yam_dual``.
    """
    from tiptop.yam import BIMANUAL_ARM_DOF

    if len(q_target) != BIMANUAL_ARM_DOF:
        raise ValueError(f"Expected a {BIMANUAL_ARM_DOF}-wide q_target for the dual embodiment, got {len(q_target)}")
    if not 0 < time_dilation_factor <= 1:
        raise ValueError(f"time_dilation_factor must be between 0 and 1, but got {time_dilation_factor}")

    client = get_robot_client()
    if motion_gen is None:
        world_cfg = WorldConfig(cuboid=list(workspace_cuboids()))
        motion_gen = get_motion_gen(world_cfg, collision_activation_distance=0.01, warmup_iters=0)

    tensor_args = TensorDeviceType()
    q_start = tensor_args.to_device(client.get_dual_joint_positions())
    q_target = tensor_args.to_device(list(q_target))

    dist = torch.norm(q_start - q_target)
    if dist <= dist_tol:
        _log.info(f"Both arms already at target joint positions with {dist=:.2f}")
        return None

    js_start, js_target = JointState.from_position(q_start), JointState.from_position(q_target)
    plan_config = MotionGenPlanConfig(time_dilation_factor=time_dilation_factor)
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    result = motion_gen.plan_single_js(js_start[None], js_target[None], plan_config)
    torch.cuda.synchronize()
    mp_duration = time.perf_counter() - start_time
    _log.info(f"Dual-arm motion planning took {mp_duration:.2f}s")
    if not result.success.all():
        raise MotionPlanningError(
            f"Could not motion plan both arms to target joint positions. Reason: {result.status}.\n"
            "You could try moving an arm in 'Programming' mode to more feasible initial joint positions."
        )

    plan = result.interpolated_plan
    dt = result.interpolation_dt
    positions = plan.position.cpu().numpy()
    velocities = plan.velocity.cpu().numpy()

    from tiptop.execute_plan import ExecutionFailure, _DualQueuedArm

    dual = _DualQueuedArm()
    if not dual.available:
        raise ExecutionFailure(
            "go_to_dual_q requires the arm server's trajectory queue -- there is no blocking "
            "fallback for driving both arms at once."
        )
    t_start = time.time()
    try:
        dual.submit(positions, velocities, float(dt))
        drained = dual.wait_done()
        if not drained.get("success"):
            dual.abort()
            raise RuntimeError(f"Failed to execute dual-arm trajectory on the robot. {drained.get('error')}")
    finally:
        dual.close()
    t_end = time.time()
    _log.info("Executed dual-arm trajectory on the robot")
    return {"positions": positions, "velocities": velocities, "dt": dt, "t_start": t_start, "t_end": t_end}


def go_to_dual_home(time_dilation_factor: float, motion_gen: MotionGen | None = None) -> dict | None:
    """Move both arms to home configuration. Returns the executed trajectory (see :func:`go_to_dual_q`)."""
    cfg = tiptop_cfg()
    return go_to_dual_q(
        q_target=list(cfg.robot.q_home), time_dilation_factor=time_dilation_factor, motion_gen=motion_gen
    )
