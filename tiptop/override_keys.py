"""The top-level ``tamp_overrides`` keys tiptop reads (the ``--curobo-overrides`` JSON).

``tiptop_websocket_server._load_curobo_overrides``, which both tiptop-run and tiptop-server load with,
rejects any other key, so a misspelled, renamed or removed setting fails at startup instead of being
silently ignored. A key belongs here exactly when some code reads it; add or remove it in the same change.
tiptop-server also rejects SESSION_ONLY_OVERRIDE_KEYS, which only tiptop-run reads.
"""

# tiptop-run session options: auto_mode (auto_mode.resolve_auto_mode) and reset_placement_region
# (scene_reset.reset_placement_region) configure its reset-or-collect loop and scene resets, which
# tiptop-server does not have, so TiptopPlanningServer rejects them instead of ignoring them.
SESSION_ONLY_OVERRIDE_KEYS = frozenset({"auto_mode", "reset_placement_region"})

SUPPORTED_OVERRIDE_KEYS = SESSION_ONLY_OVERRIDE_KEYS | frozenset(
    {
        # Perception (motion_planning._PERCEPTION_OVERRIDE_KEYS).
        "contact_threshold_m",
        "grasp_threshold",
        "m2t2_num_runs",
        "voxel_downsample_size",
        # Solver effort (motion_planning.resolve_solver_effort).
        "num_particles",
        "opt_steps_per_skeleton",
        # cuTAMP planning: TAMPConfiguration fields and run_planning's grasp soft costs.
        "traj_length_norm",
        "grasp_pose_change_weight",
        "grasp_center_weight",
        "grasp_rank_conf_weight",
        "max_motion_refine_attempts",
        "require_m2t2_grasps",
        "transit_apex_height",
        "transit_apex_min_dist",
        "time_dilation_factor",
        "time_dilation_factor_literal",
        # Teleop-posture IK branch selection (motion_planning.resolve_posture_selection, resolve_ik_num_seeds).
        "posture_selection_seeds",
        "posture_ref",
        "posture_grasp_roll",
        "posture_pos_tol",
        "posture_rot_tol",
        "ik_num_seeds",
        # cuRobo trajopt costs (motion_planning.apply_cost_overrides).
        "encoder_weight",
        "encoder_path",
        "uniform_velocity_weight",
        "rnd_novelty_weight",
        "rnd_novelty_log",
        "joint_density_weight",
        "smooth_weight",
        "bound_weight",
        "bound_activation_distance",
        "run_weight_acceleration",
        "run_weight_jerk",
        "pose_weight",
        "run_vec_weight",
        "primitive_collision_activation_distance",
        "self_collision_weight",
        "cspace_weight",
        # cuRobo trajopt model and MotionGen joint-limit scales (apply_model_overrides, _scale_kwargs).
        "horizon",
        "base_dt",
        "velocity_scale",
        "acceleration_scale",
        "jerk_scale",
        # Encoder stroke re-timing (trajectory_blending.resolve_blend_config; retime_mode is checked below).
        "retime_trajectory",
        "retime_mode",
        "retime_smoothing",
        "retime_boundary_speed",
        "retime_speed_scale",
        "retime_vel_slack",
        "retime_acc_slack",
        "retime_ops",
        "retime_max_duration_mult",
        "retime_sample_target",
        # Goal clearing (goal_clearing.resolve_clear_goal_surfaces).
        "clear_goal_surfaces",
    }
)


def check_override_keys(overrides: dict) -> None:
    """Raise ValueError for a key not in SUPPORTED_OVERRIDE_KEYS, a ``retime_mode`` other than "encoder", or
    ``retime_trajectory`` without the ``encoder_path`` that the re-timer needs."""
    unknown = sorted(set(overrides) - SUPPORTED_OVERRIDE_KEYS)
    if unknown:
        raise ValueError(
            f"Unsupported tamp override key(s) {unknown}; the supported keys are listed in "
            f"SUPPORTED_OVERRIDE_KEYS in tiptop/override_keys.py"
        )
    if "retime_mode" in overrides and overrides["retime_mode"] != "encoder":
        raise ValueError(
            f"retime_mode must be 'encoder', the only stroke re-timing mode, got {overrides['retime_mode']!r}"
        )
    if overrides.get("retime_trajectory") and not overrides.get("encoder_path"):
        raise ValueError(
            "retime_trajectory needs encoder_path, the trajectory-encoder checkpoint that times each stroke"
        )
