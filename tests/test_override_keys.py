"""The ``tamp_overrides`` whitelist (``tiptop.override_keys``) and its enforcement when overrides are loaded.

``tiptop_websocket_server._load_curobo_overrides`` is what both tiptop-run and tiptop-server load with, so a
key it rejects never reaches a resolver: an unsupported setting fails at startup instead of being ignored.
tiptop-server also rejects the tiptop-run session options and applies every other key the way tiptop-run does.
"""

import json
import re
from pathlib import Path

import pytest
import yaml

from tiptop.override_keys import SESSION_ONLY_OVERRIDE_KEYS, SUPPORTED_OVERRIDE_KEYS, check_override_keys

# DATAFARM checks tiptop out at submodules/tiptop, three levels below its task configs.
DATAFARM_CONFIGS = Path(__file__).resolve().parents[3] / "data_collection" / "configs"

ENCODER_RETIMING = {
    "encoder_weight": 25000,
    "encoder_path": "/ckpts/encoder.pt",
    "retime_trajectory": True,
    "retime_mode": "encoder",
    "retime_smoothing": 3.0e-4,
    "retime_boundary_speed": 0.07,
}

# The tiptop-run session options, as a config sets them.
SESSION_OPTIONS = {"auto_mode": True, "reset_placement_region": {"x": [0.40, 0.55], "y": [-0.2, 0.2]}}

# Names tiptop used to read, and the names that replaced them. The old names are not accepted any more.
RENAMED_KEYS = {
    "vae_path": "encoder_path",
    "vae_manifold_weight": "encoder_weight",
    "blend_trajectory": "retime_trajectory",
    "blend_mode": "retime_mode",
    "blend_smoothing": "retime_smoothing",
    "blend_boundary_speed": "retime_boundary_speed",
    "blend_ops": "retime_ops",
    "blend_max_duration_mult": "retime_max_duration_mult",
    "blend_vel_slack": "retime_vel_slack",
    "blend_acc_slack": "retime_acc_slack",
    "blend_vae_sample_target": "retime_sample_target",
    "blend_speed_scale": "retime_speed_scale",
}

# Settings of the removed vae_retiming feature and of the removed spline and flow re-timing modes.
REMOVED_KEYS = [
    ("vae_retiming", True),
    ("vae_retiming", False),
    ("retime_scale", 0.7),
    ("retime_smooth_weight", 10.0),
    ("retime_limit_weight", 500.0),
    ("blend_mode", "spline"),
    ("blend_mode", "flow"),
    ("blend_boundary_window", 0.5),
    ("blend_model_path", "checkpoints/flow_net_t64.pt"),
    ("blend_flow_steps", 60),
    ("blend_flow_retime_only", True),
    ("blend_pace", "droid"),
    ("blend_pace_scale", 1.0),
    ("blend_boundary_mode", "droid"),
    ("blend_boundary_window_sec", 0.5),
    ("blend_profile_end_sec", 1.0),
    ("blend_stats_path", "checkpoints/droid_timing_stats.npz"),
    ("blend_seed", 0),
]


def _datafarm_configs() -> list:
    configs = sorted(DATAFARM_CONFIGS.glob("*.yaml"))
    if not configs:
        return [pytest.param(None, marks=pytest.mark.skip(reason=f"no task configs at {DATAFARM_CONFIGS}"))]
    return configs


def _tamp_overrides(cfg_path: Path) -> dict:
    return yaml.safe_load(cfg_path.read_text())["tamp_overrides"]


def test_encoder_retiming_is_accepted():
    check_override_keys(ENCODER_RETIMING)
    check_override_keys({k: v for k, v in ENCODER_RETIMING.items() if k != "retime_mode"})
    check_override_keys({})


@pytest.mark.parametrize("cfg_path", _datafarm_configs(), ids=lambda p: p.stem if p else "none")
def test_every_datafarm_config_is_accepted(cfg_path):
    overrides = _tamp_overrides(cfg_path)
    check_override_keys(overrides)
    assert overrides["retime_trajectory"] is True and overrides["retime_mode"] == "encoder"


@pytest.mark.parametrize(("old", "new"), RENAMED_KEYS.items())
def test_old_names_are_rejected(old, new):
    with pytest.raises(ValueError, match=rf"\['{old}'\].*tiptop/override_keys\.py"):
        check_override_keys({**ENCODER_RETIMING, old: ENCODER_RETIMING.get(new, True)})


def test_every_new_name_is_supported():
    assert set(RENAMED_KEYS.values()) <= SUPPORTED_OVERRIDE_KEYS
    assert not set(RENAMED_KEYS) & SUPPORTED_OVERRIDE_KEYS


@pytest.mark.parametrize(("key", "value"), REMOVED_KEYS, ids=[f"{k}={v}" for k, v in REMOVED_KEYS])
def test_removed_features_are_rejected(key, value):
    with pytest.raises(ValueError, match=rf"\['{key}'\]"):
        check_override_keys({**ENCODER_RETIMING, key: value})


@pytest.mark.parametrize("mode", ["spline", "flow", "vae", "Encoder", " encoder", "", None])
def test_retime_mode_must_be_encoder(mode):
    message = f"retime_mode must be 'encoder', the only stroke re-timing mode, got {mode!r}"
    with pytest.raises(ValueError, match=re.escape(message)):
        check_override_keys({**ENCODER_RETIMING, "retime_mode": mode})


def test_retiming_needs_an_encoder_path():
    without_path = {k: v for k, v in ENCODER_RETIMING.items() if k != "encoder_path"}
    with pytest.raises(ValueError, match="retime_trajectory needs encoder_path"):
        check_override_keys(without_path)
    check_override_keys({**without_path, "retime_trajectory": False})


def test_a_misspelled_key_is_rejected():
    with pytest.raises(ValueError, match=r"\['retime_smothing'\]"):
        check_override_keys({"retime_trajectory": True, "retime_smothing": 3.0e-4})


def test_session_options_are_supported_keys():
    """tiptop-run reads them, so the shared whitelist accepts them; only tiptop-server turns them away."""
    assert set(SESSION_OPTIONS) == SESSION_ONLY_OVERRIDE_KEYS
    assert SESSION_ONLY_OVERRIDE_KEYS <= SUPPORTED_OVERRIDE_KEYS
    check_override_keys({**ENCODER_RETIMING, **SESSION_OPTIONS})


def test_the_error_lists_every_offending_key_sorted():
    overrides = {**ENCODER_RETIMING, "vae_retiming": True, "blend_mode": "spline", "retime_smothing": 1e-4}
    with pytest.raises(ValueError) as exc:
        check_override_keys(overrides)
    assert "['blend_mode', 'retime_smothing', 'vae_retiming']" in str(exc.value)
    assert "SUPPORTED_OVERRIDE_KEYS in tiptop/override_keys.py" in str(exc.value)


@pytest.fixture
def load_curobo_overrides():
    """The real loader; importing the websocket server needs the full pixi environment (cuRobo, cuTAMP)."""
    from tiptop.tiptop_websocket_server import _load_curobo_overrides

    return _load_curobo_overrides


@pytest.mark.parametrize("cfg_path", _datafarm_configs(), ids=lambda p: p.stem if p else "none")
def test_loading_every_datafarm_config(load_curobo_overrides, tmp_path, cfg_path):
    """``tiptop-run --curobo-overrides <file>.json``, as data_collection/collect.py starts it."""
    overrides = _tamp_overrides(cfg_path)
    path = tmp_path / "tamp_overrides.json"
    path.write_text(json.dumps(overrides, indent=2))
    loaded = load_curobo_overrides(str(path))
    assert loaded == overrides
    assert list(loaded) == list(overrides)


def test_loading_inline_json(load_curobo_overrides):
    assert load_curobo_overrides(json.dumps(ENCODER_RETIMING)) == ENCODER_RETIMING
    assert load_curobo_overrides(None) == {}


@pytest.mark.parametrize(
    ("extra", "match"),
    [
        ({"vae_retiming": True}, r"\['vae_retiming'\]"),
        ({"blend_mode": "flow"}, r"\['blend_mode'\]"),
        ({"retime_mode": "spline"}, "retime_mode must be 'encoder'"),
    ],
)
def test_loading_an_unsupported_file_is_rejected(load_curobo_overrides, tmp_path, extra, match):
    path = tmp_path / "tamp_overrides.json"
    path.write_text(json.dumps({**ENCODER_RETIMING, **extra}))
    with pytest.raises(ValueError, match=match):
        load_curobo_overrides(str(path))


def test_loading_keeps_the_session_options(load_curobo_overrides):
    """tiptop-run loads with the same function, and it must still get auto_mode and reset_placement_region."""
    overrides = {**ENCODER_RETIMING, **SESSION_OPTIONS}
    assert load_curobo_overrides(json.dumps(overrides)) == overrides


@pytest.fixture
def planning_server():
    """The real server class; importing it needs the full pixi environment (cuRobo, cuTAMP)."""
    from tiptop.tiptop_websocket_server import TiptopPlanningServer

    return TiptopPlanningServer


@pytest.mark.parametrize(
    "session_options",
    [{"auto_mode": True}, {"auto_mode": False}, {"reset_placement_region": SESSION_OPTIONS["reset_placement_region"]}],
    ids=["auto_mode", "auto_mode_off", "reset_placement_region"],
)
def test_the_server_rejects_session_options(planning_server, session_options):
    """The server has no reset-or-collect loop or scene reset, so it fails instead of ignoring them."""
    (key,) = session_options
    with pytest.raises(ValueError, match=rf"\['{key}'\] are tiptop-run session options"):
        planning_server(curobo_overrides=json.dumps({**ENCODER_RETIMING, **session_options}))


def test_the_server_rejection_lists_every_session_option(planning_server):
    with pytest.raises(ValueError, match=r"\['auto_mode', 'reset_placement_region'\].*tiptop/override_keys\.py"):
        planning_server(curobo_overrides=json.dumps({**ENCODER_RETIMING, **SESSION_OPTIONS}))


def test_the_server_applies_solver_effort_and_grasp_keys(planning_server):
    """Overrides win over the num_particles argument, and every key reaches the TAMPConfiguration."""
    overrides = {
        "num_particles": 512,
        "opt_steps_per_skeleton": 600,
        "grasp_rank_conf_weight": 2,
        "require_m2t2_grasps": True,
    }
    server = planning_server(num_particles=128, curobo_overrides=json.dumps(overrides))
    assert (server._config.num_particles, server._config.num_opt_steps) == (512, 600)
    assert server._config.grasp_rank_conf_weight == 2.0
    assert server._config.require_m2t2_grasps is True
    assert (server._metadata["num_particles"], server._metadata["opt_steps_per_skeleton"]) == (512, 600)


def test_the_server_defaults_without_overrides(planning_server):
    server = planning_server(num_particles=128)
    assert (server._config.num_particles, server._config.num_opt_steps) == (128, 500)
    assert server._config.grasp_rank_conf_weight is None
    assert server._config.require_m2t2_grasps is False
    assert (server._metadata["num_particles"], server._metadata["opt_steps_per_skeleton"]) == (128, 500)


def test_the_server_rejects_non_positive_solver_effort(planning_server):
    with pytest.raises(ValueError, match="num_particles and opt_steps_per_skeleton must be positive"):
        planning_server(curobo_overrides=json.dumps({"opt_steps_per_skeleton": 0}))
