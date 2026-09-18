"""Auto mode: the reset-or-collect decision, and the guard that keeps it from stalling a session.

Everything here is the pure half (``tiptop.auto_mode``) plus how Gemini's answer is read
(``perception.gemini._parse_reset_check``). The camera read and the API call around them live in
``tiptop_run._auto_mode_needs_reset``, which needs hardware.
"""

import asyncio

import pytest

from tiptop.auto_mode import MAX_CONSECUTIVE_AUTO_RESETS, resolve_auto_mode, should_reset
from tiptop.perception.gemini import _parse_reset_check


def test_auto_mode_is_off_unless_a_config_asks():
    """Every existing cfg/tamp yml has no auto_mode key and must keep prompting the operator."""
    assert resolve_auto_mode(None) is False
    assert resolve_auto_mode({}) is False
    assert resolve_auto_mode({"vae_manifold_weight": 25000}) is False
    assert resolve_auto_mode({"auto_mode": False}) is False
    assert resolve_auto_mode({"auto_mode": True}) is True


def test_a_clear_scene_collects():
    do_reset, why = should_reset(needs_reset=False, consecutive_resets=0)
    assert do_reset is False and "ready" in why


def test_a_dirty_scene_resets():
    assert should_reset(needs_reset=True, consecutive_resets=0)[0] is True


def test_a_second_reset_is_still_allowed():
    """One reset can legitimately leave work behind: _plan_largest_solvable_reset drops the objects
    it could not plan for and reports them as skipped, so the next pass should finish the job."""
    assert MAX_CONSECUTIVE_AUTO_RESETS >= 2
    assert should_reset(needs_reset=True, consecutive_resets=1)[0] is True


def test_the_streak_guard_collects_rather_than_looping_forever():
    """A scene a reset cannot fix -- an object wedged in a container -- must not stall the session."""
    do_reset, why = should_reset(needs_reset=True, consecutive_resets=MAX_CONSECUTIVE_AUTO_RESETS)
    assert do_reset is False
    assert "stall" in why and str(MAX_CONSECUTIVE_AUTO_RESETS) in why
    # ...and it stays collected however high the streak goes.
    assert should_reset(needs_reset=True, consecutive_resets=99)[0] is False


@pytest.mark.parametrize(
    "text, expected",
    [
        ('{"needs_reset": true, "reason": "the toys are on the plate"}', True),
        ('{"needs_reset": false, "reason": "toys loose on the table"}', False),
        ('```json\n{"needs_reset": true, "reason": "stacked"}\n```', True),  # fenced anyway
        ('{"needs_reset": false}', False),  # reason is optional
    ],
)
def test_parsing_geminis_answer(text, expected):
    needs_reset, reason = _parse_reset_check(text)
    assert needs_reset is expected
    assert isinstance(reason, str)


def test_a_missing_verdict_is_an_error_not_a_no():
    """ "No reset needed" and "Gemini did not answer" lead to different actions, so they must not
    collapse into the same value -- the caller turns the raise into "collect, and say why"."""
    with pytest.raises(ValueError, match="needs_reset"):
        _parse_reset_check('{"reason": "I am not sure"}')
    with pytest.raises(Exception):
        _parse_reset_check("I think the scene looks fine")


# --- the orchestration around the call (tiptop_run._auto_mode_needs_reset) ----------------------


class _FakeFrame:
    def __init__(self):
        import numpy as np

        self.rgb = np.zeros((360, 640, 3), dtype=np.uint8)


class _FakeCam:
    def read_camera(self):
        return _FakeFrame()


class _FakeContainer:
    external_cam = _FakeCam()


def test_the_decision_and_the_image_it_judged_are_saved(tmp_path, monkeypatch):
    """ "Why did it reset there?" has to be answerable after the session."""
    import json

    from tiptop import tiptop_run

    async def _fake_check(image, instruction):
        assert image.size[0] == 800, "the image must be downscaled the way detection downscales"
        return True, "the toys are already on the plate"

    monkeypatch.setattr(tiptop_run, "check_scene_needs_reset_async", _fake_check)
    needs, reason = asyncio.run(
        tiptop_run._auto_mode_needs_reset(_FakeContainer(), "place the toys on the plate", str(tmp_path))
    )
    assert needs is True and "plate" in reason

    (saved,) = list((tmp_path / "auto_mode").glob("*.json"))
    assert json.loads(saved.read_text()) == {
        "instruction": "place the toys on the plate",
        "needs_reset": True,
        "reason": "the toys are already on the plate",
    }
    assert (tmp_path / "auto_mode" / f"{saved.stem}.jpg").is_file()


def test_an_unreachable_gemini_collects_instead_of_stalling(tmp_path, monkeypatch):
    """No API key, no network: the session must keep working, with the error as the reason."""
    from tiptop import tiptop_run

    async def _boom(image, instruction):
        raise RuntimeError("no API key")

    monkeypatch.setattr(tiptop_run, "check_scene_needs_reset_async", _boom)
    needs, reason = asyncio.run(
        tiptop_run._auto_mode_needs_reset(_FakeContainer(), "place the toys on the plate", str(tmp_path))
    )
    assert needs is False
    assert "no API key" in reason
