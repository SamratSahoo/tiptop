"""Auto mode's loop: it must not block, and it must not lose the labels it skipped.

Two hard stops sit in the ordinary rollout loop, both of them blocking reads on stdin: the
success/failure prompt after every episode (``_label_rollout``) and the "next task?" prompt before
every episode (``_get_task_instruction``). Auto mode has to get past both or it collects exactly one
episode and waits -- while still obeying a stop, which arrives on that same stdin.
"""

import io
import os

import pytest

from tiptop import tiptop_run
from tiptop.tiptop_run import UserExitException


@pytest.fixture(autouse=True)
def _clean_queue(monkeypatch):
    monkeypatch.setattr(tiptop_run, "_AUTO_LABEL_QUEUE", [])
    monkeypatch.setattr(tiptop_run, "_LAST_SEGMENT_DIR", None)
    monkeypatch.setattr(tiptop_run, "_LAST_TASK", None)


def _eval_rollout(root, stamp):
    d = root / "eval" / stamp
    d.mkdir(parents=True)
    return d


# --- deferring, and draining -------------------------------------------------------------------


def test_a_deferred_rollout_stays_in_eval_and_is_remembered(tmp_path):
    d = _eval_rollout(tmp_path, "2026-09-01_10-00-00")
    tiptop_run._defer_label(d, "2026-09-01_10-00-00")
    assert tiptop_run._AUTO_LABEL_QUEUE == [(d, "2026-09-01_10-00-00")]
    assert d.is_dir(), "nothing is moved until the label arrives"


def test_the_drain_labels_oldest_first_and_moves_each_one(tmp_path, monkeypatch):
    stamps = ["2026-09-01_10-00-00", "2026-09-01_10-04-00", "2026-09-01_10-08-00"]
    for stamp in stamps:
        tiptop_run._defer_label(_eval_rollout(tmp_path, stamp), stamp)

    asked, answers = [], iter(["y", "n", "y"])

    def _fake_input(prompt=""):
        asked.append(prompt)
        return next(answers)

    processed = []
    monkeypatch.setattr("builtins.input", _fake_input)
    monkeypatch.setattr(tiptop_run, "_spawn_postprocess", processed.append)
    tiptop_run._drain_auto_label_queue(str(tmp_path))

    assert len(asked) == 3, "one prompt per deferred rollout"
    assert not tiptop_run._AUTO_LABEL_QUEUE
    assert [p.parent.name for p in processed] == ["success", "failure", "success"]
    assert [p.name for p in processed] == stamps, "oldest first"
    assert not any((tmp_path / "eval").glob("*")), "every one left eval/"


def test_the_drain_skips_a_rollout_that_is_already_gone(tmp_path, monkeypatch):
    """The operator can relabel from the Episodes page while auto mode is still running."""
    gone = _eval_rollout(tmp_path, "2026-09-01_10-00-00")
    kept = _eval_rollout(tmp_path, "2026-09-01_10-04-00")
    tiptop_run._defer_label(gone, "2026-09-01_10-00-00")
    tiptop_run._defer_label(kept, "2026-09-01_10-04-00")
    gone.rmdir()

    monkeypatch.setattr("builtins.input", lambda prompt="": "y")
    monkeypatch.setattr(tiptop_run, "_spawn_postprocess", lambda d: None)
    tiptop_run._drain_auto_label_queue(str(tmp_path))
    assert not tiptop_run._AUTO_LABEL_QUEUE
    assert (tmp_path / "success" / "2026-09-01_10-04-00").is_dir()


def test_a_ctrl_c_during_the_drain_leaves_the_rest_in_eval(tmp_path, monkeypatch):
    """Giving up must not propagate out of the session -- the rest is still on disk."""
    for stamp in ("2026-09-01_10-00-00", "2026-09-01_10-04-00"):
        tiptop_run._defer_label(_eval_rollout(tmp_path, stamp), stamp)

    def _interrupt(prompt=""):
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", _interrupt)
    monkeypatch.setattr(tiptop_run, "_clear_preempt", lambda: None)
    tiptop_run._drain_auto_label_queue(str(tmp_path))  # must not raise
    assert len(tiptop_run._AUTO_LABEL_QUEUE) == 2
    assert (tmp_path / "eval" / "2026-09-01_10-00-00").is_dir()


def test_an_empty_queue_prompts_for_nothing(tmp_path, monkeypatch):
    def _boom(prompt=""):
        raise AssertionError("must not prompt with nothing queued")

    monkeypatch.setattr("builtins.input", _boom)
    tiptop_run._drain_auto_label_queue(str(tmp_path))


# --- the non-blocking task source --------------------------------------------------------------


def test_nothing_on_stdin_repeats_the_last_task(monkeypatch):
    """The case that runs on every pass: no operator input, so just collect again."""
    monkeypatch.setattr(tiptop_run, "_LAST_TASK", "place the toys on the plate")
    monkeypatch.setattr(tiptop_run, "_pending_stdin_line", lambda: None)
    monkeypatch.delenv("TIPTOP_TASK", raising=False)
    assert tiptop_run._auto_mode_task_instruction() == "place the toys on the plate"


def test_a_bare_enter_is_a_no_op(monkeypatch):
    """The UI's "Collect another" — already what auto mode is doing."""
    monkeypatch.setattr(tiptop_run, "_LAST_TASK", "place the toys on the plate")
    monkeypatch.setattr(tiptop_run, "_pending_stdin_line", lambda: "")
    monkeypatch.delenv("TIPTOP_TASK", raising=False)
    assert tiptop_run._auto_mode_task_instruction() == "place the toys on the plate"


def test_a_graceful_stop_still_ends_the_session(monkeypatch):
    """data-collection writes "q\\n" and SIGTERMs 2 s later, so this read cannot be skipped."""
    monkeypatch.setattr(tiptop_run, "_LAST_TASK", "place the toys on the plate")
    monkeypatch.setattr(tiptop_run, "_pending_stdin_line", lambda: "q")
    monkeypatch.delenv("TIPTOP_TASK", raising=False)
    with pytest.raises(UserExitException):
        tiptop_run._auto_mode_task_instruction()


def test_a_robot_command_is_handed_back_for_the_caller_to_run(monkeypatch):
    monkeypatch.setattr(tiptop_run, "_LAST_TASK", "place the toys on the plate")
    monkeypatch.setattr(tiptop_run, "_pending_stdin_line", lambda: "reset")
    monkeypatch.delenv("TIPTOP_TASK", raising=False)
    assert tiptop_run._auto_mode_task_instruction() == "reset"
    assert tiptop_run._LAST_TASK == "place the toys on the plate", "a nudge is not a task"


def test_a_new_instruction_switches_what_auto_mode_collects(monkeypatch):
    monkeypatch.setattr(tiptop_run, "_LAST_TASK", "place the toys on the plate")
    monkeypatch.setattr(tiptop_run, "_pending_stdin_line", lambda: "pack the toys in the box")
    monkeypatch.delenv("TIPTOP_TASK", raising=False)
    assert tiptop_run._auto_mode_task_instruction() == "pack the toys in the box"
    assert tiptop_run._LAST_TASK == "pack the toys in the box"


def test_the_launch_task_is_consumed_the_ordinary_way(monkeypatch):
    """First pass of a data-collection session: TIPTOP_TASK is set and does not block."""
    monkeypatch.setenv("TIPTOP_TASK", "place the toys on the plate")
    monkeypatch.setattr(tiptop_run, "_pending_stdin_line", lambda: None)
    assert tiptop_run._auto_mode_task_instruction() == "place the toys on the plate"
    assert tiptop_run._LAST_TASK == "place the toys on the plate"


def test_with_no_task_at_all_it_asks_once(monkeypatch):
    """A terminal launch with auto_mode and no TIPTOP_TASK still has to be told what to collect."""
    monkeypatch.delenv("TIPTOP_TASK", raising=False)
    monkeypatch.setattr("builtins.input", lambda prompt="": "place the toys on the plate")
    assert tiptop_run._auto_mode_task_instruction() == "place the toys on the plate"


# --- the stdin poll itself ----------------------------------------------------------------------


def test_polling_a_stream_with_nothing_on_it_returns_none(monkeypatch):
    r, w = os.pipe()
    monkeypatch.setattr(tiptop_run.sys, "stdin", os.fdopen(r))
    try:
        assert tiptop_run._pending_stdin_line() is None
    finally:
        os.close(w)


def test_polling_reads_a_line_that_is_already_waiting(monkeypatch):
    r, w = os.pipe()
    os.write(w, b"reset\n")
    os.close(w)
    monkeypatch.setattr(tiptop_run.sys, "stdin", os.fdopen(r))
    assert tiptop_run._pending_stdin_line() == "reset"


def test_a_closed_stdin_is_not_an_error(monkeypatch):
    """Some launchers hand over something that cannot be selected; auto mode just keeps going."""
    monkeypatch.setattr(tiptop_run.sys, "stdin", None)
    assert tiptop_run._pending_stdin_line() is None
    closed = io.StringIO()
    closed.close()
    monkeypatch.setattr(tiptop_run.sys, "stdin", closed)
    assert tiptop_run._pending_stdin_line() is None
