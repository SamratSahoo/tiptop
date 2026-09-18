"""``prev_rollout.json``: the back-pointer that orders a session's footage.

A session writes two kinds of recorded segment into the same run directory -- rollouts (``eval/``,
moved to ``success|failure/`` when labelled) and scene resets (``resets/``) -- and both record
cameras. Stitching them into one continuous video needs their ORDER, which timestamps alone do not
give reliably: a rollout's directory is renamed when it is labelled, and a relabel can happen after
later segments already exist. So each segment names its predecessor as it starts.
"""

from pathlib import Path

from tiptop import tiptop_run


def _segment(root: Path, parent: str, stamp: str) -> Path:
    d = root / parent / stamp
    d.mkdir(parents=True)
    return d


def test_a_segments_kind_comes_from_its_bucket():
    assert tiptop_run._segment_kind(Path("/runs/tamp/x/resets/2026-09-01_10-00-00")) == "reset"
    assert tiptop_run._segment_kind(Path("/runs/tamp/x/success/2026-09-01_10-00-00")) == "rollout"
    assert tiptop_run._segment_kind(Path("/runs/tamp/x/eval/2026-09-01_10-00-00")) == "rollout"
    assert tiptop_run._segment_kind(None) is None


def test_the_first_segment_of_an_empty_run_has_no_predecessor(tmp_path, monkeypatch):
    monkeypatch.setattr(tiptop_run, "_LAST_SEGMENT_DIR", None)
    pointer = tiptop_run._write_prev_pointer(
        tmp_path / "eval" / "2026-09-01_10-00-00", tiptop_run._prev_segment(str(tmp_path))
    )
    assert pointer == {
        "prev_rollout": None,
        "prev_rollout_id": None,
        "prev_rollout_kind": None,
        "kind": "rollout",
    }


def test_a_reset_points_at_the_rollout_it_followed(tmp_path, monkeypatch):
    rollout = _segment(tmp_path, "success", "2026-09-01_10-00-00")
    monkeypatch.setattr(tiptop_run, "_LAST_SEGMENT_DIR", None)
    tiptop_run._note_segment(rollout)

    reset_dir = tmp_path / "resets" / "2026-09-01_10-05-00"
    pointer = tiptop_run._write_prev_pointer(reset_dir, tiptop_run._prev_segment(str(tmp_path)))
    assert pointer["prev_rollout"] == str(rollout)
    assert pointer["prev_rollout_id"] == "2026-09-01_10-00-00"
    assert pointer["prev_rollout_kind"] == "rollout"
    assert pointer["kind"] == "reset"
    import json

    assert json.loads((reset_dir / "prev_rollout.json").read_text()) == pointer


def test_a_rollout_points_at_the_reset_it_followed(tmp_path, monkeypatch):
    """The interleaved case, which is the whole point: rollout, reset, rollout, ... must chain."""
    reset_dir = _segment(tmp_path, "resets", "2026-09-01_10-05-00")
    monkeypatch.setattr(tiptop_run, "_LAST_SEGMENT_DIR", None)
    tiptop_run._note_segment(reset_dir)
    pointer = tiptop_run._write_prev_pointer(
        tmp_path / "eval" / "2026-09-01_10-09-00", tiptop_run._prev_segment(str(tmp_path))
    )
    assert pointer["prev_rollout"] == str(reset_dir) and pointer["prev_rollout_kind"] == "reset"


def test_a_restart_picks_the_newest_segment_already_on_disk(tmp_path, monkeypatch):
    """First segment of a fresh process, into a run directory that already holds footage."""
    monkeypatch.setattr(tiptop_run, "_LAST_SEGMENT_DIR", None)
    _segment(tmp_path, "success", "2026-09-01_10-00-00")
    _segment(tmp_path, "resets", "2026-09-01_10-05-00")
    newest = _segment(tmp_path, "failure", "2026-09-01_10-09-00")
    _segment(tmp_path, "resets", "2026-08-30_23-59-59")  # older, different day

    mine = tmp_path / "eval" / "2026-09-01_10-12-00"
    assert tiptop_run._prev_segment(str(tmp_path), exclude=mine) == newest


def test_a_segment_never_points_at_itself(tmp_path, monkeypatch):
    """The reset path resolves its pointer before its own directory exists; the rollout path does
    not, so the exclusion has to hold even once the directory is there."""
    monkeypatch.setattr(tiptop_run, "_LAST_SEGMENT_DIR", None)
    mine = _segment(tmp_path, "eval", "2026-09-01_10-12-00")
    assert tiptop_run._prev_segment(str(tmp_path), exclude=mine) is None


def test_the_in_process_pointer_wins_over_the_disk_scan(tmp_path, monkeypatch):
    """An unlabelled rollout left in eval/ by a previous process has a NEWER-looking name than the
    reset this process just ran; the chain must follow what actually happened."""
    monkeypatch.setattr(tiptop_run, "_LAST_SEGMENT_DIR", None)
    just_ran = _segment(tmp_path, "resets", "2026-09-01_10-05-00")
    _segment(tmp_path, "eval", "2026-09-01_23-00-00")
    tiptop_run._note_segment(just_ran)
    assert tiptop_run._prev_segment(str(tmp_path)) == just_ran
