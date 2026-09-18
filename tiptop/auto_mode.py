"""Auto mode: let the VLM decide, between rollouts, whether to reset the scene or collect.

A data-collection session is a loop of "collect an episode, put the objects back, collect another",
and the putting-back is the half an operator has to remember: the scene ends every rollout in its
GOAL state (the toys on the plate), so the next rollout has to start with a **⟲ Reset scene**. Auto
mode takes that decision over. At each task prompt it shows Gemini the workspace and asks one
question -- does this scene have to be put back before the task can be attempted again? -- then runs
either the reset or the rollout. The operator still drives the loop one step at a time (*Collect
another* / Enter) and still labels each episode; only the reset-or-collect choice moves.

Opt in per config with ``auto_mode: true`` in a ``cfg/tamp/*.yml``'s ``tamp_overrides``, the same
JSON every other knob rides (see :func:`goal_clearing.resolve_clear_goal_surfaces`).

**Why a VLM at all, when the reset already decides for itself what to move.**
``scene_reset.build_reset_goal`` reads the support relations out of the perceived geometry, and a
reset over an already-clear table correctly moves nothing. So the VLM is not deciding WHAT to move;
it decides whether to spend a reset cycle at all. That cycle is not free -- perception, a cuTAMP
plan and the arm driving to the capture pose, tens of seconds -- and running one before every single
rollout would roughly halve the session's episode rate. One Gemini call is about a second.

**The two answers fail differently, which is what the guard below is for.** A false "no reset
needed" costs one rollout that starts from a dirty scene -- recoverable, the operator labels it a
failure. A false "yes" costs a reset cycle that moves nothing, and if the VLM keeps saying yes the
session never collects anything again. ``MAX_CONSECUTIVE_AUTO_RESETS`` is the stop: after that many
resets in a row with no rollout between them, auto mode stops believing the answer and collects.
"""

import logging

_log = logging.getLogger(__name__)

# How many resets auto mode will run back-to-back before it collects an episode regardless of what
# the VLM says. Two, because the legitimate case is one: a reset that could only plan for some of the
# objects (``_plan_largest_solvable_reset`` reports the rest as skipped) leaves work for a second.
# A third in a row means the scene is not something a reset can fix -- an object wedged in a
# container, a mis-detection -- and the operator needs to see a rollout, or the session stalls.
MAX_CONSECUTIVE_AUTO_RESETS = 2


def resolve_auto_mode(overrides: dict | None) -> bool:
    """Whether this session picks reset-or-collect itself, from cfg/tamp ``tamp_overrides``.

    Off unless a config opts in with ``auto_mode: true``, so every existing config keeps prompting
    the operator for the choice.
    """
    return bool((overrides or {}).get("auto_mode"))


def should_reset(needs_reset: bool, consecutive_resets: int) -> tuple[bool, str]:
    """Turn the VLM's answer plus the reset streak into the action to take, and why.

    Pure, so the guard is testable without a camera or an API key. ``consecutive_resets`` counts the
    resets auto mode has run since the last rollout.
    """
    if not needs_reset:
        return False, "the scene is ready for a fresh attempt"
    if consecutive_resets >= MAX_CONSECUTIVE_AUTO_RESETS:
        return False, (
            f"{consecutive_resets} reset(s) in a row have not cleared the scene "
            f"(limit {MAX_CONSECUTIVE_AUTO_RESETS}); collecting anyway so the session does not stall"
        )
    return True, "the scene still holds the last attempt's result"
