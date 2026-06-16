import logging
import time

from tiptop.utils import RobotClient, get_robot_client

_log = logging.getLogger(__name__)


class ExecutionFailure(Exception):
    """Failure in executing plan on robot."""


def _gripper_state(client) -> dict | None:
    """Best-effort read of the gripper state dict ({width, is_grasped, is_moving}). None if unavailable."""
    try:
        res = client.get_gripper_state()
    except Exception:
        return None
    if not isinstance(res, dict):
        return None
    state = res.get("state", res)
    return state if isinstance(state, dict) else None


def _wait_for_gripper_settled(client, *, timeout: float = 5.0, poll: float = 0.02) -> None:
    """Block until the gripper stops moving, polling client-side.

    This mirrors the bamboo gripper server's own ``_spin_until_done`` exit condition,
    but runs here so the (single-threaded) server socket stays free *between* polls --
    letting the LeRobot gripper sampler read the position register during the motion and
    capture the open<->close ramp, which a server-side blocking command hides entirely.

    Falls back to a short fixed wait if the gripper state can't be read, so the arm never
    starts the next trajectory before the gripper has had time to actuate.
    """
    start = time.time()
    saw_moving = False
    while time.time() - start < timeout:
        time.sleep(poll)
        state = _gripper_state(client)
        if state is None or "is_moving" not in state:
            time.sleep(1.0)  # unreadable state: wait out a conservative actuation time
            return
        if state["is_moving"]:
            saw_moving = True
        elif saw_moving or (time.time() - start) > 0.25:
            # Stopped after moving, or never moved and already settled (e.g. already open).
            return
    _log.warning("Gripper did not report settled within %.1fs; continuing", timeout)


def _command_gripper(client, action: str):
    """Issue an open/close gripper command non-blocking, then wait client-side for it to
    settle. Falls back to the client's blocking command if it doesn't accept blocking=False."""
    fn = client.open_gripper if action == "open" else client.close_gripper
    try:
        result = fn(speed=1.0, blocking=False)
    except TypeError:
        # Client without a blocking flag (e.g. UR5) -- it blocks until done itself.
        return fn(speed=1.0)
    _wait_for_gripper_settled(client)
    return result


def execute_cutamp_plan(
    cutamp_plan: list[dict], client: RobotClient | None = None, timeline: list | None = None
) -> None:
    """Execute the plan from cuTAMP on the real robot.

    If ``timeline`` is provided, one entry per plan step is appended to it (in order),
    each ``{"type", "label", "t_start", "t_end"}`` with wall-clock (epoch) seconds.
    The LeRobot export uses this to place each control frame on the real execution
    clock -- including the gripper-actuation pauses that the plan timeline omits -- so
    camera frames can be aligned to states by hardware timestamp.
    """
    if client is None:
        client = get_robot_client()

    start_time = time.perf_counter()
    for step, action_dict in enumerate(cutamp_plan):
        action_start_time = time.perf_counter()
        step_t_start = time.time()
        action_type = action_dict["type"]
        action_label = action_dict["label"]

        # Form log message
        msg = f"Executing step {step + 1}/{len(cutamp_plan)}: {action_label}. Action type: {action_dict['type']}"
        if action_type == "gripper":
            msg += f" ({action_dict['action']})"
        elif action_type == "trajectory":
            msg += f" ({len(action_dict['plan'].position)} waypoints)"
        else:
            raise ValueError(f"Unknown action type in cuTAMP plan: {action_dict['type']}")
        _log.info(msg)

        # Now execute the actions
        if action_type == "gripper":
            action = action_dict["action"]
            if action not in ("open", "close"):
                raise ValueError(f"Unknown gripper action: {action}")
            # Issue non-blocking and wait client-side so the gripper position can be
            # sampled during the motion (see _command_gripper / _wait_for_gripper_settled).
            result = _command_gripper(client, action)

        elif action_type == "trajectory":
            # Extract joint position and velocity waypoints for the trajectory
            waypoints = action_dict["plan"].position.cpu().numpy()
            velocities = action_dict["plan"].velocity.cpu().numpy()
            timings = [action_dict["dt"]] * len(waypoints)
            result = client.execute_joint_impedance_path(
                joint_confs=waypoints, joint_vels=velocities, durations=timings
            )

        else:
            raise ValueError(f"Unexpected action type in cuTAMP plan: {action_dict['type']}")

        # Raise error if execution failed
        if result is None:
            raise RuntimeError("Fatal error: result should not be None")
        # if not result["success"]:
        #     raise ExecutionFailure(result["error"])

        if timeline is not None:
            timeline.append(
                {"type": action_type, "label": action_label, "t_start": step_t_start, "t_end": time.time()}
            )

        action_duration = time.perf_counter() - action_start_time
        _log.debug(f"Executing {action_type} action took {action_duration:.2f}s")

    # Now we're done executing plan open-loop without any failures on the controller side
    duration = time.perf_counter() - start_time
    _log.info(f"Real robot execution took {duration:.2f}s")
