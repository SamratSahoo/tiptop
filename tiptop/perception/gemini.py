import json
import logging
from functools import cache
from pathlib import Path

from PIL import Image
from google import genai
from google.genai import types

_log = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"


@cache
def load_prompt(prompt_name: str) -> str:
    """Load a prompt template from the prompts directory."""
    return (_PROMPTS_DIR / f"{prompt_name}.txt").read_text().strip()


@cache
def gemini_client() -> genai.Client:
    return genai.Client()


def load_json(response_text: str) -> list | dict:
    """Extract JSON string from code fencing if present."""
    cleaned_text = response_text.strip()
    if cleaned_text.startswith("```json"):
        cleaned_text = cleaned_text.replace("```json", "").replace("```", "")
    elif cleaned_text.startswith("```"):
        cleaned_text = cleaned_text.replace("```", "")

    try:
        results = json.loads(cleaned_text)
    except json.decoder.JSONDecodeError:
        _log.error(f"Invalid JSON: {cleaned_text}")
        raise
    return results


def _parse_response(response_text: str) -> tuple[list, list]:
    """Parse Gemini response text into bboxes and grounded atoms."""
    try:
        result = load_json(response_text)
    except Exception:
        raise ValueError(f"Gemini returned a non-JSON response; check for a discrepancy in your image: {response_text}")
    bboxes = result.get("bboxes", [])
    grounded_atoms = [
        {"predicate": spec["name"], "args": spec["args"]}
        for spec in result.get("predicates", [])
        if spec.get("name") and spec.get("args")
    ]
    return bboxes, grounded_atoms


def detect_and_translate(
    image: Image.Image,
    task_instruction: str,
    client: genai.Client | None = None,
    model_id: str = "gemini-robotics-er-2-preview",
    temperature: float | None = None,
) -> tuple[list[dict], list[dict]]:
    """Detect objects and translate task in a single Gemini API call.

    Args:
        image: The image to analyze.
        task_instruction: The natural language task to translate.
        client: Gemini API client. If None, a new client will be created.
        model_id: Gemini model ID to use.
        temperature: Temperature for generation.

    Returns:
        Tuple of (bboxes, grounded_atoms) where:
        - bboxes: List of detected objects with bounding boxes
        - grounded_atoms: List of predicate specifications
    """
    client = client or gemini_client()
    prompt = load_prompt("detect_and_translate").format(task_instruction=task_instruction)
    response = client.models.generate_content(
        model=model_id,
        contents=[image, prompt],
        config=types.GenerateContentConfig(
            temperature=temperature, thinking_config=types.ThinkingConfig(thinking_budget=0)
        ),
    )
    return _parse_response(response.text)


async def detect_and_translate_async(
    image: Image.Image,
    task_instruction: str,
    client: genai.Client | None = None,
    model_id: str = "gemini-robotics-er-2-preview",
    temperature: float | None = None,
) -> tuple[list[dict], list[dict]]:
    """Asynchronously detect objects and translate task in a single Gemini API call.

    Args:
        image: The image to analyze.
        task_instruction: The natural language task to translate.
        client: Gemini API client. If None, a new client will be created.
        model_id: Gemini model ID to use.
        temperature: Temperature for generation.

    Returns:
        Tuple of (bboxes, grounded_atoms) where:
        - bboxes: List of detected objects with bounding boxes.
        - grounded_atoms: List of predicate specifications.
    """
    client = client or gemini_client()
    prompt = load_prompt("detect_and_translate").format(task_instruction=task_instruction)
    response = await client.aio.models.generate_content(
        model=model_id,
        contents=[image, prompt],
        config=types.GenerateContentConfig(
            temperature=temperature, thinking_config=types.ThinkingConfig(thinking_budget=0)
        ),
    )
    return _parse_response(response.text)


def _parse_reset_check(response_text: str) -> tuple[bool, str]:
    """Parse the scene_reset_check response into ``(needs_reset, reason)``.

    Split out from the call so the shape of Gemini's answer is testable without a network round
    trip. A response missing ``needs_reset`` is an error rather than a False: "no" and "I did not
    answer" would otherwise be indistinguishable, and the caller acts on the difference.
    """
    result = load_json(response_text)
    if not isinstance(result, dict) or "needs_reset" not in result:
        raise ValueError(f"scene_reset_check returned no 'needs_reset' field: {response_text}")
    return bool(result["needs_reset"]), str(result.get("reason") or "").strip()


async def check_scene_needs_reset_async(
    image: Image.Image,
    task_instruction: str,
    client: genai.Client | None = None,
    model_id: str = "gemini-robotics-er-2-preview",
    temperature: float | None = None,
) -> tuple[bool, str]:
    """Ask whether the workspace has to be put back before the task can be attempted again.

    Returns ``(needs_reset, reason)``. Drives auto mode (``tiptop.auto_mode``), which is why it is a
    yes/no on the WHOLE scene rather than per-object: the reset it triggers works out for itself,
    from geometry, which objects to move (``tiptop.scene_reset``).
    """
    client = client or gemini_client()
    prompt = load_prompt("scene_reset_check").format(task_instruction=task_instruction)
    response = await client.aio.models.generate_content(
        model=model_id,
        contents=[image, prompt],
        config=types.GenerateContentConfig(
            temperature=temperature, thinking_config=types.ThinkingConfig(thinking_budget=0)
        ),
    )
    return _parse_reset_check(response.text)
