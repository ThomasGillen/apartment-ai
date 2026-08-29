"""Deterministic commands, LLM intent classification, and response routing."""

import json
import re

from .constants import (
    ALLOWED_CONTROL_COMMANDS,
    CONTROL_DEVICE_ID,
    CONTROL_DEVICE_NAME,
    DEFAULT_LLM_URL,
    MAX_CONVERSATION_CHARACTERS,
    NON_CONTROL_INTENTS,
)
from .errors import ControllerError
from .weather import current_date_response, current_time_response


SYSTEM_PROMPT = """
Classify one apartment request.

Available devices:
- desk_lamp (the living room lamps)

Available control actions:
- on
- off

Return exactly one of these JSON shapes:
{"intent":"control","device":"desk_lamp","action":"on"}
{"intent":"control","device":"desk_lamp","action":"off"}
{"intent":"time"}
{"intent":"date"}
{"intent":"weather"}
{"intent":"conversation"}

Use time for the current local time.
Use date for the current calendar date.
Use weather for current outdoor weather or today's forecast.
Use conversation for EVERY other input, including ordinary questions, unclear
phrases, questions about apartments or appliances, and requests to control a
device or action that is not available. A question about a device is not a
control request unless the user is explicitly asking to change its state.
Never substitute an available device for an unsupported one.

Return ONLY JSON with no extra keys or explanation.

Examples:
Turn on the living room lamps -> {"intent":"control","device":"desk_lamp","action":"on"}
What time is it? -> {"intent":"time"}
What day is it? -> {"intent":"date"}
What is the weather? -> {"intent":"weather"}
Tell me a joke -> {"intent":"conversation"}
Who wrote Dune? -> {"intent":"conversation"}
How long does a washing machine usually run? -> {"intent":"conversation"}
Turn on the oven -> {"intent":"conversation"}
"""

CONVERSATION_SYSTEM_PROMPT = """
You are a concise voice assistant running locally in an apartment.
Answer the user's message directly in no more than two short sentences.
Return plain text only, suitable for text-to-speech: no Markdown, lists, JSON,
or analysis. If the answer requires live information you do not have, briefly
say that. Never claim to have controlled a device; device actions are handled
by a separate validated path. Only the living room lamps can be turned on and
off. If a request to control anything else reaches you, say briefly that you
cannot control it and do not imply that the action happened.
/no_think
"""

DIRECT_LAMP_PATTERNS = (
    re.compile(
        r"^(?:please )?(?:the |my )?(?:living room )?"
        r"(?:lights?|lamps?|light[']s) (?P<action>on|off)(?: please)?$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:(?:can|could|would|will) you )?(?:please )?"
        r"(?:turn|switch) (?:the |my )?(?:living room )?"
        r"(?:lights?|lamps?) (?P<action>on|off)(?: please)?$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:(?:can|could|would|will) you )?(?:please )?"
        r"(?:turn|switch) (?P<action>on|off) "
        r"(?:the |my )?(?:living room )?(?:lights?|lamps?)(?: please)?$",
        re.IGNORECASE,
    ),
)


def parse_direct_lamp_request(user_text):
    """Recognize unambiguous common lamp commands without consulting the LLM."""
    normalized_text = re.sub(
        r"\s+",
        " ",
        re.sub(r"[,.!?;:]+", " ", user_text).strip(),
    )
    for pattern in DIRECT_LAMP_PATTERNS:
        match = pattern.fullmatch(normalized_text)
        if match:
            action = match.group("action").casefold()
            print(
                f"Recognized direct request: control "
                f"{CONTROL_DEVICE_NAME} -> {action}"
            )
            return "control", CONTROL_DEVICE_ID, action
    return None


def interpret_request(user_text, llm_url=DEFAULT_LLM_URL, http_client=None):
    direct_request = parse_direct_lamp_request(user_text)
    if direct_request is not None:
        return direct_request

    if http_client is None:
        try:
            import requests
        except ImportError as error:
            raise ControllerError(
                "The 'requests' package is not installed in this Python environment."
            ) from error
        http_client = requests

    payload = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        "temperature": 0,
    }
    try:
        response = http_client.post(llm_url, json=payload, timeout=60)
        response.raise_for_status()
    except Exception as error:
        raise ControllerError(
            f"Error communicating with LLM server: {error}"
        ) from error

    try:
        response_text = response.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise ControllerError("Unexpected response from LLM server.") from error

    print("Raw LLM response:")
    print(response_text)
    try:
        request = json.loads(response_text)
    except (json.JSONDecodeError, TypeError):
        print("Classifier response was not valid JSON; using conversation fallback.")
        return "conversation", None, None
    return validate_request(request)


def clean_conversation_response(response_text):
    if not isinstance(response_text, str):
        raise ControllerError("Unexpected conversational response from LLM server.")

    response_text = re.sub(
        r"<think>.*?</think>",
        " ",
        response_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    response_text = re.sub(
        r"<think>.*$",
        " ",
        response_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    response_text = re.sub(r"</?think>", " ", response_text, flags=re.IGNORECASE)
    response_text = re.sub(
        r"(?m)^\s*(?:#{1,6}\s+|[-*+]\s+|\d+[.)]\s+)",
        "",
        response_text,
    )
    response_text = response_text.replace("```", "").replace("`", "")
    response_text = response_text.replace("**", "").replace("__", "")
    response_text = " ".join(response_text.split())
    if not response_text:
        raise ControllerError("The LLM returned an empty conversational response.")

    sentences = re.split(r"(?<=[.!?])\s+", response_text)
    response_text = " ".join(sentences[:2])
    if len(response_text) > MAX_CONVERSATION_CHARACTERS:
        clipped = response_text[: MAX_CONVERSATION_CHARACTERS - 3]
        if " " in clipped:
            clipped = clipped.rsplit(" ", 1)[0]
        response_text = f"{clipped.rstrip(' ,;:-')}..."
    return response_text


def generate_conversation_response(
    user_text,
    llm_url=DEFAULT_LLM_URL,
    http_client=None,
):
    if http_client is None:
        try:
            import requests
        except ImportError as error:
            raise ControllerError(
                "The 'requests' package is not installed in this Python environment."
            ) from error
        http_client = requests

    payload = {
        "messages": [
            {"role": "system", "content": CONVERSATION_SYSTEM_PROMPT},
            {"role": "user", "content": f"{user_text}\n/no_think"},
        ],
        "temperature": 0.6,
        "max_tokens": 160,
    }
    try:
        response = http_client.post(llm_url, json=payload, timeout=60)
        response.raise_for_status()
    except Exception as error:
        raise ControllerError(
            f"Error communicating with LLM server: {error}"
        ) from error
    try:
        response_text = response.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise ControllerError(
            "Unexpected conversational response from LLM server."
        ) from error
    return clean_conversation_response(response_text)


def validate_request(request):
    if not isinstance(request, dict):
        print("Classifier response was not an object; using conversation fallback.")
        return "conversation", None, None

    intent = request.get("intent")
    if intent == "none":
        print("Normalized request: none -> conversation")
        return "conversation", None, None
    if intent in NON_CONTROL_INTENTS:
        print(f"Validated request: {intent}")
        return intent, None, None
    if intent != "control":
        print(f"Unknown intent {intent!r}; using conversation fallback.")
        return "conversation", None, None

    device = request.get("device")
    action = request.get("action")
    if device not in ALLOWED_CONTROL_COMMANDS:
        print(
            f"Unconfigured control device {device!r}; "
            "using conversation fallback."
        )
        return "conversation", None, None
    if action not in ALLOWED_CONTROL_COMMANDS[device]:
        print(
            f"Unconfigured control action {device!r} -> {action!r}; "
            "using conversation fallback."
        )
        return "conversation", None, None

    print(f"Validated request: control {CONTROL_DEVICE_NAME} -> {action}")
    return intent, device, action


def execute_request(
    request,
    device_output,
    weather,
    now=None,
    user_text=None,
    llm_url=DEFAULT_LLM_URL,
    conversation_client=None,
):
    intent, device, action = request
    if intent == "control":
        return device_output.apply(device, action)
    if intent == "time":
        return current_time_response(now=now)
    if intent == "date":
        return current_date_response(now=now)
    if intent == "weather":
        return weather.current_response()
    if intent == "conversation":
        if not user_text:
            raise ControllerError(
                "A conversational request requires the original text."
            )
        return generate_conversation_response(
            user_text,
            llm_url=llm_url,
            http_client=conversation_client,
        )
    raise ControllerError(f"No handler is configured for intent: {intent}")
