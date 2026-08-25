import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


DEFAULT_LLM_URL = "http://127.0.0.1:8080/v1/chat/completions"
DEFAULT_LED_PIN = 7
DEFAULT_WHISPER_BIN = "~/whisper.cpp/build/bin/whisper-cli"
DEFAULT_WHISPER_MODEL = "~/whisper.cpp/models/ggml-base.en.bin"

SYSTEM_PROMPT = """
You control an apartment.

Available devices:
- desk_lamp
- none

Available actions:
- on
- off
- none

Interpret the user's request.

If the user is not asking to control a device, return:
{"device":"none","action":"none"}

Return ONLY JSON.

Examples:
{"device":"desk_lamp","action":"on"}
{"device":"desk_lamp","action":"off"}
{"device":"none","action":"none"}
"""

ALLOWED_COMMANDS = {
    "desk_lamp": {"on", "off"},
    "none": {"none"},
}


class ControllerError(RuntimeError):
    """An expected controller error that can be shown to the user."""


class WhisperVoiceInput:
    """Record a short command with ALSA and transcribe it with whisper.cpp."""

    def __init__(
        self,
        whisper_bin,
        whisper_model,
        record_seconds=5,
        microphone=None,
        language="en",
        timeout=120,
    ):
        self.whisper_bin = Path(whisper_bin).expanduser()
        self.whisper_model = Path(whisper_model).expanduser()
        self.record_seconds = record_seconds
        self.microphone = microphone
        self.language = language
        self.timeout = timeout

    def check_ready(self):
        if shutil.which("arecord") is None:
            raise ControllerError(
                "Voice mode needs 'arecord'. Install the ALSA utilities package "
                "and try again."
            )

        if not self.whisper_bin.is_file():
            raise ControllerError(
                f"whisper.cpp executable not found: {self.whisper_bin}"
            )

        if not self.whisper_model.is_file():
            raise ControllerError(
                f"Whisper model not found: {self.whisper_model}"
            )

    def listen(self):
        with tempfile.TemporaryDirectory(prefix="apartment-voice-") as temp_dir:
            temp_path = Path(temp_dir)
            recording_path = temp_path / "command.wav"
            transcript_path = temp_path / "transcript"

            record_command = [
                "arecord",
                "--quiet",
                "--format",
                "S16_LE",
                "--rate",
                "16000",
                "--channels",
                "1",
                "--duration",
                str(self.record_seconds),
            ]

            if self.microphone:
                record_command.extend(["--device", self.microphone])

            record_command.append(str(recording_path))

            print(f"Recording for {self.record_seconds} seconds... speak now.")
            self._run(
                record_command,
                "Could not record from the microphone",
                timeout=self.record_seconds + 10,
            )
            print("Transcribing locally...")

            whisper_command = [
                str(self.whisper_bin),
                "--model",
                str(self.whisper_model),
                "--file",
                str(recording_path),
                "--language",
                self.language,
                "--no-timestamps",
                "--no-prints",
                "--suppress-nst",
                "--output-txt",
                "--output-file",
                str(transcript_path),
                "--prompt",
                "A spoken apartment command, such as turn the desk lamp on or off.",
            ]
            self._run(
                whisper_command,
                "whisper.cpp could not transcribe the recording",
                timeout=self.timeout,
            )

            output_file = transcript_path.with_suffix(".txt")
            if not output_file.is_file():
                raise ControllerError(
                    "whisper.cpp finished without creating a transcript."
                )

            return output_file.read_text(encoding="utf-8").strip()

    @staticmethod
    def _run(command, error_prefix, timeout):
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise ControllerError(f"{error_prefix}: timed out") from error
        except OSError as error:
            raise ControllerError(f"{error_prefix}: {error}") from error

        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            if detail:
                detail = detail.splitlines()[-1]
                raise ControllerError(f"{error_prefix}: {detail}")
            raise ControllerError(error_prefix)


class GpioLed:
    def __init__(self, pin):
        try:
            import Jetson.GPIO as gpio
        except ImportError as error:
            raise ControllerError(
                "Jetson.GPIO is not installed. Run this controller on the Jetson "
                "inside the project environment."
            ) from error

        self.gpio = gpio
        self.pin = pin
        self.gpio.setwarnings(False)
        self.gpio.setmode(self.gpio.BOARD)
        self.gpio.setup(self.pin, self.gpio.OUT, initial=self.gpio.LOW)

    def apply(self, device, action):
        if device == "none":
            print("No apartment action requested.\n")
            return

        if device != "desk_lamp":
            raise ControllerError(f"No output is configured for device: {device}")

        if action == "on":
            self.gpio.output(self.pin, self.gpio.HIGH)
            print(">>> DESK LAMP ON\n")
        elif action == "off":
            self.gpio.output(self.pin, self.gpio.LOW)
            print(">>> DESK LAMP OFF\n")

    def cleanup(self):
        self.gpio.output(self.pin, self.gpio.LOW)
        self.gpio.cleanup()


def interpret_command(user_text, llm_url=DEFAULT_LLM_URL, http_client=None):
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
        command = json.loads(response_text)
    except (json.JSONDecodeError, TypeError) as error:
        raise ControllerError("Invalid JSON from LLM.") from error

    return validate_command(command)


def validate_command(command):
    if not isinstance(command, dict):
        raise ControllerError("LLM command must be a JSON object.")

    device = command.get("device")
    action = command.get("action")

    if device not in ALLOWED_COMMANDS:
        raise ControllerError(f"Rejected unknown device: {device}")

    if action not in ALLOWED_COMMANDS[device]:
        raise ControllerError(f"Rejected invalid command: {device} -> {action}")

    print(f"Validated command: {device} -> {action}")
    return device, action


def read_command(voice_input=None):
    if voice_input is None:
        return input("Command: ").strip()

    typed_text = input(
        "Press Enter to speak, type a command, or type 'quit': "
    ).strip()
    if typed_text:
        return typed_text

    transcript = voice_input.listen()
    if transcript:
        print(f'Heard: "{transcript}"')
    else:
        print("No speech detected.\n")
    return transcript


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Control the apartment LED using typed or spoken commands."
    )
    parser.add_argument(
        "--voice",
        action="store_true",
        help="enable local push-to-talk input through whisper.cpp",
    )
    parser.add_argument(
        "--record-seconds",
        type=int,
        default=5,
        help="seconds to record after Enter is pressed (default: 5)",
    )
    parser.add_argument(
        "--microphone",
        help="optional ALSA capture device, for example plughw:2,0",
    )
    parser.add_argument(
        "--whisper-bin",
        default=DEFAULT_WHISPER_BIN,
        help=f"path to whisper-cli (default: {DEFAULT_WHISPER_BIN})",
    )
    parser.add_argument(
        "--whisper-model",
        default=DEFAULT_WHISPER_MODEL,
        help=f"path to a whisper.cpp model (default: {DEFAULT_WHISPER_MODEL})",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="spoken language code, or 'auto' to detect it (default: en)",
    )
    parser.add_argument(
        "--llm-url",
        default=DEFAULT_LLM_URL,
        help=f"local chat completions endpoint (default: {DEFAULT_LLM_URL})",
    )
    parser.add_argument(
        "--led-pin",
        type=int,
        default=DEFAULT_LED_PIN,
        help=f"physical Jetson BOARD pin (default: {DEFAULT_LED_PIN})",
    )
    args = parser.parse_args(argv)

    if args.record_seconds < 1:
        parser.error("--record-seconds must be at least 1")

    return args


def main(argv=None):
    args = parse_args(argv)
    voice_input = None
    led = None

    try:
        if args.voice:
            voice_input = WhisperVoiceInput(
                whisper_bin=args.whisper_bin,
                whisper_model=args.whisper_model,
                record_seconds=args.record_seconds,
                microphone=args.microphone,
                language=args.language,
            )
            voice_input.check_ready()

        led = GpioLed(args.led_pin)

        print("Apartment controller started.")
        if args.voice:
            print("Voice input enabled. Press Enter when you are ready to speak.")
        else:
            print("Text input enabled.")
        print("Type 'exit' or 'quit' to stop.\n")

        while True:
            try:
                user_text = read_command(voice_input)
            except ControllerError as error:
                print(f"{error}\n")
                continue

            if user_text.lower() in {"exit", "quit"}:
                print("Exiting apartment controller.")
                break

            if not user_text:
                continue

            try:
                device, action = interpret_command(user_text, args.llm_url)
                led.apply(device, action)
            except ControllerError as error:
                print(f"{error}\n")

    except ControllerError as error:
        print(f"Startup error: {error}")
        return 1
    except (KeyboardInterrupt, EOFError):
        print("\nStopping apartment controller.")
    finally:
        if led is not None:
            led.cleanup()
            print("GPIO cleaned up.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
