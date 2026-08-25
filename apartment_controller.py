import argparse
import json
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
import wave
from pathlib import Path


DEFAULT_LLM_URL = "http://127.0.0.1:8080/v1/chat/completions"
DEFAULT_LED_PIN = 7
DEFAULT_WHISPER_BIN = "~/whisper.cpp/build/bin/whisper-cli"
DEFAULT_WHISPER_MODEL = "~/whisper.cpp/models/ggml-base.en.bin"
DEFAULT_WAKE_WORD = "command"
DEFAULT_WAKE_WINDOW_SECONDS = 2
DEFAULT_TTS_MODEL = "~/apartment-ai/voices/en_US-lessac-medium.onnx"

AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1
AUDIO_SAMPLE_WIDTH = 2
WHISPER_COMMAND_PROMPT = (
    "A spoken apartment command, such as turn the desk lamp on or off."
)

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


class AlsaAudioStream:
    """Continuously drain raw microphone audio from a single arecord process."""

    def __init__(self, microphone=None, frame_ms=100, max_buffer_seconds=10):
        self.microphone = microphone
        self.frame_bytes = (
            AUDIO_SAMPLE_RATE
            * AUDIO_CHANNELS
            * AUDIO_SAMPLE_WIDTH
            * frame_ms
            // 1000
        )
        frame_capacity = max(1, max_buffer_seconds * 1000 // frame_ms)
        self.frames = queue.Queue(maxsize=frame_capacity)
        self.process = None
        self.reader_thread = None
        self.stop_event = threading.Event()

    def start(self):
        if self.process is not None and self.process.poll() is None:
            return

        command = [
            "arecord",
            "--quiet",
            "--format",
            "S16_LE",
            "--rate",
            str(AUDIO_SAMPLE_RATE),
            "--channels",
            str(AUDIO_CHANNELS),
            "--file-type",
            "raw",
        ]
        if self.microphone:
            command.extend(["--device", self.microphone])
        command.append("-")

        try:
            self.process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except OSError as error:
            raise ControllerError(
                f"Could not start continuous microphone capture: {error}"
            ) from error

        self.stop_event.clear()
        self.reader_thread = threading.Thread(
            target=self._read_audio,
            name="alsa-audio-reader",
            daemon=True,
        )
        self.reader_thread.start()

    def _read_audio(self):
        while not self.stop_event.is_set():
            data = self.process.stdout.read(self.frame_bytes)
            if not data:
                break

            try:
                self.frames.put_nowait(data)
            except queue.Full:
                try:
                    self.frames.get_nowait()
                except queue.Empty:
                    pass
                self.frames.put_nowait(data)

    def read_seconds(self, seconds):
        self.start()
        bytes_needed = int(
            seconds * AUDIO_SAMPLE_RATE * AUDIO_CHANNELS * AUDIO_SAMPLE_WIDTH
        )
        audio = bytearray()
        deadline = time.monotonic() + seconds + 10

        while len(audio) < bytes_needed:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ControllerError("Timed out while reading microphone audio.")

            try:
                audio.extend(self.frames.get(timeout=min(1, remaining)))
            except queue.Empty:
                if self.process.poll() is not None:
                    raise ControllerError(self._process_error())

        return bytes(audio[:bytes_needed])

    def discard_buffer(self):
        while True:
            try:
                self.frames.get_nowait()
            except queue.Empty:
                return

    def _process_error(self):
        detail = self.process.stderr.read().decode(errors="replace").strip()
        if detail:
            lines = detail.splitlines()
            summary = lines[0]
            if len(lines) > 1 and lines[-1] != summary:
                summary = f"{summary} ({lines[-1]})"
            return f"Continuous microphone capture stopped: {summary}"
        return "Continuous microphone capture stopped unexpectedly."

    def close(self):
        self.stop_event.set()
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)

        if self.reader_thread is not None:
            self.reader_thread.join(timeout=2)

        self.process = None
        self.reader_thread = None
        self.discard_buffer()


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

    def listen(self, record_seconds=None):
        duration = record_seconds or self.record_seconds
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
                str(AUDIO_SAMPLE_RATE),
                "--channels",
                str(AUDIO_CHANNELS),
                "--duration",
                str(duration),
            ]

            if self.microphone:
                record_command.extend(["--device", self.microphone])

            record_command.append(str(recording_path))

            print(f"Recording for {duration} seconds... speak now.")
            self._run(
                record_command,
                "Could not record from the microphone",
                timeout=duration + 10,
            )
            print("Transcribing locally...")

            return self._transcribe(
                recording_path,
                transcript_path,
                prompt=WHISPER_COMMAND_PROMPT,
            )

    def transcribe_pcm(self, pcm_audio, prompt=None, announce=False):
        with tempfile.TemporaryDirectory(prefix="apartment-voice-") as temp_dir:
            temp_path = Path(temp_dir)
            recording_path = temp_path / "stream.wav"
            transcript_path = temp_path / "transcript"

            with wave.open(str(recording_path), "wb") as wav_file:
                wav_file.setnchannels(AUDIO_CHANNELS)
                wav_file.setsampwidth(AUDIO_SAMPLE_WIDTH)
                wav_file.setframerate(AUDIO_SAMPLE_RATE)
                wav_file.writeframes(pcm_audio)

            if announce:
                print("Transcribing locally...")

            return self._transcribe(recording_path, transcript_path, prompt=prompt)

    def _transcribe(self, recording_path, transcript_path, prompt=None):
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
            "--no-fallback",
            "--output-txt",
            "--output-file",
            str(transcript_path),
        ]
        if prompt:
            whisper_command.extend(["--prompt", prompt])

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


class PiperSpeechOutput:
    """Synthesize responses with Piper and play them through Linux audio."""

    def __init__(
        self,
        model_path,
        playback_device=None,
        voice_loader=None,
        player_runner=None,
    ):
        self.model_path = Path(model_path).expanduser()
        self.config_path = Path(f"{self.model_path}.json")
        self.playback_device = playback_device
        self.player_runner = player_runner or subprocess.run

        self.players = self._available_players()
        if not self.players:
            raise ControllerError(
                "Speech output needs paplay, pw-play, or aplay. Install the "
                "PulseAudio, PipeWire, or ALSA command-line utilities."
            )
        if not self.model_path.is_file():
            raise ControllerError(f"Piper voice model not found: {self.model_path}")
        if not self.config_path.is_file():
            raise ControllerError(
                f"Piper voice configuration not found: {self.config_path}"
            )

        if voice_loader is None:
            try:
                from piper import PiperVoice
            except ImportError as error:
                raise ControllerError(
                    "Speech output needs the 'piper-tts' Python package."
                ) from error
            voice_loader = PiperVoice.load

        try:
            # Keep Piper on the CPU so Qwen and Whisper retain the Jetson GPU.
            self.voice = voice_loader(str(self.model_path), use_cuda=False)
        except Exception as error:
            raise ControllerError(f"Could not load Piper voice: {error}") from error

    def _available_players(self):
        if self.playback_device:
            if shutil.which("aplay") is None:
                raise ControllerError(
                    "--speaker-device selects an ALSA device and needs 'aplay'."
                )
            return ["aplay"]

        # paplay and pw-play follow the desktop's selected system output, which
        # includes Bluetooth sinks. aplay remains a fallback for direct ALSA.
        return [
            player
            for player in ("paplay", "pw-play", "aplay")
            if shutil.which(player) is not None
        ]

    def _play_command(self, player, speech_path):
        if player == "aplay":
            command = ["aplay", "--quiet"]
            if self.playback_device:
                command.extend(["--device", self.playback_device])
            command.append(str(speech_path))
            return command

        return [player, str(speech_path)]

    def speak(self, text):
        text = text.strip()
        if not text:
            return

        with tempfile.TemporaryDirectory(prefix="apartment-speech-") as temp_dir:
            speech_path = Path(temp_dir) / "response.wav"
            try:
                with wave.open(str(speech_path), "wb") as wav_file:
                    self.voice.synthesize_wav(text, wav_file)
            except Exception as error:
                raise ControllerError(
                    f"Could not synthesize speech response: {error}"
                ) from error

            failures = []
            for player in self.players:
                play_command = self._play_command(player, speech_path)
                try:
                    result = self.player_runner(
                        play_command,
                        capture_output=True,
                        text=True,
                        timeout=30,
                        check=False,
                    )
                except (OSError, subprocess.TimeoutExpired) as error:
                    failures.append(f"{player}: {error}")
                    continue

                if result.returncode == 0:
                    return

                detail = (result.stderr or result.stdout).strip()
                if detail:
                    failures.append(f"{player}: {detail.splitlines()[-1]}")
                else:
                    failures.append(f"{player}: playback failed")

            raise ControllerError(
                "Could not play speech response: " + "; ".join(failures)
            )


class WakeWordInput:
    """Scan a continuous microphone stream, then capture a command on wake."""

    def __init__(
        self,
        voice_input,
        wake_word=DEFAULT_WAKE_WORD,
        window_seconds=DEFAULT_WAKE_WINDOW_SECONDS,
        speech_output=None,
        stream_factory=AlsaAudioStream,
    ):
        normalized_wake_word = self._normalize(wake_word)
        if not normalized_wake_word:
            raise ControllerError("The wake word must contain a letter or number.")

        self.voice_input = voice_input
        self.wake_word = wake_word
        self.normalized_wake_word = normalized_wake_word
        self.window_seconds = window_seconds
        self.speech_output = speech_output
        self.stream = stream_factory(microphone=voice_input.microphone)
        self.announced = False

    def listen(self):
        self.stream.start()
        if self.announced:
            # Audio captured while the previous command was interpreted is stale.
            self.stream.discard_buffer()
        if not self.announced:
            print(
                f'Listening continuously for wake word "{self.wake_word}". '
                "Press Ctrl+C to stop."
            )
            self.announced = True

        overlap_bytes = (
            min(1, self.window_seconds)
            * AUDIO_SAMPLE_RATE
            * AUDIO_CHANNELS
            * AUDIO_SAMPLE_WIDTH
        )
        previous_audio = b""

        while True:
            wake_audio = self.stream.read_seconds(self.window_seconds)
            wake_transcript = self.voice_input.transcribe_pcm(
                previous_audio + wake_audio
            )

            if not self._contains_wake_word(wake_transcript):
                previous_audio = wake_audio[-overlap_bytes:]
                continue

            # Drop anything captured while Whisper was detecting the wake word.
            # This makes the next recording begin after the ready message.
            self.stream.discard_buffer()
            print(f'\aWake word detected in: "{wake_transcript}"')
            if self.speech_output is not None:
                self.speech_output.speak("Ready.")
                # Prevent the spoken response from entering the command audio.
                self.stream.discard_buffer()

            print(
                f"Recording command for {self.voice_input.record_seconds} "
                "seconds... speak now."
            )
            command_audio = self.stream.read_seconds(
                self.voice_input.record_seconds
            )
            transcript = self.voice_input.transcribe_pcm(
                command_audio,
                prompt=WHISPER_COMMAND_PROMPT,
                announce=True,
            )

            if transcript:
                print(f'Heard: "{transcript}"')
            else:
                print("No speech detected. Resuming wake-word listening.\n")
            return transcript

    def _contains_wake_word(self, transcript):
        normalized_transcript = self._normalize(transcript)
        return (
            f" {self.normalized_wake_word} "
            in f" {normalized_transcript} "
        )

    @staticmethod
    def _normalize(text):
        return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))

    def close(self):
        self.stream.close()


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
            return None

        if device != "desk_lamp":
            raise ControllerError(f"No output is configured for device: {device}")

        if action == "on":
            self.gpio.output(self.pin, self.gpio.HIGH)
            print(">>> DESK LAMP ON\n")
            return "Desk lamp turned on."
        elif action == "off":
            self.gpio.output(self.pin, self.gpio.LOW)
            print(">>> DESK LAMP OFF\n")
            return "Desk lamp turned off."

        return None

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


def read_command(voice_input=None, wake_input=None):
    if wake_input is not None:
        return wake_input.listen()

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
    input_mode = parser.add_mutually_exclusive_group()
    input_mode.add_argument(
        "--voice",
        action="store_true",
        help="enable local push-to-talk input through whisper.cpp",
    )
    input_mode.add_argument(
        "--wake-listen",
        "--always-listen",
        dest="wake_listen",
        action="store_true",
        help="continuously listen for a wake word before recording a command",
    )
    parser.add_argument(
        "--record-seconds",
        type=int,
        default=5,
        help="seconds to record for each spoken command (default: 5)",
    )
    parser.add_argument(
        "--microphone",
        help="optional ALSA capture device, for example plughw:2,0",
    )
    parser.add_argument(
        "--wake-word",
        default=DEFAULT_WAKE_WORD,
        help=f"wake phrase for --wake-listen (default: {DEFAULT_WAKE_WORD})",
    )
    parser.add_argument(
        "--wake-window-seconds",
        type=int,
        default=DEFAULT_WAKE_WINDOW_SECONDS,
        help=(
            "audio window used to detect the wake phrase "
            f"(default: {DEFAULT_WAKE_WINDOW_SECONDS})"
        ),
    )
    parser.add_argument(
        "--speak",
        action="store_true",
        help="speak local ready and action responses with Piper",
    )
    parser.add_argument(
        "--tts-model",
        default=DEFAULT_TTS_MODEL,
        help=f"path to a Piper ONNX voice model (default: {DEFAULT_TTS_MODEL})",
    )
    parser.add_argument(
        "--speaker-device",
        help="force a direct ALSA playback device, for example plughw:3,0",
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
    if args.wake_window_seconds < 1:
        parser.error("--wake-window-seconds must be at least 1")
    if not WakeWordInput._normalize(args.wake_word):
        parser.error("--wake-word must contain a letter or number")

    return args


def main(argv=None):
    args = parse_args(argv)
    voice_input = None
    wake_input = None
    speech_output = None
    led = None

    try:
        if args.voice or args.wake_listen:
            voice_input = WhisperVoiceInput(
                whisper_bin=args.whisper_bin,
                whisper_model=args.whisper_model,
                record_seconds=args.record_seconds,
                microphone=args.microphone,
                language=args.language,
            )
            voice_input.check_ready()

        if args.speak:
            print("Loading local Piper voice...")
            speech_output = PiperSpeechOutput(
                model_path=args.tts_model,
                playback_device=args.speaker_device,
            )

        if args.wake_listen:
            wake_input = WakeWordInput(
                voice_input=voice_input,
                wake_word=args.wake_word,
                window_seconds=args.wake_window_seconds,
                speech_output=speech_output,
            )

        led = GpioLed(args.led_pin)

        print("Apartment controller started.")
        if args.wake_listen:
            print(
                f'Wake-word input enabled. Say "{args.wake_word}", wait for '
                "the ready message, then speak the command."
            )
        elif args.voice:
            print("Voice input enabled. Press Enter when you are ready to speak.")
        else:
            print("Text input enabled.")
        if speech_output is not None:
            print("Local speech responses enabled.")
        print("Type 'exit' or 'quit' to stop.\n")

        while True:
            try:
                user_text = read_command(
                    voice_input=voice_input if args.voice else None,
                    wake_input=wake_input,
                )
            except ControllerError as error:
                print(f"{error}\n")
                if wake_input is not None:
                    wake_input.close()
                    print("Restarting the wake listener in two seconds.\n")
                    time.sleep(2)
                continue

            if user_text.lower() in {"exit", "quit"}:
                print("Exiting apartment controller.")
                break

            if not user_text:
                continue

            try:
                device, action = interpret_command(user_text, args.llm_url)
                action_response = led.apply(device, action)
            except ControllerError as error:
                print(f"{error}\n")
                continue

            if speech_output is not None and action_response:
                try:
                    speech_output.speak(action_response)
                except ControllerError as error:
                    print(f"Speech response failed: {error}\n")

    except ControllerError as error:
        print(f"Startup error: {error}")
        return 1
    except (KeyboardInterrupt, EOFError):
        print("\nStopping apartment controller.")
    finally:
        if wake_input is not None:
            wake_input.close()
        if led is not None:
            led.cleanup()
            print("GPIO cleaned up.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
