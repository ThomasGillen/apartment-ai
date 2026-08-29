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
from datetime import datetime
from pathlib import Path


DEFAULT_LLM_URL = "http://127.0.0.1:8080/v1/chat/completions"
DEFAULT_LED_PIN = 7
DEFAULT_WHISPER_BIN = "~/whisper.cpp/build/bin/whisper-cli"
DEFAULT_WHISPER_MODEL = "~/whisper.cpp/models/ggml-base.en.bin"
DEFAULT_WAKE_WORD = "command"
DEFAULT_WAKE_WINDOW_SECONDS = 2
DEFAULT_TTS_MODEL = "~/apartment-ai/voices/en_US-lessac-medium.onnx"
OPEN_METEO_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ATTRIBUTION = "Weather data by Open-Meteo: https://open-meteo.com/"

AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1
AUDIO_SAMPLE_WIDTH = 2
WHISPER_REQUEST_PROMPT = (
    "A spoken apartment request or general question, such as turn the desk "
    "lamp on, ask for the time or date, ask about the weather, or ask a "
    "short conversational question."
)

SYSTEM_PROMPT = """
Classify one apartment request.

Available devices:
- desk_lamp

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
Turn on the lamp -> {"intent":"control","device":"desk_lamp","action":"on"}
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
by a separate validated path. Only desk_lamp on and off are configured. If a
request to control anything else reaches you, say briefly that you cannot
control it and do not imply that the action happened.
/no_think
"""

ALLOWED_CONTROL_COMMANDS = {
    "desk_lamp": {"on", "off"},
}
INFORMATION_INTENTS = {"time", "date", "weather"}
NON_CONTROL_INTENTS = INFORMATION_INTENTS | {"conversation"}
MAX_CONVERSATION_CHARACTERS = 320


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
    """Record a short request with ALSA and transcribe it with whisper.cpp."""

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
            recording_path = temp_path / "request.wav"
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
                prompt=WHISPER_REQUEST_PROMPT,
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


def _run_audio_query(command, command_runner=None):
    runner = command_runner or subprocess.run
    try:
        return runner(
            command,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def get_system_audio_sinks(command_runner=None):
    """Return PulseAudio/PipeWire output sinks with friendly descriptions."""
    if shutil.which("pactl") is None:
        return []

    result = _run_audio_query(
        ["pactl", "--format=json", "list", "sinks"],
        command_runner=command_runner,
    )
    if result is not None and result.returncode == 0:
        try:
            raw_sinks = json.loads(result.stdout)
        except (json.JSONDecodeError, TypeError):
            raw_sinks = None

        if isinstance(raw_sinks, list):
            sinks = []
            for raw_sink in raw_sinks:
                if not isinstance(raw_sink, dict) or not raw_sink.get("name"):
                    continue
                sinks.append(
                    {
                        "index": str(raw_sink.get("index", "")),
                        "name": str(raw_sink["name"]),
                        "description": str(
                            raw_sink.get("description") or raw_sink["name"]
                        ),
                        "state": str(raw_sink.get("state", "unknown")),
                    }
                )
            return sinks

    # Older pactl versions may not support JSON output.
    result = _run_audio_query(
        ["pactl", "list", "sinks"],
        command_runner=command_runner,
    )
    if result is None or result.returncode != 0:
        return []

    sinks = []
    current = None
    for line in result.stdout.splitlines():
        stripped = line.strip()
        match = re.fullmatch(r"Sink #(\d+)", stripped)
        if match:
            if current and current.get("name"):
                current["description"] = current.get("description") or current["name"]
                sinks.append(current)
            current = {
                "index": match.group(1),
                "name": "",
                "description": "",
                "state": "unknown",
            }
        elif current is not None and stripped.startswith("Name:"):
            current["name"] = stripped.partition(":")[2].strip()
        elif current is not None and stripped.startswith("Description:"):
            current["description"] = stripped.partition(":")[2].strip()
        elif current is not None and stripped.startswith("State:"):
            current["state"] = stripped.partition(":")[2].strip()

    if current and current.get("name"):
        current["description"] = current.get("description") or current["name"]
        sinks.append(current)
    return sinks


def get_default_system_sink(command_runner=None):
    if shutil.which("pactl") is None:
        return None

    result = _run_audio_query(
        ["pactl", "get-default-sink"],
        command_runner=command_runner,
    )
    if result is not None and result.returncode == 0:
        default_sink = result.stdout.strip()
        if default_sink:
            return default_sink

    result = _run_audio_query(
        ["pactl", "info"],
        command_runner=command_runner,
    )
    if result is not None and result.returncode == 0:
        for line in result.stdout.splitlines():
            if line.strip().startswith("Default Sink:"):
                return line.partition(":")[2].strip() or None
    return None


def resolve_system_sink(selection, sinks):
    """Resolve an index, sink ID, or friendly description to one sink."""
    requested = selection.strip()
    if requested.casefold() in {"default", "system", "auto"}:
        return None

    for sink in sinks:
        if requested == sink["name"]:
            return sink

    normalized = requested.casefold()
    exact_matches = [
        sink
        for sink in sinks
        if normalized
        in {
            sink.get("index", "").casefold(),
            sink.get("description", "").casefold(),
        }
    ]
    if len(exact_matches) == 1:
        return exact_matches[0]

    partial_matches = [
        sink
        for sink in sinks
        if normalized in sink.get("name", "").casefold()
        or normalized in sink.get("description", "").casefold()
    ]
    if len(partial_matches) == 1:
        return partial_matches[0]

    if exact_matches or partial_matches:
        matches = exact_matches or partial_matches
        descriptions = ", ".join(
            f'{sink["description"]} ({sink["name"]})' for sink in matches
        )
        raise ControllerError(
            f'Speaker selection "{selection}" is ambiguous: {descriptions}'
        )

    raise ControllerError(
        f'System audio speaker not found: "{selection}". '
        "Run with --list-speakers to see available choices."
    )


def print_available_speakers(command_runner=None):
    sinks = get_system_audio_sinks(command_runner=command_runner)
    if not sinks:
        raise ControllerError(
            "No Linux system audio speakers were found. Confirm that pactl can "
            "connect to the current user's audio session."
        )

    default_sink = get_default_system_sink(command_runner=command_runner)
    print("Available Linux system audio speakers:\n")
    for sink in sinks:
        marker = " [default]" if sink["name"] == default_sink else ""
        state = sink.get("state", "unknown").lower()
        print(f'  {sink["description"]}{marker}')
        print(f'    sink: {sink["name"]}')
        print(f'    state: {state}\n')

    print('Select one with --speaker "DESCRIPTION" or --speaker "SINK".')


def get_alsa_microphones(command_runner=None):
    """Return physical ALSA capture devices with stable named selectors."""
    if shutil.which("arecord") is None:
        return []

    result = _run_audio_query(
        ["arecord", "--list-devices"],
        command_runner=command_runner,
    )
    if result is None or result.returncode != 0:
        return []

    microphones = []
    device_pattern = re.compile(
        r"^card\s+(?P<card_index>\d+):\s+"
        r"(?P<card_id>\S+)\s+\[(?P<card_description>[^]]+)\],\s+"
        r"device\s+(?P<device_index>\d+):\s+"
        r"(?P<device_name>.*?)\s+\[(?P<device_description>[^]]*)\]\s*$"
    )
    for line in result.stdout.splitlines():
        match = device_pattern.match(line.strip())
        if not match:
            continue

        microphone = match.groupdict()
        microphone["selector"] = (
            f'plughw:CARD={microphone["card_id"]},'
            f'DEV={microphone["device_index"]}'
        )
        microphones.append(microphone)
    return microphones


def resolve_microphone(selection, microphones=None):
    """Resolve a friendly microphone name while preserving direct ALSA names."""
    if selection is None:
        return None

    requested = selection.strip()
    if not requested:
        raise ControllerError("Microphone selection cannot be empty.")
    if requested.casefold() in {"default", "system", "auto"}:
        return None

    # Existing ALSA selectors remain valid and skip device discovery.
    if ":" in requested:
        return requested

    microphones = microphones if microphones is not None else get_alsa_microphones()
    for microphone in microphones:
        if requested == microphone["selector"]:
            return microphone["selector"]

    normalized = requested.casefold()
    exact_matches = [
        microphone
        for microphone in microphones
        if normalized
        in {
            microphone.get("card_index", "").casefold(),
            microphone.get("card_id", "").casefold(),
            microphone.get("card_description", "").casefold(),
            microphone.get("device_description", "").casefold(),
        }
    ]
    if len(exact_matches) == 1:
        return exact_matches[0]["selector"]

    partial_matches = [
        microphone
        for microphone in microphones
        if any(
            normalized in microphone.get(field, "").casefold()
            for field in (
                "card_id",
                "card_description",
                "device_name",
                "device_description",
                "selector",
            )
        )
    ]
    if len(partial_matches) == 1:
        return partial_matches[0]["selector"]

    if exact_matches or partial_matches:
        matches = exact_matches or partial_matches
        descriptions = ", ".join(
            f'{microphone["card_description"]} '
            f'({microphone["selector"]})'
            for microphone in matches
        )
        raise ControllerError(
            f'Microphone selection "{selection}" is ambiguous: {descriptions}'
        )

    raise ControllerError(
        f'ALSA microphone not found: "{selection}". '
        "Run with --list-microphones to see available choices."
    )


def print_available_microphones(command_runner=None):
    microphones = get_alsa_microphones(command_runner=command_runner)
    if not microphones:
        raise ControllerError(
            "No ALSA microphones were found. Confirm that the input is attached "
            "and visible in arecord --list-devices."
        )

    print("Available ALSA microphones:\n")
    for microphone in microphones:
        print(f'  {microphone["card_description"]}')
        print(f'    input: {microphone["selector"]}')
        print(f'    device: {microphone["device_description"]}\n')

    print(
        'Select one with --microphone "DESCRIPTION" or '
        '--microphone "INPUT".'
    )


class PiperSpeechOutput:
    """Synthesize responses with Piper and play them through Linux audio."""

    def __init__(
        self,
        model_path,
        playback_device=None,
        playback_sink=None,
        voice_loader=None,
        player_runner=None,
        sink_lister=None,
    ):
        self.model_path = Path(model_path).expanduser()
        self.config_path = Path(f"{self.model_path}.json")
        self.playback_device = playback_device
        self.playback_sink = None
        self.speaker_label = "Linux system default"
        self.player_runner = player_runner or subprocess.run

        if playback_device and playback_sink:
            raise ControllerError(
                "Choose either --speaker or --speaker-device, not both."
            )
        if playback_device:
            self.speaker_label = f"direct ALSA device {playback_device}"
        elif playback_sink:
            sinks = (sink_lister or get_system_audio_sinks)()
            resolved_sink = resolve_system_sink(playback_sink, sinks)
            if resolved_sink is not None:
                self.playback_sink = resolved_sink["name"]
                self.speaker_label = resolved_sink["description"]

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

        if self.playback_sink:
            if shutil.which("paplay") is None:
                raise ControllerError(
                    "Manual system speaker selection needs 'paplay'. Install "
                    "the PulseAudio command-line utilities."
                )
            return ["paplay"]

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

        if player == "paplay" and self.playback_sink:
            return [
                "paplay",
                f"--device={self.playback_sink}",
                str(speech_path),
            ]

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
    """Scan a continuous microphone stream, then capture a request on wake."""

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
            # Audio captured while the previous request was handled is stale.
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
                # Prevent the spoken response from entering the request audio.
                self.stream.discard_buffer()

            print(
                f"Recording request for {self.voice_input.record_seconds} "
                "seconds... speak now."
            )
            request_audio = self.stream.read_seconds(
                self.voice_input.record_seconds
            )
            transcript = self.voice_input.transcribe_pcm(
                request_audio,
                prompt=WHISPER_REQUEST_PROMPT,
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


WEATHER_CODE_DESCRIPTIONS = {
    0: "clear skies",
    1: "mainly clear skies",
    2: "partly cloudy skies",
    3: "overcast skies",
    45: "fog",
    48: "freezing fog",
    51: "light drizzle",
    53: "drizzle",
    55: "heavy drizzle",
    56: "light freezing drizzle",
    57: "freezing drizzle",
    61: "light rain",
    63: "rain",
    65: "heavy rain",
    66: "light freezing rain",
    67: "freezing rain",
    71: "light snow",
    73: "snow",
    75: "heavy snow",
    77: "snow grains",
    80: "light rain showers",
    81: "rain showers",
    82: "heavy rain showers",
    85: "light snow showers",
    86: "heavy snow showers",
    95: "thunderstorms",
    96: "thunderstorms with light hail",
    99: "thunderstorms with heavy hail",
}


def current_time_response(now=None):
    current = now or datetime.now()
    return f"It is {current.strftime('%I:%M %p').lstrip('0')}."


def current_date_response(now=None):
    current = now or datetime.now()
    date_text = current.strftime("%A, %B %d, %Y").replace(" 0", " ")
    return f"Today is {date_text}."


class OpenMeteoWeather:
    """Fetch current conditions for one configured location on demand."""

    def __init__(self, location=None, http_client=None):
        self.location = location.strip() if location else None
        self.http_client = http_client
        self.resolved_location = None

    def current_response(self):
        location = self._resolve_location()
        weather = self._request_json(
            OPEN_METEO_FORECAST_URL,
            params={
                "latitude": location["latitude"],
                "longitude": location["longitude"],
                "current": (
                    "temperature_2m,apparent_temperature,weather_code"
                ),
                "daily": (
                    "temperature_2m_max,temperature_2m_min,"
                    "precipitation_probability_max"
                ),
                "temperature_unit": "fahrenheit",
                "timezone": "auto",
                "forecast_days": 1,
            },
            error_prefix="Could not get current weather",
        )

        try:
            current = weather["current"]
            daily = weather["daily"]
            temperature = round(float(current["temperature_2m"]))
            apparent = round(float(current["apparent_temperature"]))
            weather_code = int(current["weather_code"])
            high = round(float(daily["temperature_2m_max"][0]))
            low = round(float(daily["temperature_2m_min"][0]))
            precipitation = round(
                float(daily["precipitation_probability_max"][0])
            )
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise ControllerError(
                "Weather service returned an unexpected response."
            ) from error

        description = WEATHER_CODE_DESCRIPTIONS.get(
            weather_code,
            "unclassified conditions",
        )
        response = (
            f"In {location['label']}, it is {temperature} degrees Fahrenheit "
            f"with {description}. Today's high is {high} and the low is {low}, "
            f"with a {precipitation} percent chance of precipitation."
        )
        if abs(apparent - temperature) >= 3:
            response += f" It feels like {apparent} degrees."
        return response

    def _resolve_location(self):
        if self.resolved_location is not None:
            return self.resolved_location
        if not self.location:
            raise ControllerError(
                "Weather location is not configured. Start with "
                '--weather-location "City, State".'
            )

        result = self._request_json(
            OPEN_METEO_GEOCODING_URL,
            params={
                "name": self.location,
                "count": 1,
                "language": "en",
                "format": "json",
            },
            error_prefix="Could not look up weather location",
        )
        locations = result.get("results")
        if not isinstance(locations, list) or not locations:
            raise ControllerError(
                f'Weather location not found: "{self.location}". Try a city '
                "with its state or country."
            )

        selected = locations[0]
        try:
            latitude = float(selected["latitude"])
            longitude = float(selected["longitude"])
            name = str(selected["name"])
        except (KeyError, TypeError, ValueError) as error:
            raise ControllerError(
                "Weather location service returned an unexpected response."
            ) from error

        admin_area = str(selected.get("admin1") or "")
        label = name
        if admin_area and admin_area.casefold() != name.casefold():
            label = f"{name}, {admin_area}"
        self.resolved_location = {
            "latitude": latitude,
            "longitude": longitude,
            "label": label,
        }
        return self.resolved_location

    def _request_json(self, url, params, error_prefix):
        client = self.http_client
        if client is None:
            try:
                import requests
            except ImportError as error:
                raise ControllerError(
                    "The 'requests' package is not installed in this environment."
                ) from error
            client = requests

        try:
            response = client.get(url, params=params, timeout=10)
            response.raise_for_status()
            payload = response.json()
        except Exception as error:
            raise ControllerError(f"{error_prefix}: {error}") from error
        if not isinstance(payload, dict):
            raise ControllerError(f"{error_prefix}: unexpected response")
        return payload


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


def interpret_request(user_text, llm_url=DEFAULT_LLM_URL, http_client=None):
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
        raise ControllerError(
            "Unexpected conversational response from LLM server."
        )

    # Qwen's soft /no_think switch can still produce an empty thinking block,
    # so remove both complete and truncated blocks before speech output.
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
    response_text = re.sub(
        r"</?think>",
        " ",
        response_text,
        flags=re.IGNORECASE,
    )

    # Remove common Markdown presentation syntax because Piper should receive
    # natural spoken text rather than headings, bullets, or formatting marks.
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
    # Older classifier prompts used "none" for unmatched requests. Treat any
    # legacy-shaped output as conversation so it cannot dead-end.
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

    print(f"Validated request: control {device} -> {action}")
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


def read_request(voice_input=None, wake_input=None):
    if wake_input is not None:
        return wake_input.listen()

    if voice_input is None:
        return input("Request: ").strip()

    typed_text = input(
        "Press Enter to speak, type a request, or type 'quit': "
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
        description=(
            "Handle apartment controls, information requests, and short "
            "conversations using typed or spoken input."
        )
    )
    input_mode = parser.add_mutually_exclusive_group()
    input_mode.add_argument(
        "--voice",
        action="store_true",
        help="enable local push-to-talk input through whisper.cpp",
    )
    input_mode.add_argument(
        "--listen",
        dest="listen",
        action="store_true",
        help="continuously listen for a wake word before recording a request",
    )
    parser.add_argument(
        "--record-seconds",
        type=int,
        default=5,
        help="seconds to record for each spoken request (default: 5)",
    )
    parser.add_argument(
        "--microphone",
        help=(
            "microphone description or ALSA input from --list-microphones, "
            "for example plughw:CARD=Microphone,DEV=0"
        ),
    )
    parser.add_argument(
        "--list-microphones",
        "--list-inputs",
        dest="list_microphones",
        action="store_true",
        help="list ALSA microphone names and stable inputs, then exit",
    )
    parser.add_argument(
        "--list-audio",
        action="store_true",
        help="list both microphone inputs and speaker outputs, then exit",
    )
    parser.add_argument(
        "--wake-word",
        default=DEFAULT_WAKE_WORD,
        help=f"wake phrase for --listen (default: {DEFAULT_WAKE_WORD})",
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
        help=(
            "speak local ready, action, information, and conversational "
            "responses with Piper"
        ),
    )
    parser.add_argument(
        "--tts-model",
        default=DEFAULT_TTS_MODEL,
        help=f"path to a Piper ONNX voice model (default: {DEFAULT_TTS_MODEL})",
    )
    parser.add_argument(
        "--list-speakers",
        action="store_true",
        help="list wired and Bluetooth Linux audio speakers, then exit",
    )
    speaker_selection = parser.add_mutually_exclusive_group()
    speaker_selection.add_argument(
        "--speaker",
        help="Linux speaker description, sink ID, or index from --list-speakers",
    )
    speaker_selection.add_argument(
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
        "--weather-location",
        help=(
            'city or postal code for weather responses, for example "Boston, MA"'
        ),
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
    if (
        (args.speaker or args.speaker_device)
        and not args.speak
        and not args.list_speakers
    ):
        parser.error("--speaker and --speaker-device require --speak")

    return args


def main(argv=None):
    args = parse_args(argv)
    voice_input = None
    wake_input = None
    speech_output = None
    weather = None
    led = None

    try:
        if args.list_audio or args.list_speakers or args.list_microphones:
            show_speakers = args.list_audio or args.list_speakers
            show_microphones = args.list_audio or args.list_microphones
            if show_speakers:
                print_available_speakers()
            if show_speakers and show_microphones:
                print()
            if show_microphones:
                print_available_microphones()
            return 0

        if args.voice or args.listen:
            microphone = resolve_microphone(args.microphone)
            voice_input = WhisperVoiceInput(
                whisper_bin=args.whisper_bin,
                whisper_model=args.whisper_model,
                record_seconds=args.record_seconds,
                microphone=microphone,
                language=args.language,
            )
            voice_input.check_ready()

        if args.speak:
            print("Loading local Piper voice...")
            speech_output = PiperSpeechOutput(
                model_path=args.tts_model,
                playback_device=args.speaker_device,
                playback_sink=args.speaker,
            )

        if args.listen:
            wake_input = WakeWordInput(
                voice_input=voice_input,
                wake_word=args.wake_word,
                window_seconds=args.wake_window_seconds,
                speech_output=speech_output,
            )

        weather = OpenMeteoWeather(args.weather_location)
        led = GpioLed(args.led_pin)

        print("Apartment controller started.")
        if args.listen:
            print(
                f'Listen mode enabled. Say "{args.wake_word}", wait for '
                "the ready message, then speak the request."
            )
        elif args.voice:
            print("Voice input enabled. Press Enter when you are ready to speak.")
        else:
            print("Text input enabled.")
        if voice_input is not None and voice_input.microphone:
            print(f"Microphone input: {voice_input.microphone}.")
        if speech_output is not None:
            print(
                "Local speech responses enabled: "
                f"{speech_output.speaker_label}."
            )
        if args.weather_location:
            print(f"Weather location: {args.weather_location}.")
        print("Type 'exit' or 'quit' to stop.\n")

        while True:
            try:
                user_text = read_request(
                    voice_input=voice_input if args.voice else None,
                    wake_input=wake_input,
                )
            except ControllerError as error:
                print(f"{error}\n")
                if wake_input is not None:
                    wake_input.close()
                    print("Restarting the listener in two seconds.\n")
                    time.sleep(2)
                continue

            if user_text.lower() in {"exit", "quit"}:
                print("Exiting apartment controller.")
                break

            if not user_text:
                continue

            request = None
            try:
                request = interpret_request(user_text, args.llm_url)
                response_text = execute_request(
                    request,
                    led,
                    weather,
                    user_text=user_text,
                    llm_url=args.llm_url,
                )
            except ControllerError as error:
                print(f"{error}\n")
                if speech_output is not None:
                    failure_response = (
                        "Sorry, I couldn't get the weather. Check the configured "
                        "location and internet connection."
                        if request is not None and request[0] == "weather"
                        else "Sorry, I couldn't complete that request."
                    )
                    try:
                        speech_output.speak(failure_response)
                    except ControllerError as speech_error:
                        print(f"Speech response failed: {speech_error}\n")
                continue

            if response_text and request[0] != "control":
                print(f">>> {response_text}\n")
                if request[0] == "weather":
                    print(f"{OPEN_METEO_ATTRIBUTION}\n")
            if speech_output is not None and response_text:
                try:
                    speech_output.speak(response_text)
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
