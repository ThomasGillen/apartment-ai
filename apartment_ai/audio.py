"""Microphone capture, Whisper transcription, and Linux audio discovery."""

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

from .constants import (
    AUDIO_CHANNELS,
    AUDIO_SAMPLE_RATE,
    AUDIO_SAMPLE_WIDTH,
    WHISPER_REQUEST_PROMPT,
)
from .errors import ControllerError


class AlsaAudioStream:
    """Continuously drain raw microphone audio from a single arecord process."""

    def __init__(self, microphone=None, frame_ms=100, max_buffer_seconds=10):
        self.microphone = microphone
        self.frame_ms = frame_ms
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

    def read_frame(self, timeout=2):
        """Read one short microphone frame from the continuously drained stream."""
        self.start()
        try:
            return self.frames.get(timeout=timeout)
        except queue.Empty as error:
            if self.process.poll() is not None:
                raise ControllerError(self._process_error()) from error
            raise ControllerError(
                "Timed out while reading microphone audio."
            ) from error

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
            audio.extend(self.read_frame(timeout=min(1, remaining)))

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
            raise ControllerError(f"Whisper model not found: {self.whisper_model}")

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

    result = _run_audio_query(["pactl", "info"], command_runner=command_runner)
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
        print(f"    state: {state}\n")
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
