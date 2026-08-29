"""Local Piper speech synthesis and Linux playback."""

import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

from .audio import get_system_audio_sinks, resolve_system_sink
from .errors import ControllerError


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
            return ["paplay", f"--device={self.playback_sink}", str(speech_path)]
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
                failures.append(
                    f"{player}: {detail.splitlines()[-1]}"
                    if detail
                    else f"{player}: playback failed"
                )

            raise ControllerError(
                "Could not play speech response: " + "; ".join(failures)
            )
