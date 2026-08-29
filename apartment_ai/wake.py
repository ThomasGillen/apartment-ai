"""Continuous local wake-phrase detection."""

import math
import re
import sys
from array import array
from collections import deque
from statistics import median

from .audio import AlsaAudioStream
from .constants import (
    AUDIO_SAMPLE_WIDTH,
    DEFAULT_WAKE_SENSITIVITY,
    DEFAULT_WAKE_THRESHOLD,
    DEFAULT_WAKE_WINDOW_SECONDS,
    DEFAULT_WAKE_WORD,
    OPENWAKEWORD_DOWNLOAD_NAME,
    OPENWAKEWORD_FRAME_MS,
    OPENWAKEWORD_MODEL_NAME,
    WAKE_ACKNOWLEDGEMENT,
    WAKE_AUDIO_FRAME_MS,
    WAKE_CALIBRATION_SECONDS,
    WAKE_END_SILENCE_SECONDS,
    WAKE_MIN_SPEECH_SECONDS,
    WAKE_PRE_ROLL_SECONDS,
    WAKE_SENSITIVITY_SETTINGS,
    WHISPER_REQUEST_PROMPT,
)
from .errors import ControllerError


def download_openwakeword_model(downloader=None):
    """Download only the local models needed for the Hey Jarvis detector."""
    if downloader is None:
        try:
            from openwakeword.utils import download_models
        except ImportError as error:
            raise ControllerError(
                "openWakeWord is not installed. Activate the project virtual "
                "environment and run: python -m pip install -r requirements.txt"
            ) from error
        downloader = download_models

    print('Downloading the local openWakeWord "Hey Jarvis" model...')
    try:
        downloader(model_names=[OPENWAKEWORD_DOWNLOAD_NAME])
    except Exception as error:
        raise ControllerError(
            f"Could not download the openWakeWord model: {error}"
        ) from error
    print("Wake model download complete.")


class OpenWakeWordInput:
    """Detect Hey Jarvis locally, then use Whisper for the spoken request."""

    def __init__(
        self,
        voice_input,
        threshold=DEFAULT_WAKE_THRESHOLD,
        debug=False,
        speech_output=None,
        stream_factory=AlsaAudioStream,
        model_loader=None,
        sample_converter=None,
    ):
        if not 0 < threshold <= 1:
            raise ControllerError("The wake threshold must be above 0 and at most 1.")

        if model_loader is None:
            try:
                from openwakeword.model import Model
            except ImportError as error:
                raise ControllerError(
                    "Listen mode needs openWakeWord. Activate the project virtual "
                    "environment and run: python -m pip install -r requirements.txt"
                ) from error
            model_loader = Model

        if sample_converter is None:
            try:
                import numpy as np
            except ImportError as error:
                raise ControllerError(
                    "Listen mode needs NumPy. Reinstall the project requirements."
                ) from error

            def sample_converter(pcm):
                return np.frombuffer(pcm, dtype=np.int16)

        try:
            self.model = model_loader(
                wakeword_models=[OPENWAKEWORD_MODEL_NAME],
                inference_framework="tflite",
            )
        except Exception as error:
            raise ControllerError(
                "Could not load the openWakeWord Hey Jarvis model. Run the "
                "one-time setup command: python apartment_controller.py "
                f"--download-wake-model. Details: {error}"
            ) from error

        self.voice_input = voice_input
        self.threshold = threshold
        self.debug = debug
        self.speech_output = speech_output
        self.sample_converter = sample_converter
        self.stream = stream_factory(
            microphone=voice_input.microphone,
            frame_ms=OPENWAKEWORD_FRAME_MS,
        )
        self.announced = False

    def listen(self):
        self.stream.start()
        if self.announced:
            self.stream.discard_buffer()
            self.model.reset()
        if not self.announced:
            print(
                'Listening continuously for "Hey Jarvis" with openWakeWord. '
                "Press Ctrl+C to stop."
            )
            self.announced = True

        debug_frames = 0
        debug_peak = 0.0
        while True:
            frame = self.stream.read_frame()
            try:
                scores = self.model.predict(self.sample_converter(frame))
            except Exception as error:
                raise ControllerError(
                    f"openWakeWord could not process microphone audio: {error}"
                ) from error
            score = self._wake_score(scores)

            if self.debug:
                debug_frames += 1
                debug_peak = max(debug_peak, score)
                if debug_frames >= round(2000 / OPENWAKEWORD_FRAME_MS):
                    print(
                        "Wake model score: "
                        f"peak {debug_peak:.3f}, trigger {self.threshold:.3f}"
                    )
                    debug_frames = 0
                    debug_peak = 0.0
            if score < self.threshold:
                continue

            self.stream.discard_buffer()
            print(f"\aWake phrase detected (score {score:.3f}).")
            if self.speech_output is not None:
                self.speech_output.speak(WAKE_ACKNOWLEDGEMENT)
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
                print("No speech detected. Resuming listening.\n")
            return transcript

    @staticmethod
    def _wake_score(scores):
        if not isinstance(scores, dict) or not scores:
            return 0.0
        normalized_target = OPENWAKEWORD_MODEL_NAME.replace("_", " ")
        for name, score in scores.items():
            if str(name).casefold().replace("_", " ") == normalized_target:
                return float(score)
        if len(scores) == 1:
            return float(next(iter(scores.values())))
        return 0.0

    def close(self):
        self.stream.close()


class WhisperWakeWordInput:
    """Fallback detector that transcribes complete wake phrases with Whisper."""

    def __init__(
        self,
        voice_input,
        wake_word=DEFAULT_WAKE_WORD,
        window_seconds=DEFAULT_WAKE_WINDOW_SECONDS,
        sensitivity=DEFAULT_WAKE_SENSITIVITY,
        debug=False,
        speech_output=None,
        stream_factory=AlsaAudioStream,
    ):
        normalized_wake_word = self._normalize(wake_word)
        if not normalized_wake_word:
            raise ControllerError("The wake word must contain a letter or number.")
        if sensitivity not in WAKE_SENSITIVITY_SETTINGS:
            raise ControllerError(f"Unknown wake sensitivity: {sensitivity}")

        self.voice_input = voice_input
        self.wake_word = wake_word
        self.normalized_wake_word = normalized_wake_word
        self.window_seconds = window_seconds
        self.sensitivity = sensitivity
        self.debug = debug
        self.speech_output = speech_output
        self.stream = stream_factory(microphone=voice_input.microphone)
        stream_frame_ms = getattr(self.stream, "frame_ms", WAKE_AUDIO_FRAME_MS)
        self.frame_ms = (
            stream_frame_ms
            if isinstance(stream_frame_ms, (int, float)) and stream_frame_ms > 0
            else WAKE_AUDIO_FRAME_MS
        )
        self.noise_floor = None
        self.announced = False

    def listen(self):
        self.stream.start()
        if self.announced:
            self.stream.discard_buffer()
        if self.noise_floor is None:
            print("Calibrating microphone noise for one second... stay quiet.")
            self._calibrate_noise()
        if not self.announced:
            print(
                f'Listening continuously for wake word "{self.wake_word}". '
                "Press Ctrl+C to stop."
            )
            self.announced = True

        while True:
            wake_audio = self._capture_wake_utterance()
            wake_transcript, wake_matched = self._transcribe_wake_candidate(
                wake_audio
            )
            if not wake_matched:
                if self.debug:
                    displayed = wake_transcript or "<no transcription>"
                    print(f'Wake candidate rejected: "{displayed}"')
                self.stream.discard_buffer()
                continue

            self.stream.discard_buffer()
            print(f'\aWake word detected in: "{wake_transcript}"')
            if self.speech_output is not None:
                self.speech_output.speak(WAKE_ACKNOWLEDGEMENT)
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

    def _transcribe_wake_candidate(self, wake_audio):
        transcript = self.voice_input.transcribe_pcm(wake_audio)
        if self._contains_wake_word(transcript):
            return transcript, True
        if not self._is_plausible_wake_candidate(transcript):
            return transcript, False

        focused_prompt = (
            "A short activation phrase may be spoken. The activation phrase is "
            f'"{self.wake_word}."'
        )
        focused_transcript = self.voice_input.transcribe_pcm(
            wake_audio,
            prompt=focused_prompt,
        )
        if self.debug:
            displayed = focused_transcript or "<no transcription>"
            print(f'Focused wake retry heard: "{displayed}"')
        if self._contains_wake_word(focused_transcript):
            return focused_transcript, True
        return transcript, False

    def _calibrate_noise(self):
        self.stream.discard_buffer()
        calibration_frames = max(
            1,
            round(WAKE_CALIBRATION_SECONDS * 1000 / self.frame_ms),
        )
        levels = [
            self._pcm_rms(self.stream.read_frame())
            for _ in range(calibration_frames)
        ]
        self.noise_floor = max(1.0, float(median(levels)))
        if self.debug:
            print(f"Wake noise floor: {self.noise_floor:.0f} RMS")

    def _capture_wake_utterance(self):
        pre_roll_frames = max(
            1,
            round(WAKE_PRE_ROLL_SECONDS * 1000 / self.frame_ms),
        )
        speech_start_frames = max(
            1,
            round(WAKE_MIN_SPEECH_SECONDS * 1000 / self.frame_ms),
        )
        end_silence_frames = max(
            1,
            round(WAKE_END_SILENCE_SECONDS * 1000 / self.frame_ms),
        )
        max_utterance_frames = max(
            speech_start_frames + end_silence_frames,
            round(
                (self.window_seconds + WAKE_END_SILENCE_SECONDS)
                * 1000
                / self.frame_ms
            ),
        )
        pre_roll = deque(maxlen=pre_roll_frames + speech_start_frames)
        utterance = []
        consecutive_speech = 0
        consecutive_silence = 0
        debug_interval_frames = max(1, round(2 * 1000 / self.frame_ms))
        debug_frames = 0
        debug_peak = 0.0

        while True:
            frame = self.stream.read_frame()
            level = self._pcm_rms(frame)
            threshold = self._wake_energy_threshold()
            is_speech = level >= threshold
            if not utterance:
                if self.debug:
                    debug_frames += 1
                    debug_peak = max(debug_peak, level)
                    if debug_frames >= debug_interval_frames:
                        print(
                            "Wake audio level: "
                            f"peak {debug_peak:.0f}, trigger {threshold:.0f} RMS"
                        )
                        debug_frames = 0
                        debug_peak = 0.0
                pre_roll.append(frame)
                if is_speech:
                    consecutive_speech += 1
                else:
                    consecutive_speech = 0
                    self._update_noise_floor(level)
                if consecutive_speech >= speech_start_frames:
                    utterance.extend(pre_roll)
                continue

            utterance.append(frame)
            if is_speech:
                consecutive_silence = 0
            else:
                consecutive_silence += 1
            if consecutive_silence >= end_silence_frames:
                return b"".join(utterance)
            if len(utterance) >= max_utterance_frames:
                return b"".join(utterance)

    def _wake_energy_threshold(self):
        ratio, margin = WAKE_SENSITIVITY_SETTINGS[self.sensitivity]
        noise_floor = self.noise_floor or 1.0
        return max(40.0, noise_floor * ratio, noise_floor + margin)

    def _update_noise_floor(self, level):
        if self.noise_floor is None:
            self.noise_floor = max(1.0, level)
            return
        self.noise_floor = max(1.0, 0.98 * self.noise_floor + 0.02 * level)

    @staticmethod
    def _pcm_rms(pcm_audio):
        if not pcm_audio:
            return 0.0
        usable_bytes = len(pcm_audio) - (len(pcm_audio) % AUDIO_SAMPLE_WIDTH)
        samples = array("h")
        samples.frombytes(pcm_audio[:usable_bytes])
        if sys.byteorder != "little":
            samples.byteswap()
        if not samples:
            return 0.0
        return math.sqrt(sum(sample * sample for sample in samples) / len(samples))

    def _contains_wake_word(self, transcript):
        transcript_tokens = self._normalize(transcript).split()
        wake_tokens = self.normalized_wake_word.split()
        if not transcript_tokens:
            return False
        wake_length = len(wake_tokens)
        for start in range(len(transcript_tokens) - wake_length + 1):
            if transcript_tokens[start : start + wake_length] == wake_tokens:
                return True

        expected = "".join(wake_tokens)
        if len(expected) < 5:
            return False
        allowed_edits = 1 if len(expected) <= 8 else 2
        for candidate_length in range(max(1, wake_length - 1), wake_length + 2):
            for start in range(len(transcript_tokens) - candidate_length + 1):
                candidate = "".join(
                    transcript_tokens[start : start + candidate_length]
                )
                if abs(len(candidate) - len(expected)) > allowed_edits:
                    continue
                if self._edit_distance(candidate, expected) <= allowed_edits:
                    return True
        return False

    def _is_plausible_wake_candidate(self, transcript):
        transcript_tokens = self._normalize(transcript).split()
        wake_tokens = self.normalized_wake_word.split()
        if not transcript_tokens:
            return False
        for heard_token in transcript_tokens:
            for wake_token in wake_tokens:
                if heard_token == wake_token:
                    return True
                if (
                    abs(len(heard_token) - len(wake_token)) <= 1
                    and self._edit_distance(heard_token, wake_token) <= 1
                ):
                    return True
        return False

    @staticmethod
    def _edit_distance(left, right):
        if len(left) < len(right):
            left, right = right, left
        previous = list(range(len(right) + 1))
        for left_index, left_character in enumerate(left, start=1):
            current = [left_index]
            for right_index, right_character in enumerate(right, start=1):
                current.append(
                    min(
                        current[-1] + 1,
                        previous[right_index] + 1,
                        previous[right_index - 1]
                        + (left_character != right_character),
                    )
                )
            previous = current
        return previous[-1]

    @staticmethod
    def _normalize(text):
        return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))

    def close(self):
        self.stream.close()
