"""Command-line parsing and apartment controller orchestration."""

import argparse
import time

from .audio import (
    WhisperVoiceInput,
    print_available_microphones,
    print_available_speakers,
    resolve_microphone,
)
from .constants import (
    DEFAULT_LED_PIN,
    DEFAULT_LLM_URL,
    DEFAULT_TTS_MODEL,
    DEFAULT_WAKE_ENGINE,
    DEFAULT_WAKE_SENSITIVITY,
    DEFAULT_WAKE_THRESHOLD,
    DEFAULT_WAKE_WINDOW_SECONDS,
    DEFAULT_WAKE_WORD,
    DEFAULT_WHISPER_BIN,
    DEFAULT_WHISPER_MODEL,
    OPENWAKEWORD_MODEL_NAME,
    OPEN_METEO_ATTRIBUTION,
    WAKE_ACKNOWLEDGEMENT,
    WAKE_SENSITIVITY_SETTINGS,
)
from .devices import create_device_output
from .errors import ControllerError
from .intent import execute_request, interpret_request
from .speech import PiperSpeechOutput
from .wake import (
    OpenWakeWordInput,
    WhisperWakeWordInput,
    download_openwakeword_model,
)
from .weather import OpenMeteoWeather


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
        help='continuously listen for local "Hey Jarvis" activation',
    )
    parser.add_argument(
        "--download-wake-model",
        action="store_true",
        help='download the local openWakeWord "Hey Jarvis" model, then exit',
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
        "--wake-engine",
        choices=("openwakeword", "whisper"),
        default=DEFAULT_WAKE_ENGINE,
        help=(
            "wake detector for --listen: dedicated openWakeWord model or "
            f"Whisper fallback (default: {DEFAULT_WAKE_ENGINE})"
        ),
    )
    parser.add_argument(
        "--wake-word",
        default=DEFAULT_WAKE_WORD,
        help=(
            "wake phrase used by --wake-engine whisper "
            f'(default: "{DEFAULT_WAKE_WORD}")'
        ),
    )
    parser.add_argument(
        "--wake-threshold",
        type=float,
        default=DEFAULT_WAKE_THRESHOLD,
        help=(
            "openWakeWord activation score from above 0 through 1; lower is "
            f"more sensitive (default: {DEFAULT_WAKE_THRESHOLD})"
        ),
    )
    parser.add_argument(
        "--wake-window-seconds",
        type=int,
        default=DEFAULT_WAKE_WINDOW_SECONDS,
        help=(
            "maximum wake-phrase length for the Whisper fallback "
            f"(default: {DEFAULT_WAKE_WINDOW_SECONDS})"
        ),
    )
    parser.add_argument(
        "--wake-sensitivity",
        choices=tuple(WAKE_SENSITIVITY_SETTINGS),
        default=DEFAULT_WAKE_SENSITIVITY,
        help=(
            "speech sensitivity for the Whisper fallback: low, normal, or high "
            f"(default: {DEFAULT_WAKE_SENSITIVITY})"
        ),
    )
    parser.add_argument(
        "--wake-debug",
        action="store_true",
        help="show wake scores, or fallback audio levels and rejected phrases",
    )
    parser.add_argument(
        "--speak",
        action="store_true",
        help=(
            "speak the local wake acknowledgement, action, information, and "
            "conversational responses with Piper"
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
        help=f"path to a Whisper model (default: {DEFAULT_WHISPER_MODEL})",
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
        help='city or postal code for weather, for example "Boston, MA"',
    )
    parser.add_argument(
        "--gpio",
        action="store_true",
        help="use the Jetson GPIO LED instead of the default Shelly outlets",
    )
    parser.add_argument(
        "--outlet",
        dest="outlet_hosts",
        action="append",
        metavar="HOST",
        help=(
            "Shelly hostname/IP, or NAME=HOST; repeat for multiple outlets "
            "(default: lamp-1 and lamp-2)"
        ),
    )
    parser.add_argument(
        "--led-pin",
        type=int,
        default=DEFAULT_LED_PIN,
        help=(
            "physical Jetson BOARD pin used with --gpio "
            f"(default: {DEFAULT_LED_PIN})"
        ),
    )
    args = parser.parse_args(argv)

    if args.record_seconds < 1:
        parser.error("--record-seconds must be at least 1")
    if args.wake_window_seconds < 1:
        parser.error("--wake-window-seconds must be at least 1")
    if not 0 < args.wake_threshold <= 1:
        parser.error("--wake-threshold must be above 0 and at most 1")
    if not WhisperWakeWordInput._normalize(args.wake_word):
        parser.error("--wake-word must contain a letter or number")
    if (
        args.wake_engine == "openwakeword"
        and WhisperWakeWordInput._normalize(args.wake_word)
        != OPENWAKEWORD_MODEL_NAME
    ):
        parser.error(
            'the openWakeWord model uses "hey jarvis"; select '
            "--wake-engine whisper to use a custom --wake-word"
        )
    if (
        (args.speaker or args.speaker_device)
        and not args.speak
        and not args.list_speakers
    ):
        parser.error("--speaker and --speaker-device require --speak")
    if args.gpio and args.outlet_hosts:
        parser.error("--outlet cannot be combined with --gpio")
    return args


def main(argv=None):
    args = parse_args(argv)
    voice_input = None
    wake_input = None
    speech_output = None
    device_output = None

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

        if args.download_wake_model:
            download_openwakeword_model()
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
            if args.wake_engine == "openwakeword":
                wake_input = OpenWakeWordInput(
                    voice_input=voice_input,
                    threshold=args.wake_threshold,
                    debug=args.wake_debug,
                    speech_output=speech_output,
                )
            else:
                wake_input = WhisperWakeWordInput(
                    voice_input=voice_input,
                    wake_word=args.wake_word,
                    window_seconds=args.wake_window_seconds,
                    sensitivity=args.wake_sensitivity,
                    debug=args.wake_debug,
                    speech_output=speech_output,
                )

        weather = OpenMeteoWeather(args.weather_location)
        device_output = create_device_output(args)

        print("Apartment controller started.")
        if args.listen:
            wake_phrase = (
                "Hey Jarvis"
                if args.wake_engine == "openwakeword"
                else args.wake_word
            )
            print(
                f'Listen mode enabled. Say "{wake_phrase}", wait for '
                f'"{WAKE_ACKNOWLEDGEMENT}", then speak the request.'
            )
            if args.wake_engine == "openwakeword":
                print(
                    "Wake detector: openWakeWord "
                    f"(threshold {args.wake_threshold:.2f})."
                )
            else:
                print("Wake detector: Whisper fallback.")
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
        if args.gpio:
            print(f"Device output: GPIO LED on BOARD pin {args.led_pin}.")
        else:
            outlet_names = ", ".join(
                outlet.name for outlet in device_output.outlets
            )
            print(f"Device output: Shelly outlets {outlet_names}.")
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
                    device_output,
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
        if device_output is not None:
            device_output.cleanup()
            if args.gpio:
                print("GPIO cleaned up.")
    return 0
