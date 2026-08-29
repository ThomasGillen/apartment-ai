"""Compatibility entry point for the Apartment AI controller."""

from apartment_ai.audio import (
    AlsaAudioStream,
    WhisperVoiceInput,
    get_alsa_microphones,
    get_default_system_sink,
    get_system_audio_sinks,
    print_available_microphones,
    print_available_speakers,
    resolve_microphone,
    resolve_system_sink,
)
from apartment_ai.cli import main, parse_args, read_request
from apartment_ai.devices import (
    GpioLed,
    ShellyOutlet,
    ShellyOutletGroup,
    create_device_output,
    parse_outlet_target,
)
from apartment_ai.errors import ControllerError
from apartment_ai.intent import (
    clean_conversation_response,
    execute_request,
    generate_conversation_response,
    interpret_request,
    parse_direct_lamp_request,
    validate_request,
)
from apartment_ai.speech import PiperSpeechOutput
from apartment_ai.wake import (
    OpenWakeWordInput,
    WhisperWakeWordInput,
    download_openwakeword_model,
)
from apartment_ai.weather import (
    OpenMeteoWeather,
    current_date_response,
    current_time_response,
)

__all__ = [
    "AlsaAudioStream",
    "ControllerError",
    "GpioLed",
    "OpenMeteoWeather",
    "OpenWakeWordInput",
    "PiperSpeechOutput",
    "ShellyOutlet",
    "ShellyOutletGroup",
    "WhisperVoiceInput",
    "WhisperWakeWordInput",
    "clean_conversation_response",
    "create_device_output",
    "current_date_response",
    "current_time_response",
    "download_openwakeword_model",
    "execute_request",
    "generate_conversation_response",
    "get_alsa_microphones",
    "get_default_system_sink",
    "get_system_audio_sinks",
    "interpret_request",
    "main",
    "parse_args",
    "parse_direct_lamp_request",
    "parse_outlet_target",
    "print_available_microphones",
    "print_available_speakers",
    "read_request",
    "resolve_microphone",
    "resolve_system_sink",
    "validate_request",
]


if __name__ == "__main__":
    raise SystemExit(main())
