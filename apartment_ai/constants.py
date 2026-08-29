DEFAULT_LLM_URL = "http://127.0.0.1:8080/v1/chat/completions"
DEFAULT_LED_PIN = 7
DEFAULT_WHISPER_BIN = "~/whisper.cpp/build/bin/whisper-cli"
DEFAULT_WHISPER_MODEL = "~/whisper.cpp/models/ggml-base.en.bin"
DEFAULT_WAKE_WORD = "hey jarvis"
DEFAULT_WAKE_ENGINE = "openwakeword"
DEFAULT_WAKE_THRESHOLD = 0.5
DEFAULT_WAKE_WINDOW_SECONDS = 2
DEFAULT_WAKE_SENSITIVITY = "normal"
OPENWAKEWORD_MODEL_NAME = "hey jarvis"
OPENWAKEWORD_DOWNLOAD_NAME = "hey_jarvis"
OPENWAKEWORD_FRAME_MS = 80
WAKE_ACKNOWLEDGEMENT = "What's up?"
DEFAULT_TTS_MODEL = "~/apartment-ai/voices/en_US-lessac-medium.onnx"
DEFAULT_OUTLET_HOSTS = ("lamp-1", "lamp-2")
DEFAULT_OUTLET_CHANNEL = 0
DEFAULT_OUTLET_TIMEOUT = 5
SHELLY_PASSWORD_ENV = "SHELLY_PASSWORD"
OPEN_METEO_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ATTRIBUTION = "Weather data by Open-Meteo: https://open-meteo.com/"

AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1
AUDIO_SAMPLE_WIDTH = 2
WAKE_AUDIO_FRAME_MS = 100
WAKE_CALIBRATION_SECONDS = 1
WAKE_PRE_ROLL_SECONDS = 0.3
WAKE_END_SILENCE_SECONDS = 0.5
WAKE_MIN_SPEECH_SECONDS = 0.2
WAKE_SENSITIVITY_SETTINGS = {
    "low": (3.0, 150),
    "normal": (2.0, 75),
    "high": (1.5, 40),
}

CONTROL_DEVICE_ID = "desk_lamp"
CONTROL_DEVICE_NAME = "living room lamps"
WHISPER_REQUEST_PROMPT = (
    "A spoken apartment request or general question, such as turn the living "
    "room lamps on, ask for the time or date, ask about the weather, or ask "
    "a short conversational question."
)

ALLOWED_CONTROL_COMMANDS = {
    CONTROL_DEVICE_ID: {"on", "off"},
}
INFORMATION_INTENTS = {"time", "date", "weather"}
NON_CONTROL_INTENTS = INFORMATION_INTENTS | {"conversation"}
MAX_CONVERSATION_CHARACTERS = 320
