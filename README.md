# Apartment AI

A local-first apartment automation project for an NVIDIA Jetson Orin Nano. A
dedicated openWakeWord model listens locally for “Hey Jarvis.” After it
activates, the spoken request is transcribed by `whisper.cpp`, classified by
Qwen running through `llama.cpp`, validated by Python, and handled as a device
control, a time/date/weather response, or a short local conversational response.

No microphone audio, transcript, or apartment request is sent to a cloud API.
Time and date remain fully local. A weather request sends only the configured
place name and resolved coordinates to Open-Meteo; it therefore needs internet
access. Internet is otherwise needed only for initial package and model
downloads, Shelly onboarding, and device firmware updates.

## Architecture

```text
USB microphone
    ↓
ALSA arecord → 16 kHz mono PCM
    ↓
openWakeWord → local “Hey Jarvis” activation
    ↓
whisper.cpp → post-activation request transcript
    ↓
Qwen3-1.7B on llama.cpp → JSON request intent
    ↓
Python allow-list validation
    ├─ control → local Shelly RPC → lamp-1 and lamp-2
    │             or --gpio → Jetson GPIO test LED
    ├─ time/date → Jetson system clock
    ├─ weather → Open-Meteo using the configured location
    └─ every other request → Qwen3-1.7B → short sanitized text
                              ↓
        response → Piper → Linux system audio → speaker
```

The LLM never accesses the Shelly outlets, GPIO, the system clock, or the
weather service directly. Voice and typed requests both enter the same
validation and response path.

## One-time setup versus daily use

| Component | One-time setup | Normal use |
| --- | --- | --- |
| `llama.cpp` | Build `llama-server` with CUDA | Keep one server process running |
| Qwen3-1.7B | Downloaded automatically on first server launch | Loaded from the local cache |
| openWakeWord | Install the pinned package and download the “Hey Jarvis” model | Scores short microphone frames locally on CPU while idle |
| `whisper.cpp` | Build `whisper-cli` with CUDA | Python launches it after activation to transcribe the request |
| Whisper model | Download `ggml-base.en.bin` once | Loaded locally for transcription |
| Piper | Install `piper-tts` and download one voice | Speaks the wake acknowledgement, action, information, and conversational responses on CPU |
| Shelly outlets | Join Wi-Fi, name `lamp-1` and `lamp-2`, and reserve their IPs | Controlled directly over the local network |
| Open-Meteo | No account or API key for personal use | Queried only for a weather request |
| Python project | Create `.venv` and install requirements | Run push-to-talk or continuous listen mode |

After setup, normal operation still uses only two terminals: one for
`llama-server` and one for the apartment controller. Whisper does not need its
own server or terminal.

## Hardware

- NVIDIA Jetson Orin Nano with JetPack and the CUDA toolkit installed
- USB microphone or another ALSA-compatible capture device
- USB, HDMI, or another ALSA-compatible speaker
- Two Shelly Plug US Gen4 outlets named `lamp-1` and `lamp-2`
- Two plug-in lamps whose physical switches can remain on
- Optional breadboard LED, 220–1000 Ω resistor, and jumper wires for `--gpio`

Optional GPIO test wiring:

- Jetson physical pin 7 — GPIO output for the LED
- Jetson physical pin 6 — ground

Wire the resistor in series with the LED. The LED anode connects toward pin 7
and the cathode connects toward ground. Power down the Jetson before changing
the wiring. The Shelly outlets do not connect to Jetson GPIO.

## Full one-time setup

Run the following sections on the Jetson, not on a separate Windows computer.
The commands assume this project is at `~/apartment-ai`, `llama.cpp` will be at
`~/llama.cpp`, and `whisper.cpp` will be at `~/whisper.cpp`. The documented
Jetson environment is JetPack 7.2.1 with Jetson Linux 39.2.1 and CUDA 13.2.1.

### 1. Confirm JetPack and CUDA

JetPack should already be installed. Confirm that the Jetson is ARM64 and that
the CUDA compiler is available:

```bash
uname -m
nvcc --version
python3 --version
```

`uname -m` should report `aarch64`. If `nvcc` is missing, repair the JetPack/CUDA
installation before building either local AI project.

### 2. Install system packages

```bash
sudo apt update
sudo apt install -y \
  alsa-utils \
  build-essential \
  cmake \
  curl \
  git \
  libssl-dev \
  pipewire-bin \
  pulseaudio-utils \
  python3-pip \
  python3-venv
```

These provide the C/C++ build tools, Python environment support, ALSA microphone
tools, and HTTPS support for downloading the Qwen model.

### 3. Create the Python environment

Place or clone this repository at `~/apartment-ai`, then run:

```bash
cd ~/apartment-ai
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python apartment_controller.py --download-wake-model
```

The requirements install `requests`, NVIDIA's `Jetson.GPIO` package, Piper
text-to-speech, and a pinned openWakeWord revision with current ARM64 LiteRT
support. The final command downloads only the feature files and “Hey Jarvis”
wake model needed by this project. Both steps happen once, not each time the
controller starts. The included pretrained wake models are licensed for
non-commercial use under CC BY-NC-SA 4.0.

### 4. Configure the Shelly outlets

Use the Shelly Smart Control app to join both outlets to the same local network
as the Jetson. Give them distinct names even though they operate as one group:

```text
lamp-1
lamp-2
```

Install current firmware and reserve both addresses in the router. Test each
reserved address from the Jetson, replacing the examples with the actual IPs:

```bash
curl http://192.168.1.50/rpc/Shelly.GetDeviceInfo
curl "http://192.168.1.50/rpc/Switch.GetStatus?id=0"

curl http://192.168.1.51/rpc/Shelly.GetDeviceInfo
curl "http://192.168.1.51/rpc/Switch.GetStatus?id=0"
```

The controller initially tries the default network names `lamp-1` and
`lamp-2`. An app display name is not guaranteed to become a resolvable hostname,
so IP overrides are the most reliable daily command:

```bash
python apartment_controller.py \
  --outlet lamp-1=192.168.1.50 \
  --outlet lamp-2=192.168.1.51
```

Repeat `--outlet` for every member of the logical living-room-lamp group. If
Shelly authentication is enabled, store the shared device password in the
process environment instead of the repository:

```bash
read -rsp "Shelly password: " SHELLY_PASSWORD
export SHELLY_PASSWORD
echo
```

The username is always `admin`. This prompt keeps the password out of terminal
output and shell history; it remains available only in that shell environment.

### 5. Optional: configure GPIO permissions

With the project environment still active:

```bash
sudo groupadd -f -r gpio
sudo usermod -a -G gpio "$USER"

GPIO_RULE_PATH="$(python -c 'from pathlib import Path; import Jetson.GPIO as GPIO; print(Path(GPIO.__file__).with_name("99-gpio.rules"))')"
sudo cp "$GPIO_RULE_PATH" /etc/udev/rules.d/99-gpio.rules
sudo udevadm control --reload-rules
sudo udevadm trigger
```

This permits GPIO access without running the entire controller as root.

### 6. Optional: configure physical pin 7 with Jetson-IO

Launch NVIDIA's header configuration tool:

```bash
sudo /opt/nvidia/jetson-io/jetson-io.py
```

In the interface:

1. Select the 40-pin header.
2. Choose the option to configure header pins manually.
3. Ensure physical pin 7 is configured as GPIO.
4. Save the pin changes and reboot.

The reboot also activates the new `gpio` group membership. After reconnecting,
confirm it with:

```bash
groups
```

The output should include `gpio`.

### 7. Optional: test the LED and GPIO

```bash
cd ~/apartment-ai
source .venv/bin/activate
python gpio_test.py
```

The LED should alternate on and off every two seconds. Press Ctrl+C to stop; the
cleanup code leaves the LED off. This hardware is used only when the controller
starts with `--gpio`.

### 8. Build llama.cpp for Qwen

`llama.cpp` is the local LLM runtime. It builds the HTTP server that the Python
controller calls.

```bash
cd ~
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp

cmake -S . -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_OPENSSL=ON

cmake --build build \
  --config Release \
  --target llama-server \
  --parallel 1
```

If `~/llama.cpp` already exists, skip the `git clone` line, enter that directory,
and rerun the `cmake` commands. Completed build work will be reused.

The expected executable is:

```text
~/llama.cpp/build/bin/llama-server
```

Architecture `87` targets the Jetson Orin GPU. The single build job is
intentional: parallel CUDA compilation can exhaust the Jetson's shared memory.

The Qwen model does not require a separate manual download command. The first
server launch with `-hf ggml-org/Qwen3-1.7B-GGUF:Q4_K_M` downloads and caches
the Q4_K_M model. Later launches use the local cached copy. The 1.7B model is
the default because it leaves more shared-memory and power headroom for
request transcription, wake detection, and the desktop audio services.

### 9. Build whisper.cpp for speech-to-text

`whisper.cpp` is separate from `llama.cpp`. It builds the command-line program
that converts recorded audio into text.

```bash
cd ~
git clone https://github.com/ggml-org/whisper.cpp.git
cd whisper.cpp

cmake -S . -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build \
  --config Release \
  --target whisper-cli \
  --parallel 1

sh ./models/download-ggml-model.sh base.en
```

If `~/whisper.cpp` already exists, skip the `git clone` line, enter that
directory, and rerun the configure, single-job build, and model-download
commands.

The controller's default voice paths are:

```text
~/whisper.cpp/build/bin/whisper-cli
~/whisper.cpp/models/ggml-base.en.bin
```

Test the build with the sample included in `whisper.cpp`:

```bash
./build/bin/whisper-cli \
  --model models/ggml-base.en.bin \
  --file samples/jfk.wav \
  --no-timestamps
```

A successful run prints an English transcription.

### 10. Check the microphone

List available capture devices:

```bash
arecord -l
python apartment_controller.py --list-microphones
```

The controller listing shows both a friendly description and a stable selector
based on the ALSA card name. Unlike a numeric card position, the named selector
normally remains unchanged after a reboot.

First try the default microphone:

```bash
arecord -f S16_LE -r 16000 -c 1 -d 5 mic-test.wav
paplay mic-test.wav
```

If the controller lists `onn USB Microphone`, test it by friendly name or with
the stable input it displays:

```bash
arecord \
  -D 'plughw:CARD=Microphone,DEV=0' \
  -f S16_LE \
  -r 16000 \
  -c 1 \
  -d 5 \
  mic-test.wav

paplay mic-test.wav
```

Replace the example selector with the `input:` value printed by
`--list-microphones`.

### 11. Download a Piper voice and test the speaker

The default speech model is the English `en_US-lessac-medium` voice:

```bash
cd ~/apartment-ai
source .venv/bin/activate
mkdir -p voices
python -m piper.download_voices \
  --data-dir voices \
  en_US-lessac-medium
```

This creates the two files expected by the controller:

```text
~/apartment-ai/voices/en_US-lessac-medium.onnx
~/apartment-ai/voices/en_US-lessac-medium.onnx.json
```

Generate and play a test response:

```bash
python -m piper \
  --data-dir voices \
  --model en_US-lessac-medium \
  -f tts-test.wav \
  -- "What's up?"

paplay tts-test.wav
```

`paplay` follows Linux's selected system output, including a Bluetooth speaker.
If it is unavailable, try `pw-play tts-test.wav`. Direct ALSA playback with
`aplay` does not normally route to a Bluetooth sink managed by the desktop.

List the friendly names and exact sink IDs for every wired and Bluetooth
speaker currently visible to Linux:

```bash
python apartment_controller.py --list-speakers
```

This listing exits without loading Piper, Whisper, the LLM, or a device output,
so it can also be run remotely over SSH.

To list both microphone inputs and speaker outputs in one remote command:

```bash
python apartment_controller.py --list-audio
```

### 12. Run the software tests

```bash
cd ~/apartment-ai
source .venv/bin/activate
python -m unittest -v
```

These tests simulate the voice, LLM classification, conversational response,
clock, and weather boundaries. They do not record from the real microphone,
contact the live weather service, or change GPIO or a real outlet.

## Daily operation

Only two terminals are needed after the one-time setup.

### Terminal 1: start the local LLM server

```bash
cd ~/llama.cpp

./build/bin/llama-server \
  -hf ggml-org/Qwen3-1.7B-GGUF:Q4_K_M \
  -c 1024 \
  -np 1 \
  -b 64 \
  -ub 64 \
  --host 127.0.0.1 \
  --port 8080
```

Leave this terminal running. The first launch needs internet access while Qwen
is downloaded; subsequent launches use the cached model. Do not add `-ngl 99`:
current llama.cpp automatically chooses how many layers fit on the GPU, while a
manual value disables that fitting. The controller needs only one server slot,
and its short JSON requests do not need a large context or batch buffer.

### Terminal 2 option A: push-to-talk mode

For the default microphone:

```bash
cd ~/apartment-ai
source .venv/bin/activate
python apartment_controller.py --voice
```

The prompt shows:

```text
Press Enter to speak, type a request, or type 'quit':
```

Press Enter, then say a short request such as “turn on my living room lamps.”
The controller records for five seconds, transcribes the audio locally,
displays what it heard, sends the transcript to Qwen, validates Qwen's JSON
response, and sends the action to both Shelly outlets only if the control
request is allowed.
Each outlet is queried afterward, and the controller reports success only when
both confirm the requested state.

If the router does not resolve the default names, use the reserved IP addresses:

```bash
python apartment_controller.py \
  --voice \
  --outlet lamp-1=192.168.1.50 \
  --outlet lamp-2=192.168.1.51
```

For a non-default microphone:

```bash
python apartment_controller.py \
  --voice \
  --microphone "onn USB Microphone"
```

The friendly name is resolved to a stable `plughw:CARD=...,DEV=...` selector at
startup. Existing numeric selectors such as `plughw:2,0` remain supported.

To use a seven-second recording window:

```bash
python apartment_controller.py --voice --record-seconds 7
```

Typing text at the voice prompt bypasses recording for that request. Type
`exit` or `quit`, or press Ctrl+C, to stop. Network outlets retain their current
state when the controller exits.

To use the original GPIO LED instead of the outlets:

```bash
python apartment_controller.py --voice --gpio
```

GPIO cleanup forces the test LED off when the controller exits.

### Terminal 2 option B: continuous listen mode

The continuous mode uses the same build and model; it needs no additional
server or terminal:

```bash
cd ~/apartment-ai
source .venv/bin/activate
python apartment_controller.py --listen --speak
```

The default wake phrase is “Hey Jarvis.” Use it in two stages:

1. Say “Hey Jarvis.”
2. Wait for the spoken “What’s up?” response and the `Recording request...`
   message.
3. Say “turn on my living room lamps.”

The microphone remains open through one `arecord` process. While idle,
openWakeWord evaluates 80 ms audio frames with its dedicated “Hey Jarvis” model
on the CPU. It detects the phrase directly from audio, so a Whisper spelling
such as “Hey Department” can no longer cause the activation to be missed.
Whisper is not launched for idle audio.

When the wake phrase score reaches the default `0.5` threshold, old buffered
audio is discarded and a fresh five-second request is recorded. The microphone
audio containing the spoken “What’s up?” response is also discarded before
recording begins. Whisper transcribes only this post-activation request, and
only that transcript enters the Qwen interpretation and validation path.

After both outlets confirm an allowed action, Piper says either “Living room
lamps turned on” or “Living room lamps turned off.” Device confirmations remain
fixed Python responses, not conversational LLM output.

If the phrase is still missed, lower the openWakeWord threshold slightly. Lower
values make activation easier but can increase accidental activations:

```bash
python apartment_controller.py \
  --listen \
  --speak \
  --wake-threshold 0.4
```

For a non-default microphone:

```bash
python apartment_controller.py \
  --listen \
  --speak \
  --microphone "onn USB Microphone"
```

Add `--wake-debug` to print the highest model score seen every two seconds and
the active threshold. This diagnostic mode does not save audio.

The previous Whisper-based wake detector remains available as a fallback and
supports a custom phrase:

```bash
python apartment_controller.py \
  --listen \
  --speak \
  --wake-engine whisper \
  --wake-word "hey apartment" \
  --wake-sensitivity high
```

With that fallback only, `--wake-sensitivity`, `--wake-window-seconds`, and
custom `--wake-word` values have their previous meanings. The openWakeWord
engine is fixed to its pretrained “Hey Jarvis” phrase.

The controller automatically tries `paplay`, then `pw-play`, so the selected
Linux system output is used for Bluetooth. It falls back to direct ALSA only if
neither system player is available.

To select a particular wired or Bluetooth speaker without changing the Linux
desktop default, first list the available choices:

```bash
python apartment_controller.py --list-speakers
```

Then use either its friendly description:

```bash
python apartment_controller.py \
  --listen \
  --speak \
  --microphone "onn USB Microphone" \
  --speaker "AB13X USB Audio"
```

Or use the exact sink ID shown by the listing, which is useful when two
speakers have similar names:

```bash
python apartment_controller.py \
  --listen \
  --speak \
  --microphone "onn USB Microphone" \
  --speaker "bluez_output.XX_XX_XX_XX_XX_XX.1"
```

`--speaker` keeps playback inside Linux system audio, so it supports Bluetooth,
resampling, and device sharing. It works from an SSH terminal as long as that
user's Linux audio session is running.

For a headless system with no PulseAudio/PipeWire session, the advanced direct
ALSA option remains available:

```bash
python apartment_controller.py \
  --listen \
  --speak \
  --microphone "onn USB Microphone" \
  --speaker-device plughw:3,0
```

Direct ALSA bypasses Linux system audio and can report `Device or resource busy`
when the desktop audio service already owns the hardware. Prefer `--speaker`
whenever the desired output appears in `--list-speakers`.

Listen mode still performs more work than push-to-talk, but the small dedicated
wake model avoids running Whisper during silence. Audio stays local; temporary
request recordings and transcripts are deleted after each request. Press
Ctrl+C to stop.

### Time, date, and weather responses

Information requests work in text, push-to-talk, and continuous listen modes.
For example, after the wake word and “What’s up?” response, say:

```text
What time is it?
What is today's date?
What is the weather?
```

Time and date use the Jetson's local system clock and require no additional
option. Weather needs a configured city or postal code:

```bash
python apartment_controller.py \
  --listen \
  --speak \
  --weather-location "Boston, MA"
```

Include a state or country when a city name is ambiguous. The first weather
request resolves that name to coordinates; later requests during the same run
reuse the result. Each weather response includes current conditions, current
temperature, today's high and low, and maximum precipitation probability in
Fahrenheit and percent. No API key is required for personal use.

Weather results are displayed with an Open-Meteo attribution line. Open-Meteo
API data is provided under the CC BY 4.0 license; the attribution is displayed
in the terminal but is not read aloud by Piper.

For these information requests, the LLM selects only the request type. Python
reads the clock and builds the weather sentence, so the LLM cannot invent the
reported time or temperature.

### Conversational fallback

If a request is an ordinary question rather than a configured command or live
information request, the controller asks the same local Qwen server for a short
answer. For example:

```text
Who wrote Dune?
Tell me a short joke.
Why is the sky blue?
How long does a washing machine usually run?
```

The conversational answer is limited to two short sentences, stripped of
thinking blocks and common Markdown formatting, displayed in the terminal, and
spoken when `--speak` is enabled. This adds a second sequential LLM request, so
it can respond a little more slowly than a lamp, time, date, or weather request.

Conversation is the catch-all fallback, including for apartment or appliance
questions and requests the classifier cannot match to a dedicated handler. It
is currently one request at a time; previous exchanges are not kept as chat
history. Qwen has no live internet access. Current time, date, and weather still
use their dedicated Python handlers. If an unconfigured control request reaches
the conversational path, Qwen is instructed to say it cannot perform that
action and never to imply it happened. Treat general factual answers as
unverified local model output.

### Text-only mode

The original text input remains available:

```bash
cd ~/apartment-ai
source .venv/bin/activate
python apartment_controller.py
```

## Request safety

The only allowed request intents are:

```text
control → desk_lamp (living room lamps) → on
control → desk_lamp (living room lamps) → off
time
date
weather
conversation
```

Only the exact allow-listed living-room-lamp control shapes can enter the
device output path. By default, every valid action is sent to `lamp-1` and
`lamp-2`, and each reported output state must match before success is announced.
A failure still allows the other outlet to be attempted and reports that the
group may be in a partial state. Invalid classifier JSON, non-object results,
unknown intents, unknown devices, invalid actions, and the legacy `none` intent
all route to conversation instead. Microphone, transcription, LLM connection,
and malformed server-envelope errors stop the request without changing an
output. Time, date, weather, and conversation cannot enter the device output
path. In `--gpio` mode, the LED is forced off during normal controller shutdown;
network outlets retain their state.

## Useful options

```bash
python apartment_controller.py --help
```

- `--voice` enables push-to-talk input.
- `--listen` continuously listens locally for “Hey Jarvis.”
- `--download-wake-model` downloads the “Hey Jarvis” model once, then exits.
- `--wake-threshold N` tunes openWakeWord from above 0 through 1; lower values
  activate more easily.
- `--wake-debug` prints live openWakeWord peak scores and the trigger threshold.
- `--wake-engine whisper` selects the previous transcription-based fallback.
- `--wake-word PHRASE`, `--wake-window-seconds N`, and
  `--wake-sensitivity {low,normal,high}` configure only the Whisper fallback.
- `--speak` enables the local Piper wake acknowledgement, action, information,
  and conversational responses.
- `--tts-model PATH` selects a different Piper ONNX voice model.
- `--list-speakers` lists wired and Bluetooth system-audio outputs, then exits.
- `--speaker NAME` selects a system speaker by description, sink ID, or index.
- `--speaker-device plughw:CARD,DEVICE` forces a direct ALSA playback device.
- `--list-microphones` (or `--list-inputs`) lists friendly ALSA inputs, then exits.
- `--list-audio` lists microphone inputs and speaker outputs together, then exits.
- `--microphone NAME` selects an input by description or stable ALSA selector.
- `--record-seconds N` changes the recording window.
- `--language auto` enables Whisper language detection.
- `--whisper-bin PATH` selects a different `whisper-cli` executable.
- `--whisper-model PATH` selects a different Whisper model.
- `--llm-url URL` selects a different chat-completions endpoint.
- `--weather-location PLACE` selects the city or postal code used for weather.
- `--outlet HOST` replaces the default outlets; repeat it for every group member.
- `--outlet NAME=HOST` preserves a friendly name while using a reserved IP.
- `--gpio` uses the original Jetson GPIO LED instead of Shelly outlets.
- `--led-pin N` changes the physical BOARD pin used with `--gpio`.

## Troubleshooting

### Build ends with `Killed` or `Error 137`

The Jetson ran out of memory during compilation. Stop `llama-server` and other
large applications, then rerun the relevant configure command and build with:

```bash
cmake --build build --config Release --parallel 1
```

CMake reuses completed object files; deleting or cloning the repository again is
normally unnecessary. If a single-job build still fails, inspect memory and
swap with:

```bash
free -h
swapon --show
```

### `llama-server` loads and then prints `Killed`

Linux most likely terminated the server because the Jetson exhausted shared
RAM. Confirm that with:

```bash
free -h
swapon --show
sudo journalctl -k -b --no-pager | grep -Ei "oom|out of memory|killed process"
```

Use the daily-operation command documented above. In particular, remove
`-ngl 99`, use one slot with `-np 1`, and keep the context and batch sizes at
`-c 1024 -b 64 -ub 64`. Also stop browsers and other memory-heavy programs
while testing.

The documented 1.7B Q4 model is the stable baseline. If you experiment with a
larger model and the server is killed, return to this command:

```bash
./build/bin/llama-server \
  -hf ggml-org/Qwen3-1.7B-GGUF:Q4_K_M \
  -c 1024 \
  -np 1 \
  -b 64 \
  -ub 64 \
  --host 127.0.0.1 \
  --port 8080
```

The Python controller uses the same local HTTP endpoint, so switching models
does not require a controller change.

### `whisper.cpp executable not found`

Confirm that this file exists:

```bash
ls -l ~/whisper.cpp/build/bin/whisper-cli
```

If `whisper.cpp` is installed elsewhere, supply `--whisper-bin` and
`--whisper-model` explicitly.

### `arecord` cannot find the microphone

Run `python apartment_controller.py --list-microphones`. Select the microphone
by its friendly description, or copy its stable `plughw:CARD=...,DEV=...`
input. If nothing is listed, run `arecord -l`, reconnect the USB microphone,
and try another USB port.

### Wake word is missed or triggers accidentally

Speak “Hey Jarvis” by itself and wait for “What’s up?” before saying the
apartment request. Add `--wake-debug` and watch the peak score while speaking.
If the peak approaches but does not reach `0.5`, try `--wake-threshold 0.4`,
then `0.35` if necessary. Raise the threshold if unrelated speech activates it.
Confirm the selected microphone with `--list-microphones` and a playback test
before replacing hardware.

If openWakeWord cannot load, rerun:

```bash
source ~/apartment-ai/.venv/bin/activate
python -m pip install -r ~/apartment-ai/requirements.txt
cd ~/apartment-ai
python apartment_controller.py --download-wake-model
```

The older detector is available with `--wake-engine whisper`; its
`--wake-sensitivity`, `--wake-window-seconds`, and custom `--wake-word` options
still work. Push-to-talk mode remains available with `--voice`.

### Piper model is missing or no speech is heard

Confirm that both `voices/en_US-lessac-medium.onnx` and its `.onnx.json`
configuration exist. Run `python apartment_controller.py --list-speakers`, then
test the chosen sink with `--speaker`. For a Bluetooth speaker, also test the
generated file with `paplay tts-test.wav`. Bluetooth outputs often do not appear
in `aplay -l`; reserve `--speaker-device` for a headless direct-ALSA setup where
Linux system audio is not already using the hardware.

### Time or date is wrong

Time and date responses use the Jetson's system clock. Check its time, time zone,
and network synchronization:

```bash
timedatectl status
sudo timedatectl set-timezone America/New_York
sudo timedatectl set-ntp true
```

Replace `America/New_York` with the correct time zone. Wait until `timedatectl`
reports that the system clock is synchronized before testing another response.

### Weather is unavailable or uses the wrong place

Start the controller with a specific location, including a state or country when
needed:

```bash
python apartment_controller.py \
  --listen \
  --speak \
  --weather-location "Boston, MA"
```

Weather requires an internet connection. A missing location, failed location
match, network timeout, or unexpected service response is rejected without
changing the device output. Restart the controller after changing the location.

### Shelly outlet cannot be reached

An app name such as `lamp-1` may not be registered as a hostname by the router.
Test the reserved IP directly from the Jetson:

```bash
curl http://192.168.1.50/rpc/Shelly.GetDeviceInfo
curl "http://192.168.1.50/rpc/Switch.GetStatus?id=0"
```

If the IP works, start the controller with explicit mappings:

```bash
python apartment_controller.py \
  --listen \
  --speak \
  --outlet lamp-1=192.168.1.50 \
  --outlet lamp-2=192.168.1.51
```

If the response is `401 Unauthorized`, set `SHELLY_PASSWORD` to the password
configured on both devices. If neither the name nor IP works, confirm that the
Jetson and outlets are on the same LAN and that the Wi-Fi network does not use
guest/client isolation.

### GPIO permission error

Run `groups` and confirm that `gpio` is present. If it is missing after running
the permission commands, reboot the Jetson. Also confirm that
`/etc/udev/rules.d/99-gpio.rules` exists.

### Controller cannot reach the LLM

Make sure Terminal 1 is still running. From another terminal, check:

```bash
curl http://127.0.0.1:8080/health
```

The controller expects the OpenAI-compatible chat-completions endpoint at
`http://127.0.0.1:8080/v1/chat/completions`.

## Upstream references

- [llama.cpp CUDA build documentation](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)
- [llama-server documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
- [Qwen3-1.7B GGUF model](https://huggingface.co/ggml-org/Qwen3-1.7B-GGUF)
- [Qwen3 thinking-mode controls](https://huggingface.co/Qwen/Qwen3-32B-GGUF)
- [whisper.cpp documentation](https://github.com/ggml-org/whisper.cpp)
- [whisper.cpp model downloads](https://github.com/ggml-org/whisper.cpp/blob/master/models/README.md)
- [Piper local text-to-speech](https://github.com/OHF-Voice/piper1-gpl)
- [openWakeWord](https://github.com/dscripka/openWakeWord)
- [Open-Meteo forecast API](https://open-meteo.com/en/docs)
- [Open-Meteo geocoding API](https://open-meteo.com/en/docs/geocoding-api)
- [Open-Meteo data license](https://open-meteo.com/en/licence)
- [Shelly Plug US Gen4 documentation](https://us.shelly.com/blogs/documentation/shelly-plug-us-gen4)
- [Shelly local Switch RPC API](https://shelly-api-docs.shelly.cloud/gen2/ComponentsAndServices/Switch/)
- [Shelly local API authentication](https://shelly-api-docs.shelly.cloud/gen2/General/Authentication/)
- [NVIDIA Jetson.GPIO setup](https://github.com/NVIDIA/jetson-gpio)
- [NVIDIA JetPack 7.2.1 release information](https://developer.nvidia.com/embedded/jetpack/downloads)
- [NVIDIA Jetson-IO documentation for Jetson Linux 39.2](https://docs.nvidia.com/jetson/archives/r39.2/DeveloperGuide/HR/ConfiguringTheJetsonExpansionHeaders.html)
- [NVIDIA CUDA compute capabilities](https://developer.nvidia.com/cuda/gpus)

## Next milestones

- Stop recording automatically when speech ends
- Evaluate a custom apartment-specific wake model after real-room testing
- Add optional multi-turn conversational memory
- Home Assistant integration
- Environmental and presence sensors
