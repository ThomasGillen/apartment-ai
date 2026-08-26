# Apartment AI

A fully local apartment automation project for an NVIDIA Jetson Orin Nano. A
microphone command is transcribed by `whisper.cpp`, interpreted by Qwen running
through `llama.cpp`, validated by Python, and represented by a breadboard LED on
Jetson GPIO.

No microphone audio, transcript, or apartment command is sent to a cloud API.
Internet access is only needed during initial package and model downloads.

## Architecture

```text
USB microphone
    ↓
ALSA arecord → temporary 16 kHz mono WAV
    ↓
whisper.cpp → transcript
    ↓
Qwen3-1.7B on llama.cpp → JSON device command
    ↓
Python allow-list validation
    ↓
Jetson GPIO physical pin 7
    ↓
LED standing in for desk_lamp
    ↓
Piper confirmation → Linux system audio → speaker
```

The LLM never accesses GPIO directly. Voice and typed commands both enter the
same validation and output path.

## One-time setup versus daily use

| Component | One-time setup | Normal use |
| --- | --- | --- |
| `llama.cpp` | Build `llama-server` with CUDA | Keep one server process running |
| Qwen3-1.7B | Downloaded automatically on first server launch | Loaded from the local cache |
| `whisper.cpp` | Build `whisper-cli` with CUDA | Python launches it for recordings and wake-word checks |
| Whisper model | Download `ggml-base.en.bin` once | Loaded locally for transcription |
| Piper | Install `piper-tts` and download one voice | Speaks fixed ready/action responses on CPU |
| Python project | Create `.venv` and install requirements | Run push-to-talk or continuous listen mode |

After setup, normal operation still uses only two terminals: one for
`llama-server` and one for the apartment controller. Whisper does not need its
own server or terminal.

## Hardware

- NVIDIA Jetson Orin Nano with JetPack and the CUDA toolkit installed
- USB microphone or another ALSA-compatible capture device
- USB, HDMI, or another ALSA-compatible speaker
- Breadboard LED
- 220–1000 Ω current-limiting resistor
- Jumper wires

Current physical wiring:

- Jetson physical pin 7 — GPIO output for the LED
- Jetson physical pin 6 — ground

Wire the resistor in series with the LED. The LED anode connects toward pin 7
and the cathode connects toward ground. Power down the Jetson before changing
the wiring.

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
```

The requirements install `requests`, NVIDIA's `Jetson.GPIO` package, and the
local Piper text-to-speech package. This installation happens once, not each
time the controller starts.

### 4. Configure GPIO permissions

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

### 5. Configure physical pin 7 with Jetson-IO

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

### 6. Test the LED and GPIO

```bash
cd ~/apartment-ai
source .venv/bin/activate
python gpio_test.py
```

The LED should alternate on and off every two seconds. Press Ctrl+C to stop; the
cleanup code leaves the LED off.

### 7. Build llama.cpp for Qwen

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
continuous Whisper transcription and the desktop audio services.

### 8. Build whisper.cpp for speech-to-text

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

### 9. Check the microphone

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

### 10. Download a Piper voice and test the speaker

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
  -- "Ready."

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

This listing exits without loading Piper, Whisper, the LLM, or GPIO, so it can
also be run remotely over SSH.

To list both microphone inputs and speaker outputs in one remote command:

```bash
python apartment_controller.py --list-audio
```

### 11. Run the software tests

```bash
cd ~/apartment-ai
source .venv/bin/activate
python -m unittest -v
```

These tests simulate the voice and LLM boundaries. They do not record from the
real microphone or change GPIO.

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
and its short JSON commands do not need a large context or batch buffer.

### Terminal 2 option A: push-to-talk mode

For the default microphone:

```bash
cd ~/apartment-ai
source .venv/bin/activate
python apartment_controller.py --voice
```

The prompt shows:

```text
Press Enter to speak, type a command, or type 'quit':
```

Press Enter, then say a short command such as “turn on my desk lamp.” The
controller records for five seconds, transcribes the audio locally, displays
what it heard, sends the transcript to Qwen, validates Qwen's JSON response, and
changes the LED only if the command is allowed.

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

Typing text at the voice prompt bypasses recording for that command. Type
`exit` or `quit`, or press Ctrl+C, to stop. The cleanup path leaves the LED off.

### Terminal 2 option B: continuous listen mode

The continuous mode uses the same build and model; it needs no additional
server or terminal:

```bash
cd ~/apartment-ai
source .venv/bin/activate
python apartment_controller.py --listen --speak
```

The default wake word is `command`. Use it in two stages:

1. Say “command.”
2. Wait for the spoken “Ready” response and the `Recording command...` message.
3. Say “turn on my desk lamp.”

The microphone remains open through one `arecord` process. Python continuously
buffers local audio while Whisper examines overlapping two-second windows. When
the wake word is detected, old buffered audio is discarded and a fresh
five-second command is recorded. The microphone audio containing the spoken
“Ready” response is discarded before recording begins. Only the post-wake
command transcript enters the Qwen interpretation, validation, and GPIO path.

After an allowed GPIO action succeeds, Piper says either “Desk lamp turned on”
or “Desk lamp turned off.” These are fixed Python responses, not raw LLM output.

Use a more distinctive wake phrase if `command` triggers too easily:

```bash
python apartment_controller.py \
  --listen \
  --speak \
  --wake-word "hey apartment"
```

For a non-default microphone or a longer wake-detection window:

```bash
python apartment_controller.py \
  --listen \
  --speak \
  --microphone "onn USB Microphone" \
  --wake-window-seconds 3
```

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

Listen mode performs more Whisper inference than push-to-talk, so it keeps the
Jetson busier. All audio and temporary transcripts remain local and are
deleted after each check. Press Ctrl+C to stop.

### Text-only mode

The original text input remains available:

```bash
cd ~/apartment-ai
source .venv/bin/activate
python apartment_controller.py
```

## Command safety

The only allowed device/action pairs are:

```text
desk_lamp → on
desk_lamp → off
none      → none
```

Malformed JSON, unknown devices, invalid actions, microphone failures,
transcription failures, and LLM connection errors are rejected without changing
GPIO. The LED is also forced off during normal controller shutdown.

## Useful options

```bash
python apartment_controller.py --help
```

- `--voice` enables push-to-talk input.
- `--listen` enables continuous listen mode with wake-word detection.
- `--wake-word PHRASE` changes the default `command` wake phrase.
- `--wake-window-seconds N` changes the wake-detection audio window.
- `--speak` enables local Piper ready and action responses.
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
- `--led-pin N` changes the physical BOARD pin used for the LED.

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

Speak the wake phrase by itself and wait for the ready message before saying the
apartment command. Try a distinctive phrase such as `hey apartment`, or increase
`--wake-window-seconds` from 2 to 3. Push-to-talk mode remains available when
continuous Whisper scanning uses too much GPU time.

### Piper model is missing or no speech is heard

Confirm that both `voices/en_US-lessac-medium.onnx` and its `.onnx.json`
configuration exist. Run `python apartment_controller.py --list-speakers`, then
test the chosen sink with `--speaker`. For a Bluetooth speaker, also test the
generated file with `paplay tts-test.wav`. Bluetooth outputs often do not appear
in `aplay -l`; reserve `--speaker-device` for a headless direct-ALSA setup where
Linux system audio is not already using the hardware.

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
- [whisper.cpp documentation](https://github.com/ggml-org/whisper.cpp)
- [whisper.cpp model downloads](https://github.com/ggml-org/whisper.cpp/blob/master/models/README.md)
- [Piper local text-to-speech](https://github.com/OHF-Voice/piper1-gpl)
- [NVIDIA Jetson.GPIO setup](https://github.com/NVIDIA/jetson-gpio)
- [NVIDIA JetPack 7.2.1 release information](https://developer.nvidia.com/embedded/jetpack/downloads)
- [NVIDIA Jetson-IO documentation for Jetson Linux 39.2](https://docs.nvidia.com/jetson/archives/r39.2/DeveloperGuide/HR/ConfiguringTheJetsonExpansionHeaders.html)
- [NVIDIA CUDA compute capabilities](https://developer.nvidia.com/cuda/gpus)

## Next milestones

- Stop recording automatically when speech ends
- Replace continuous Whisper scanning with a smaller dedicated wake-word model
- Add optional conversational responses beyond fixed confirmations
- Home Assistant integration
- Smart plugs and multiple lights
- Environmental and presence sensors
