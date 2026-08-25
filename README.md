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
Qwen3-4B on llama.cpp → JSON device command
    ↓
Python allow-list validation
    ↓
Jetson GPIO physical pin 7
    ↓
LED standing in for desk_lamp
```

The LLM never accesses GPIO directly. Voice and typed commands both enter the
same validation and output path.

## One-time setup versus daily use

| Component | One-time setup | Normal use |
| --- | --- | --- |
| `llama.cpp` | Build `llama-server` with CUDA | Keep one server process running |
| Qwen3-4B | Downloaded automatically on first server launch | Loaded from the local cache |
| `whisper.cpp` | Build `whisper-cli` with CUDA | Python launches it for each recording |
| Whisper model | Download `ggml-base.en.bin` once | Loaded locally for transcription |
| Python project | Create `.venv` and install requirements | Run `apartment_controller.py --voice` |

After setup, normal operation still uses only two terminals: one for
`llama-server` and one for the apartment controller. Whisper does not need its
own server or terminal.

## Hardware

- NVIDIA Jetson Orin Nano with JetPack and the CUDA toolkit installed
- USB microphone or another ALSA-compatible capture device
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

The requirements install `requests` and NVIDIA's `Jetson.GPIO` Python package.
This installation happens once, not each time the controller starts.

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
server launch with `-hf Qwen/Qwen3-4B-GGUF:Q4_K_M` downloads and caches the
Q4_K_M model. Later launches use the local cached copy.

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
```

First try the default microphone:

```bash
arecord -f S16_LE -r 16000 -c 1 -d 5 mic-test.wav
aplay mic-test.wav
```

If the USB microphone is listed as card 2, device 0, select it explicitly:

```bash
arecord -D plughw:2,0 -f S16_LE -r 16000 -c 1 -d 5 mic-test.wav
aplay mic-test.wav
```

Replace `2,0` with the card and device numbers reported by `arecord -l`.

### 10. Run the software tests

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
  -hf Qwen/Qwen3-4B-GGUF:Q4_K_M \
  -ngl 99 \
  -c 4096 \
  --host 127.0.0.1 \
  --port 8080
```

Leave this terminal running. The first launch needs internet access while Qwen
is downloaded; subsequent launches use the cached model.

### Terminal 2: start the apartment controller

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
python apartment_controller.py --voice --microphone plughw:2,0
```

To use a seven-second recording window:

```bash
python apartment_controller.py --voice --record-seconds 7
```

Typing text at the voice prompt bypasses recording for that command. Type
`exit` or `quit`, or press Ctrl+C, to stop. The cleanup path leaves the LED off.

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
- `--microphone plughw:CARD,DEVICE` selects an ALSA capture device.
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

### `whisper.cpp executable not found`

Confirm that this file exists:

```bash
ls -l ~/whisper.cpp/build/bin/whisper-cli
```

If `whisper.cpp` is installed elsewhere, supply `--whisper-bin` and
`--whisper-model` explicitly.

### `arecord` cannot find the microphone

Run `arecord -l`, test the device with a short recording, and pass its
`plughw:CARD,DEVICE` value through `--microphone`.

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
- [Qwen3-4B GGUF model](https://huggingface.co/Qwen/Qwen3-4B-GGUF)
- [whisper.cpp documentation](https://github.com/ggml-org/whisper.cpp)
- [whisper.cpp model downloads](https://github.com/ggml-org/whisper.cpp/blob/master/models/README.md)
- [NVIDIA Jetson.GPIO setup](https://github.com/NVIDIA/jetson-gpio)
- [NVIDIA JetPack 7.2.1 release information](https://developer.nvidia.com/embedded/jetpack/downloads)
- [NVIDIA Jetson-IO documentation for Jetson Linux 39.2](https://docs.nvidia.com/jetson/archives/r39.2/DeveloperGuide/HR/ConfiguringTheJetsonExpansionHeaders.html)
- [NVIDIA CUDA compute capabilities](https://developer.nvidia.com/cuda/gpus)

## Next milestones

- Stop recording automatically when speech ends
- Optional wake word and always-listening mode
- Text-to-speech confirmation
- Home Assistant integration
- Smart plugs and multiple lights
- Environmental and presence sensors
