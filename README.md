# Apartment AI

A local apartment automation project running on an NVIDIA Jetson Orin Nano.

The current version uses a locally hosted LLM to interpret natural-language commands and convert them into structured device commands. The Jetson then uses its GPIO pins to control physical hardware.

Currently, the project supports controlling a breadboard LED as a stand-in for a desk lamp.

## Current Architecture

```text
User command
    ↓
Qwen3-4B
running locally with llama.cpp
    ↓
JSON command
    ↓
Python validation
    ↓
Jetson GPIO
    ↓
LED / device
```

Example:

```text
Command: turn on my desk lamp
```

LLM output:

```json
{
  "device": "desk_lamp",
  "action": "on"
}
```

The Python controller validates the command and sets the corresponding Jetson GPIO output HIGH or LOW.

## Hardware

Current hardware:

- NVIDIA Jetson Orin Nano
- JetPack 7.2.1
- Breadboard
- LED
- Current-limiting resistor
- Jumper wires

Current GPIO configuration:

- Jetson physical pin 7 — GPIO output
- Jetson physical pin 6 — GND

Pin 7 must first be configured as GPIO using NVIDIA's Jetson-IO utility.

## Software

The project currently uses:

- `llama.cpp`
- Qwen3-4B GGUF (`Q4_K_M`)
- Python
- `requests`
- `Jetson.GPIO`

The LLM runs entirely locally on the Jetson.

## Running the Project

Two terminals are required.

### 1. Start the LLM Server

From the `llama.cpp` directory:

```bash
cd ~/llama.cpp

./build/bin/llama-server \
  -hf Qwen/Qwen3-4B-GGUF:Q4_K_M \
  -ngl 99 \
  -c 4096 \
  --host 127.0.0.1 \
  --port 8080
```

This starts the Qwen model locally and exposes the llama.cpp API at:

```text
http://127.0.0.1:8080
```

Leave this terminal running.

### 2. Activate the Python Environment

Open another terminal:

```bash
cd ~/apartment-ai
source .venv/bin/activate
```

The terminal should now show the virtual environment as active:

```text
(.venv)
```

### 3. Start the Apartment Controller

Run:

```bash
python apartment_controller.py
```

The program will display:

```text
Command:
```

Enter a command such as:

```text
turn on my desk lamp
```

or:

```text
turn off the desk light
```

The LLM interprets the request and returns JSON. The Python controller validates the response before changing the GPIO state.

Type:

```text
exit
```

or:

```text
quit
```

to stop the controller.

## Command Safety

The LLM does not directly control GPIO.

Its response is first parsed and checked against a list of allowed devices and actions.

Currently allowed:

```text
desk_lamp
  ├── on
  └── off
```

Unrelated prompts can return:

```json
{
  "device": "none",
  "action": "none"
}
```

Only validated commands reach the GPIO control code.

## Current Status

Working:

- Local Qwen3-4B inference on Jetson
- CUDA-accelerated inference through llama.cpp
- Local llama.cpp API server
- Python-to-LLM communication
- Natural-language command interpretation
- JSON command output
- Command validation
- Jetson GPIO control
- Physical LED on/off control

## Planned Expansion

Possible next steps include:

```text
Text input
    ↓
Voice input
    ↓
Speech-to-text
    ↓
Local LLM
    ↓
Apartment controller
    ↓
Home Assistant
    ↓
Smart plugs / lights / sensors
```

Future features may include:

- Local microphone input
- Local speech-to-text
- Text-to-speech responses
- Home Assistant integration
- Smart plug control for normal wall-powered lamps
- Multiple rooms and devices
- Environmental sensors
- Presence detection
- Automated lighting
- ROS 2 integration for more complex distributed hardware
- ESP32/STM32 remote nodes
- Computer vision using the Jetson