import requests
import json
import Jetson.GPIO as GPIO

# Configuration

URL = "http://127.0.0.1:8080/v1/chat/completions"

# Physical pin 7 on Jetson 40-pin header
LED_PIN = 7

SYSTEM_PROMPT = """
You control an apartment.

Available devices:
- desk_lamp
- none

Available actions:
- on
- off
- none

Interpret the user's request.

If the user is not asking to control a device, return:
{"device":"none","action":"none"}

Return ONLY JSON.

Examples:
{"device":"desk_lamp","action":"on"}
{"device":"desk_lamp","action":"off"}
{"device":"none","action":"none"}
"""

allowed_commands = {
    "desk_lamp": {"on", "off"},
    "none": {"none"}
}

# GPIO setup
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)

GPIO.setup(
    LED_PIN,
    GPIO.OUT,
    initial=GPIO.LOW
)

# Main program
print("Apartment controller started.")
print("Type 'exit' or 'quit' to stop.\n")

try:
    while True:
        user_text = input("Command: ").strip()
        # Exit command
        if user_text.lower() in {"exit", "quit"}:
            print("Exiting apartment controller.")
            break

        # Ignore empty input
        if not user_text:
            continue

        # Send command to local LLM
        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": user_text
                }
            ],
            "temperature": 0
        }

        try:
            response = requests.post(
                URL,
                json=payload,
                timeout=60
            )
            response.raise_for_status()

        except requests.RequestException as e:
            print(f"Error communicating with LLM server: {e}\n")
            continue

        # Extract LLM response
        try:
            text = response.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            print("Unexpected response from LLM server.\n")
            continue

        print("Raw LLM response:")
        print(text)

        # Parse JSON
        try:
            command = json.loads(text)

        except json.JSONDecodeError:
            print("Invalid JSON from LLM.\n")
            continue

        device = command.get("device")
        action = command.get("action")

        # Validate command
        if device not in allowed_commands:
            print(f"Rejected unknown device: {device}\n")
            continue

        if action not in allowed_commands[device]:
            print(
                f"Rejected invalid command: "
                f"{device} -> {action}\n"
            )
            continue
        print(f"Validated command: {device} -> {action}")

        # Execute command
        if device == "none":
            print("No apartment action requested.\n")
            continue

        if device == "desk_lamp":
            if action == "on":
                GPIO.output(
                    LED_PIN,
                    GPIO.HIGH
                )
                print(">>> DESK LAMP ON\n")

            elif action == "off":
                GPIO.output(
                    LED_PIN,
                    GPIO.LOW
                )
                print(">>> DESK LAMP OFF\n")

except KeyboardInterrupt:
    print("\nStopping apartment controller.")

finally:
    # Always leave the LED off when program exits
    GPIO.output(
        LED_PIN,
        GPIO.LOW
    )
    GPIO.cleanup()
    print("GPIO cleaned up.")