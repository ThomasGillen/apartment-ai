"""Blink the optional GPIO test LED until interrupted."""

import time

import Jetson.GPIO as GPIO

from apartment_ai.constants import DEFAULT_LED_PIN


def main():
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(DEFAULT_LED_PIN, GPIO.OUT, initial=GPIO.LOW)

    try:
        while True:
            GPIO.output(DEFAULT_LED_PIN, GPIO.HIGH)
            print("LED ON")
            time.sleep(2)
            GPIO.output(DEFAULT_LED_PIN, GPIO.LOW)
            print("LED OFF")
            time.sleep(2)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        GPIO.output(DEFAULT_LED_PIN, GPIO.LOW)
        GPIO.cleanup()


if __name__ == "__main__":
    main()
