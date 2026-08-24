import Jetson.GPIO as GPIO
import time

GPIO.setwarnings(False)

LED_PIN = 7

GPIO.setmode(GPIO.BOARD)
GPIO.setup(LED_PIN, GPIO.OUT, initial=GPIO.LOW)

try:
    while True:
        GPIO.output(LED_PIN, GPIO.HIGH)
        print("LED ON")
        time.sleep(2)

        GPIO.output(LED_PIN, GPIO.LOW)
        print("LED OFF")
        time.sleep(2)

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    GPIO.output(LED_PIN, GPIO.LOW)
    GPIO.cleanup()