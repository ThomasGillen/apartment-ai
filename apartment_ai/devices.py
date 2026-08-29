"""Validated Shelly outlet and Jetson GPIO device outputs."""

import os
import re

from .constants import (
    CONTROL_DEVICE_ID,
    CONTROL_DEVICE_NAME,
    DEFAULT_OUTLET_CHANNEL,
    DEFAULT_OUTLET_HOSTS,
    DEFAULT_OUTLET_TIMEOUT,
    SHELLY_PASSWORD_ENV,
)
from .errors import ControllerError


class ShellyOutlet:
    """Control and verify one Shelly switch through its local HTTP RPC API."""

    def __init__(
        self,
        name,
        host=None,
        channel=DEFAULT_OUTLET_CHANNEL,
        password=None,
        http_client=None,
        auth=None,
        timeout=DEFAULT_OUTLET_TIMEOUT,
    ):
        self.name = name
        self.host = (host or name).strip().rstrip("/")
        if not self.host:
            raise ControllerError("Shelly outlet host cannot be empty.")
        self.base_url = (
            self.host
            if re.match(r"^https?://", self.host, flags=re.IGNORECASE)
            else f"http://{self.host}"
        )
        self.channel = channel
        self.http_client = http_client
        self.timeout = timeout
        self.auth = auth

        if self.auth is None and password:
            try:
                from requests.auth import HTTPDigestAuth
            except ImportError as error:
                raise ControllerError(
                    "Shelly authentication needs the 'requests' package."
                ) from error
            self.auth = HTTPDigestAuth("admin", password)

    def _client(self):
        if self.http_client is not None:
            return self.http_client
        try:
            import requests
        except ImportError as error:
            raise ControllerError(
                "Shelly outlet control needs the 'requests' package."
            ) from error
        self.http_client = requests
        return self.http_client

    def _request(self, method, params=None):
        url = f"{self.base_url}/rpc/{method}"
        request_options = {
            "params": params or {},
            "timeout": self.timeout,
        }
        if self.auth is not None:
            request_options["auth"] = self.auth

        try:
            response = self._client().get(url, **request_options)
            if getattr(response, "status_code", None) == 401:
                raise ControllerError(
                    f'Authentication failed for outlet "{self.name}". Set '
                    f"{SHELLY_PASSWORD_ENV} to the device password."
                )
            response.raise_for_status()
            payload = response.json()
        except ControllerError:
            raise
        except Exception as error:
            raise ControllerError(
                f'Could not reach outlet "{self.name}" at {self.host}: {error}'
            ) from error

        if not isinstance(payload, dict):
            raise ControllerError(
                f'Outlet "{self.name}" returned an unexpected response.'
            )
        if payload.get("error"):
            raise ControllerError(
                f'Outlet "{self.name}" rejected {method}: {payload["error"]}'
            )
        return payload

    def get_power(self):
        status = self._request("Switch.GetStatus", params={"id": self.channel})
        errors = status.get("errors") or []
        if errors:
            raise ControllerError(
                f'Outlet "{self.name}" reported an error: '
                f'{", ".join(map(str, errors))}'
            )
        if not isinstance(status.get("output"), bool):
            raise ControllerError(
                f'Outlet "{self.name}" did not report its output state.'
            )
        return status["output"]

    def set_power(self, enabled):
        desired_state = bool(enabled)
        self._request(
            "Switch.Set",
            params={
                "id": self.channel,
                "on": "true" if desired_state else "false",
                "tag": "apartment-ai",
            },
        )
        actual_state = self.get_power()
        if actual_state != desired_state:
            expected = "on" if desired_state else "off"
            actual = "on" if actual_state else "off"
            raise ControllerError(
                f'Outlet "{self.name}" should be {expected} but reported {actual}.'
            )


class ShellyOutletGroup:
    """Treat multiple Shelly outlets as one allow-listed apartment device."""

    def __init__(self, outlets):
        self.outlets = list(outlets)
        if not self.outlets:
            raise ControllerError("At least one Shelly outlet must be configured.")

    def apply(self, device, action):
        if device != CONTROL_DEVICE_ID:
            raise ControllerError(f"No output is configured for device: {device}")
        if action not in {"on", "off"}:
            raise ControllerError(
                f"Unsupported {CONTROL_DEVICE_NAME} action: {action}"
            )

        failures = []
        successful = []
        for outlet in self.outlets:
            try:
                outlet.set_power(action == "on")
                successful.append(outlet.name)
            except ControllerError as error:
                failures.append(str(error))

        if failures:
            failure_summary = "; ".join(
                failure.rstrip(".") for failure in failures
            )
            partial_warning = (
                f" Updated before the failure: {', '.join(successful)}."
                if successful
                else ""
            )
            raise ControllerError(
                f"Could not turn every living room lamp {action}: "
                f"{failure_summary}.{partial_warning}"
            )

        outlet_names = ", ".join(outlet.name for outlet in self.outlets)
        print(
            f">>> {CONTROL_DEVICE_NAME.upper()} {action.upper()} "
            f"({outlet_names})\n"
        )
        return f"{CONTROL_DEVICE_NAME.capitalize()} turned {action}."

    @staticmethod
    def cleanup():
        return None


def parse_outlet_target(target):
    """Parse HOST or NAME=HOST while keeping friendly labels for failures."""
    value = target.strip()
    if not value:
        raise ControllerError("Outlet target cannot be empty.")
    if "=" not in value:
        return value, value
    name, host = (part.strip() for part in value.split("=", 1))
    if not name or not host:
        raise ControllerError(
            f'Invalid outlet target "{target}"; use HOST or NAME=HOST.'
        )
    return name, host


def create_device_output(args):
    if args.gpio:
        return GpioLed(args.led_pin)
    targets = args.outlet_hosts or DEFAULT_OUTLET_HOSTS
    password = os.environ.get(SHELLY_PASSWORD_ENV)
    outlets = []
    for target in targets:
        name, host = parse_outlet_target(target)
        outlets.append(ShellyOutlet(name=name, host=host, password=password))
    return ShellyOutletGroup(outlets)


class GpioLed:
    def __init__(self, pin):
        try:
            import Jetson.GPIO as gpio
        except ImportError as error:
            raise ControllerError(
                "Jetson.GPIO is not installed. Run this controller on the Jetson "
                "inside the project environment."
            ) from error

        self.gpio = gpio
        self.pin = pin
        self.gpio.setwarnings(False)
        self.gpio.setmode(self.gpio.BOARD)
        self.gpio.setup(self.pin, self.gpio.OUT, initial=self.gpio.LOW)

    def apply(self, device, action):
        if device != CONTROL_DEVICE_ID:
            raise ControllerError(f"No output is configured for device: {device}")
        if action == "on":
            self.gpio.output(self.pin, self.gpio.HIGH)
            print(f">>> {CONTROL_DEVICE_NAME.upper()} ON\n")
            return f"{CONTROL_DEVICE_NAME.capitalize()} turned on."
        if action == "off":
            self.gpio.output(self.pin, self.gpio.LOW)
            print(f">>> {CONTROL_DEVICE_NAME.upper()} OFF\n")
            return f"{CONTROL_DEVICE_NAME.capitalize()} turned off."
        return None

    def cleanup(self):
        self.gpio.output(self.pin, self.gpio.LOW)
        self.gpio.cleanup()
