"""Local clock responses and Open-Meteo weather lookups."""

from datetime import datetime

from .constants import OPEN_METEO_FORECAST_URL, OPEN_METEO_GEOCODING_URL
from .errors import ControllerError


WEATHER_CODE_DESCRIPTIONS = {
    0: "clear skies",
    1: "mainly clear skies",
    2: "partly cloudy skies",
    3: "overcast skies",
    45: "fog",
    48: "freezing fog",
    51: "light drizzle",
    53: "drizzle",
    55: "heavy drizzle",
    56: "light freezing drizzle",
    57: "freezing drizzle",
    61: "light rain",
    63: "rain",
    65: "heavy rain",
    66: "light freezing rain",
    67: "freezing rain",
    71: "light snow",
    73: "snow",
    75: "heavy snow",
    77: "snow grains",
    80: "light rain showers",
    81: "rain showers",
    82: "heavy rain showers",
    85: "light snow showers",
    86: "heavy snow showers",
    95: "thunderstorms",
    96: "thunderstorms with light hail",
    99: "thunderstorms with heavy hail",
}


def current_time_response(now=None):
    current = now or datetime.now()
    return f"It is {current.strftime('%I:%M %p').lstrip('0')}."


def current_date_response(now=None):
    current = now or datetime.now()
    date_text = current.strftime("%A, %B %d, %Y").replace(" 0", " ")
    return f"Today is {date_text}."


class OpenMeteoWeather:
    """Fetch current conditions for one configured location on demand."""

    def __init__(self, location=None, http_client=None):
        self.location = location.strip() if location else None
        self.http_client = http_client
        self.resolved_location = None

    def current_response(self):
        location = self._resolve_location()
        weather = self._request_json(
            OPEN_METEO_FORECAST_URL,
            params={
                "latitude": location["latitude"],
                "longitude": location["longitude"],
                "current": "temperature_2m,apparent_temperature,weather_code",
                "daily": (
                    "temperature_2m_max,temperature_2m_min,"
                    "precipitation_probability_max"
                ),
                "temperature_unit": "fahrenheit",
                "timezone": "auto",
                "forecast_days": 1,
            },
            error_prefix="Could not get current weather",
        )

        try:
            current = weather["current"]
            daily = weather["daily"]
            temperature = round(float(current["temperature_2m"]))
            apparent = round(float(current["apparent_temperature"]))
            weather_code = int(current["weather_code"])
            high = round(float(daily["temperature_2m_max"][0]))
            low = round(float(daily["temperature_2m_min"][0]))
            precipitation = round(
                float(daily["precipitation_probability_max"][0])
            )
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise ControllerError(
                "Weather service returned an unexpected response."
            ) from error

        description = WEATHER_CODE_DESCRIPTIONS.get(
            weather_code,
            "unclassified conditions",
        )
        response = (
            f"In {location['label']}, it is {temperature} degrees Fahrenheit "
            f"with {description}. Today's high is {high} and the low is {low}, "
            f"with a {precipitation} percent chance of precipitation."
        )
        if abs(apparent - temperature) >= 3:
            response += f" It feels like {apparent} degrees."
        return response

    def _resolve_location(self):
        if self.resolved_location is not None:
            return self.resolved_location
        if not self.location:
            raise ControllerError(
                "Weather location is not configured. Start with "
                '--weather-location "City, State".'
            )

        result = self._request_json(
            OPEN_METEO_GEOCODING_URL,
            params={
                "name": self.location,
                "count": 1,
                "language": "en",
                "format": "json",
            },
            error_prefix="Could not look up weather location",
        )
        locations = result.get("results")
        if not isinstance(locations, list) or not locations:
            raise ControllerError(
                f'Weather location not found: "{self.location}". Try a city '
                "with its state or country."
            )

        selected = locations[0]
        try:
            latitude = float(selected["latitude"])
            longitude = float(selected["longitude"])
            name = str(selected["name"])
        except (KeyError, TypeError, ValueError) as error:
            raise ControllerError(
                "Weather location service returned an unexpected response."
            ) from error

        admin_area = str(selected.get("admin1") or "")
        label = name
        if admin_area and admin_area.casefold() != name.casefold():
            label = f"{name}, {admin_area}"
        self.resolved_location = {
            "latitude": latitude,
            "longitude": longitude,
            "label": label,
        }
        return self.resolved_location

    def _request_json(self, url, params, error_prefix):
        client = self.http_client
        if client is None:
            try:
                import requests
            except ImportError as error:
                raise ControllerError(
                    "The 'requests' package is not installed in this environment."
                ) from error
            client = requests

        try:
            response = client.get(url, params=params, timeout=10)
            response.raise_for_status()
            payload = response.json()
        except Exception as error:
            raise ControllerError(f"{error_prefix}: {error}") from error
        if not isinstance(payload, dict):
            raise ControllerError(f"{error_prefix}: unexpected response")
        return payload
