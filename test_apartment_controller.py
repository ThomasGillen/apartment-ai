import json
import subprocess
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

from apartment_controller import (
    AlsaAudioStream,
    ControllerError,
    GpioLed,
    OpenMeteoWeather,
    PiperSpeechOutput,
    ShellyOutlet,
    ShellyOutletGroup,
    WakeWordInput,
    WhisperVoiceInput,
    clean_conversation_response,
    create_device_output,
    current_date_response,
    current_time_response,
    execute_request,
    generate_conversation_response,
    get_alsa_microphones,
    get_system_audio_sinks,
    interpret_request,
    main,
    parse_args,
    parse_outlet_target,
    resolve_microphone,
    resolve_system_sink,
    validate_request,
)


class RequestValidationTests(unittest.TestCase):
    def test_allows_known_lamp_command(self):
        self.assertEqual(
            validate_request(
                {"intent": "control", "device": "desk_lamp", "action": "on"}
            ),
            ("control", "desk_lamp", "on"),
        )

    def test_allows_non_control_intents(self):
        for intent in ("time", "date", "weather", "conversation"):
            with self.subTest(intent=intent):
                self.assertEqual(
                    validate_request({"intent": intent}),
                    (intent, None, None),
                )

    def test_normalizes_legacy_none_to_conversation(self):
        self.assertEqual(
            validate_request({"intent": "none"}),
            ("conversation", None, None),
        )

    def test_routes_unknown_device_and_action_to_conversation(self):
        for request in (
            {"intent": "control", "device": "oven", "action": "on"},
            {
                "intent": "control",
                "device": "desk_lamp",
                "action": "dim",
            },
        ):
            with self.subTest(request=request):
                self.assertEqual(
                    validate_request(request),
                    ("conversation", None, None),
                )

    def test_routes_unknown_intent_to_conversation(self):
        self.assertEqual(
            validate_request({"intent": "unsupported"}),
            ("conversation", None, None),
        )

    def test_routes_non_object_json_to_conversation(self):
        self.assertEqual(
            validate_request(["desk_lamp", "on"]),
            ("conversation", None, None),
        )

    def test_interpret_request_posts_text_and_returns_valid_request(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "intent": "control",
                                "device": "desk_lamp",
                                "action": "off",
                            }
                        )
                    }
                }
            ]
        }
        client = Mock()
        client.post.return_value = response

        result = interpret_request(
            "turn it off", "http://localhost/test", http_client=client
        )

        self.assertEqual(result, ("control", "desk_lamp", "off"))
        request_body = client.post.call_args.kwargs["json"]
        self.assertEqual(request_body["messages"][-1]["content"], "turn it off")

    def test_invalid_classifier_json_uses_conversation_fallback(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "choices": [{"message": {"content": "not JSON"}}]
        }
        client = Mock()
        client.post.return_value = response

        result = interpret_request(
            "How long does laundry take?",
            "http://localhost/test",
            http_client=client,
        )

        self.assertEqual(result, ("conversation", None, None))


class ConversationResponseTests(unittest.TestCase):
    @staticmethod
    def response_with(content):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "choices": [{"message": {"content": content}}]
        }
        return response

    def test_generates_short_clean_response(self):
        client = Mock()
        client.post.return_value = self.response_with(
            "<think>Private reasoning.</think>\n**Frank Herbert** wrote Dune."
        )

        result = generate_conversation_response(
            "Who wrote Dune?",
            "http://localhost/test",
            http_client=client,
        )

        self.assertEqual(result, "Frank Herbert wrote Dune.")
        request_body = client.post.call_args.kwargs["json"]
        self.assertIn("/no_think", request_body["messages"][-1]["content"])
        self.assertIn(
            "do not imply that the action happened",
            request_body["messages"][0]["content"],
        )
        self.assertEqual(request_body["max_tokens"], 160)

    def test_limits_response_to_two_sentences(self):
        result = clean_conversation_response(
            "First sentence. Second sentence! Third sentence?"
        )

        self.assertEqual(result, "First sentence. Second sentence!")

    def test_rejects_empty_response_after_thinking_is_removed(self):
        with self.assertRaisesRegex(ControllerError, "empty conversational"):
            clean_conversation_response("<think>Only hidden reasoning.</think>")


class InformationResponseTests(unittest.TestCase):
    def test_formats_local_time_and_date(self):
        now = datetime(2026, 8, 27, 9, 5)

        self.assertEqual(current_time_response(now), "It is 9:05 AM.")
        self.assertEqual(
            current_date_response(now),
            "Today is Thursday, August 27, 2026.",
        )

    def test_executes_information_without_touching_gpio(self):
        output = Mock()
        weather = Mock()
        weather.current_response.return_value = "Weather response."
        now = datetime(2026, 8, 27, 21, 15)

        self.assertEqual(
            execute_request(("time", None, None), output, weather, now=now),
            "It is 9:15 PM.",
        )
        self.assertEqual(
            execute_request(("weather", None, None), output, weather, now=now),
            "Weather response.",
        )
        output.apply.assert_not_called()

    def test_executes_control_through_device_output(self):
        output = Mock()
        output.apply.return_value = "Living room lamps turned on."

        response = execute_request(
            ("control", "desk_lamp", "on"),
            output,
            Mock(),
        )

        self.assertEqual(response, "Living room lamps turned on.")
        output.apply.assert_called_once_with("desk_lamp", "on")

    def test_executes_conversation_without_touching_gpio(self):
        output = Mock()
        client = Mock()
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "choices": [
                {"message": {"content": "Dune was written by Frank Herbert."}}
            ]
        }
        client.post.return_value = response

        result = execute_request(
            ("conversation", None, None),
            output,
            Mock(),
            user_text="Who wrote Dune?",
            conversation_client=client,
        )

        self.assertEqual(result, "Dune was written by Frank Herbert.")
        output.apply.assert_not_called()

class WeatherResponseTests(unittest.TestCase):
    def test_builds_current_weather_response_from_location(self):
        location_response = Mock()
        location_response.raise_for_status.return_value = None
        location_response.json.return_value = {
            "results": [
                {
                    "name": "Boston",
                    "admin1": "Massachusetts",
                    "latitude": 42.36,
                    "longitude": -71.06,
                }
            ]
        }
        forecast_response = Mock()
        forecast_response.raise_for_status.return_value = None
        forecast_response.json.return_value = {
            "current": {
                "temperature_2m": 72.4,
                "apparent_temperature": 75.6,
                "weather_code": 2,
            },
            "daily": {
                "temperature_2m_max": [78.2],
                "temperature_2m_min": [61.4],
                "precipitation_probability_max": [20],
            },
        }
        client = Mock()
        client.get.side_effect = [location_response, forecast_response]
        weather = OpenMeteoWeather("Boston, MA", http_client=client)

        response = weather.current_response()

        self.assertEqual(
            response,
            "In Boston, Massachusetts, it is 72 degrees Fahrenheit with partly "
            "cloudy skies. Today's high is 78 and the low is 61, with a 20 "
            "percent chance of precipitation. It feels like 76 degrees.",
        )
        self.assertEqual(client.get.call_count, 2)
        forecast_params = client.get.call_args_list[1].kwargs["params"]
        self.assertEqual(forecast_params["temperature_unit"], "fahrenheit")
        self.assertEqual(forecast_params["forecast_days"], 1)

    def test_requires_a_configured_weather_location(self):
        weather = OpenMeteoWeather()

        with self.assertRaisesRegex(ControllerError, "--weather-location"):
            weather.current_response()

    def test_reports_location_that_cannot_be_found(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"results": []}
        client = Mock()
        client.get.return_value = response
        weather = OpenMeteoWeather("Not A Real City", http_client=client)

        with self.assertRaisesRegex(ControllerError, "not found"):
            weather.current_response()

    def test_reports_weather_network_failure(self):
        client = Mock()
        client.get.side_effect = TimeoutError("offline")
        weather = OpenMeteoWeather("Boston, MA", http_client=client)

        with self.assertRaisesRegex(ControllerError, "Could not look up"):
            weather.current_response()


class VoiceInputTests(unittest.TestCase):
    @patch("apartment_controller.subprocess.run")
    def test_records_then_transcribes(self, run):
        def fake_run(command, **kwargs):
            if Path(command[0]).name == "whisper-cli":
                output_base = Path(command[command.index("--output-file") + 1])
                output_base.with_suffix(".txt").write_text(
                    "turn the living room lamps on\n", encoding="utf-8"
                )
            return subprocess.CompletedProcess(command, 0, "", "")

        run.side_effect = fake_run
        voice = WhisperVoiceInput(
            whisper_bin="/opt/whisper-cli",
            whisper_model="/opt/ggml-base.en.bin",
            record_seconds=4,
            microphone="plughw:2,0",
        )

        transcript = voice.listen()

        self.assertEqual(transcript, "turn the living room lamps on")
        record_command = run.call_args_list[0].args[0]
        whisper_command = run.call_args_list[1].args[0]
        self.assertIn("--duration", record_command)
        self.assertIn("plughw:2,0", record_command)
        self.assertIn("--output-txt", whisper_command)


class ContinuousAudioTests(unittest.TestCase):
    @patch("apartment_controller.subprocess.Popen")
    def test_starts_arecord_with_supported_raw_file_type_option(self, popen):
        process = Mock()
        process.poll.return_value = 1
        process.stdout.read.return_value = b""
        popen.return_value = process
        stream = AlsaAudioStream(microphone="plughw:2,0")

        stream.start()
        stream.reader_thread.join(timeout=1)

        command = popen.call_args.args[0]
        self.assertIn("--file-type", command)
        self.assertNotIn("--type", command)
        self.assertIn("plughw:2,0", command)
        self.assertEqual(command[-1], "-")

    def test_reads_requested_duration_from_buffer(self):
        stream = AlsaAudioStream()
        stream.process = Mock()
        stream.process.poll.return_value = None
        stream.frames.put(b"\x00" * 64000)

        audio = stream.read_seconds(2)

        self.assertEqual(len(audio), 64000)


class WakeWordInputTests(unittest.TestCase):
    def test_waits_for_whole_wake_word_then_records_command(self):
        stream = Mock()
        stream.read_seconds.side_effect = [b"noise", b"wake", b"command"]
        voice = Mock()
        voice.microphone = "plughw:2,0"
        voice.record_seconds = 5
        voice.transcribe_pcm.side_effect = [
            "background speech",
            "please say COMMAND!",
            "turn the living room lamps on",
        ]
        stream_factory = Mock(return_value=stream)
        wake_input = WakeWordInput(
            voice_input=voice,
            wake_word="command",
            window_seconds=2,
            stream_factory=stream_factory,
        )

        transcript = wake_input.listen()

        self.assertEqual(transcript, "turn the living room lamps on")
        self.assertEqual(
            [call.args[0] for call in stream.read_seconds.call_args_list],
            [2, 2, 5],
        )
        stream.discard_buffer.assert_called_once_with()
        command_call = voice.transcribe_pcm.call_args_list[-1]
        self.assertTrue(command_call.kwargs["prompt"])
        self.assertTrue(command_call.kwargs["announce"])

    def test_does_not_match_wake_word_inside_another_word(self):
        voice = Mock(microphone=None, record_seconds=5)
        wake_input = WakeWordInput(
            voice_input=voice,
            wake_word="command",
            stream_factory=Mock(),
        )

        self.assertFalse(wake_input._contains_wake_word("commander reporting"))
        self.assertTrue(wake_input._contains_wake_word("Command, please"))

    def test_says_acknowledgement_then_discards_its_own_speaker_audio(self):
        stream = Mock()
        stream.read_seconds.side_effect = [b"wake", b"command"]
        voice = Mock(microphone=None, record_seconds=5)
        voice.transcribe_pcm.side_effect = ["command", "lights on"]
        speech_output = Mock()
        wake_input = WakeWordInput(
            voice_input=voice,
            speech_output=speech_output,
            stream_factory=Mock(return_value=stream),
        )

        transcript = wake_input.listen()

        self.assertEqual(transcript, "lights on")
        speech_output.speak.assert_called_once_with("What's up?")
        self.assertEqual(stream.discard_buffer.call_count, 2)


class SpeechOutputTests(unittest.TestCase):
    @staticmethod
    def _write_test_audio(_text, wav_file):
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(22050)
        wav_file.writeframes(b"\x00\x00")

    @patch("apartment_controller.shutil.which")
    def test_loads_piper_on_cpu_and_plays_selected_alsa_device(self, which):
        which.side_effect = lambda player: (
            "/usr/bin/aplay" if player == "aplay" else None
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "voice.onnx"
            model_path.write_bytes(b"model")
            Path(f"{model_path}.json").write_text("{}", encoding="utf-8")

            voice = Mock()

            def synthesize(text, wav_file):
                self.assertEqual(text, "What's up?")
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(22050)
                wav_file.writeframes(b"\x00\x00")

            voice.synthesize_wav.side_effect = synthesize
            voice_loader = Mock(return_value=voice)
            player_runner = Mock(
                return_value=subprocess.CompletedProcess([], 0, "", "")
            )
            speech_output = PiperSpeechOutput(
                model_path=model_path,
                playback_device="plughw:3,0",
                voice_loader=voice_loader,
                player_runner=player_runner,
            )

            speech_output.speak("What's up?")

            voice_loader.assert_called_once_with(str(model_path), use_cuda=False)
            play_command = player_runner.call_args.args[0]
            self.assertIn("--device", play_command)
            self.assertIn("plughw:3,0", play_command)
            self.assertEqual(Path(play_command[-1]).name, "response.wav")

    @patch("apartment_controller.shutil.which")
    def test_prefers_system_audio_for_bluetooth_output(self, which):
        available = {
            "paplay": "/usr/bin/paplay",
            "pw-play": "/usr/bin/pw-play",
            "aplay": "/usr/bin/aplay",
        }
        which.side_effect = available.get

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "voice.onnx"
            model_path.write_bytes(b"model")
            Path(f"{model_path}.json").write_text("{}", encoding="utf-8")
            voice = Mock()
            voice.synthesize_wav.side_effect = self._write_test_audio
            player_runner = Mock(
                return_value=subprocess.CompletedProcess([], 0, "", "")
            )
            speech_output = PiperSpeechOutput(
                model_path=model_path,
                voice_loader=Mock(return_value=voice),
                player_runner=player_runner,
            )

            speech_output.speak("What's up?")

            play_command = player_runner.call_args.args[0]
            self.assertEqual(play_command[0], "paplay")
            self.assertEqual(Path(play_command[-1]).name, "response.wav")

    @patch("apartment_controller.shutil.which")
    def test_routes_to_manually_selected_friendly_system_speaker(self, which):
        which.side_effect = lambda player: (
            "/usr/bin/paplay" if player == "paplay" else None
        )
        sinks = [
            {
                "index": "52",
                "name": "bluez_output.11_22_33.a2dp-sink",
                "description": "Living Room Speaker",
                "state": "RUNNING",
            }
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "voice.onnx"
            model_path.write_bytes(b"model")
            Path(f"{model_path}.json").write_text("{}", encoding="utf-8")
            voice = Mock()
            voice.synthesize_wav.side_effect = self._write_test_audio
            player_runner = Mock(
                return_value=subprocess.CompletedProcess([], 0, "", "")
            )
            speech_output = PiperSpeechOutput(
                model_path=model_path,
                playback_sink="living room",
                sink_lister=Mock(return_value=sinks),
                voice_loader=Mock(return_value=voice),
                player_runner=player_runner,
            )

            speech_output.speak("What's up?")

            play_command = player_runner.call_args.args[0]
            self.assertEqual(play_command[0], "paplay")
            self.assertIn(
                "--device=bluez_output.11_22_33.a2dp-sink",
                play_command,
            )
            self.assertEqual(speech_output.speaker_label, "Living Room Speaker")

    @patch("apartment_controller.shutil.which")
    def test_falls_back_when_first_system_player_fails(self, which):
        available = {
            "paplay": "/usr/bin/paplay",
            "pw-play": "/usr/bin/pw-play",
        }
        which.side_effect = available.get

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "voice.onnx"
            model_path.write_bytes(b"model")
            Path(f"{model_path}.json").write_text("{}", encoding="utf-8")
            voice = Mock()
            voice.synthesize_wav.side_effect = self._write_test_audio
            player_runner = Mock(
                side_effect=[
                    subprocess.CompletedProcess([], 1, "", "no Pulse server"),
                    subprocess.CompletedProcess([], 0, "", ""),
                ]
            )
            speech_output = PiperSpeechOutput(
                model_path=model_path,
                voice_loader=Mock(return_value=voice),
                player_runner=player_runner,
            )

            speech_output.speak("What's up?")

            self.assertEqual(player_runner.call_count, 2)
            self.assertEqual(player_runner.call_args_list[0].args[0][0], "paplay")
            self.assertEqual(player_runner.call_args_list[1].args[0][0], "pw-play")


class SystemAudioSinkTests(unittest.TestCase):
    @patch("apartment_controller.shutil.which", return_value="/usr/bin/pactl")
    def test_reads_system_speakers_from_pactl_json(self, _which):
        pactl_output = json.dumps(
            [
                {
                    "index": 7,
                    "name": "alsa_output.usb-AB13X",
                    "description": "AB13X USB Audio",
                    "state": "SUSPENDED",
                },
                {
                    "index": 9,
                    "name": "bluez_output.11_22_33.a2dp-sink",
                    "description": "Living Room Speaker",
                    "state": "RUNNING",
                },
            ]
        )
        runner = Mock(
            return_value=subprocess.CompletedProcess([], 0, pactl_output, "")
        )

        sinks = get_system_audio_sinks(command_runner=runner)

        self.assertEqual(len(sinks), 2)
        self.assertEqual(sinks[0]["description"], "AB13X USB Audio")
        self.assertEqual(sinks[1]["index"], "9")

    def test_resolves_sink_by_index_description_or_unique_partial_name(self):
        sinks = [
            {
                "index": "7",
                "name": "alsa_output.usb-AB13X",
                "description": "AB13X USB Audio",
            },
            {
                "index": "9",
                "name": "bluez_output.11_22_33.a2dp-sink",
                "description": "Living Room Speaker",
            },
        ]

        self.assertEqual(resolve_system_sink("7", sinks), sinks[0])
        self.assertEqual(
            resolve_system_sink("Living Room Speaker", sinks),
            sinks[1],
        )
        self.assertEqual(resolve_system_sink("bluez_output", sinks), sinks[1])

    def test_rejects_unknown_speaker_with_listing_hint(self):
        with self.assertRaisesRegex(ControllerError, "--list-speakers"):
            resolve_system_sink("Kitchen", [])


class MicrophoneSelectionTests(unittest.TestCase):
    @patch("apartment_controller.shutil.which", return_value="/usr/bin/arecord")
    def test_reads_stable_microphone_selectors_from_arecord(self, _which):
        arecord_output = """\
**** List of CAPTURE Hardware Devices ****
card 0: Microphone [onn USB Microphone], device 0: USB Audio [USB Audio]
  Subdevices: 1/1
  Subdevice #0: subdevice #0
card 2: Webcam [Conference Webcam], device 1: USB Audio [USB Audio]
  Subdevices: 1/1
"""
        runner = Mock(
            return_value=subprocess.CompletedProcess([], 0, arecord_output, "")
        )

        microphones = get_alsa_microphones(command_runner=runner)

        self.assertEqual(len(microphones), 2)
        self.assertEqual(
            microphones[0]["selector"],
            "plughw:CARD=Microphone,DEV=0",
        )
        self.assertEqual(microphones[1]["card_description"], "Conference Webcam")

    def test_resolves_friendly_microphone_to_stable_alsa_selector(self):
        microphones = [
            {
                "card_index": "0",
                "card_id": "Microphone",
                "card_description": "onn USB Microphone",
                "device_index": "0",
                "device_name": "USB Audio",
                "device_description": "USB Audio",
                "selector": "plughw:CARD=Microphone,DEV=0",
            }
        ]

        self.assertEqual(
            resolve_microphone("onn USB", microphones),
            "plughw:CARD=Microphone,DEV=0",
        )
        self.assertEqual(
            resolve_microphone("0", microphones),
            "plughw:CARD=Microphone,DEV=0",
        )

    def test_preserves_existing_direct_alsa_microphone_selector(self):
        self.assertEqual(
            resolve_microphone("plughw:2,0"),
            "plughw:2,0",
        )

    def test_rejects_unknown_microphone_with_listing_hint(self):
        with self.assertRaisesRegex(ControllerError, "--list-microphones"):
            resolve_microphone("Kitchen", [])


class GpioLedTests(unittest.TestCase):
    def test_returns_fixed_spoken_confirmation_after_action(self):
        led = GpioLed.__new__(GpioLed)
        led.gpio = Mock(HIGH=1, LOW=0)
        led.pin = 7

        self.assertEqual(
            led.apply("desk_lamp", "on"),
            "Living room lamps turned on.",
        )
        self.assertEqual(
            led.apply("desk_lamp", "off"),
            "Living room lamps turned off.",
        )


class ShellyOutletTests(unittest.TestCase):
    @staticmethod
    def response(payload, status_code=200):
        response = Mock(status_code=status_code)
        response.raise_for_status.return_value = None
        response.json.return_value = payload
        return response

    def test_sets_and_verifies_local_outlet_state(self):
        client = Mock()
        client.get.side_effect = [
            self.response({"was_on": False}),
            self.response({"id": 0, "output": True}),
        ]
        outlet = ShellyOutlet(
            name="lamp-1",
            host="192.168.1.50",
            http_client=client,
        )

        outlet.set_power(True)

        set_call = client.get.call_args_list[0]
        self.assertEqual(
            set_call.args[0],
            "http://192.168.1.50/rpc/Switch.Set",
        )
        self.assertEqual(set_call.kwargs["params"]["on"], "true")
        self.assertEqual(set_call.kwargs["params"]["tag"], "apartment-ai")
        status_call = client.get.call_args_list[1]
        self.assertEqual(
            status_call.args[0],
            "http://192.168.1.50/rpc/Switch.GetStatus",
        )

    def test_rejects_unconfirmed_outlet_state(self):
        client = Mock()
        client.get.side_effect = [
            self.response({"was_on": False}),
            self.response({"id": 0, "output": False}),
        ]
        outlet = ShellyOutlet(
            name="lamp-1",
            host="192.168.1.50",
            http_client=client,
        )

        with self.assertRaisesRegex(ControllerError, "should be on"):
            outlet.set_power(True)

    def test_reports_authentication_failure_with_environment_hint(self):
        client = Mock()
        client.get.return_value = self.response({}, status_code=401)
        outlet = ShellyOutlet(
            name="lamp-1",
            host="192.168.1.50",
            http_client=client,
        )

        with self.assertRaisesRegex(ControllerError, "SHELLY_PASSWORD"):
            outlet.get_power()

    def test_group_updates_every_outlet_as_one_logical_device(self):
        first = Mock(name="first")
        first.name = "lamp-1"
        second = Mock(name="second")
        second.name = "lamp-2"
        group = ShellyOutletGroup([first, second])

        self.assertEqual(
            group.apply("desk_lamp", "off"),
            "Living room lamps turned off.",
        )
        first.set_power.assert_called_once_with(False)
        second.set_power.assert_called_once_with(False)

    def test_group_attempts_every_outlet_and_reports_partial_failure(self):
        first = Mock(name="first")
        first.name = "lamp-1"
        first.set_power.side_effect = ControllerError("lamp-1 is offline")
        second = Mock(name="second")
        second.name = "lamp-2"
        group = ShellyOutletGroup([first, second])

        with self.assertRaisesRegex(ControllerError, "lamp-1 is offline"):
            group.apply("desk_lamp", "on")

        second.set_power.assert_called_once_with(True)

    def test_parses_friendly_name_and_host_override(self):
        self.assertEqual(parse_outlet_target("lamp-1"), ("lamp-1", "lamp-1"))
        self.assertEqual(
            parse_outlet_target("lamp-1=192.168.1.50"),
            ("lamp-1", "192.168.1.50"),
        )


class CommandLineTests(unittest.TestCase):
    def test_listen_enables_continuous_mode(self):
        args = parse_args(["--listen", "--wake-word", "hey apartment"])

        self.assertTrue(args.listen)
        self.assertFalse(args.voice)
        self.assertEqual(args.wake_word, "hey apartment")

    def test_voice_and_listen_modes_are_mutually_exclusive(self):
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            parse_args(["--voice", "--listen"])

    def test_accepts_friendly_system_speaker_with_speech_enabled(self):
        args = parse_args(["--speak", "--speaker", "Living Room Speaker"])

        self.assertEqual(args.speaker, "Living Room Speaker")
        self.assertIsNone(args.speaker_device)

    def test_accepts_weather_location(self):
        args = parse_args(["--weather-location", "Boston, MA"])

        self.assertEqual(args.weather_location, "Boston, MA")

    def test_shelly_outlets_are_the_default_output(self):
        args = parse_args([])

        self.assertFalse(args.gpio)
        self.assertIsNone(args.outlet_hosts)

    @patch.dict("apartment_controller.os.environ", {}, clear=True)
    def test_default_output_builds_the_two_named_outlets(self):
        output = create_device_output(parse_args([]))

        self.assertIsInstance(output, ShellyOutletGroup)
        self.assertEqual(
            [outlet.name for outlet in output.outlets],
            ["lamp-1", "lamp-2"],
        )

    def test_gpio_selects_led_output(self):
        args = parse_args(["--gpio", "--led-pin", "11"])

        self.assertTrue(args.gpio)
        self.assertEqual(args.led_pin, 11)

    def test_accepts_repeated_outlet_overrides(self):
        args = parse_args(
            [
                "--outlet",
                "lamp-1=192.168.1.50",
                "--outlet",
                "lamp-2=192.168.1.51",
            ]
        )

        self.assertEqual(
            args.outlet_hosts,
            ["lamp-1=192.168.1.50", "lamp-2=192.168.1.51"],
        )

    def test_rejects_outlet_overrides_in_gpio_mode(self):
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            parse_args(["--gpio", "--outlet", "lamp-1"])

    @patch("apartment_controller.print_available_speakers")
    def test_speaker_listing_exits_before_hardware_initialization(self, listing):
        self.assertEqual(main(["--list-speakers"]), 0)
        listing.assert_called_once_with()

    @patch("apartment_controller.print_available_microphones")
    def test_microphone_listing_exits_before_hardware_initialization(self, listing):
        self.assertEqual(main(["--list-microphones"]), 0)
        listing.assert_called_once_with()

    @patch("apartment_controller.print_available_microphones")
    @patch("apartment_controller.print_available_speakers")
    def test_combined_audio_listing_exits_before_hardware_initialization(
        self,
        speakers,
        microphones,
    ):
        self.assertEqual(main(["--list-audio"]), 0)
        speakers.assert_called_once_with()
        microphones.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
