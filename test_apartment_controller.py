import json
import subprocess
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from apartment_controller import (
    AlsaAudioStream,
    ControllerError,
    WakeWordInput,
    WhisperVoiceInput,
    interpret_command,
    parse_args,
    validate_command,
)


class CommandValidationTests(unittest.TestCase):
    def test_allows_known_lamp_command(self):
        self.assertEqual(
            validate_command({"device": "desk_lamp", "action": "on"}),
            ("desk_lamp", "on"),
        )

    def test_rejects_unknown_device(self):
        with self.assertRaisesRegex(ControllerError, "unknown device"):
            validate_command({"device": "oven", "action": "on"})

    def test_rejects_non_object_json(self):
        with self.assertRaisesRegex(ControllerError, "JSON object"):
            validate_command(["desk_lamp", "on"])

    def test_interpret_command_posts_text_and_returns_valid_command(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {"device": "desk_lamp", "action": "off"}
                        )
                    }
                }
            ]
        }
        client = Mock()
        client.post.return_value = response

        result = interpret_command(
            "turn it off", "http://localhost/test", http_client=client
        )

        self.assertEqual(result, ("desk_lamp", "off"))
        request_body = client.post.call_args.kwargs["json"]
        self.assertEqual(request_body["messages"][-1]["content"], "turn it off")


class VoiceInputTests(unittest.TestCase):
    @patch("apartment_controller.subprocess.run")
    def test_records_then_transcribes(self, run):
        def fake_run(command, **kwargs):
            if Path(command[0]).name == "whisper-cli":
                output_base = Path(command[command.index("--output-file") + 1])
                output_base.with_suffix(".txt").write_text(
                    "turn the desk lamp on\n", encoding="utf-8"
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

        self.assertEqual(transcript, "turn the desk lamp on")
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
            "turn the desk lamp on",
        ]
        stream_factory = Mock(return_value=stream)
        wake_input = WakeWordInput(
            voice_input=voice,
            wake_word="command",
            window_seconds=2,
            stream_factory=stream_factory,
        )

        transcript = wake_input.listen()

        self.assertEqual(transcript, "turn the desk lamp on")
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


class CommandLineTests(unittest.TestCase):
    def test_always_listen_alias_enables_wake_mode(self):
        args = parse_args(["--always-listen", "--wake-word", "hey apartment"])

        self.assertTrue(args.wake_listen)
        self.assertFalse(args.voice)
        self.assertEqual(args.wake_word, "hey apartment")


if __name__ == "__main__":
    unittest.main()
