import json
import subprocess
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from apartment_controller import (
    ControllerError,
    WhisperVoiceInput,
    interpret_command,
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


if __name__ == "__main__":
    unittest.main()
