from __future__ import annotations

import subprocess
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from pydantic import BaseModel

from src.llm_client import (
    LLMClient,
    decode_process_output,
    describe_exception,
    is_powershell_connection_error,
)


class ExampleOutput(BaseModel):
    value: str


class PowerShellTransportTests(unittest.TestCase):
    def test_decodes_utf8_model_output(self) -> None:
        self.assertEqual(decode_process_output("中文故事线".encode("utf-8")), "中文故事线")

    def test_decodes_windows_chinese_error_output(self) -> None:
        message = "无法连接到远程服务器"
        self.assertEqual(decode_process_output(message.encode("gb18030")), message)

    def test_transport_preserves_real_error_instead_of_none_strip(self) -> None:
        client = LLMClient.__new__(LLMClient)
        client.config = SimpleNamespace(root=Path("."))
        completed = subprocess.CompletedProcess(
            args=[], returncode=1, stdout=b"", stderr="无法连接到远程服务器".encode("gb18030")
        )

        with patch("src.llm_client.subprocess.run", return_value=completed):
            with self.assertRaisesRegex(RuntimeError, "无法连接到远程服务器"):
                client._generate_via_powershell({"messages": []})

    def test_transport_decodes_successful_utf8_response(self) -> None:
        client = LLMClient.__new__(LLMClient)
        client.config = SimpleNamespace(root=Path("."))
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="{\"plans\":[]}".encode("utf-8"), stderr=b""
        )

        with patch("src.llm_client.subprocess.run", return_value=completed):
            self.assertEqual(client._generate_via_powershell({"messages": []}), '{"plans":[]}')

    def test_sdk_transport_assembles_streamed_content(self) -> None:
        create = Mock(
            return_value=[
                SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(content='{"value"'))]
                ),
                SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(content=':"ok"}'))]
                ),
                SimpleNamespace(choices=[]),
            ]
        )
        client = LLMClient.__new__(LLMClient)
        client.client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )

        content = client._generate_via_sdk({"model": "test-model", "messages": []})

        self.assertEqual(content, '{"value":"ok"}')
        create.assert_called_once_with(model="test-model", messages=[], stream=True)

    def test_classifies_connection_close_for_sdk_fallback(self) -> None:
        self.assertTrue(
            is_powershell_connection_error(
                "PowerShell transport failed: The underlying connection was closed unexpectedly"
            )
        )
        self.assertFalse(
            is_powershell_connection_error("PowerShell transport failed: HTTP 401 Unauthorized")
        )

    def test_describe_exception_includes_nested_windows_cause(self) -> None:
        try:
            try:
                raise PermissionError(13, "socket access denied")
            except PermissionError as cause:
                raise ConnectionError("Connection error") from cause
        except ConnectionError as exc:
            detail = describe_exception(exc)

        self.assertIn("ConnectionError: Connection error", detail)
        self.assertIn("PermissionError", detail)
        self.assertIn("socket access denied", detail)

    def test_generate_falls_back_to_sdk_after_powershell_connection_error(self) -> None:
        client = LLMClient.__new__(LLMClient)
        client.config = SimpleNamespace(
            root=Path(".").resolve(),
            transport="powershell",
            generation=SimpleNamespace(max_retries=1, seed=42),
            components={
                "dataset_blueprint_planner": SimpleNamespace(
                    model="test-model", temperature=0.1, max_completion_tokens=128
                )
            },
        )
        client.records = []
        client._powershell_unavailable = False

        with (
            patch.object(
                client,
                "_generate_via_powershell",
                side_effect=RuntimeError(
                    "PowerShell transport failed: The underlying connection was closed unexpectedly"
                ),
            ) as powershell_generate,
            patch.object(client, "_generate_via_sdk", return_value='{"value":"ok"}') as sdk_generate,
        ):
            result = client.generate("dataset_blueprint_planner", {}, ExampleOutput)

        self.assertEqual(result.value, "ok")
        self.assertTrue(client._powershell_unavailable)
        powershell_generate.assert_called_once()
        sdk_generate.assert_called_once()
        self.assertEqual([record.status for record in client.records], ["error", "success"])

    def test_generate_does_not_fallback_for_http_api_error(self) -> None:
        client = LLMClient.__new__(LLMClient)
        client.config = SimpleNamespace(
            root=Path(".").resolve(),
            transport="powershell",
            generation=SimpleNamespace(max_retries=1, seed=42),
            components={
                "dataset_blueprint_planner": SimpleNamespace(
                    model="test-model", temperature=0.1, max_completion_tokens=128
                )
            },
        )
        client.records = []
        client._powershell_unavailable = False

        with (
            patch.object(
                client,
                "_generate_via_powershell",
                side_effect=RuntimeError("PowerShell transport failed: HTTP 401 Unauthorized"),
            ),
            patch.object(client, "_generate_via_sdk") as sdk_generate,
        ):
            with self.assertRaisesRegex(RuntimeError, "HTTP 401 Unauthorized"):
                client.generate("dataset_blueprint_planner", {}, ExampleOutput)

        self.assertFalse(client._powershell_unavailable)
        sdk_generate.assert_not_called()

    def test_sdk_connection_retry_uses_backoff_then_succeeds(self) -> None:
        client = LLMClient.__new__(LLMClient)
        client.config = SimpleNamespace(
            root=Path(".").resolve(),
            transport="openai_sdk",
            generation=SimpleNamespace(max_retries=2, seed=42),
            components={
                "dataset_blueprint_planner": SimpleNamespace(
                    model="test-model", temperature=0.1, max_completion_tokens=128
                )
            },
        )
        client.records = []
        client._powershell_unavailable = False

        with (
            patch.object(
                client,
                "_generate_via_sdk",
                side_effect=[ConnectionError("Connection error"), '{"value":"ok"}'],
            ) as sdk_generate,
            patch("src.llm_client.time.sleep") as sleep,
        ):
            result = client.generate("dataset_blueprint_planner", {}, ExampleOutput)

        self.assertEqual(result.value, "ok")
        self.assertEqual(sdk_generate.call_count, 2)
        sleep.assert_called_once_with(1)
        self.assertEqual([record.status for record in client.records], ["error", "success"])


if __name__ == "__main__":
    unittest.main()
