from __future__ import annotations

import subprocess
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.llm_client import LLMClient, decode_process_output


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


if __name__ == "__main__":
    unittest.main()
