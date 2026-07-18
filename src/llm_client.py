from __future__ import annotations

import json
import subprocess
import base64
import locale
from pathlib import Path
from typing import TypeVar

from openai import OpenAI
from pydantic import BaseModel

from .config import AppConfig
from .models import CallRecord

T = TypeVar("T", bound=BaseModel)


def decode_process_output(value: bytes | str | None) -> str:
    """Decode PowerShell pipes without assuming the Windows error code page."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    encodings = ("utf-8-sig", locale.getpreferredencoding(False), "gb18030")
    for encoding in dict.fromkeys(encodings):
        try:
            return value.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    return value.decode("utf-8", errors="replace")


class LLMClient:
    def __init__(self, config: AppConfig) -> None:
        kwargs = {"api_key": config.api_key}
        if config.base_url:
            kwargs["base_url"] = config.base_url
        self.client = OpenAI(**kwargs)
        self.config = config
        self.records: list[CallRecord] = []

    def generate(self, component: str, payload: dict, output_model: type[T]) -> T:
        cc = self.config.components[component]
        prompt_path = self.config.root / "prompts" / f"{component}.txt"
        system_prompt = prompt_path.read_text(encoding="utf-8")
        last_error: Exception | None = None
        for attempt in range(1, self.config.generation.max_retries + 1):
            try:
                request = {
                    "model": cc.model,
                    "temperature": cc.temperature,
                    "seed": self.config.generation.seed,
                    "max_completion_tokens": cc.max_completion_tokens,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                    ],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": output_model.__name__.lower(),
                            "strict": True,
                            "schema": output_model.model_json_schema(),
                        },
                    },
                }
                if self.config.transport == "powershell":
                    content = self._generate_via_powershell(request)
                else:
                    response = self.client.chat.completions.create(**request)
                    content = response.choices[0].message.content
                if not content:
                    raise ValueError("model returned empty content")
                parsed = output_model.model_validate_json(content)
                self.records.append(CallRecord(component=component, model=cc.model, attempt=attempt, status="success"))
                return parsed
            except Exception as exc:  # retry API, decoding, and schema failures uniformly
                last_error = exc
                self.records.append(CallRecord(component=component, model=cc.model, attempt=attempt, status="error", error=str(exc)[:500]))
        raise RuntimeError(f"{component} failed after retries: {last_error}")

    def _generate_via_powershell(self, request: dict) -> str:
        script = self.config.root / "scripts" / "invoke_openai.ps1"
        encoded_request = base64.b64encode(
            json.dumps(request, ensure_ascii=False).encode("utf-8")
        )
        completed = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(script)],
            input=encoded_request,
            capture_output=True,
            timeout=300,
            check=False,
        )
        stdout = decode_process_output(completed.stdout).strip()
        stderr = decode_process_output(completed.stderr).strip()
        if completed.returncode != 0:
            error = stderr or stdout or f"process exited with code {completed.returncode}"
            raise RuntimeError(f"PowerShell transport failed: {error[:500]}")
        return stdout
