from __future__ import annotations

import json
import subprocess
import base64
import locale
import time
from pathlib import Path
from typing import TypeVar

from openai import OpenAI
from pydantic import BaseModel

from .config import AppConfig
from .models import CallRecord

T = TypeVar("T", bound=BaseModel)

POWERSHELL_CONNECTION_ERROR_MARKERS = (
    "underlying connection was closed",
    "connection was closed unexpectedly",
    "connection reset",
    "connection aborted",
    "connection error",
    "could not create ssl/tls secure channel",
    "ssl connection could not be established",
    "tls",
    "unable to connect to the remote server",
    "无法连接到远程服务器",
    "timed out",
    "timeout",
    "name resolution",
    "remote name could not be resolved",
    "winerror 10013",
    "winerror 10054",
    "winerror 10060",
)


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


def describe_exception(error: BaseException | str, *, max_depth: int = 6) -> str:
    """Preserve useful SDK/httpx/Windows causes instead of only `Connection error`."""
    if isinstance(error, str):
        return error
    parts: list[str] = []
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and len(parts) < max_depth and id(current) not in seen:
        seen.add(id(current))
        message = str(current).strip() or repr(current)
        parts.append(f"{type(current).__name__}: {message}")
        current = current.__cause__ or current.__context__
    return " <- ".join(parts)


def is_powershell_connection_error(error: BaseException | str) -> bool:
    """Return whether a transport failed before receiving a usable HTTP response."""
    message = describe_exception(error).lower()
    return any(marker in message for marker in POWERSHELL_CONNECTION_ERROR_MARKERS)


class LLMClient:
    def __init__(self, config: AppConfig) -> None:
        kwargs = {"api_key": config.api_key}
        if config.base_url:
            kwargs["base_url"] = config.base_url
        self.client = OpenAI(**kwargs)
        self.config = config
        self.records: list[CallRecord] = []
        self._powershell_unavailable = False

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
                if self.config.transport == "powershell" and not self._powershell_unavailable:
                    try:
                        content = self._generate_via_powershell(request)
                    except RuntimeError as exc:
                        if not is_powershell_connection_error(exc):
                            raise
                        # Windows PowerShell 5.1 can fail during TLS negotiation with
                        # otherwise healthy OpenAI-compatible gateways. Once that is
                        # observed, use the SDK for this and all later calls in the run.
                        self._powershell_unavailable = True
                        self.records.append(
                            CallRecord(
                                component=component,
                                model=cc.model,
                                attempt=attempt,
                                status="error",
                                error=f"PowerShell connection failed; switched to openai_sdk: {exc}"[:500],
                            )
                        )
                        content = self._generate_via_sdk(request)
                else:
                    content = self._generate_via_sdk(request)
                if not content:
                    raise ValueError("model returned empty content")
                parsed = output_model.model_validate_json(content)
                self.records.append(CallRecord(component=component, model=cc.model, attempt=attempt, status="success"))
                return parsed
            except Exception as exc:  # retry API, decoding, and schema failures uniformly
                last_error = exc
                error_detail = describe_exception(exc)
                self.records.append(
                    CallRecord(
                        component=component,
                        model=cc.model,
                        attempt=attempt,
                        status="error",
                        error=error_detail[:500],
                    )
                )
                if (
                    attempt < self.config.generation.max_retries
                    and is_powershell_connection_error(exc)
                ):
                    delay = min(2 ** (attempt - 1), 8)
                    print(
                        f"retrying {component} after connection error; "
                        f"next attempt {attempt + 1}/{self.config.generation.max_retries} "
                        f"in {delay}s",
                        flush=True,
                    )
                    time.sleep(delay)
        detail = describe_exception(last_error) if last_error else "unknown error"
        raise RuntimeError(f"{component} failed after retries: {detail}") from last_error

    def _generate_via_sdk(self, request: dict) -> str | None:
        # Long structured Blueprint/Plan generations can remain silent for over a
        # minute. Some OpenAI-compatible gateways close such non-streaming requests
        # before sending response headers. Streaming keeps the connection active and
        # the assembled text is validated by the exact same Pydantic schema below.
        stream = self.client.chat.completions.create(**request, stream=True)
        content_parts: list[str] = []
        for event in stream:
            if not event.choices:
                continue
            piece = event.choices[0].delta.content
            if piece:
                content_parts.append(piece)
        return "".join(content_parts)

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
