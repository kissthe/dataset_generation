from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


@dataclass(frozen=True)
class ComponentConfig:
    model: str
    temperature: float
    max_completion_tokens: int


@dataclass(frozen=True)
class GenerationConfig:
    session_count: int
    planner_batch_size: int
    min_rounds: int
    max_rounds: int
    context_sessions: int
    seed: int
    max_retries: int
    max_revision_cycles: int
    run_eval: bool
    stop_after_planning: bool = False
    run_eval_examples: bool = False


@dataclass(frozen=True)
class BlueprintConstraints:
    """Deterministic coverage knobs; `none` sessions are the remaining slots."""

    encode_association_count: int = 1
    triggered_recall_count: int = 1
    memory_update_count: int = 1
    control_count: int = 1
    required_cue_types: tuple[str, ...] = ("object", "scene", "utterance")
    eval_triggered_count: int = 2
    eval_insufficient_evidence_count: int = 2
    eval_not_triggered_count: int = 2


@dataclass(frozen=True)
class ValidationConfig:
    """Optional post-generation validation stages.

    Structured model parsing is always enforced by the LLM client. These flags
    only control the extra validation/revision stages after SessionWriter.
    """

    structure: bool = False
    semantic: bool = False
    naturalness: bool = False
    qa: bool = False


@dataclass(frozen=True)
class AppConfig:
    root: Path
    dataset_id_prefix: str
    transport: str
    generation: GenerationConfig
    blueprint_constraints: BlueprintConstraints
    validation: ValidationConfig
    components: dict[str, ComponentConfig]
    api_key: str
    base_url: str | None

    @classmethod
    def load(cls, path: Path) -> "AppConfig":
        raw = json.loads(path.read_text(encoding="utf-8"))
        api_key = os.getenv("openai_api_key") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing openai_api_key/OPENAI_API_KEY environment variable")
        base_url = os.getenv("base_url") or os.getenv("BASE_URL") or None
        # OpenAI-compatible gateways normally expose their API below /v1. A host-only
        # environment value is common, so normalize it without changing explicit paths.
        if base_url:
            parsed = urlsplit(base_url)
            if parsed.path in ("", "/"):
                base_url = urlunsplit((parsed.scheme, parsed.netloc, "/v1", parsed.query, parsed.fragment))
        generation_raw = dict(raw["generation"])
        # Older snapshots stored a single eval_count. Label-specific Blueprint
        # constraints now own this value.
        generation_raw.pop("eval_count", None)
        blueprint_raw = dict(raw.get("blueprint_constraints", {}))
        if "required_cue_types" in blueprint_raw:
            blueprint_raw["required_cue_types"] = tuple(blueprint_raw["required_cue_types"])
        return cls(
            root=path.parent.resolve(),
            dataset_id_prefix=raw["dataset_id_prefix"],
            transport=raw.get("transport", "openai_sdk"),
            generation=GenerationConfig(**generation_raw),
            blueprint_constraints=BlueprintConstraints(**blueprint_raw),
            validation=ValidationConfig(**raw.get("validation", {})),
            components={k: ComponentConfig(**v) for k, v in raw["components"].items()},
            api_key=api_key,
            base_url=base_url,
        )
