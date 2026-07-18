from __future__ import annotations

import hashlib
import re
from typing import Any
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


DEFAULT_ASSISTANT_PERSONA = "用户的普通朋友"

DECEASED_RELATION_TERMS = (
    "外祖父", "外祖母", "祖父", "祖母", "爷爷", "奶奶", "姥爷", "姥姥",
    "父亲", "母亲", "爸爸", "妈妈", "哥哥", "姐姐", "弟弟", "妹妹",
    "丈夫", "妻子", "伴侣", "朋友", "老师", "宠物",
)


def _stable_slug(value: str) -> str:
    """Create a deterministic ASCII identifier while keeping readable names."""
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or hashlib.sha256(value.encode("utf-8")).hexdigest()[:10]


class CharacterProfile(StrictModel):
    user_id: str = ""
    name: str
    identity: str = ""
    daily_scenes: list[str] = Field(default_factory=list)
    conversation_style: str = "自然、口语化，像日常聊天"
    interests: list[str] = Field(default_factory=list)
    persona_summary: str = ""
    traits: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def fill_user_id(self) -> "CharacterProfile":
        if not self.user_id:
            self.user_id = f"user-{_stable_slug(self.name)}"
        return self


class SessionOutline(StrictModel):
    session_id: str
    core_content: str
    function: str


class EvalOutline(StrictModel):
    outline_id: str
    description: str
    target_label: Literal["triggered", "not_triggered", "insufficient_evidence"]
    target_emotion: Literal["fear", "anxiety", "sadness", "anger", "comfort", "nostalgia", "happiness", "neutral"]
    history_cutoff: str


class CueSpec(StrictModel):
    exact: list[str] = []
    related: list[str] = []
    non_triggering: list[str] = []


class CaseSpec(StrictModel):
    case_id: str = ""
    character_profile: CharacterProfile
    core_emotional_event: str
    forbidden_facts: list[str] = Field(default_factory=list)
    cues: CueSpec = Field(default_factory=CueSpec)
    session_outlines: list[SessionOutline] = Field(default_factory=list)
    eval_outlines: list[EvalOutline] = Field(default_factory=list)
    assistant_persona: str = DEFAULT_ASSISTANT_PERSONA

    @model_validator(mode="before")
    @classmethod
    def accept_minimal_spec(cls, value: Any) -> Any:
        """Accept {name, core_emotional_event} as the minimal public input."""
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        flat_name = normalized.pop("name", None)
        profile = normalized.get("character_profile")
        if profile is None:
            if not flat_name:
                raise ValueError("spec requires name or character_profile.name")
            normalized["character_profile"] = {"name": flat_name}
        elif flat_name:
            profile = dict(profile)
            if profile.get("name") and profile["name"] != flat_name:
                raise ValueError("name conflicts with character_profile.name")
            profile["name"] = flat_name
            normalized["character_profile"] = profile

        if not normalized.get("case_id"):
            profile_name = normalized["character_profile"].get("name", "case")
            normalized["case_id"] = f"case-{_stable_slug(profile_name)}"
        return normalized

    def planner_brief(self) -> dict[str, Any]:
        """Return only the stable story seeds that SessionPlanner can use."""
        profile = self.character_profile
        brief: dict[str, Any] = {
            "name": profile.name,
            "core_emotional_event": self.core_emotional_event,
        }
        optional_values = {
            "identity": profile.identity,
            "daily_scenes": profile.daily_scenes,
            "conversation_style": profile.conversation_style,
            "interests": profile.interests,
            "persona_summary": profile.persona_summary,
            "traits": profile.traits,
            "assistant_persona": self.assistant_persona,
            "forbidden_facts": self.forbidden_facts,
            "session_outlines": [outline.model_dump() for outline in self.session_outlines],
        }
        for key, item in optional_values.items():
            if item and item != DEFAULT_ASSISTANT_PERSONA and item != "自然、口语化，像日常聊天":
                brief[key] = item
        return brief

    def deceased_entities(self) -> list[str]:
        """Extract explicit deceased relations for the always-on fact firewall."""
        event = self.core_emotional_event
        death_terms = r"(?:去世|离世|过世|死亡|病逝|已经不在了)"
        return [
            relation for relation in DECEASED_RELATION_TERMS
            if re.search(rf"{re.escape(relation)}.{{0,6}}{death_terms}", event)
        ]

    def immutable_facts(self) -> list[str]:
        """Return facts whose direct implications may not be reversed by a plan."""
        facts = [f"核心事件原文（不可写成相反事实）：{self.core_emotional_event}"]
        for entity in self.deceased_entities():
            facts.append(
                f"{entity}已经去世；后续不得安排与其探望、见面、通话、约时间，"
                "也不得写成对方仍能回复消息。回忆、悼念或祭扫不属于现实互动。"
            )
        facts.extend(f"禁止新增或断言：{fact}" for fact in self.forbidden_facts)
        return facts


class LifeAnchor(StrictModel):
    """Planner-created private scaffold for a coherent ordinary life."""

    identity: str
    recurring_scenes: list[str]
    interests: list[str]
    ongoing_threads: list[str]


class SessionPlan(StrictModel):
    session_id: str
    date: str
    topic: str
    story_beat: str
    outline_function: str
    round_count: int = Field(ge=1, le=20)
    session_type: Literal["daily_life", "core_echo", "core_event"]
    scene: str
    user_intent: str
    continuity_hook: str
    life_thread: str
    thread_progress: str
    interaction_mode: Literal[
        "share", "vent", "ask_opinion", "small_decision", "ask_advice", "result_update"
    ]

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_plan(cls, value: Any) -> Any:
        """Keep old checkpoints readable while requiring richer new LLM output."""
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        normalized.setdefault("session_type", "daily_life")
        normalized.setdefault("scene", "")
        normalized.setdefault("user_intent", "")
        normalized.setdefault("continuity_hook", "")
        normalized.setdefault("life_thread", "one_off")
        normalized.setdefault("thread_progress", "")
        normalized.setdefault("interaction_mode", "share")
        return normalized


class SessionPlanList(StrictModel):
    plans: list[SessionPlan]
    life_anchor: LifeAnchor | None

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_plan_list(cls, value: Any) -> Any:
        """Old saved plan files predate the internal life anchor."""
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        normalized.setdefault("life_anchor", None)
        return normalized


class Turn(StrictModel):
    turn_id: str
    round_id: str
    speaker: Literal["user", "assistant"]
    text: str
    image_id: list[str]
    image_dir: str
    image_caption: list[str]


class Session(StrictModel):
    session_id: str
    topic: str
    date: str
    turns: list[Turn]
    summary: str


class VerificationIssue(StrictModel):
    turn_id: str
    type: str
    description: str


class VerificationResult(StrictModel):
    result: Literal["pass", "revise"]
    issues: list[VerificationIssue]


class ImageRef(StrictModel):
    image_id: str
    image_path: str
    image_caption: str


class CueOption(StrictModel):
    cue_id: str
    name: str


class CurrentInput(StrictModel):
    input_type: Literal["text", "image", "text_image"]
    text: str
    image_refs: list[ImageRef]
    cue_type: str
    cue_options: list[CueOption]


class Gold(StrictModel):
    trigger_label: Literal["triggered", "not_triggered", "insufficient_evidence"]
    trigger_cue_id: str
    evidence_turn_ids: list[str]
    current_emotion: Literal["fear", "anxiety", "sadness", "anger", "comfort", "nostalgia", "happiness", "neutral"]

    @model_validator(mode="after")
    def enforce_empty_rules(self) -> "Gold":
        if self.trigger_label == "triggered":
            if self.trigger_cue_id == "none" or not self.evidence_turn_ids:
                raise ValueError("triggered gold requires cue and evidence")
        elif self.trigger_cue_id != "none" or self.evidence_turn_ids:
            raise ValueError("non-triggered gold requires cue=none and empty evidence")
        return self


class EvalSample(StrictModel):
    sample_id: str
    eval_user_id: str
    current_input: CurrentInput
    gold: Gold


class EvalCandidate(StrictModel):
    current_input: CurrentInput
    gold: Gold
    history_cutoff: str


class EvalCandidateList(StrictModel):
    candidates: list[EvalCandidate]


class EvalSelection(StrictModel):
    reject_all: bool
    selected_index: int
    issues: list[str]


class Benchmark(StrictModel):
    dataset_id: str
    character_profile: CharacterProfile
    dialogues: list[dict]
    eval_samples: list[EvalSample]


class CallRecord(StrictModel):
    component: str
    model: str
    attempt: int
    status: Literal["success", "error"]
    error: str | None = None


class QAReport(StrictModel):
    dataset_id: str
    passed: bool
    checks: dict[str, bool]
    errors: list[str]
    call_records: list[CallRecord]
