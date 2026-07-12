from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CharacterProfile(StrictModel):
    user_id: str
    name: str
    identity: str
    daily_scenes: list[str]
    conversation_style: str
    interests: list[str]


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
    exact: list[str]
    related: list[str]
    non_triggering: list[str]


class CaseSpec(StrictModel):
    case_id: str
    character_profile: CharacterProfile
    core_emotional_event: str
    forbidden_facts: list[str]
    cues: CueSpec
    session_outlines: list[SessionOutline]
    eval_outlines: list[EvalOutline]


class SessionPlan(StrictModel):
    session_id: str
    date: str
    topic: str
    story_beat: str
    outline_function: str
    round_count: int = Field(ge=1, le=20)


class SessionPlanList(StrictModel):
    plans: list[SessionPlan]


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
