from __future__ import annotations

from .llm_client import LLMClient
from .models import (
    CaseSpec, EvalCandidateList, EvalOutline, EvalSelection, Session,
    SessionPlanList, VerificationResult,
)


class SessionPlanner:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def run(self, case: CaseSpec, min_rounds: int, max_rounds: int,
            session_count: int, id_prefix: str, start_index: int = 0,
            prior_plans: list[dict] | None = None) -> SessionPlanList:
        return self.llm.generate("session_planner", {
            "case_spec": case.model_dump(),
            "session_count": session_count,
            "id_prefix": id_prefix,
            "start_index": start_index,
            "constraints": {"min_rounds": min_rounds, "max_rounds": max_rounds},
            "prior_plans": prior_plans or [],
        }, SessionPlanList)


class SessionWriter:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def run(self, case: CaseSpec, plan: dict, recent_sessions: list[Session]) -> Session:
        return self.llm.generate("session_writer", {
            "case_spec": case.model_dump(),
            "session_plan": plan,
            "recent_history": [s.model_dump() for s in recent_sessions],
        }, Session)


class SessionVerifier:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def run(self, case: CaseSpec, plan: dict, session: Session, recent_sessions: list[Session]) -> VerificationResult:
        return self.llm.generate("session_verifier", {
            "case_spec": case.model_dump(), "session_plan": plan,
            "session": session.model_dump(),
            "recent_history": [s.model_dump() for s in recent_sessions],
        }, VerificationResult)


class SessionReviser:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def run(self, case: CaseSpec, plan: dict, session: Session, issues: VerificationResult) -> Session:
        return self.llm.generate("session_reviser", {
            "case_spec": case.model_dump(), "session_plan": plan,
            "session": session.model_dump(), "issues": issues.model_dump(),
        }, Session)


class NaturalnessChecker:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def run(self, case: CaseSpec, session: Session) -> VerificationResult:
        return self.llm.generate("naturalness_checker", {
            "conversation_style": case.character_profile.conversation_style,
            "session": session.model_dump(),
        }, VerificationResult)


class EvalGenerator:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def run(self, case: CaseSpec, sessions: list[Session], outline: EvalOutline) -> EvalCandidateList:
        visible = []
        for session in sessions:
            visible.append(session)
            if session.session_id == outline.history_cutoff:
                break
        return self.llm.generate("eval_generator", {
            "case_spec": case.model_dump(),
            "eval_outline": outline.model_dump(),
            "visible_sessions": [s.model_dump() for s in visible],
            "candidate_count": 3,
        }, EvalCandidateList)


class EvalVerifier:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def run(self, case: CaseSpec, sessions: list[Session], outline: EvalOutline, candidates: EvalCandidateList) -> EvalSelection:
        return self.llm.generate("eval_verifier", {
            "case_spec": case.model_dump(),
            "eval_outline": outline.model_dump(),
            "sessions": [s.model_dump() for s in sessions],
            "candidates": candidates.model_dump(),
        }, EvalSelection)
