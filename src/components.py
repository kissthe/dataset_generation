from __future__ import annotations

import re
from typing import Any

from .llm_client import LLMClient
from .models import (
    CaseSpec, EvalCandidateList, EvalOutline, EvalSelection, LifeAnchor, Session,
    SessionPlanList, VerificationResult,
)


def _hard_fact_conflicts(case: CaseSpec, value: Any) -> list[str]:
    """Catch a small set of irreversible contradictions even when QA is disabled."""
    text = str(value)
    conflicts: list[str] = []
    for entity in case.deceased_entities():
        escaped = re.escape(entity)
        direct_interaction = re.search(
            rf"(?:{escaped}.{{0,24}}(?:下次|以后|明天|准备|计划|打算).{{0,24}}"
            rf"(?:探望|看望|拜访|见面|约好|打电话|通话|联系)"
            rf"|(?:下次|以后|明天|准备|计划|打算).{{0,24}}"
            rf"(?:探望|看望|拜访|见面|约好|打电话|通话|联系).{{0,24}}{escaped}"
            rf"|{escaped}.{{0,12}}(?:最近怎么样|身体怎么样|今天还好吗))",
            text,
        )
        implied_visit = re.search(
            r"(?:下次|准备|计划|打算).{0,18}(?:探望|看望|拜访)|(?:探望|看望|拜访)前"
            r"|我来看看你|约好我再去",
            text,
        )
        deceased_context = entity in text or ("最后一次通话" in text and "遗憾" in text)
        if direct_interaction or (deceased_context and implied_visit):
            conflicts.append(f"{entity}已经去世，但内容安排了后续现实互动")
    return conflicts


class SessionPlanner:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    @staticmethod
    def build_payload(case: CaseSpec, min_rounds: int, max_rounds: int,
                      session_count: int, total_session_count: int, id_prefix: str,
                      start_index: int = 0, prior_plans: list[dict] | None = None,
                      life_anchor: LifeAnchor | None = None) -> dict:
        return {
            "planner_brief": case.planner_brief(),
            "immutable_facts": case.immutable_facts(),
            "established_life_anchor": life_anchor.model_dump() if life_anchor else None,
            "planning_window": {
                "batch_size": session_count,
                "total_session_count": total_session_count,
                "start_index": start_index,
                "id_prefix": id_prefix,
            },
            "constraints": {"min_rounds": min_rounds, "max_rounds": max_rounds},
            "prior_plans": prior_plans or [],
        }

    def run(self, case: CaseSpec, min_rounds: int, max_rounds: int,
            session_count: int, total_session_count: int, id_prefix: str,
            start_index: int = 0, prior_plans: list[dict] | None = None,
            life_anchor: LifeAnchor | None = None) -> SessionPlanList:
        payload = self.build_payload(
            case, min_rounds, max_rounds, session_count, total_session_count,
            id_prefix, start_index, prior_plans, life_anchor,
        )
        generated = None
        for attempt in range(2):
            generated = self.llm.generate("session_planner", payload, SessionPlanList)
            current_batch = generated.plans[:session_count]
            conflicts = _hard_fact_conflicts(
                case, {"plans": [plan.model_dump() for plan in current_batch]}
            )
            if life_anchor is None and generated.life_anchor is None:
                conflicts.append("首批计划没有生成 life_anchor")
            elif life_anchor is not None and generated.life_anchor != life_anchor:
                conflicts.append("后续批次改写了 established_life_anchor")
            if not conflicts:
                break
            payload = {
                **payload,
                "consistency_feedback": {
                    "attempt": attempt + 1,
                    "conflicts": conflicts,
                    "instruction": "重写冲突计划；immutable_facts 高于故事节拍，不得反转硬事实。",
                },
            }
        else:
            raise ValueError(f"planner violated immutable facts after retry: {conflicts}")
        assert generated is not None
        if case.session_outlines:
            expected_ids = [outline.session_id for outline in case.session_outlines[:session_count]]
        else:
            expected_ids = [f"{id_prefix}-S{index + 1:02d}" for index in range(start_index, start_index + session_count)]

        if len(generated.plans) < session_count:
            raise ValueError(
                f"planner returned only {len(generated.plans)} plans for a batch of {session_count}"
            )

        exact_by_id = {plan.session_id: plan for plan in generated.plans}
        if all(session_id in exact_by_id for session_id in expected_ids):
            selected = [exact_by_id[session_id] for session_id in expected_ids]
        else:
            by_number = {}
            duplicate_numbers = set()
            for plan in generated.plans:
                match = re.search(r"S[-_]?0*(\d+)$", plan.session_id, flags=re.IGNORECASE)
                if not match:
                    continue
                number = int(match.group(1))
                if number in by_number:
                    duplicate_numbers.add(number)
                by_number[number] = plan
            expected_numbers = list(range(start_index + 1, start_index + session_count + 1))
            if not duplicate_numbers.intersection(expected_numbers) and all(
                number in by_number for number in expected_numbers
            ):
                selected = [by_number[number] for number in expected_numbers]
            else:
                selected = generated.plans[:session_count]

        normalized = [
            plan.model_copy(update={"session_id": expected_id})
            for plan, expected_id in zip(selected, expected_ids)
        ]
        actual_ids = [plan.session_id for plan in selected]
        if len(generated.plans) != session_count or actual_ids != expected_ids:
            print(
                f"planner returned {len(generated.plans)} plans with non-canonical batch IDs; "
                f"normalized to {expected_ids}",
                flush=True,
            )
        return SessionPlanList(
            plans=normalized,
            life_anchor=life_anchor or generated.life_anchor,
        )


class SessionWriter:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def run(self, case: CaseSpec, plan: dict, recent_sessions: list[Session],
            story_so_far: list[dict] | None = None,
            life_anchor: LifeAnchor | None = None) -> Session:
        payload = {
            "case_spec": case.model_dump(),
            "immutable_facts": case.immutable_facts(),
            "life_anchor": life_anchor.model_dump() if life_anchor else None,
            "session_plan": plan,
            "story_so_far": story_so_far or [],
            "recent_history": [s.model_dump() for s in recent_sessions],
        }
        session = None
        for attempt in range(2):
            session = self.llm.generate("session_writer", payload, Session)
            conflicts = _hard_fact_conflicts(case, session.model_dump())
            if not conflicts:
                return session
            payload = {
                **payload,
                "consistency_feedback": {
                    "attempt": attempt + 1,
                    "conflicts": conflicts,
                    "instruction": "忽略 SessionPlan 中与 immutable_facts 冲突的部分，并重写对话。",
                },
            }
        raise ValueError(f"writer violated immutable facts after retry: {conflicts}")


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
