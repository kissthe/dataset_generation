from __future__ import annotations

import json
from pathlib import Path

from .components import NaturalnessChecker, SessionPlanner, SessionReviser, SessionVerifier, SessionWriter
from .config import AppConfig
from .llm_client import LLMClient
from .models import Benchmark, CaseSpec, Session, VerificationIssue, VerificationResult
from .qa import validate_sessions


class GenerationPipeline:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.llm = LLMClient(config)
        self.planner = SessionPlanner(self.llm)
        self.writer = SessionWriter(self.llm)
        self.verifier = SessionVerifier(self.llm)
        self.reviser = SessionReviser(self.llm)
        self.naturalness = NaturalnessChecker(self.llm)

    def run(self, case_path: Path, output_dir: Path) -> tuple[Path, Path]:
        case = CaseSpec.model_validate_json(case_path.read_text(encoding="utf-8"))
        cfg = self.config.generation
        expected = [x.session_id for x in case.session_outlines[:cfg.session_count]]
        output_dir.mkdir(parents=True, exist_ok=True)
        plan_path = output_dir / "session_plans.json"
        if plan_path.exists():
            from .models import SessionPlanList
            plans = SessionPlanList.model_validate_json(plan_path.read_text(encoding="utf-8"))
            print(f"resumed {len(plans.plans)} saved plans", flush=True)
        else:
            all_plans = []
            outlines = case.session_outlines[:cfg.session_count]
            for start in range(0, len(outlines), cfg.planner_batch_size):
                batch_case = case.model_copy(update={
                    "session_outlines": outlines[start:start + cfg.planner_batch_size],
                    "eval_outlines": [],
                })
                batch = self.planner.run(
                    batch_case, cfg.min_rounds, cfg.max_rounds,
                    prior_plans=[p.model_dump() for p in all_plans],
                )
                all_plans.extend(batch.plans)
                print(f"planned {len(all_plans)}/{len(outlines)} sessions", flush=True)
            from .models import SessionPlanList
            plans = SessionPlanList(plans=all_plans)
        plans.plans = [p for p in plans.plans if p.session_id in expected]
        plans.plans.sort(key=lambda p: expected.index(p.session_id))
        if [p.session_id for p in plans.plans] != expected:
            raise ValueError(f"planner did not return the expected session IDs; got {[p.session_id for p in plans.plans]}")

        (output_dir / "session_plans.json").write_text(plans.model_dump_json(indent=2), encoding="utf-8")
        text_plan = "\n".join(f"{p.session_id} | {p.date} | {p.topic} | {p.round_count} rounds | {p.story_beat}" for p in plans.plans)
        (output_dir / "session_plans.txt").write_text(text_plan, encoding="utf-8")

        checkpoint_path = output_dir / "checkpoint_sessions.json"
        sessions: list[Session] = []
        if checkpoint_path.exists():
            saved = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            sessions = [Session.model_validate(item) for item in saved]
            if [s.session_id for s in sessions] != expected[:len(sessions)]:
                raise ValueError("checkpoint sessions are not a valid prefix of the case")
            print(f"resumed {len(sessions)}/{len(plans.plans)} completed sessions", flush=True)
        for plan in plans.plans[len(sessions):]:
            print(f"generating {plan.session_id}...", flush=True)
            recent = sessions[-cfg.context_sessions:]
            session = self.writer.run(case, plan.model_dump(), recent)
            for cycle in range(cfg.max_revision_cycles):
                structural = self._structural_verdict(session, plan.round_count)
                verdict = structural if structural.result == "revise" else self.verifier.run(
                    case, plan.model_dump(), session, recent
                )
                if verdict.result == "pass":
                    break
                session = self.reviser.run(case, plan.model_dump(), session, verdict)
            final_structure = self._structural_verdict(session, plan.round_count)
            if final_structure.result != "pass":
                raise RuntimeError(f"{plan.session_id} failed deterministic structure checks: {final_structure.model_dump()}")
            natural = self.naturalness.run(case, session)
            if natural.result == "revise":
                revised = self.reviser.run(case, plan.model_dump(), session, natural)
                if self._structural_verdict(revised, plan.round_count).result == "pass":
                    session = revised
            sessions.append(session)
            print(f"completed {plan.session_id} ({len(sessions)}/{len(plans.plans)})", flush=True)
            checkpoint_path.write_text(
                json.dumps([s.model_dump() for s in sessions], ensure_ascii=False, indent=2), encoding="utf-8"
            )

        dataset_id = f"{self.config.dataset_id_prefix}-{case.case_id}"
        dialogues = []
        for s in sessions:
            item = s.model_dump()
            item.pop("summary", None)
            dialogues.append(item)
        benchmark = Benchmark(dataset_id=dataset_id, character_profile=case.character_profile, dialogues=dialogues, eval_samples=[])
        qa = validate_sessions(dataset_id, case, sessions, self.llm.records)
        benchmark_path = output_dir / "benchmark.json"
        qa_path = output_dir / "qa_report.json"
        benchmark_path.write_text(benchmark.model_dump_json(indent=2), encoding="utf-8")
        qa_path.write_text(qa.model_dump_json(indent=2), encoding="utf-8")
        return benchmark_path, qa_path

    @staticmethod
    def _structural_verdict(session: Session, round_count: int) -> VerificationResult:
        issues: list[VerificationIssue] = []
        expected_turns = round_count * 2
        if len(session.turns) != expected_turns:
            issues.append(VerificationIssue(
                turn_id=session.session_id, type="turn_count",
                description=f"需要 {expected_turns} 条 turns（{round_count} rounds），实际为 {len(session.turns)} 条。",
            ))
        seen: set[str] = set()
        for index, turn in enumerate(session.turns):
            number = index + 1
            expected_turn_id = f"{session.session_id}_T{number:02d}"
            expected_round_id = f"{session.session_id}_R{index // 2 + 1:02d}"
            expected_speaker = "user" if index % 2 == 0 else "assistant"
            if turn.turn_id != expected_turn_id or turn.round_id != expected_round_id or turn.speaker != expected_speaker or turn.turn_id in seen:
                issues.append(VerificationIssue(
                    turn_id=turn.turn_id, type="turn_structure",
                    description=f"应为 turn_id={expected_turn_id}, round_id={expected_round_id}, speaker={expected_speaker}。",
                ))
            seen.add(turn.turn_id)
        return VerificationResult(result="revise" if issues else "pass", issues=issues)
