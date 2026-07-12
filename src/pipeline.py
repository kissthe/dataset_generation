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
        raw_case_spec = case_path.read_text(encoding="utf-8")
        case = CaseSpec.model_validate_json(raw_case_spec)
        cfg = self.config.generation
        output_dir.mkdir(parents=True, exist_ok=True)
        audit_dir = output_dir / "logs"
        audit_dir.mkdir(parents=True, exist_ok=True)
        self._write_log(audit_dir / "00_original_case_spec.json", {
            "source_path": str(case_path),
            "raw_case_spec": raw_case_spec,
            "case_spec": case.model_dump(),
            "generation_config": {
                "session_count": cfg.session_count,
                "planner_batch_size": cfg.planner_batch_size,
                "min_rounds": cfg.min_rounds,
                "max_rounds": cfg.max_rounds,
                "context_sessions": cfg.context_sessions,
                "seed": cfg.seed,
                "max_retries": cfg.max_retries,
                "max_revision_cycles": cfg.max_revision_cycles,
                "run_eval": cfg.run_eval,
            },
        })
        plan_path = output_dir / "session_plans.json"
        if case.session_outlines:
            expected = [x.session_id for x in case.session_outlines[:cfg.session_count]]
        else:
            id_prefix = case.case_id.split('-')[-1].upper()
            expected = [f"{id_prefix}-S{i+1:02d}" for i in range(cfg.session_count)]
        if plan_path.exists():
            from .models import SessionPlanList
            plans = SessionPlanList.model_validate_json(plan_path.read_text(encoding="utf-8"))
            print(f"resumed {len(plans.plans)} saved plans", flush=True)
            self._write_log(audit_dir / "01_planner_resumed.json", {
                "action": "reused_existing_session_plans",
                "source_path": str(plan_path),
                "output": plans.model_dump(),
            })
        else:
            all_plans = []
            if case.session_outlines:
                # Backward-compatible path: outlines provided in case spec.
                outlines = case.session_outlines[:cfg.session_count]
                for start in range(0, len(outlines), cfg.planner_batch_size):
                    batch_case = case.model_copy(update={
                        "session_outlines": outlines[start:start + cfg.planner_batch_size],
                        "eval_outlines": [],
                    })
                    planner_input = {
                        "case_spec": batch_case.model_dump(),
                        "constraints": {"min_rounds": cfg.min_rounds, "max_rounds": cfg.max_rounds},
                        "prior_plans": [p.model_dump() for p in all_plans],
                    }
                    batch = self.planner.run(batch_case, cfg.min_rounds, cfg.max_rounds,
                                             session_count=len(outlines[start:start + cfg.planner_batch_size]),
                                             id_prefix="", start_index=start,
                                             prior_plans=planner_input["prior_plans"])
                    self._write_log(audit_dir / f"01_planner_batch_{start // cfg.planner_batch_size + 1:02d}.json", {
                        "component": "session_planner", "llm": self._llm_metadata("session_planner"),
                        "input": planner_input, "output": batch.model_dump(),
                    })
                    all_plans.extend(batch.plans)
                    print(f"planned {len(all_plans)}/{len(outlines)} sessions", flush=True)
            else:
                # New path: planner generates sessions from scratch based on core_emotional_event.
                id_prefix = case.case_id.split('-')[-1].upper()
                total = cfg.session_count
                for start in range(0, total, cfg.planner_batch_size):
                    batch_count = min(cfg.planner_batch_size, total - start)
                    planner_input = {
                        "case_spec": case.model_dump(),
                        "session_count": batch_count,
                        "id_prefix": id_prefix,
                        "start_index": start,
                        "constraints": {"min_rounds": cfg.min_rounds, "max_rounds": cfg.max_rounds},
                        "prior_plans": [p.model_dump() for p in all_plans],
                    }
                    batch = self.planner.run(case, cfg.min_rounds, cfg.max_rounds,
                                             session_count=batch_count, id_prefix=id_prefix,
                                             start_index=start,
                                             prior_plans=planner_input["prior_plans"])
                    self._write_log(audit_dir / f"01_planner_batch_{start // cfg.planner_batch_size + 1:02d}.json", {
                        "component": "session_planner", "llm": self._llm_metadata("session_planner"),
                        "input": planner_input, "output": batch.model_dump(),
                    })
                    all_plans.extend(batch.plans)
                    print(f"planned {len(all_plans)}/{total} sessions", flush=True)
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
            session_log_dir = audit_dir / "sessions" / plan.session_id
            session_log_dir.mkdir(parents=True, exist_ok=True)
            writer_input = {
                "case_spec": case.model_dump(), "session_plan": plan.model_dump(),
                "recent_history": [s.model_dump() for s in recent],
            }
            session = self.writer.run(case, plan.model_dump(), recent)
            self._write_log(session_log_dir / "01_writer.json", {
                "component": "session_writer", "llm": self._llm_metadata("session_writer"),
                "input": writer_input, "output": session.model_dump(),
            })
            for cycle in range(cfg.max_revision_cycles):
                structural = self._structural_verdict(session, plan.round_count)
                self._write_log(session_log_dir / f"{cycle + 2:02d}_cycle_{cycle + 1}_structural_verifier.json", {
                    "component": "deterministic_structural_verifier",
                    "input": {"session": session.model_dump(), "expected_round_count": plan.round_count},
                    "output": structural.model_dump(),
                })
                if structural.result == "revise":
                    verdict = structural
                    verifier_kind = "deterministic_structural_verifier"
                else:
                    verdict = self.verifier.run(case, plan.model_dump(), session, recent)
                    verifier_kind = "session_verifier"
                    self._write_log(session_log_dir / f"{cycle + 2:02d}_cycle_{cycle + 1}_session_verifier.json", {
                        "component": verifier_kind, "llm": self._llm_metadata("session_verifier"),
                        "input": {
                            "case_spec": case.model_dump(), "session_plan": plan.model_dump(),
                            "session": session.model_dump(), "recent_history": [s.model_dump() for s in recent],
                        },
                        "output": verdict.model_dump(),
                    })
                if verdict.result == "pass":
                    break
                before_revision = session
                session = self.reviser.run(case, plan.model_dump(), session, verdict)
                self._write_log(session_log_dir / f"{cycle + 2:02d}_cycle_{cycle + 1}_reviser.json", {
                    "component": "session_reviser", "llm": self._llm_metadata("session_reviser"),
                    "triggered_by": verifier_kind,
                    "input": {
                        "case_spec": case.model_dump(), "session_plan": plan.model_dump(),
                        "session": before_revision.model_dump(), "issues": verdict.model_dump(),
                    },
                    "output": session.model_dump(),
                })
            final_structure = self._structural_verdict(session, plan.round_count)
            if final_structure.result != "pass":
                raise RuntimeError(f"{plan.session_id} failed deterministic structure checks: {final_structure.model_dump()}")
            natural = self.naturalness.run(case, session)
            self._write_log(session_log_dir / "90_naturalness_checker.json", {
                "component": "naturalness_checker", "llm": self._llm_metadata("naturalness_checker"),
                "input": {"conversation_style": case.character_profile.conversation_style, "session": session.model_dump()},
                "output": natural.model_dump(),
            })
            if natural.result == "revise":
                revised = self.reviser.run(case, plan.model_dump(), session, natural)
                post_natural_structure = self._structural_verdict(revised, plan.round_count)
                self._write_log(session_log_dir / "91_naturalness_reviser.json", {
                    "component": "session_reviser", "llm": self._llm_metadata("session_reviser"),
                    "triggered_by": "naturalness_checker",
                    "input": {
                        "case_spec": case.model_dump(), "session_plan": plan.model_dump(),
                        "session": session.model_dump(), "issues": natural.model_dump(),
                    },
                    "output": revised.model_dump(),
                    "post_revision_structural_verdict": post_natural_structure.model_dump(),
                })
                if post_natural_structure.result == "pass":
                    session = revised
            self._write_log(session_log_dir / "99_final_session.json", {
                "component": "pipeline", "output": session.model_dump(),
            })
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
        qa = validate_sessions(dataset_id, case, sessions, self.llm.records, expected)
        benchmark_path = output_dir / "benchmark.json"
        qa_path = output_dir / "qa_report.json"
        benchmark_path.write_text(benchmark.model_dump_json(indent=2), encoding="utf-8")
        qa_path.write_text(qa.model_dump_json(indent=2), encoding="utf-8")
        self._write_log(audit_dir / "99_pipeline_result.json", {
            "benchmark_path": str(benchmark_path), "qa_path": str(qa_path),
            "qa": qa.model_dump(), "eval_generated": False,
        })
        self._write_log_index(audit_dir, case.case_id)
        return benchmark_path, qa_path

    def _llm_metadata(self, component: str) -> dict:
        component_config = self.config.components[component]
        prompt_path = self.config.root / "prompts" / f"{component}.txt"
        return {
            "model": component_config.model,
            "temperature": component_config.temperature,
            "max_completion_tokens": component_config.max_completion_tokens,
            "prompt_file": str(prompt_path),
            "prompt": prompt_path.read_text(encoding="utf-8"),
        }

    @staticmethod
    def _write_log_index(audit_dir: Path, case_id: str) -> None:
        lines = [
            f"# Generation logs: {case_id}", "",
            "这些日志按真实执行顺序保存了 Spec、Planner、各 Session 组件输入/输出以及最终 QA。", "",
            "| Log | Component |", "|---|---|",
        ]
        for path in sorted(audit_dir.rglob("*.json")):
            relative = path.relative_to(audit_dir).as_posix()
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                component = payload.get("component", "pipeline")
            except (OSError, json.JSONDecodeError):
                component = "unknown"
            lines.append(f"| [{relative}]({relative}) | `{component}` |")
        (audit_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    @staticmethod
    def _write_log(path: Path, value: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

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
