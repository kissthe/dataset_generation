from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .components import (
    DatasetBlueprintPlanner, EvalGenerator, EvalVerifier, EvidenceResolver,
    NaturalnessChecker, SessionPlanner, SessionReviser, SessionVerifier, SessionWriter,
    blueprint_id_for, build_eval_candidate_list, canonical_eval_ids, blueprint_fingerprint,
    canonical_session_ids, eval_label_targets, memory_role_targets, plan_fingerprint,
    validate_blueprint_constraints, validate_dataset_blueprint, validate_eval_candidate,
)
from .config import AppConfig, BlueprintConstraints
from .gold_finalizer import GoldFinalizer
from .llm_client import LLMClient, describe_exception, is_powershell_connection_error
from .models import (
    ArtifactReuseProvenance, Benchmark, BlueprintCandidate,
    BlueprintCandidateSelection, CandidateSelection, CaseSpec, DatasetBlueprint,
    EvalGenerationResult, EvalSample, PlanCandidate,
    PlanningCandidateManifest, Session, SessionPlanList, VerificationIssue,
    VerificationResult,
)
from .qa import validate_sessions

class GenerationPipeline:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.llm = LLMClient(config)
        self.blueprint_planner = DatasetBlueprintPlanner(self.llm)
        self.planner = SessionPlanner(self.llm)
        self.writer = SessionWriter(self.llm)
        self.verifier = SessionVerifier(self.llm)
        self.reviser = SessionReviser(self.llm)
        self.naturalness = NaturalnessChecker(self.llm)
        self.eval_generator = EvalGenerator(self.llm)
        self.eval_resolver = EvidenceResolver(self.llm)
        self.eval_verifier = EvalVerifier(self.llm)
        self.gold_finalizer = GoldFinalizer(config.generation.seed)

    def prepare_blueprint_candidates(
        self, case_path: Path, output_dir: Path, candidate_count: int = 3
    ) -> Path:
        """Stage 1: generate reviewable Blueprint options and stop."""
        if not 1 <= candidate_count <= 8:
            raise ValueError("blueprint candidate count must be between 1 and 8")
        case = CaseSpec.model_validate_json(case_path.read_text(encoding="utf-8"))
        cfg = self.config.generation
        total = (
            min(len(case.session_outlines), cfg.session_count)
            if case.session_outlines else cfg.session_count
        )
        constraints = self.config.blueprint_constraints
        errors = validate_blueprint_constraints(total, constraints)
        if errors:
            raise ValueError("invalid blueprint constraints: " + "; ".join(errors))
        session_ids = canonical_session_ids(case, total)
        eval_count = sum(eval_label_targets(constraints=constraints).values())
        output_dir.mkdir(parents=True, exist_ok=True)
        logs_dir = output_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / "planning_candidates.json"
        if manifest_path.exists():
            saved_manifest = PlanningCandidateManifest.model_validate_json(
                manifest_path.read_text(encoding="utf-8")
            )
            expected_blueprint_id = blueprint_id_for(
                case, session_ids, eval_count, constraints
            )
            if (
                saved_manifest.version != 2
                or saved_manifest.case_id != case.case_id
                or saved_manifest.session_ids != session_ids
                or saved_manifest.blueprint_candidate_count != candidate_count
                or len(saved_manifest.blueprint_candidates) != candidate_count
                or any(
                    item.blueprint.blueprint_id != expected_blueprint_id
                    for item in saved_manifest.blueprint_candidates
                )
            ):
                raise ValueError(
                    "saved planning candidates do not match this Spec/config; "
                    "use a new output directory"
                )
            print(
                f"resumed {candidate_count} Blueprint candidates",
                flush=True,
            )
            print(f"blueprint candidates: {manifest_path}", flush=True)
            return manifest_path

        blueprint_goals = (
            "以日常物件和可重复出现的细节为主，线索清晰克制",
            "以场景、声音和时间节奏为主，强调渐进式情绪回响",
            "以话语、关系互动和生活变化为主，强调前后对照",
            "以低强度生活线索为主，减少直白说明",
            "以多感官线索为主，保持证据边界明确",
            "以空间与时间的重复模式为主，保持线索自然出现",
            "以轻度日常线索为主，突出正负样本边界",
            "以关系语境和生活节点为主，减少刻意触发",
        )

        blueprint_candidates: list[BlueprintCandidate] = []
        for index in range(candidate_count):
            context = {
                "candidate_number": index + 1,
                "candidate_count": candidate_count,
                "variant_goal": blueprint_goals[index % len(blueprint_goals)],
                "instruction": "与其他候选形成明显差异，但仍严格遵守覆盖约束。",
            }
            print(f"generating Blueprint candidate {index + 1}/{candidate_count}...", flush=True)
            blueprint = self.blueprint_planner.run(
                case, session_ids, eval_count, constraints, context
            )
            cue_names = [
                cue.canonical_form
                for memory in blueprint.emotion_memory_map
                for cue in memory.cue_seeds
            ][:4]
            candidate = BlueprintCandidate(
                candidate_id=f"BP-{index + 1:02d}",
                title=f"Blueprint {index + 1} · {blueprint.life_anchor.identity}",
                summary="；".join(cue_names) or blueprint.life_anchor.identity,
                blueprint=blueprint,
            )
            blueprint_candidates.append(candidate)
            self._write_log(
                logs_dir / f"01_blueprint_candidate_{index + 1:02d}.json",
                {
                    "component": "dataset_blueprint_planner",
                    "candidate_id": candidate.candidate_id,
                    "candidate_context": context,
                    "output": candidate.model_dump(),
                },
            )
            print(f"completed Blueprint candidate {index + 1}/{candidate_count}", flush=True)

        manifest = PlanningCandidateManifest(
            case_id=case.case_id,
            session_ids=session_ids,
            blueprint_candidate_count=candidate_count,
            blueprint_candidates=blueprint_candidates,
            plan_candidates=[],
        )
        manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        self._write_log(logs_dir / "99_blueprint_candidate_result.json", {
            "mode": "blueprint_candidate_review",
            "candidate_manifest_path": str(manifest_path),
            "blueprint_candidate_count": len(blueprint_candidates),
        })
        print(f"blueprint candidates: {manifest_path}", flush=True)
        return manifest_path

    def prepare_plan_candidates(
        self, case_path: Path, output_dir: Path,
        blueprint_candidate_id: str, candidate_count: int = 3,
    ) -> Path:
        """Stage 2: generate multiple complete Plans from one selected Blueprint."""
        if not 1 <= candidate_count <= 8:
            raise ValueError("plan candidate count must be between 1 and 8")
        manifest_path = output_dir / "planning_candidates.json"
        if not manifest_path.exists():
            raise ValueError("generate Blueprint candidates before generating Plans")
        manifest = PlanningCandidateManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
        if manifest.version != 2:
            raise ValueError("legacy 3x3 candidates cannot be resumed; use a new output directory")
        blueprints = {item.candidate_id: item for item in manifest.blueprint_candidates}
        if blueprint_candidate_id not in blueprints:
            raise ValueError(f"unknown Blueprint candidate: {blueprint_candidate_id}")
        if (output_dir / "checkpoint_sessions.json").exists() or manifest.selection:
            raise ValueError("cannot regenerate Plan candidates after Writer has started")
        if (
            manifest.blueprint_selection
            and manifest.blueprint_selection.blueprint_candidate_id == blueprint_candidate_id
            and manifest.plan_candidate_count == candidate_count
            and len(manifest.plan_candidates) == candidate_count
            and all(
                item.blueprint_candidate_id == blueprint_candidate_id
                for item in manifest.plan_candidates
            )
        ):
            print(
                f"resumed {candidate_count} Plan candidates for {blueprint_candidate_id}",
                flush=True,
            )
            print(f"plan candidates: {manifest_path}", flush=True)
            return manifest_path

        case = CaseSpec.model_validate_json(case_path.read_text(encoding="utf-8"))
        if case.case_id != manifest.case_id:
            raise ValueError("CaseSpec does not match the Blueprint candidate manifest")
        cfg = self.config.generation
        total = len(manifest.session_ids)
        selected_blueprint = blueprints[blueprint_candidate_id].blueprint
        reuse_path = output_dir / "reuse_provenance.json"
        if reuse_path.exists():
            reuse = ArtifactReuseProvenance.model_validate_json(
                reuse_path.read_text(encoding="utf-8")
            )
            if (
                reuse.case_id != case.case_id
                or reuse.blueprint_fingerprint != blueprint_fingerprint(selected_blueprint)
            ):
                raise ValueError("reused Blueprint does not match its provenance")
        expected_ids = [slot.session_id for slot in selected_blueprint.session_slots]
        if expected_ids != manifest.session_ids:
            raise ValueError("selected Blueprint session IDs do not match the candidate manifest")
        logs_dir = output_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        plan_goals = (
            "生活感强、节奏舒缓，以连续的小事推动关系",
            "场景变化丰富，适度穿插未完成事项和后续回访",
            "人物主动性更强，以选择、反馈和状态变化推进故事",
            "更克制内敛，重视日常观察和微小情绪变化",
            "更轻快自然，用兴趣和生活任务建立长期连续性",
            "时间跨度更清晰，让记忆建立、回响和更新自然分隔",
            "对话动机变化更丰富，减少连续求建议",
            "生活线更集中，用少量持续事项形成稳定连贯性",
        )

        plan_candidates: list[PlanCandidate] = []
        id_prefix = (
            manifest.session_ids[0].rsplit("-S", 1)[0]
            if manifest.session_ids else "SESSION"
        )
        for index in range(candidate_count):
            context = {
                "candidate_number": index + 1,
                "candidate_count": candidate_count,
                "variant_goal": plan_goals[index % len(plan_goals)],
                "instruction": (
                    "严格落实已选 Blueprint 的 life anchor、Session slots、"
                    "memory role、cue 和 evidence goal，同时形成不同的叙事方案。"
                ),
            }
            print(f"generating Plan candidate {index + 1}/{candidate_count}...", flush=True)
            all_plans = []
            start = 0
            adaptive_batch_size = cfg.planner_batch_size
            while start < total:
                batch_count = min(adaptive_batch_size, total - start)
                try:
                    batch_slots = selected_blueprint.session_slots[start:start + batch_count]
                    batch = self.planner.run(
                        case, cfg.min_rounds, cfg.max_rounds,
                        session_count=batch_count,
                        total_session_count=total,
                        id_prefix=id_prefix,
                        start_index=start,
                        prior_plans=[plan.model_dump() for plan in all_plans],
                        life_anchor=selected_blueprint.life_anchor,
                        blueprint=selected_blueprint,
                        session_slots=batch_slots,
                        candidate_context=context,
                    )
                except RuntimeError as exc:
                    if batch_count <= 1 or not is_powershell_connection_error(exc):
                        raise
                    adaptive_batch_size = max(1, batch_count // 2)
                    print(
                        f"Plan candidate connection failed; reducing batch size "
                        f"from {batch_count} to {adaptive_batch_size} and retrying",
                        flush=True,
                    )
                    continue
                all_plans.extend(batch.plans)
                start += batch_count
                print(
                    f"planned candidate {index + 1}: {len(all_plans)}/{total} sessions",
                    flush=True,
                )
            plan_list = SessionPlanList(
                plans=all_plans,
                life_anchor=selected_blueprint.life_anchor,
                blueprint_id=selected_blueprint.blueprint_id,
                blueprint_fingerprint=blueprint_fingerprint(selected_blueprint),
            )
            self._validate_plans_against_blueprint(plan_list.plans, selected_blueprint)
            topics = [plan.topic for plan in plan_list.plans[:4]]
            candidate = PlanCandidate(
                candidate_id=f"PLAN-{index + 1:02d}",
                blueprint_candidate_id=blueprint_candidate_id,
                title=(
                    f"Plan {index + 1} · "
                    f"{plan_goals[index % len(plan_goals)].split('，')[0]}"
                ),
                summary=" → ".join(topics),
                plan=plan_list,
            )
            plan_candidates.append(candidate)
            self._write_log(
                logs_dir / (
                    f"02_plan_candidate_{blueprint_candidate_id}_{index + 1:02d}.json"
                ),
                {
                    "component": "session_planner",
                    "blueprint_candidate_id": blueprint_candidate_id,
                    "candidate_id": candidate.candidate_id,
                    "candidate_context": context,
                    "output": candidate.model_dump(),
                },
            )
            print(f"completed Plan candidate {index + 1}/{candidate_count}", flush=True)

        manifest = manifest.model_copy(update={
            "plan_candidate_count": candidate_count,
            "plan_candidates": plan_candidates,
            "blueprint_selection": BlueprintCandidateSelection(
                blueprint_candidate_id=blueprint_candidate_id,
                selected_at=datetime.now(timezone.utc).isoformat(),
            ),
            "selection": None,
        })
        manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        self._write_log(logs_dir / "99_plan_candidate_result.json", {
            "mode": "plan_candidate_review",
            "candidate_manifest_path": str(manifest_path),
            "plan_candidate_count": len(plan_candidates),
            "blueprint_candidate_id": blueprint_candidate_id,
        })
        print(f"plan candidates: {manifest_path}", flush=True)
        return manifest_path

    def select_candidates(
        self, output_dir: Path, blueprint_candidate_id: str, plan_candidate_id: str
    ) -> tuple[Path, Path]:
        """Materialize one Plan that was generated from the selected Blueprint."""
        manifest_path = output_dir / "planning_candidates.json"
        if not manifest_path.exists():
            raise ValueError("planning_candidates.json does not exist")
        manifest = PlanningCandidateManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
        blueprints = {item.candidate_id: item for item in manifest.blueprint_candidates}
        plans = {item.candidate_id: item for item in manifest.plan_candidates}
        if blueprint_candidate_id not in blueprints:
            raise ValueError(f"unknown Blueprint candidate: {blueprint_candidate_id}")
        if plan_candidate_id not in plans:
            raise ValueError(f"unknown Plan candidate: {plan_candidate_id}")
        selected_plan_candidate = plans[plan_candidate_id]
        if (
            not manifest.blueprint_selection
            or manifest.blueprint_selection.blueprint_candidate_id != blueprint_candidate_id
            or selected_plan_candidate.blueprint_candidate_id != blueprint_candidate_id
        ):
            raise ValueError("the selected Plan was not generated from this Blueprint")
        selection = CandidateSelection(
            blueprint_candidate_id=blueprint_candidate_id,
            plan_candidate_id=plan_candidate_id,
            selected_at=datetime.now(timezone.utc).isoformat(),
        )
        if (output_dir / "checkpoint_sessions.json").exists() and (
            manifest.selection is None
            or manifest.selection.blueprint_candidate_id != blueprint_candidate_id
            or manifest.selection.plan_candidate_id != plan_candidate_id
        ):
            raise ValueError(
                "cannot change the Blueprint/Plan combination after Writer has started"
            )
        selected_blueprint = blueprints[blueprint_candidate_id].blueprint
        selected_plans = selected_plan_candidate.plan
        expected_fingerprint = blueprint_fingerprint(selected_blueprint)
        if (
            selected_plans.blueprint_id != selected_blueprint.blueprint_id
            or selected_plans.blueprint_fingerprint != expected_fingerprint
        ):
            raise ValueError("the selected Plan references a different Blueprint")
        self._validate_plans_against_blueprint(selected_plans.plans, selected_blueprint)
        blueprint_path = output_dir / "dataset_blueprint.json"
        plan_path = output_dir / "session_plans.json"
        blueprint_path.write_text(
            selected_blueprint.model_dump_json(indent=2), encoding="utf-8"
        )
        plan_path.write_text(selected_plans.model_dump_json(indent=2), encoding="utf-8")
        manifest = manifest.model_copy(update={"selection": selection})
        manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        (output_dir / "candidate_selection.json").write_text(
            selection.model_dump_json(indent=2), encoding="utf-8"
        )
        print(
            f"selected {blueprint_candidate_id} + {plan_candidate_id}; continuing pipeline",
            flush=True,
        )
        return blueprint_path, plan_path

    def run(self, case_path: Path, output_dir: Path) -> tuple[Path, Path | None]:
        raw_case_spec = case_path.read_text(encoding="utf-8")
        case = CaseSpec.model_validate_json(raw_case_spec)
        cfg = self.config.generation
        run_eval_examples = getattr(cfg, "run_eval_examples", False)
        total = min(len(case.session_outlines), cfg.session_count) if case.session_outlines else cfg.session_count
        blueprint_constraints = getattr(self.config, "blueprint_constraints", None)
        if blueprint_constraints is None:
            legacy_targets = memory_role_targets(total)
            blueprint_constraints = BlueprintConstraints(
                encode_association_count=legacy_targets["encode_association"],
                triggered_recall_count=legacy_targets["triggered_recall"],
                memory_update_count=legacy_targets["memory_update"],
                control_count=legacy_targets["control"],
            )
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
                "run_eval_examples": run_eval_examples,
                "stop_after_planning": cfg.stop_after_planning,
                "eval_count": sum(eval_label_targets(constraints=blueprint_constraints).values()),
            },
            "blueprint_constraints": {
                "encode_association_count": blueprint_constraints.encode_association_count,
                "triggered_recall_count": blueprint_constraints.triggered_recall_count,
                "memory_update_count": blueprint_constraints.memory_update_count,
                "control_count": blueprint_constraints.control_count,
                "required_cue_types": list(blueprint_constraints.required_cue_types),
                "eval_triggered_count": blueprint_constraints.eval_triggered_count,
                "eval_insufficient_evidence_count": blueprint_constraints.eval_insufficient_evidence_count,
                "eval_not_triggered_count": blueprint_constraints.eval_not_triggered_count,
            },
            "validation_config": self._validation_settings(),
        })
        constraint_errors = validate_blueprint_constraints(total, blueprint_constraints)
        if constraint_errors:
            raise ValueError("invalid blueprint constraints: " + "; ".join(constraint_errors))
        expected = canonical_session_ids(case, total)
        eval_count = sum(eval_label_targets(constraints=blueprint_constraints).values())
        blueprint_path = output_dir / "dataset_blueprint.json"
        plan_path = output_dir / "session_plans.json"
        reuse_path = output_dir / "reuse_provenance.json"
        reuse = (
            ArtifactReuseProvenance.model_validate_json(
                reuse_path.read_text(encoding="utf-8")
            )
            if reuse_path.exists() else None
        )
        if reuse and reuse.case_id != case.case_id:
            raise ValueError("reuse provenance belongs to a different CaseSpec")
        if reuse:
            self._write_log(audit_dir / "00_reuse_provenance.json", {
                "component": "pipeline",
                "action": "validated_reused_artifacts",
                "output": reuse.model_dump(),
            })

        blueprint: DatasetBlueprint | None = None
        blueprint_schema_migrated = False
        if blueprint_path.exists():
            raw_blueprint_payload = json.loads(blueprint_path.read_text(encoding="utf-8"))
            blueprint_schema_migrated = any(
                "historical_emotion" in memory
                or any("distinguishing_detail" in cue for cue in memory.get("cue_seeds", []))
                for memory in raw_blueprint_payload.get("emotion_memory_map", [])
            ) or any(
                any(field in slot for field in (
                    "emotion_intensity", "disclosure_level", "supports_eval_outlines"
                ))
                for slot in raw_blueprint_payload.get("session_slots", [])
            )
            blueprint = DatasetBlueprint.model_validate(raw_blueprint_payload)
            expected_blueprint_id = blueprint_id_for(
                case, expected, eval_count, blueprint_constraints
            )
            legacy_blueprint_id = blueprint_id_for(case, expected, eval_count)
            if (
                blueprint_schema_migrated
                and blueprint.blueprint_id == legacy_blueprint_id
                and blueprint.blueprint_id != expected_blueprint_id
            ):
                blueprint = blueprint.model_copy(update={"blueprint_id": expected_blueprint_id})
            blueprint_errors = validate_dataset_blueprint(
                blueprint,
                expected_blueprint_id=expected_blueprint_id,
                expected_session_ids=expected,
                expected_eval_ids=canonical_eval_ids(case, eval_count),
                role_targets=memory_role_targets(total, blueprint_constraints),
                label_targets=eval_label_targets(constraints=blueprint_constraints),
                required_cue_types=blueprint_constraints.required_cue_types,
            )
            if blueprint_errors:
                raise ValueError(
                    "saved dataset_blueprint.json does not match this Spec/config: "
                    + "; ".join(blueprint_errors)
                )
            if blueprint_schema_migrated:
                blueprint_path.write_text(blueprint.model_dump_json(indent=2), encoding="utf-8")
            print(f"resumed dataset blueprint with {len(blueprint.session_slots)} session slots", flush=True)
            self._write_log(audit_dir / "01_dataset_blueprint_resumed.json", {
                "component": "dataset_blueprint_planner",
                "action": "reused_existing_dataset_blueprint",
                "source_path": str(blueprint_path),
                "output": blueprint.model_dump(),
            })
        elif not plan_path.exists():
            blueprint_input = self.blueprint_planner.build_payload(
                case, expected, eval_count, blueprint_constraints
            )
            blueprint = self.blueprint_planner.run(
                case, expected, eval_count, blueprint_constraints
            )
            blueprint_path.write_text(blueprint.model_dump_json(indent=2), encoding="utf-8")
            self._write_log(audit_dir / "01_dataset_blueprint_planner.json", {
                "component": "dataset_blueprint_planner",
                "llm": self._llm_metadata("dataset_blueprint_planner"),
                "input": blueprint_input,
                "raw_output": (
                    getattr(self.blueprint_planner, "last_raw_blueprint", None).model_dump()
                    if getattr(self.blueprint_planner, "last_raw_blueprint", None) else None
                ),
                "normalization_notes": getattr(
                    self.blueprint_planner, "last_normalization_notes", []
                ),
                "output": blueprint.model_dump(),
            })
            print(
                f"blueprint ready: {len(blueprint.session_slots)} session slots, "
                f"{len(blueprint.eval_outlines)} eval outlines",
                flush=True,
            )
        else:
            print("resuming legacy plans without a dataset blueprint", flush=True)

        if reuse:
            if blueprint is None:
                raise ValueError("reuse provenance requires dataset_blueprint.json")
            if blueprint_fingerprint(blueprint) != reuse.blueprint_fingerprint:
                raise ValueError("reused Blueprint fingerprint does not match provenance")

        if plan_path.exists():
            from .models import SessionPlanList
            plans = SessionPlanList.model_validate_json(plan_path.read_text(encoding="utf-8"))
            if blueprint is not None:
                expected_fingerprint = blueprint_fingerprint(blueprint)
                blueprint_reference_changed = (
                    plans.blueprint_id != blueprint.blueprint_id
                    or plans.blueprint_fingerprint != expected_fingerprint
                )
                if blueprint_reference_changed and not blueprint_schema_migrated:
                    raise ValueError(
                        "session_plans.json was created from a different DatasetBlueprint; "
                        "use a new output directory or regenerate the plans"
                    )
                self._validate_plans_against_blueprint(plans.plans, blueprint)
                if blueprint_reference_changed and blueprint_schema_migrated:
                    plans = plans.model_copy(update={
                        "blueprint_id": blueprint.blueprint_id,
                        "blueprint_fingerprint": expected_fingerprint,
                    })
            print(f"resumed {len(plans.plans)} saved plans", flush=True)
            self._write_log(audit_dir / "02_planner_resumed.json", {
                "component": "session_planner",
                "action": "reused_existing_session_plans",
                "source_path": str(plan_path),
                "output": plans.model_dump(),
            })
        else:
            all_plans = []
            life_anchor = blueprint.life_anchor if blueprint else None
            id_prefix = expected[0].rsplit("-S", 1)[0] if expected else "SESSION"
            start = 0
            planner_batch_number = 0
            adaptive_batch_size = cfg.planner_batch_size
            while start < total:
                batch_count = min(adaptive_batch_size, total - start)
                batch_slots = (
                    blueprint.session_slots[start:start + batch_count]
                    if blueprint else []
                )
                planner_input = self.planner.build_payload(
                    case, cfg.min_rounds, cfg.max_rounds, batch_count, total,
                    id_prefix, start, [p.model_dump() for p in all_plans], life_anchor,
                    blueprint, batch_slots,
                )
                try:
                    batch = self.planner.run(
                        case, cfg.min_rounds, cfg.max_rounds,
                        session_count=batch_count, total_session_count=total,
                        id_prefix=id_prefix, start_index=start,
                        prior_plans=planner_input["prior_plans"], life_anchor=life_anchor,
                        blueprint=blueprint, session_slots=batch_slots,
                    )
                except RuntimeError as exc:
                    if batch_count <= 1 or not is_powershell_connection_error(exc):
                        raise
                    reduced_batch_size = max(1, batch_count // 2)
                    print(
                        f"planner batch connection failed; reducing batch size "
                        f"from {batch_count} to {reduced_batch_size} and retrying",
                        flush=True,
                    )
                    adaptive_batch_size = reduced_batch_size
                    continue
                planner_batch_number += 1
                self._write_log(
                    audit_dir / f"02_planner_batch_{planner_batch_number:02d}.json",
                    {
                        "component": "session_planner",
                        "llm": self._llm_metadata("session_planner"),
                        "input": planner_input,
                        "output": batch.model_dump(),
                    },
                )
                all_plans.extend(batch.plans)
                start += batch_count
                print(f"planned {len(all_plans)}/{total} sessions", flush=True)
            from .models import SessionPlanList
            plans = SessionPlanList(
                plans=all_plans,
                life_anchor=life_anchor,
                blueprint_id=blueprint.blueprint_id if blueprint else "",
                blueprint_fingerprint=blueprint_fingerprint(blueprint) if blueprint else "",
            )
        plans.plans = [p for p in plans.plans if p.session_id in expected]
        plans.plans.sort(key=lambda p: expected.index(p.session_id))
        if [p.session_id for p in plans.plans] != expected:
            raise ValueError(f"planner did not return the expected session IDs; got {[p.session_id for p in plans.plans]}")
        if blueprint is not None:
            self._validate_plans_against_blueprint(plans.plans, blueprint)
        if reuse and reuse.plan_fingerprint:
            if plan_fingerprint(plans) != reuse.plan_fingerprint:
                raise ValueError("reused Plan fingerprint does not match provenance")

        (output_dir / "session_plans.json").write_text(plans.model_dump_json(indent=2), encoding="utf-8")
        text_plan = "\n".join(
            f"{p.session_id} | {p.date} | {p.session_type} | {p.memory_role} | {p.topic} | {p.round_count} rounds\n"
            f"  scene: {p.scene}\n  intent: {p.user_intent}\n  beat: {p.story_beat}\n"
            f"  memory/cue: {p.memory_id}/{p.cue_id}\n  evidence: {p.evidence_goal or '-'}\n"
            f"  next: {p.continuity_hook or '-'}"
            for p in plans.plans
        )
        (output_dir / "session_plans.txt").write_text(text_plan, encoding="utf-8")

        if cfg.stop_after_planning:
            self._write_log(audit_dir / "99_pipeline_result.json", {
                "mode": "planner_only",
                "session_plans_path": str(plan_path),
                "dataset_blueprint_path": str(blueprint_path) if blueprint else None,
                "planned_session_count": len(plans.plans),
                "validation_config": self._validation_settings(),
            })
            self._write_log_index(audit_dir, case.case_id)
            print(f"plans: {plan_path}", flush=True)
            return plan_path, None

        checkpoint_path = output_dir / "checkpoint_sessions.json"
        sessions: list[Session] = []
        if checkpoint_path.exists():
            saved = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            sessions = [Session.model_validate(item) for item in saved]
            if [s.session_id for s in sessions] != expected[:len(sessions)]:
                raise ValueError("checkpoint sessions are not a valid prefix of the case")
            if reuse and reuse.reused_session_ids:
                saved_ids = [session.session_id for session in sessions]
                if saved_ids[:len(reuse.reused_session_ids)] != reuse.reused_session_ids:
                    raise ValueError("checkpoint does not contain the reused Session prefix")
                plan_by_id = {plan.session_id: plan for plan in plans.plans}
                for session in sessions[:len(reuse.reused_session_ids)]:
                    plan = plan_by_id[session.session_id]
                    if session.date != plan.date or session.topic != plan.topic:
                        raise ValueError(
                            f"reused {session.session_id} does not match its selected Plan"
                        )
            print(f"resumed {len(sessions)}/{len(plans.plans)} completed sessions", flush=True)
        failed_plans: list = []
        for plan in plans.plans[len(sessions):]:
            print(f"generating {plan.session_id}...", flush=True)
            recent = sessions[-cfg.context_sessions:]
            story_so_far = [
                {
                    "session_id": session.session_id,
                    "date": session.date,
                    "topic": session.topic,
                    "summary": session.summary,
                }
                for session in sessions
            ]
            session_log_dir = audit_dir / "sessions" / plan.session_id
            session_log_dir.mkdir(parents=True, exist_ok=True)
            try:
                session = self._generate_one_session(
                    case, plan, recent, cfg, session_log_dir,
                    story_so_far=story_so_far, life_anchor=plans.life_anchor,
                    blueprint=blueprint,
                )
            except Exception as exc:
                err_msg = f"{plan.session_id} failed and skipped: {exc}"
                print(f"ERROR: {err_msg}", flush=True)
                self._write_log(session_log_dir / "98_session_error.json", {
                    "component": "pipeline", "error": err_msg,
                    "session_id": plan.session_id,
                })
                failed_plans.append(plan)
                continue
            sessions.append(session)
            print(f"completed {plan.session_id} ({len(sessions)}/{len(plans.plans)})", flush=True)
            checkpoint_path.write_text(
                json.dumps([s.model_dump() for s in sessions], ensure_ascii=False, indent=2), encoding="utf-8"
            )

        eval_results: list[EvalGenerationResult] = []
        eval_candidates_path = output_dir / "eval_candidates.json"
        eval_generation_error: str | None = None
        run_eval_candidates = cfg.run_eval or run_eval_examples
        if run_eval_candidates:
            if blueprint is None:
                eval_generation_error = "EvalGenerator requires dataset_blueprint.json"
            else:
                completed_ids = {session.session_id for session in sessions}
                missing_sessions = [session_id for session_id in expected if session_id not in completed_ids]
                if missing_sessions:
                    eval_generation_error = (
                        "EvalGenerator was skipped because historical sessions are incomplete: "
                        + ", ".join(missing_sessions)
                    )
                else:
                    eval_log_root = audit_dir / "evals"
                    checkpoint_eval_path = output_dir / "checkpoint_eval_candidates.json"
                    resume_eval_path = (
                        eval_candidates_path if eval_candidates_path.exists()
                        else checkpoint_eval_path if checkpoint_eval_path.exists() else None
                    )
                    if resume_eval_path:
                        saved_eval = json.loads(resume_eval_path.read_text(encoding="utf-8"))
                        if (
                            isinstance(saved_eval, dict)
                            and saved_eval.get("blueprint_id")
                            and saved_eval["blueprint_id"] != blueprint.blueprint_id
                        ):
                            raise ValueError(
                                "saved EvalGenerator candidates belong to a different Blueprint"
                            )
                        saved_results = saved_eval.get("results", []) if isinstance(saved_eval, dict) else saved_eval
                        eval_results = [
                            EvalGenerationResult.model_validate(item) for item in saved_results
                        ]
                        expected_outline_prefix = [
                            outline.outline_id
                            for outline in blueprint.eval_outlines[:len(eval_results)]
                        ]
                        if [item.outline_id for item in eval_results] != expected_outline_prefix:
                            raise ValueError("saved EvalGenerator candidates are not a valid blueprint prefix")
                        for saved_result, saved_outline in zip(
                            eval_results, blueprint.eval_outlines
                        ):
                            for candidate in saved_result.candidates:
                                if (
                                    candidate.target_label != saved_outline.target_label
                                    or candidate.target_emotion != saved_outline.target_emotion
                                    or candidate.blueprint_cue_id != saved_outline.cue_id
                                    or candidate.history_cutoff != saved_outline.history_cutoff
                                ):
                                    raise ValueError(
                                        "saved EvalGenerator candidate metadata does not match "
                                        f"Blueprint outline {saved_outline.outline_id}"
                                    )
                        print(
                            f"resumed {len(eval_results)}/{len(blueprint.eval_outlines)} eval outlines",
                            flush=True,
                        )
                    for outline in blueprint.eval_outlines[len(eval_results):]:
                        print(f"generating eval {outline.outline_id}...", flush=True)
                        result = self.eval_generator.run(case, sessions, outline, blueprint)
                        eval_results.append(result)
                        eval_log_dir = eval_log_root / outline.outline_id
                        visible_session_ids: list[str] = []
                        for visible_session in sessions:
                            visible_session_ids.append(visible_session.session_id)
                            if visible_session.session_id == outline.history_cutoff:
                                break
                        self._write_log(eval_log_dir / "01_generator.json", {
                            "component": "eval_generator",
                            "llm": self._llm_metadata("eval_generator"),
                            "input": {
                                "eval_outline": outline.model_dump(),
                                "history_cutoff": outline.history_cutoff,
                                "visible_session_ids": visible_session_ids,
                                "candidate_count": 3,
                            },
                            "output": result.model_dump(),
                        })
                        checkpoint_eval_path.write_text(
                            json.dumps(
                                [item.model_dump() for item in eval_results],
                                ensure_ascii=False, indent=2,
                            ),
                            encoding="utf-8",
                        )
                        print(
                            f"completed eval {outline.outline_id} "
                            f"({len(eval_results)}/{len(blueprint.eval_outlines)})",
                            flush=True,
                        )
                    eval_candidates_path.write_text(
                        json.dumps({
                            "stage": "generator_only",
                            "blueprint_id": blueprint.blueprint_id,
                            "results": [item.model_dump() for item in eval_results],
                        }, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
        if eval_generation_error:
            print(f"ERROR: {eval_generation_error}", flush=True)

        eval_examples: list[EvalSample] = []
        eval_examples_path = output_dir / "eval_examples.json"
        eval_example_error: str | None = None
        if run_eval_examples:
            if eval_generation_error:
                eval_example_error = eval_generation_error
            elif blueprint is None or len(eval_results) != len(blueprint.eval_outlines):
                eval_example_error = "Eval examples require a complete set of Generator candidates"
            else:
                checkpoint_examples_path = output_dir / "checkpoint_eval_examples.json"
                resume_examples_path = (
                    eval_examples_path if eval_examples_path.exists()
                    else checkpoint_examples_path if checkpoint_examples_path.exists() else None
                )
                if resume_examples_path:
                    saved_examples = json.loads(
                        resume_examples_path.read_text(encoding="utf-8")
                    )
                    if (
                        isinstance(saved_examples, dict)
                        and saved_examples.get("blueprint_id")
                        and saved_examples["blueprint_id"] != blueprint.blueprint_id
                    ):
                        raise ValueError(
                            "saved Eval examples belong to a different Blueprint"
                        )
                    saved_samples = (
                        saved_examples.get("eval_examples", [])
                        if isinstance(saved_examples, dict) else saved_examples
                    )
                    eval_examples = [EvalSample.model_validate(item) for item in saved_samples]
                    expected_example_prefix = [
                        outline.outline_id
                        for outline in blueprint.eval_outlines[:len(eval_examples)]
                    ]
                    if [item.sample_id for item in eval_examples] != expected_example_prefix:
                        raise ValueError(
                            "saved Eval examples are not a valid Blueprint prefix"
                        )
                    for saved_sample, saved_outline in zip(
                        eval_examples, blueprint.eval_outlines
                    ):
                        expected_cue = (
                            saved_outline.cue_id
                            if saved_outline.target_label == "triggered" else "none"
                        )
                        if (
                            saved_sample.history_cutoff != saved_outline.history_cutoff
                            or saved_sample.gold.trigger_label != saved_outline.target_label
                            or saved_sample.gold.current_emotion != saved_outline.target_emotion
                            or saved_sample.gold.trigger_cue_id != expected_cue
                        ):
                            raise ValueError(
                                "saved Eval example metadata does not match Blueprint outline "
                                f"{saved_outline.outline_id}"
                            )
                    print(
                        f"resumed {len(eval_examples)}/{len(blueprint.eval_outlines)} "
                        "final Eval examples",
                        flush=True,
                    )

                max_eval_cycles = max(1, cfg.max_revision_cycles)
                for outline_index in range(len(eval_examples), len(blueprint.eval_outlines)):
                    outline = blueprint.eval_outlines[outline_index]
                    generation = eval_results[outline_index]
                    eval_log_dir = audit_dir / "evals" / outline.outline_id
                    try:
                        resolution = self.eval_resolver.run(sessions, outline, blueprint)
                        self._write_log(eval_log_dir / "02_evidence_resolver.json", {
                            "component": "eval_resolver",
                            "llm": (
                                self._llm_metadata("eval_resolver")
                                if outline.target_label == "triggered" else None
                            ),
                            "input": {
                                "eval_outline": outline.model_dump(),
                                "required_evidence_session_ids": (
                                    outline.required_evidence_session_ids
                                ),
                            },
                            "output": resolution.model_dump(),
                        })
                    except Exception as exc:
                        eval_example_error = (
                            f"{outline.outline_id} evidence resolution failed: {exc}"
                        )
                        self._write_log(eval_log_dir / "98_eval_example_error.json", {
                            "component": "eval_resolver",
                            "error": eval_example_error,
                        })
                        break

                    finalized = None
                    cycle_errors: list[str] = []
                    regenerate_candidates = False
                    for cycle in range(1, max_eval_cycles + 1):
                        if cycle > 1 and regenerate_candidates:
                            try:
                                generation = self.eval_generator.run(
                                    case, sessions, outline, blueprint
                                )
                            except Exception as exc:
                                error_detail = describe_exception(exc)
                                cycle_errors.append(
                                    f"EvalGenerator retry failed: {error_detail}"
                                )
                                self._write_log(
                                    eval_log_dir / f"01_generator_retry_{cycle:02d}_error.json",
                                    {
                                        "component": "eval_generator",
                                        "error": error_detail,
                                    },
                                )
                                regenerate_candidates = True
                                continue
                            eval_results[outline_index] = generation
                            eval_candidates_path.write_text(
                                json.dumps({
                                    "stage": "generator_only",
                                    "blueprint_id": blueprint.blueprint_id,
                                    "results": [item.model_dump() for item in eval_results],
                                }, ensure_ascii=False, indent=2),
                                encoding="utf-8",
                            )
                            self._write_log(
                                eval_log_dir / f"01_generator_retry_{cycle:02d}.json",
                                {
                                    "component": "eval_generator",
                                    "reason": cycle_errors[-1] if cycle_errors else "retry",
                                    "output": generation.model_dump(),
                                },
                            )
                            regenerate_candidates = False

                        candidates = build_eval_candidate_list(
                            generation, outline, resolution
                        )
                        precheck_issues = [
                            validate_eval_candidate(candidate, outline, blueprint)
                            for candidate in candidates.candidates
                        ]
                        self._write_log(
                            eval_log_dir / f"03_precheck_cycle_{cycle:02d}.json",
                            {"issues_by_candidate": precheck_issues},
                        )
                        if all(precheck_issues):
                            cycle_errors.append(
                                "all candidates failed deterministic precheck: "
                                + repr(precheck_issues)
                            )
                            regenerate_candidates = True
                            continue

                        try:
                            selection = self.eval_verifier.run(
                                case, sessions, outline, candidates, blueprint,
                                precheck_issues=precheck_issues,
                            )
                        except Exception as exc:
                            error_detail = describe_exception(exc)
                            cycle_errors.append(
                                f"EvalVerifier request failed: {error_detail}"
                            )
                            self._write_log(
                                eval_log_dir / f"04_verifier_cycle_{cycle:02d}_error.json",
                                {
                                    "component": "eval_verifier",
                                    "error": error_detail,
                                    "retry_same_candidates": True,
                                },
                            )
                            # generate() has already exhausted its configured API
                            # retries. A transport failure says nothing about
                            # candidate quality, so keep the candidates/checkpoint
                            # and return control to the Web retry button.
                            regenerate_candidates = False
                            break
                        self._write_log(
                            eval_log_dir / f"04_verifier_cycle_{cycle:02d}.json",
                            {
                                "component": "eval_verifier",
                                "llm": self._llm_metadata("eval_verifier"),
                                "input": {
                                    "outline_id": outline.outline_id,
                                    "precheck_issues": precheck_issues,
                                },
                                "output": selection.model_dump(),
                            },
                        )
                        if selection.reject_all:
                            cycle_errors.append(
                                "verifier rejected all candidates: "
                                + "; ".join(selection.issues)
                            )
                            regenerate_candidates = True
                            continue
                        if precheck_issues[selection.selected_index]:
                            cycle_errors.append(
                                "verifier selected a candidate with deterministic errors: "
                                + "; ".join(precheck_issues[selection.selected_index])
                            )
                            regenerate_candidates = True
                            continue
                        try:
                            finalized = self.gold_finalizer.finalize(
                                case, sessions,
                                candidates.candidates[selection.selected_index],
                                outline_index + 1,
                                sample_id=outline.outline_id,
                            )
                        except Exception as exc:
                            cycle_errors.append(f"finalizer rejected candidate: {exc}")
                            regenerate_candidates = True
                            continue
                        self._write_log(eval_log_dir / "05_finalizer.json", {
                            "component": "gold_finalizer",
                            "selected_index": selection.selected_index,
                            "output": finalized.model_dump(),
                        })
                        break

                    if finalized is None:
                        eval_example_error = (
                            f"{outline.outline_id} could not produce a valid Eval example "
                            f"after {max_eval_cycles} cycles: " + " | ".join(cycle_errors)
                        )
                        self._write_log(eval_log_dir / "98_eval_example_error.json", {
                            "component": "eval_example_pipeline",
                            "error": eval_example_error,
                        })
                        break

                    eval_examples.append(finalized)
                    checkpoint_examples_path.write_text(
                        json.dumps(
                            [item.model_dump() for item in eval_examples],
                            ensure_ascii=False, indent=2,
                        ),
                        encoding="utf-8",
                    )
                    print(
                        f"completed Eval example {outline.outline_id} "
                        f"({len(eval_examples)}/{len(blueprint.eval_outlines)})",
                        flush=True,
                    )

                if not eval_example_error and len(eval_examples) == len(blueprint.eval_outlines):
                    eval_examples_path.write_text(
                        json.dumps({
                            "stage": "finalized",
                            "blueprint_id": blueprint.blueprint_id,
                            "eval_examples": [item.model_dump() for item in eval_examples],
                        }, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
        if eval_example_error:
            print(f"ERROR: {eval_example_error}", flush=True)

        dataset_id = f"{self.config.dataset_id_prefix}-{case.case_id}"
        dialogues = []
        for s in sessions:
            item = s.model_dump()
            item.pop("summary", None)
            dialogues.append(item)
        finalized_examples = (
            eval_examples
            if blueprint is not None
            and len(eval_examples) == len(blueprint.eval_outlines)
            and not eval_example_error
            else []
        )
        benchmark = Benchmark(
            dataset_id=dataset_id,
            character_profile=case.character_profile,
            dialogues=dialogues,
            eval_samples=finalized_examples,
        )
        # expected for QA only includes sessions that actually completed;
        # failed sessions are reported separately so QA does not false-flag id mismatch.
        completed_expected = [s.session_id for s in sessions]
        benchmark_path = output_dir / "benchmark.json"
        qa_path = output_dir / "qa_report.json"
        benchmark_path.write_text(benchmark.model_dump_json(indent=2), encoding="utf-8")
        qa = None
        if self.config.validation.qa:
            qa = validate_sessions(dataset_id, case, sessions, self.llm.records, completed_expected)
            qa_path.write_text(qa.model_dump_json(indent=2), encoding="utf-8")
        elif qa_path.exists():
            # Do not leave a stale report suggesting QA ran for this configuration.
            qa_path.unlink()
        self._write_log(audit_dir / "99_pipeline_result.json", {
            "benchmark_path": str(benchmark_path),
            "qa_path": str(qa_path) if qa else None,
            "qa": qa.model_dump() if qa else None,
            "validation_config": self._validation_settings(),
            "eval_generated": bool(eval_results),
            "eval_stage": (
                "finalized" if finalized_examples
                else "generator_only" if run_eval_candidates and eval_results
                else "disabled"
            ),
            "eval_candidates_path": str(eval_candidates_path) if eval_results else None,
            "eval_generation_error": eval_generation_error,
            "eval_examples_path": str(eval_examples_path) if finalized_examples else None,
            "eval_example_error": eval_example_error,
            "failed_session_ids": [p.session_id for p in failed_plans],
            "completed_session_count": len(sessions),
            "planned_session_count": len(plans.plans),
        })
        self._write_log_index(audit_dir, case.case_id)
        return benchmark_path, qa_path if qa else None

    def _generate_one_session(self, case, plan, recent, cfg, session_log_dir,
                              story_so_far: list[dict] | None = None,
                              life_anchor=None, blueprint=None) -> Session:
        """Generate one session, then run only the enabled validation stages.

        Raises on failure so the caller can decide to skip or abort.
        """
        writer_input = {
            "case_spec": case.model_dump(), "session_plan": plan.model_dump(),
            "immutable_facts": case.immutable_facts(),
            "life_anchor": life_anchor.model_dump() if life_anchor else None,
            "dataset_blueprint": blueprint.model_dump() if blueprint else None,
            "story_so_far": story_so_far or [],
            "recent_history": [s.model_dump() for s in recent],
        }
        session = self.writer.run(
            case, plan.model_dump(), recent,
            story_so_far=story_so_far, life_anchor=life_anchor, blueprint=blueprint,
        )
        self._write_log(session_log_dir / "01_writer.json", {
            "component": "session_writer", "llm": self._llm_metadata("session_writer"),
            "input": writer_input, "output": session.model_dump(),
        })
        if self.config.validation.structure or self.config.validation.semantic:
            session = self._run_revision_cycle(case, plan, recent, session, cfg, session_log_dir)
        if self.config.validation.naturalness:
            session = self._run_naturalness_stage(case, plan, session, session_log_dir)
        self._write_log(session_log_dir / "99_final_session.json", {
            "component": "pipeline",
            "validation_config": self._validation_settings(),
            "output": session.model_dump(),
        })
        return session

    def _run_revision_cycle(self, case, plan, recent, session, cfg, session_log_dir) -> Session:
        """Run enabled structural/semantic checks and their minimal revisions."""
        for cycle in range(cfg.max_revision_cycles):
            verdict = None
            verifier_kind = ""
            if self.config.validation.structure:
                structural = self._structural_verdict(session, plan.round_count)
                self._write_log(session_log_dir / f"{cycle + 2:02d}_cycle_{cycle + 1}_structural_verifier.json", {
                    "component": "deterministic_structural_verifier",
                    "input": {"session": session.model_dump(), "expected_round_count": plan.round_count},
                    "output": structural.model_dump(),
                })
                if structural.result == "revise":
                    verdict = structural
                    verifier_kind = "deterministic_structural_verifier"
            if verdict is None and self.config.validation.semantic:
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
            if verdict is None or verdict.result == "pass":
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
        if self.config.validation.structure:
            final_structure = self._structural_verdict(session, plan.round_count)
            if final_structure.result != "pass":
                raise RuntimeError(f"{plan.session_id} failed deterministic structure checks: {final_structure.model_dump()}")
        return session

    def _run_naturalness_stage(self, case, plan, session, session_log_dir) -> Session:
        """Run the optional naturalness check and one minimal revision."""
        natural = self.naturalness.run(case, session)
        self._write_log(session_log_dir / "90_naturalness_checker.json", {
            "component": "naturalness_checker", "llm": self._llm_metadata("naturalness_checker"),
            "input": {"conversation_style": case.character_profile.conversation_style, "session": session.model_dump()},
            "output": natural.model_dump(),
        })
        if natural.result != "revise":
            return session
        revised = self.reviser.run(case, plan.model_dump(), session, natural)
        post_structure = self._structural_verdict(revised, plan.round_count) if self.config.validation.structure else None
        self._write_log(session_log_dir / "91_naturalness_reviser.json", {
            "component": "session_reviser", "llm": self._llm_metadata("session_reviser"),
            "triggered_by": "naturalness_checker",
            "input": {
                "case_spec": case.model_dump(), "session_plan": plan.model_dump(),
                "session": session.model_dump(), "issues": natural.model_dump(),
            },
            "output": revised.model_dump(),
            "post_revision_structural_verdict": post_structure.model_dump() if post_structure else None,
        })
        if post_structure is None or post_structure.result == "pass":
            return revised
        return session

    def _validation_settings(self) -> dict[str, bool]:
        validation = self.config.validation
        return {
            "structure": validation.structure,
            "semantic": validation.semantic,
            "naturalness": validation.naturalness,
            "qa": validation.qa,
        }

    @staticmethod
    def _validate_plans_against_blueprint(plans: list, blueprint: DatasetBlueprint) -> None:
        """Prevent a resumed or generated local plan from drifting off its slot."""
        slot_by_id = {slot.session_id: slot for slot in blueprint.session_slots}
        fields = (
            "memory_role", "memory_id", "cue_id", "evidence_goal", "target_emotion",
            "relative_to_past", "depends_on_sessions",
        )
        errors: list[str] = []
        for plan in plans:
            slot = slot_by_id.get(plan.session_id)
            if slot is None:
                errors.append(f"{plan.session_id} 没有对应的蓝图槽位")
                continue
            changed = [field for field in fields if getattr(plan, field) != getattr(slot, field)]
            if changed:
                errors.append(f"{plan.session_id} 改写了蓝图字段：{', '.join(changed)}")
        if errors:
            raise ValueError(
                "session_plans.json is incompatible with dataset_blueprint.json: "
                + "; ".join(errors)
            )

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
