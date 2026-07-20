import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from streamlit.testing.v1 import AppTest

from src.components import blueprint_fingerprint
from src.models import (
    BlueprintCandidate, BlueprintCandidateSelection, BlueprintEvalOutline, CueSeed,
    CandidateSelection, DatasetBlueprint, EmotionMemory, LifeAnchor, PlanCandidate,
    PlanningCandidateManifest, Session, SessionPlan, SessionPlanList, SessionSlot,
    Turn,
)
from src.pipeline import GenerationPipeline


def _blueprint(cue_id: str = "C01", role: str = "encode_association") -> DatasetBlueprint:
    return DatasetBlueprint(
        blueprint_id="BP-CANONICAL",
        life_anchor=LifeAnchor(
            identity="普通上班族",
            recurring_scenes=["通勤", "家中"],
            interests=["做饭"],
            ongoing_threads=["练习家常菜"],
        ),
        emotion_memory_map=[EmotionMemory(
            memory_id="M01",
            event_summary="一次未完成的告别",
            emotion="sadness",
            emotional_meaning="仍留有遗憾",
            cue_seeds=[
                CueSeed(cue_id="C01", cue_type="object", canonical_form="旧杯子", related_forms=[], personal_meaning="旧日联系"),
                CueSeed(cue_id="C02", cue_type="scene", canonical_form="雨夜厨房", related_forms=[], personal_meaning="当时的场景"),
                CueSeed(cue_id="C03", cue_type="utterance", canonical_form="到家说一声", related_forms=[], personal_meaning="熟悉的叮嘱"),
            ],
        )],
        session_slots=[SessionSlot(
            session_id="CASE-S01",
            memory_role=role,
            memory_id="M01" if role != "control" else "none",
            cue_id=cue_id if role != "control" else "none",
            evidence_goal="建立个人关联" if role != "control" else "保持无触发",
            target_emotion="sadness" if role != "control" else "neutral",
            relative_to_past="not_applicable",
            depends_on_sessions=[],
        )],
        eval_outlines=[BlueprintEvalOutline(
            outline_id="CASE-E01",
            target_label="not_triggered",
            target_emotion="neutral",
            history_cutoff="CASE-S01",
            memory_id="none",
            cue_id="none",
            cue_specificity="none",
            emotion_explicitness="none",
            required_evidence_session_ids=[],
            current_input_goal="普通近况",
            negative_reason="没有已建立的个人线索",
        )],
    )


def _plan_for_blueprint(blueprint: DatasetBlueprint) -> SessionPlanList:
    slot = blueprint.session_slots[0]
    return SessionPlanList(
        plans=[SessionPlan(
            session_id="CASE-S01",
            date="2026-01-01",
            topic="第一次尝试做汤",
            story_beat="下班后分享做汤失败的小插曲",
            outline_function="建立可延续的生活线",
            round_count=4,
            session_type="daily_life",
            scene="晚饭后的厨房",
            user_intent="和朋友分享近况",
            continuity_hook="周末再试一次",
            life_thread="学习做饭",
            thread_progress="第一次尝试",
            interaction_mode="share",
            memory_role=slot.memory_role,
            memory_id=slot.memory_id,
            cue_id=slot.cue_id,
            evidence_goal=slot.evidence_goal,
            target_emotion=slot.target_emotion,
            relative_to_past=slot.relative_to_past,
            depends_on_sessions=slot.depends_on_sessions,
        )],
        life_anchor=blueprint.life_anchor,
        blueprint_id=blueprint.blueprint_id,
        blueprint_fingerprint=blueprint_fingerprint(blueprint),
    )


def test_select_candidates_materializes_canonical_files(tmp_path: Path) -> None:
    blueprint = _blueprint()
    manifest = PlanningCandidateManifest(
        case_id="case",
        session_ids=["CASE-S01"],
        blueprint_candidate_count=1,
        plan_candidate_count=1,
        blueprint_candidates=[BlueprintCandidate(
            candidate_id="BP-01", title="蓝图一", summary="旧杯子", blueprint=blueprint
        )],
        plan_candidates=[PlanCandidate(
            candidate_id="PLAN-01", blueprint_candidate_id="BP-01",
            title="计划一", summary="做汤", plan=_plan_for_blueprint(blueprint)
        )],
        blueprint_selection=BlueprintCandidateSelection(
            blueprint_candidate_id="BP-01", selected_at="2026-01-01T00:00:00Z"
        ),
    )
    (tmp_path / "planning_candidates.json").write_text(
        manifest.model_dump_json(indent=2), encoding="utf-8"
    )
    pipeline = object.__new__(GenerationPipeline)

    blueprint_path, plan_path = pipeline.select_candidates(tmp_path, "BP-01", "PLAN-01")

    assert blueprint_path.exists()
    selected = SessionPlanList.model_validate_json(plan_path.read_text(encoding="utf-8"))
    assert selected.blueprint_id == "BP-CANONICAL"
    assert selected.plans[0].topic == "第一次尝试做汤"
    saved_manifest = PlanningCandidateManifest.model_validate_json(
        (tmp_path / "planning_candidates.json").read_text(encoding="utf-8")
    )
    assert saved_manifest.selection is not None
    assert saved_manifest.selection.plan_candidate_id == "PLAN-01"


def test_select_candidates_rejects_plan_from_another_blueprint(tmp_path: Path) -> None:
    first = _blueprint()
    second = _blueprint(role="control").model_copy(update={"blueprint_id": "BP-OTHER"})
    manifest = PlanningCandidateManifest(
        case_id="case",
        session_ids=["CASE-S01"],
        blueprint_candidate_count=2,
        plan_candidate_count=1,
        blueprint_candidates=[
            BlueprintCandidate(
                candidate_id="BP-01", title="蓝图一", summary="旧杯子", blueprint=first
            ),
            BlueprintCandidate(
                candidate_id="BP-02", title="蓝图二", summary="控制场景", blueprint=second
            ),
        ],
        plan_candidates=[PlanCandidate(
            candidate_id="PLAN-01", blueprint_candidate_id="BP-01",
            title="计划一", summary="做汤", plan=_plan_for_blueprint(first),
        )],
        blueprint_selection=BlueprintCandidateSelection(
            blueprint_candidate_id="BP-01", selected_at="2026-01-01T00:00:00Z"
        ),
    )
    (tmp_path / "planning_candidates.json").write_text(
        manifest.model_dump_json(indent=2), encoding="utf-8"
    )
    pipeline = object.__new__(GenerationPipeline)

    with pytest.raises(ValueError, match="not generated from this Blueprint"):
        pipeline.select_candidates(tmp_path, "BP-02", "PLAN-01")


def test_prepare_plans_uses_only_the_selected_blueprint(tmp_path: Path) -> None:
    first = _blueprint()
    second = _blueprint(role="control").model_copy(update={"blueprint_id": "BP-OTHER"})
    manifest = PlanningCandidateManifest(
        case_id="case",
        session_ids=["CASE-S01"],
        blueprint_candidate_count=2,
        blueprint_candidates=[
            BlueprintCandidate(
                candidate_id="BP-01", title="蓝图一", summary="旧杯子", blueprint=first
            ),
            BlueprintCandidate(
                candidate_id="BP-02", title="蓝图二", summary="控制场景", blueprint=second
            ),
        ],
        plan_candidates=[],
    )
    (tmp_path / "planning_candidates.json").write_text(
        manifest.model_dump_json(indent=2), encoding="utf-8"
    )
    case_path = tmp_path / "case_spec.json"
    case_path.write_text(
        '{"case_id":"case","name":"林澄",'
        '"core_emotional_event":"林澄错过奶奶去世前最后一次通话，留有遗憾。"}',
        encoding="utf-8",
    )

    class FakePlanner:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def run(self, *args, **kwargs) -> SessionPlanList:
            self.calls.append(kwargs)
            return _plan_for_blueprint(kwargs["blueprint"])

    pipeline = object.__new__(GenerationPipeline)
    pipeline.config = SimpleNamespace(generation=SimpleNamespace(
        planner_batch_size=1, min_rounds=2, max_rounds=8,
    ))
    pipeline.planner = FakePlanner()

    pipeline.prepare_plan_candidates(
        case_path, tmp_path, blueprint_candidate_id="BP-02", candidate_count=2
    )

    saved = PlanningCandidateManifest.model_validate_json(
        (tmp_path / "planning_candidates.json").read_text(encoding="utf-8")
    )
    assert saved.blueprint_selection is not None
    assert saved.blueprint_selection.blueprint_candidate_id == "BP-02"
    assert len(saved.plan_candidates) == 2
    assert {item.blueprint_candidate_id for item in saved.plan_candidates} == {"BP-02"}
    assert all(call["blueprint"].blueprint_id == "BP-OTHER" for call in pipeline.planner.calls)
    assert pipeline.planner.calls[0]["candidate_context"] != pipeline.planner.calls[1]["candidate_context"]


def test_web_candidate_review_renders_both_sequential_stages(tmp_path: Path) -> None:
    blueprint = _blueprint()
    manifest = PlanningCandidateManifest(
        case_id="case",
        session_ids=["CASE-S01"],
        blueprint_candidate_count=1,
        blueprint_candidates=[BlueprintCandidate(
            candidate_id="BP-01", title="蓝图一", summary="旧杯子", blueprint=blueprint
        )],
        plan_candidates=[],
    )
    manifest_path = tmp_path / "planning_candidates.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    app = AppTest.from_file("web_app.py").run(timeout=30)
    app.session_state["last_output"] = str(tmp_path)
    app.session_state["plan_candidate_count"] = 2
    app.run(timeout=30)

    assert not app.exception
    generate_button = next(
        button for button in app.button
        if button.label == "选定 BP-01，生成 2 个 Plan"
    )

    plan = _plan_for_blueprint(blueprint)
    completed_manifest = manifest.model_copy(update={
        "plan_candidate_count": 2,
        "plan_candidates": [
            PlanCandidate(
                candidate_id=f"PLAN-{index:02d}", blueprint_candidate_id="BP-01",
                title=f"计划{index}", summary="做汤", plan=plan,
            )
            for index in (1, 2)
        ],
        "blueprint_selection": BlueprintCandidateSelection(
            blueprint_candidate_id="BP-01", selected_at="2026-01-01T00:00:00Z"
        ),
    })
    runtime_dir = tmp_path / ".runtime"
    runtime_dir.mkdir()
    (runtime_dir / "config.json").write_text(
        Path("config.json").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (tmp_path / "case_spec.json").write_text(
        '{"case_id":"case","name":"林澄",'
        '"core_emotional_event":"林澄错过奶奶去世前最后一次通话，留有遗憾。"}',
        encoding="utf-8",
    )

    class SuccessfulProcess:
        stdout: list[str] = []

        @staticmethod
        def poll() -> int:
            return 0

        @staticmethod
        def wait() -> int:
            return 0

    def start_successful_process(*args, **kwargs) -> SuccessfulProcess:
        manifest_path.write_text(
            completed_manifest.model_dump_json(indent=2), encoding="utf-8"
        )
        return SuccessfulProcess()

    with patch("subprocess.Popen", side_effect=start_successful_process):
        generate_button.click().run(timeout=30)

    assert not app.exception
    assert any(button.label == "选定这个 Plan 并继续生成" for button in app.button)
    assert any("已基于 BP-01 生成 2 个 Plan 候选" in item.value for item in app.success)

    continue_button = next(
        button for button in app.button
        if button.label == "选定这个 Plan 并继续生成"
    )
    session = Session(
        session_id="CASE-S01",
        topic="第一次尝试做汤",
        date="2026-01-01",
        turns=[
            Turn(
                turn_id="CASE-S01-T01", round_id="CASE-S01-R01", speaker="user",
                text="今天第一次试着做汤。", image_id=[], image_dir="",
                image_caption=[],
            ),
            Turn(
                turn_id="CASE-S01-T02", round_id="CASE-S01-R01", speaker="assistant",
                text="味道怎么样？", image_id=[], image_dir="", image_caption=[],
            ),
        ],
        summary="第一次尝试做汤。",
    )
    selected_manifest = completed_manifest.model_copy(update={
        "selection": CandidateSelection(
            blueprint_candidate_id="BP-01",
            plan_candidate_id="PLAN-01",
            selected_at="2026-01-01T00:00:01Z",
        ),
    })

    def complete_sessions(*args, **kwargs) -> SuccessfulProcess:
        manifest_path.write_text(
            selected_manifest.model_dump_json(indent=2), encoding="utf-8"
        )
        (tmp_path / "dataset_blueprint.json").write_text(
            blueprint.model_dump_json(indent=2), encoding="utf-8"
        )
        (tmp_path / "session_plans.json").write_text(
            plan.model_dump_json(indent=2), encoding="utf-8"
        )
        (tmp_path / "checkpoint_sessions.json").write_text(
            f"[{session.model_dump_json()}]", encoding="utf-8"
        )
        return SuccessfulProcess()

    with patch("subprocess.Popen", side_effect=complete_sessions):
        continue_button.click().run(timeout=30)

    assert not app.exception
    assert any(button.label == "生成 Eval 候选" for button in app.button)
    assert any(
        "Session 已全部生成完成，可以继续生成 Eval 候选" in item.value
        for item in app.success
    )


def test_web_reuses_historical_blueprint_plan_and_session_prefix(tmp_path: Path) -> None:
    source_dir = tmp_path / "source-run"
    source_dir.mkdir()
    blueprint = _blueprint()
    plan = _plan_for_blueprint(blueprint)
    session = Session(
        session_id="CASE-S01",
        topic="第一次尝试做汤",
        date="2026-01-01",
        turns=[
            Turn(
                turn_id="CASE-S01-T01", round_id="CASE-S01-R01", speaker="user",
                text="今天第一次试着做汤。", image_id=[], image_dir="", image_caption=[],
            ),
            Turn(
                turn_id="CASE-S01-T02", round_id="CASE-S01-R01", speaker="assistant",
                text="味道怎么样？", image_id=[], image_dir="", image_caption=[],
            ),
        ],
        summary="第一次尝试做汤。",
    )
    case_text = (
        '{"case_id":"case","name":"林澄",'
        '"core_emotional_event":"林澄错过奶奶去世前最后一次通话，留有遗憾。"}'
    )
    (source_dir / "case_spec.json").write_text(case_text, encoding="utf-8")
    (source_dir / "dataset_blueprint.json").write_text(
        blueprint.model_dump_json(indent=2), encoding="utf-8"
    )
    (source_dir / "session_plans.json").write_text(
        plan.model_dump_json(indent=2), encoding="utf-8"
    )
    (source_dir / "checkpoint_sessions.json").write_text(
        f"[{session.model_dump_json()}]", encoding="utf-8"
    )
    runtime_dir = source_dir / ".runtime"
    runtime_dir.mkdir()
    (runtime_dir / "config.json").write_text(
        Path("config.json").read_text(encoding="utf-8"), encoding="utf-8"
    )

    app = AppTest.from_file("web_app.py").run(timeout=30)
    app.session_state["output_root"] = str(tmp_path)
    app.session_state["run_name"] = "reuse-target"
    app.run(timeout=30)

    reuse_button = next(
        button for button in app.button
        if button.label == "复用完整 Session 并继续后续阶段"
    )

    class SuccessfulProcess:
        stdout: list[str] = []

        @staticmethod
        def poll() -> int:
            return 0

        @staticmethod
        def wait() -> int:
            return 0

    with patch("subprocess.Popen", return_value=SuccessfulProcess()):
        reuse_button.click().run(timeout=30)

    target_dir = tmp_path / "reuse-target"
    assert not app.exception
    assert (target_dir / "dataset_blueprint.json").exists()
    assert (target_dir / "session_plans.json").exists()
    assert (target_dir / "checkpoint_sessions.json").exists()
    provenance = json.loads(
        (target_dir / "reuse_provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["blueprint_source_run"] == str(source_dir)
    assert provenance["plan_source_run"] == str(source_dir)
    assert provenance["session_source_run"] == str(source_dir)
    assert provenance["reused_session_ids"] == ["CASE-S01"]

    eval_candidate_model = next(
        option for option in app.selectbox(key="model_eval_candidates").options
        if option != app.selectbox(key="model_eval_candidates").value
    )
    app.selectbox(key="model_eval_candidates").set_value(eval_candidate_model).run(timeout=30)
    eval_candidate_button = next(
        button for button in app.button if button.label == "生成 Eval 候选"
    )

    def complete_eval_candidates(*args, **kwargs) -> SuccessfulProcess:
        (target_dir / "eval_candidates.json").write_text(
            json.dumps({
                "stage": "generator_only",
                "results": [{
                    "outline_id": "CASE-E01",
                    "candidates": [{
                        "current_input": {
                            "text": "刚看到那只旧杯子，我忽然停了一下。",
                            "cue_options": [{"cue_id": "C01", "name": "旧杯子"}],
                        },
                        "target_label": "triggered",
                        "target_emotion": "nostalgia",
                        "blueprint_cue_id": "C01",
                    }],
                }],
            }),
            encoding="utf-8",
        )
        return SuccessfulProcess()

    with patch("subprocess.Popen", side_effect=complete_eval_candidates):
        eval_candidate_button.click().run(timeout=30)

    runtime_config = json.loads(
        (target_dir / ".runtime" / "config.json").read_text(encoding="utf-8")
    )
    assert runtime_config["components"]["eval_generator"]["model"] == eval_candidate_model
    assert runtime_config["generation"]["run_eval"] is True
    assert runtime_config["generation"]["run_eval_examples"] is False
    assert any(button.label == "生成正式 Eval Examples" for button in app.button)
    assert app.selectbox(key="post_eval_outline_reuse-target").value == "CASE-E01"
    assert any(
        "刚看到那只旧杯子" in item.value for item in app.markdown
    )

    eval_example_model = next(
        option for option in app.selectbox(key="model_eval_examples").options
        if option != app.selectbox(key="model_eval_examples").value
    )
    app.selectbox(key="model_eval_examples").set_value(eval_example_model).run(timeout=30)
    final_example_button = next(
        button for button in app.button if button.label == "生成正式 Eval Examples"
    )

    def fail_eval_examples(*args, **kwargs) -> SuccessfulProcess:
        log_dir = target_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        (log_dir / "99_pipeline_result.json").write_text(
            json.dumps({
                "failed_session_ids": [],
                "eval_generation_error": None,
                "eval_example_error": "E02 deterministic precheck failed",
            }),
            encoding="utf-8",
        )
        return SuccessfulProcess()

    with patch("subprocess.Popen", side_effect=fail_eval_examples):
        final_example_button.click().run(timeout=30)

    assert any(
        "E02 deterministic precheck failed" in item.value for item in app.error
    )
    final_example_button = next(
        button for button in app.button if button.label == "生成正式 Eval Examples"
    )

    def complete_eval_examples(*args, **kwargs) -> SuccessfulProcess:
        (target_dir / "eval_examples.json").write_text(
            json.dumps({"stage": "finalized", "eval_examples": [{}]}),
            encoding="utf-8",
        )
        (target_dir / "logs" / "99_pipeline_result.json").write_text(
            json.dumps({
                "failed_session_ids": [],
                "eval_generation_error": None,
                "eval_example_error": None,
            }),
            encoding="utf-8",
        )
        return SuccessfulProcess()

    with patch("subprocess.Popen", side_effect=complete_eval_examples):
        final_example_button.click().run(timeout=30)

    runtime_config = json.loads(
        (target_dir / ".runtime" / "config.json").read_text(encoding="utf-8")
    )
    assert runtime_config["components"]["eval_resolver"]["model"] == eval_example_model
    assert runtime_config["components"]["eval_verifier"]["model"] == eval_example_model
    assert runtime_config["generation"]["run_eval_examples"] is True
