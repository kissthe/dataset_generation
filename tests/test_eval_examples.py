from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.components import (
    EvidenceResolver, blueprint_fingerprint, blueprint_id_for,
)
from src.config import BlueprintConstraints, ValidationConfig
from src.gold_finalizer import GoldFinalizer
from src.models import (
    BlueprintEvalOutline, CaseSpec, CueOption, CueSeed, CurrentInput,
    DatasetBlueprint, EmotionMemory, EvalDraftCandidate, EvalGenerationResult,
    EvalSelection, EvidenceResolution, LifeAnchor, Session, SessionPlan,
    SessionPlanList, SessionSlot, Turn,
)
from src.pipeline import GenerationPipeline


class _ResolverLLM:
    def __init__(self, turn_ids: list[str]) -> None:
        self.turn_ids = turn_ids
        self.payload = None

    def generate(self, _component, payload, _model):
        self.payload = payload
        return EvidenceResolution(
            outline_id="model-made-id", evidence_turn_ids=self.turn_ids
        )


class _Generator:
    def __init__(self) -> None:
        self.calls = 0

    def run(self, _case, _sessions, outline, _blueprint):
        self.calls += 1
        current = CurrentInput(
            input_type="text",
            text="刚在柜子里看到那只缺口蓝杯子，手一下停住了。",
            image_refs=[], cue_type="object",
            cue_options=[
                CueOption(cue_id="C01", name="缺口蓝杯子"),
                CueOption(cue_id="local_01", name="厨房柜子"),
            ],
        )
        return EvalGenerationResult(
            outline_id=outline.outline_id,
            candidates=[
                EvalDraftCandidate(
                    current_input=current,
                    target_label=outline.target_label,
                    target_emotion=outline.target_emotion,
                    blueprint_cue_id=outline.cue_id,
                    history_cutoff=outline.history_cutoff,
                )
                for _ in range(3)
            ],
        )


class _Resolver:
    def run(self, _sessions, outline, _blueprint):
        return EvidenceResolution(
            outline_id=outline.outline_id,
            evidence_turn_ids=["A-S01_T01"],
        )


class _Verifier:
    def __init__(self, failures: int = 0) -> None:
        self.failures = failures
        self.calls = 0

    def run(self, *_args, **_kwargs):
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError("eval_verifier failed after retries: APITimeoutError")
        return EvalSelection(reject_all=False, selected_index=0, issues=[])


def _fixture():
    case = CaseSpec.model_validate({
        "case_id": "case-a", "name": "林澄",
        "core_emotional_event": "错过了一次重要的告别电话。",
    })
    session_id = "A-S01"
    constraints = BlueprintConstraints(
        encode_association_count=1,
        triggered_recall_count=0,
        memory_update_count=0,
        control_count=0,
        required_cue_types=("object",),
        eval_triggered_count=1,
        eval_insufficient_evidence_count=0,
        eval_not_triggered_count=0,
    )
    blueprint_id = blueprint_id_for(case, [session_id], 1, constraints)
    outline = BlueprintEvalOutline(
        outline_id="case-a-E01", target_label="triggered",
        target_emotion="sadness", history_cutoff=session_id,
        memory_id="M01", cue_id="C01", cue_specificity="exact",
        emotion_explicitness="behavioral",
        required_evidence_session_ids=[session_id],
        current_input_goal="缺口蓝杯子再次出现并带来局部停顿", negative_reason="",
    )
    slot = SessionSlot(
        session_id=session_id, memory_role="encode_association",
        memory_id="M01", cue_id="C01",
        evidence_goal="用户说明杯子和未接电话的个人关联",
        target_emotion="sadness", relative_to_past="not_applicable",
        depends_on_sessions=[],
    )
    blueprint = DatasetBlueprint(
        blueprint_id=blueprint_id,
        life_anchor=LifeAnchor(
            identity="普通上班族", recurring_scenes=["家中"],
            interests=["做饭"], ongoing_threads=["练习家常菜"],
        ),
        emotion_memory_map=[EmotionMemory(
            memory_id="M01", event_summary=case.core_emotional_event,
            emotion="sadness", emotional_meaning="那通电话无法重新接起",
            cue_seeds=[
                CueSeed(cue_id="C01", cue_type="object", canonical_form="缺口蓝杯子",
                        related_forms=["旧蓝杯"], personal_meaning="和未接电话相连"),
                CueSeed(cue_id="C02", cue_type="object", canonical_form="旧杯垫",
                        related_forms=["软木杯垫"], personal_meaning="和当时的桌面相连"),
                CueSeed(cue_id="C03", cue_type="object", canonical_form="纸质电话簿",
                        related_forms=["旧通讯录"], personal_meaning="和没拨出的回电相连"),
            ],
        )],
        session_slots=[slot], eval_outlines=[outline],
    )
    session = Session(
        session_id=session_id, topic="整理旧杯子", date="2026-01-01",
        turns=[
            Turn(
                turn_id="A-S01_T01", round_id="A-S01_R01", speaker="user",
                text="这只缺口蓝杯子一直会让我想到那通没接到的电话。",
                image_id=[], image_dir="", image_caption=[],
            ),
            Turn(
                turn_id="A-S01_T02", round_id="A-S01_R01", speaker="assistant",
                text="原来你每次停一下，是因为它连着那段遗憾。",
                image_id=[], image_dir="", image_caption=[],
            ),
        ],
        summary="用户建立缺口蓝杯子与未接电话的个人关联。",
    )
    return case, constraints, blueprint, outline, slot, session


class EvalExampleTests(unittest.TestCase):
    def test_resolver_locks_outline_and_accepts_only_scoped_user_turns(self) -> None:
        _case, _constraints, blueprint, outline, _slot, session = _fixture()
        llm = _ResolverLLM(["A-S01_T01"])

        result = EvidenceResolver(llm).run([session], outline, blueprint)

        self.assertEqual(result.outline_id, outline.outline_id)
        self.assertEqual(result.evidence_turn_ids, ["A-S01_T01"])
        self.assertEqual(
            llm.payload["required_evidence_sessions"][0]["session_id"], "A-S01"
        )

    def test_resolver_rejects_assistant_turn_as_evidence(self) -> None:
        _case, _constraints, blueprint, outline, _slot, session = _fixture()
        with self.assertRaisesRegex(ValueError, "non-user"):
            EvidenceResolver(_ResolverLLM(["A-S01_T02"])).run(
                [session], outline, blueprint
            )

    def _run_eval_pipeline(self, verifier):
        case, constraints, blueprint, outline, slot, session = _fixture()
        plan = SessionPlan(
            session_id=slot.session_id, date=session.date, topic=session.topic,
            story_beat="整理杯子时自然说起它的个人意义。",
            outline_function="建立历史证据。", round_count=1,
            session_type="core_echo", scene="晚饭后的厨房",
            user_intent="和朋友说起整理东西时的停顿", continuity_hook="",
            life_thread="one_off", thread_progress="完成整理",
            interaction_mode="share", **slot.model_dump(exclude={"session_id"}),
        )
        generation = SimpleNamespace(
            session_count=1, planner_batch_size=1, min_rounds=1, max_rounds=1,
            context_sessions=1, seed=42, max_retries=1, max_revision_cycles=2,
            run_eval=False, run_eval_examples=True, stop_after_planning=False,
        )
        pipeline = GenerationPipeline.__new__(GenerationPipeline)
        pipeline.config = SimpleNamespace(
            generation=generation, blueprint_constraints=constraints,
            validation=ValidationConfig(), dataset_id_prefix="test",
        )
        generator = _Generator()
        pipeline.eval_generator = generator
        pipeline.eval_resolver = _Resolver()
        pipeline.eval_verifier = verifier
        pipeline.gold_finalizer = GoldFinalizer(42)
        pipeline._llm_metadata = lambda _component: {}
        pipeline.llm = SimpleNamespace(records=[])

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            case_path = root / "case.json"
            output = root / "output"
            output.mkdir()
            case_path.write_text(case.model_dump_json(indent=2), encoding="utf-8")
            (output / "dataset_blueprint.json").write_text(
                blueprint.model_dump_json(indent=2), encoding="utf-8"
            )
            (output / "session_plans.json").write_text(
                SessionPlanList(
                    plans=[plan], life_anchor=blueprint.life_anchor,
                    blueprint_id=blueprint.blueprint_id,
                    blueprint_fingerprint=blueprint_fingerprint(blueprint),
                ).model_dump_json(indent=2),
                encoding="utf-8",
            )
            (output / "checkpoint_sessions.json").write_text(
                json.dumps([session.model_dump()], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            artifact, qa = pipeline.run(case_path, output)
            benchmark = json.loads(artifact.read_text(encoding="utf-8"))
            examples = (
                json.loads((output / "eval_examples.json").read_text(encoding="utf-8"))[
                    "eval_examples"
                ]
                if (output / "eval_examples.json").exists() else []
            )
            pipeline_result = json.loads(
                (output / "logs" / "99_pipeline_result.json").read_text(encoding="utf-8")
            )

        return qa, benchmark, examples, pipeline_result, generator, outline

    def test_pipeline_writes_final_eval_examples_and_benchmark_samples(self) -> None:
        qa, benchmark, examples, _result, generator, outline = self._run_eval_pipeline(
            _Verifier()
        )

        self.assertIsNone(qa)
        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0]["sample_id"], outline.outline_id)
        self.assertEqual(examples[0]["history_cutoff"], "A-S01")
        self.assertEqual(examples[0]["gold"]["evidence_turn_ids"], ["A-S01_T01"])
        self.assertEqual(benchmark["eval_samples"], examples)
        self.assertEqual(generator.calls, 1)

    def test_verifier_timeout_is_checkpointed_instead_of_crashing_pipeline(self) -> None:
        qa, benchmark, examples, result, generator, _outline = self._run_eval_pipeline(
            _Verifier(failures=1)
        )

        self.assertIsNone(qa)
        self.assertEqual(examples, [])
        self.assertEqual(benchmark["eval_samples"], [])
        self.assertIn("APITimeoutError", result["eval_example_error"])
        self.assertEqual(generator.calls, 1)


if __name__ == "__main__":
    unittest.main()
