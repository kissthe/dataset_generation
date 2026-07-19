from __future__ import annotations

import unittest
from pathlib import Path

from src.components import EvalGenerator
from src.models import (
    BlueprintEvalOutline, CaseSpec, CueOption, CueSeed, CurrentInput,
    DatasetBlueprint, EmotionMemory, EvalDraftCandidate, EvalGenerationResult,
    LifeAnchor, Session, SessionSlot, Turn,
)


ROOT = Path(__file__).parents[1]


class _EvalLLM:
    def __init__(self) -> None:
        self.payload = None

    def generate(self, _component, payload, _model):
        self.payload = payload
        current_input = CurrentInput(
            input_type="text", text="刚看到一只缺口蓝杯子，心里忽然沉了一下。",
            image_refs=[], cue_type="object",
            cue_options=[CueOption(cue_id="C01", name="缺口蓝杯子")],
        )
        return EvalGenerationResult(
            outline_id="model-invented-id",
            candidates=[
                EvalDraftCandidate(
                    current_input=current_input, target_label="not_triggered",
                    target_emotion="neutral", blueprint_cue_id="wrong",
                    history_cutoff="wrong",
                )
                for _ in range(3)
            ],
        )


class EvalGeneratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.case = CaseSpec.model_validate({
            "case_id": "case-a", "name": "林澄",
            "core_emotional_event": "错过了一次重要的告别电话。",
        })
        self.outline = BlueprintEvalOutline(
            outline_id="case-a-E01", target_label="triggered", target_emotion="sadness",
            history_cutoff="A-S01", memory_id="M01", cue_id="C01",
            cue_specificity="exact", emotion_explicitness="implicit",
            required_evidence_session_ids=["A-S01"],
            current_input_goal="已知杯子再次出现并带来局部悲伤", negative_reason="",
        )
        self.blueprint = DatasetBlueprint(
            blueprint_id="case-a-BP-test",
            life_anchor=LifeAnchor(
                identity="普通上班族", recurring_scenes=["家中"],
                interests=["做饭"], ongoing_threads=["练习家常菜"],
            ),
            emotion_memory_map=[EmotionMemory(
                memory_id="M01", event_summary="错过了一次重要的告别电话。",
                emotion="sadness", emotional_meaning="这通电话无法重新接起。",
                cue_seeds=[
                    CueSeed(cue_id="C01", cue_type="object", canonical_form="缺口蓝杯子",
                            related_forms=["旧蓝杯"], personal_meaning="当时桌上放着这只杯子"),
                    CueSeed(cue_id="C02", cue_type="scene", canonical_form="傍晚电话响过又停",
                            related_forms=["黄昏铃声"], personal_meaning="和错过电话的时刻相连"),
                    CueSeed(cue_id="C03", cue_type="utterance", canonical_form="怎么没接电话",
                            related_forms=["刚才没听见吗"], personal_meaning="像过去那句没有答上的追问"),
                ],
            )],
            session_slots=[SessionSlot(
                session_id="A-S01", memory_role="encode_association",
                memory_id="M01", cue_id="C01", evidence_goal="用户说清杯子与电话的关联",
                target_emotion="sadness", relative_to_past="not_applicable",
                depends_on_sessions=[],
            )],
            eval_outlines=[self.outline],
        )
        self.session = Session(
            session_id="A-S01", topic="旧杯子", date="2026-01-01",
            turns=[
                Turn(turn_id="A-S01_T01", round_id="A-S01_R01", speaker="user",
                     text="那只杯子会让我想到那通没接到的电话。",
                     image_id=[], image_dir="", image_caption=[]),
                Turn(turn_id="A-S01_T02", round_id="A-S01_R01", speaker="assistant",
                     text="难怪你每次看到它都会停一下。",
                     image_id=[], image_dir="", image_caption=[]),
            ],
            summary="用户说明旧杯子与未接电话的关联。",
        )

    def test_generator_only_creates_candidates_and_locks_outline_fields(self) -> None:
        llm = _EvalLLM()
        result = EvalGenerator(llm).run(
            self.case, [self.session], self.outline, self.blueprint
        )

        self.assertEqual(result.outline_id, "case-a-E01")
        self.assertEqual(len(result.candidates), 3)
        self.assertTrue(all(item.target_label == "triggered" for item in result.candidates))
        self.assertTrue(all(item.target_emotion == "sadness" for item in result.candidates))
        self.assertTrue(all(item.blueprint_cue_id == "C01" for item in result.candidates))
        self.assertFalse(llm.payload["stage_boundary"]["resolver_enabled"])
        self.assertEqual(len(llm.payload["visible_sessions"]), 1)

    def test_generator_schema_has_no_gold_or_evidence_turn_ids(self) -> None:
        schema_text = str(EvalGenerationResult.model_json_schema())
        prompt = (ROOT / "prompts" / "eval_generator.txt").read_text(encoding="utf-8")

        self.assertNotIn("evidence_turn_ids", schema_text)
        self.assertNotIn("'gold'", schema_text.lower())
        self.assertIn("不运行 EvidenceResolver、EvalVerifier 或 GoldFinalizer", prompt)


if __name__ == "__main__":
    unittest.main()
