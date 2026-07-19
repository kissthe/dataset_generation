from __future__ import annotations

import unittest
from pathlib import Path

from src.components import (
    DatasetBlueprintPlanner, compile_dataset_blueprint, canonical_session_ids,
    eval_label_targets, memory_role_targets, validate_dataset_blueprint,
)
from src.config import BlueprintConstraints
from src.models import CaseSpec, DatasetBlueprint, SessionPlanBatch


ROOT = Path(__file__).parents[1]


class DatasetBlueprintChainTests(unittest.TestCase):
    def test_session_ids_are_assigned_by_code_not_case_or_llm(self) -> None:
        case = CaseSpec.model_validate({
            "case_id": "case-a",
            "name": "林澄",
            "core_emotional_event": "一次未完成的告别。",
            "session_outlines": [{
                "session_id": "HUMAN-WROTE-THIS",
                "core_content": "旧版内容种子",
                "function": "旧版功能",
            }],
        })

        self.assertEqual(canonical_session_ids(case, 3), ["A-S01", "A-S02", "A-S03"])
        self.assertNotIn("session_id", case.planner_brief()["session_outlines"][0])

    def test_default_blueprint_coverage_is_deterministic(self) -> None:
        self.assertEqual(memory_role_targets(10), {
            "none": 6,
            "encode_association": 1,
            "triggered_recall": 1,
            "memory_update": 1,
            "control": 1,
        })
        self.assertEqual(eval_label_targets(6), {
            "triggered": 2,
            "insufficient_evidence": 2,
            "not_triggered": 2,
        })

    def test_blueprint_payload_contains_program_owned_ids(self) -> None:
        case = CaseSpec.model_validate({
            "case_id": "case-a", "name": "林澄",
            "core_emotional_event": "一次未完成的告别。",
        })
        session_ids = canonical_session_ids(case, 10)

        payload = DatasetBlueprintPlanner.build_payload(case, session_ids, 6)

        self.assertEqual(payload["assigned_ids"]["session_ids"], session_ids)
        self.assertEqual(
            payload["assigned_ids"]["eval_outline_ids"],
            [f"case-a-E{index:02d}" for index in range(1, 7)],
        )
        self.assertEqual(payload["coverage_targets"]["memory_roles"]["none"], 6)

    def test_structured_outputs_separate_blueprint_from_local_plans(self) -> None:
        blueprint_schema = DatasetBlueprint.model_json_schema()
        plan_batch_schema = SessionPlanBatch.model_json_schema()

        self.assertIn("emotion_memory_map", blueprint_schema["required"])
        self.assertIn("session_slots", blueprint_schema["required"])
        self.assertNotIn("life_anchor", plan_batch_schema["required"])
        self.assertEqual(plan_batch_schema["required"], ["plans"])

        schema_text = str(blueprint_schema)
        for removed_field in (
            "historical_emotion", "intensity", "distinguishing_detail",
            "emotion_intensity", "disclosure_level", "supports_eval_outlines",
        ):
            self.assertNotIn(removed_field, schema_text)

    def test_custom_blueprint_constraints_reach_payload_and_validation(self) -> None:
        case = CaseSpec.model_validate({
            "case_id": "case-a", "name": "林澄",
            "core_emotional_event": "一次未完成的告别。",
        })
        constraints = BlueprintConstraints(
            encode_association_count=1,
            triggered_recall_count=0,
            memory_update_count=0,
            control_count=1,
            required_cue_types=("sound",),
            eval_triggered_count=1,
            eval_insufficient_evidence_count=0,
            eval_not_triggered_count=1,
        )

        payload = DatasetBlueprintPlanner.build_payload(
            case, canonical_session_ids(case, 4), constraints=constraints
        )

        self.assertEqual(payload["coverage_targets"]["required_cue_types"], ["sound"])
        self.assertEqual(payload["coverage_targets"]["memory_roles"], {
            "none": 2,
            "encode_association": 1,
            "triggered_recall": 0,
            "memory_update": 0,
            "control": 1,
        })
        self.assertEqual(payload["coverage_targets"]["eval_labels"], {
            "triggered": 1,
            "insufficient_evidence": 0,
            "not_triggered": 1,
        })

    def test_blueprint_prompt_plants_multiple_cue_types_but_defers_final_gold(self) -> None:
        prompt = (ROOT / "prompts" / "dataset_blueprint_planner.txt").read_text(encoding="utf-8")

        self.assertIn("object", prompt)
        self.assertIn("scene", prompt)
        self.assertIn("utterance", prompt)
        self.assertIn("真实 evidence_turn_ids 必须等所有对话完成后再解析", prompt)
        self.assertIn("ID 已经由程序自动生成", prompt)

    def test_program_compiles_structurally_inconsistent_llm_blueprint(self) -> None:
        case = CaseSpec.model_validate({
            "case_id": "case-a", "name": "林澄",
            "core_emotional_event": "一次未完成的告别。",
        })
        session_ids = canonical_session_ids(case, 5)
        payload = DatasetBlueprintPlanner.build_payload(case, session_ids, 3)
        assigned = payload["assigned_ids"]
        targets = payload["coverage_targets"]
        raw = DatasetBlueprint.model_validate({
            "blueprint_id": "llm-made-the-wrong-id",
            "life_anchor": {
                "identity": "普通上班族", "recurring_scenes": ["通勤", "家中"],
                "interests": ["做饭"], "ongoing_threads": ["练习家常菜"],
            },
            "emotion_memory_map": [{
                "memory_id": "MEMORY-X", "event_summary": "一次未完成的告别。",
                "historical_emotion": {
                    "emotion": "sadness", "intensity": 4, "meaning": "仍有遗憾",
                },
                "cue_seeds": [
                    {"cue_id": "OBJ-X", "cue_type": "object", "canonical_form": "旧杯子",
                     "related_forms": ["缺口杯"], "personal_meaning": "和告别有关",
                     "distinguishing_detail": "杯沿有一个小缺口"},
                    {"cue_id": "SCENE-X", "cue_type": "scene", "canonical_form": "雨夜厨房",
                     "related_forms": ["雨天窗边"], "personal_meaning": "和告别有关",
                     "distinguishing_detail": "雨声与厨房灯同时出现"},
                    {"cue_id": "WORD-X", "cue_type": "utterance", "canonical_form": "到家说一声",
                     "related_forms": ["到了告诉我"], "personal_meaning": "和告别有关",
                     "distinguishing_detail": "过去常听到的叮嘱"},
                ],
            }],
            "session_slots": [
                {"session_id": session_ids[0], "memory_role": "control", "memory_id": "BAD", "cue_id": "BAD",
                 "evidence_goal": "", "target_emotion": "neutral", "emotion_intensity": 0,
                 "relative_to_past": "not_applicable", "disclosure_level": "none",
                 "depends_on_sessions": [], "supports_eval_outlines": []},
                {"session_id": session_ids[1], "memory_role": "encode_association", "memory_id": "BAD", "cue_id": "BAD",
                 "evidence_goal": "建立关联", "target_emotion": "sadness", "emotion_intensity": 4,
                 "relative_to_past": "not_applicable", "disclosure_level": "clear",
                 "depends_on_sessions": [], "supports_eval_outlines": []},
                {"session_id": session_ids[2], "memory_role": "triggered_recall", "memory_id": "BAD", "cue_id": "BAD",
                 "evidence_goal": "触发回忆", "target_emotion": "sadness", "emotion_intensity": 3,
                 "relative_to_past": "same", "disclosure_level": "clear",
                 "depends_on_sessions": [session_ids[0]], "supports_eval_outlines": []},
                {"session_id": session_ids[3], "memory_role": "control", "memory_id": "BAD", "cue_id": "BAD",
                 "evidence_goal": "", "target_emotion": "neutral", "emotion_intensity": 0,
                 "relative_to_past": "not_applicable", "disclosure_level": "none",
                 "depends_on_sessions": [], "supports_eval_outlines": []},
                {"session_id": session_ids[4], "memory_role": "none", "memory_id": "none", "cue_id": "none",
                 "evidence_goal": "", "target_emotion": "sadness", "emotion_intensity": 2,
                 "relative_to_past": "not_applicable", "disclosure_level": "hint",
                 "depends_on_sessions": [], "supports_eval_outlines": []},
            ],
            "eval_outlines": [
                {"outline_id": "wrong-e1", "target_label": "triggered", "target_emotion": "sadness",
                 "history_cutoff": session_ids[-1], "memory_id": "BAD", "cue_id": "BAD",
                 "cue_specificity": "exact", "emotion_explicitness": "implicit",
                 "required_evidence_session_ids": [session_ids[1]], "current_input_goal": "旧杯子再次出现",
                 "negative_reason": ""},
                {"outline_id": "wrong-e2", "target_label": "insufficient_evidence", "target_emotion": "anxiety",
                 "history_cutoff": session_ids[-1], "memory_id": "BAD", "cue_id": "BAD",
                 "cue_specificity": "unseen_control", "emotion_explicitness": "behavioral",
                 "required_evidence_session_ids": [], "current_input_goal": "陌生声音带来反应",
                 "negative_reason": "缺少证据"},
                {"outline_id": "wrong-e3", "target_label": "not_triggered", "target_emotion": "neutral",
                 "history_cutoff": session_ids[-1], "memory_id": "none", "cue_id": "none",
                 "cue_specificity": "none", "emotion_explicitness": "none",
                 "required_evidence_session_ids": [], "current_input_goal": "普通日常",
                 "negative_reason": "没有反应"},
            ],
        })

        compiled, notes = compile_dataset_blueprint(
            raw,
            expected_blueprint_id=assigned["blueprint_id"],
            expected_session_ids=assigned["session_ids"],
            expected_eval_ids=assigned["eval_outline_ids"],
            role_targets=targets["memory_roles"],
            label_targets=targets["eval_labels"],
        )
        errors = validate_dataset_blueprint(
            compiled,
            expected_blueprint_id=assigned["blueprint_id"],
            expected_session_ids=assigned["session_ids"],
            expected_eval_ids=assigned["eval_outline_ids"],
            role_targets=targets["memory_roles"],
            label_targets=targets["eval_labels"],
        )

        self.assertEqual(errors, [])
        self.assertTrue(notes)
        self.assertEqual(
            [slot.memory_role for slot in compiled.session_slots],
            [item["memory_role"] for item in targets["session_role_schedule"]],
        )
        self.assertEqual([cue.cue_id for cue in compiled.emotion_memory_map[0].cue_seeds], ["C01", "C02", "C03"])


if __name__ == "__main__":
    unittest.main()
