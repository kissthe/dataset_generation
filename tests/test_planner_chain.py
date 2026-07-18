from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.components import SessionPlanner
from src.config import ValidationConfig
from src.models import CaseSpec, LifeAnchor, SessionPlan, SessionPlanList
from src.pipeline import GenerationPipeline


ROOT = Path(__file__).parents[1]


class PlannerChainTests(unittest.TestCase):
    def test_minimal_flat_spec_is_normalized(self) -> None:
        spec = CaseSpec.model_validate({
            "name": "林澄",
            "core_emotional_event": "错过亲人最后一次电话，留有遗憾。",
        })

        self.assertEqual(spec.character_profile.name, "林澄")
        self.assertTrue(spec.case_id.startswith("case-"))
        self.assertTrue(spec.character_profile.user_id.startswith("user-"))
        self.assertEqual(
            spec.planner_brief(),
            {
                "name": "林澄",
                "core_emotional_event": "错过亲人最后一次电话，留有遗憾。",
            },
        )

    def test_existing_rich_spec_remains_supported(self) -> None:
        spec = CaseSpec.model_validate_json(
            (ROOT / "cases" / "case_b.json").read_text(encoding="utf-8")
        )

        brief = spec.planner_brief()
        self.assertEqual(brief["identity"], "产品设计师")
        self.assertEqual(brief["interests"], ["产品设计", "跑步"])
        self.assertNotIn("cues", brief)
        self.assertNotIn("eval_outlines", brief)

    def test_death_in_core_event_becomes_an_immutable_fact(self) -> None:
        spec = CaseSpec.model_validate({
            "name": "林澄",
            "core_emotional_event": "林澄错过奶奶去世前最后一次通话，留有遗憾。",
        })

        self.assertEqual(spec.deceased_entities(), ["奶奶"])
        self.assertTrue(any("不得安排与其探望" in fact for fact in spec.immutable_facts()))

    def test_planner_payload_exposes_global_batch_context(self) -> None:
        spec = CaseSpec.model_validate({"name": "Maya", "core_emotional_event": "一次未完成的告别。"})
        payload = SessionPlanner.build_payload(
            spec, min_rounds=4, max_rounds=6, session_count=5,
            total_session_count=10, id_prefix="M", start_index=5,
            prior_plans=[{"session_id": "M-S05"}],
        )

        self.assertEqual(payload["planner_brief"], spec.planner_brief())
        self.assertEqual(payload["planning_window"]["batch_size"], 5)
        self.assertEqual(payload["planning_window"]["total_session_count"], 10)
        self.assertEqual(payload["planning_window"]["start_index"], 5)
        self.assertEqual(payload["prior_plans"], [{"session_id": "M-S05"}])
        self.assertIn("immutable_facts", payload)
        self.assertIsNone(payload["established_life_anchor"])

    def test_rich_plan_fields_are_required_for_new_llm_output(self) -> None:
        schema = SessionPlanList.model_json_schema()
        required = set(schema["$defs"]["SessionPlan"]["required"])

        self.assertTrue({
            "session_type", "scene", "user_intent", "continuity_hook",
            "life_thread", "thread_progress", "interaction_mode",
        } <= required)
        self.assertIn("life_anchor", schema["required"])

    def test_planner_prompt_keeps_third_parties_out_of_the_chat(self) -> None:
        prompt = (ROOT / "prompts" / "session_planner.txt").read_text(encoding="utf-8")

        self.assertIn("实际人物是 user 的同一个固定朋友", prompt)
        self.assertIn("不得称其为“助手”“assistant”或“AI”", prompt)
        self.assertIn("下班和同事吐槽，晚上又和朋友聊起此事", prompt)
        self.assertIn("ask_advice 最多占 30%", prompt)
        self.assertIn("已经去世的人不能在后续被探望", prompt)

    def test_planner_retries_a_plan_that_reverses_a_death_fact(self) -> None:
        spec = CaseSpec.model_validate({
            "name": "林澄",
            "core_emotional_event": "林澄错过奶奶去世前最后一次通话，留有遗憾。",
        })

        class RetryingLLM:
            def __init__(self):
                self.calls = 0

            def generate(self, _component, _payload, _model):
                self.calls += 1
                bad = self.calls == 1
                return SessionPlanList(plans=[SessionPlan(
                    session_id="LIN-S01", date="2026-01-01", topic="周末安排",
                    story_beat="下次去探望奶奶前先列一张便签。" if bad else "周末去墓园前买一束花。",
                    outline_function="承接核心事件。", round_count=4,
                    session_type="core_event", scene="周末早上整理随身物品",
                    user_intent="和朋友聊聊周末安排", continuity_hook="",
                    life_thread="one_off", thread_progress="", interaction_mode="share",
                )], life_anchor=LifeAnchor(
                    identity="普通上班族", recurring_scenes=["通勤", "家中"],
                    interests=["散步"], ongoing_threads=["记录周末散步路线"],
                ))

        llm = RetryingLLM()
        plans = SessionPlanner(llm).run(
            spec, min_rounds=4, max_rounds=6, session_count=1,
            total_session_count=1, id_prefix="LIN",
        )

        self.assertEqual(llm.calls, 2)
        self.assertIn("墓园", plans.plans[0].story_beat)

    def test_planner_discards_future_plans_returned_in_current_batch(self) -> None:
        spec = CaseSpec.model_validate({"name": "Maya", "core_emotional_event": "一次未完成的告别。"})

        class OverGeneratingLLM:
            def generate(self, _component, _payload, _model):
                return SessionPlanList(plans=[
                    SessionPlan(
                        session_id=f"MAYA-S{index:02d}", date=f"2026-01-{index:02d}",
                        topic=f"日常 {index}", story_beat="朋友之间聊一件小事。",
                        outline_function="普通生活片段。", round_count=4,
                        session_type="daily_life", scene="傍晚回家路上",
                        user_intent="和朋友分享小事", continuity_hook="",
                    )
                    for index in range(1, 11)
                ], life_anchor=LifeAnchor(
                    identity="普通上班族", recurring_scenes=["通勤", "家中"],
                    interests=["做饭"], ongoing_threads=["学会几道家常饭"],
                ))

        plans = SessionPlanner(OverGeneratingLLM()).run(
            spec, min_rounds=4, max_rounds=6, session_count=5,
            total_session_count=10, id_prefix="MAYA", start_index=0,
        )

        self.assertEqual([plan.session_id for plan in plans.plans], [f"MAYA-S{i:02d}" for i in range(1, 6)])

    def test_planner_normalizes_ids_with_a_different_prefix(self) -> None:
        spec = CaseSpec.model_validate({"name": "Maya", "core_emotional_event": "一次未完成的告别。"})

        class WrongPrefixLLM:
            def generate(self, _component, _payload, _model):
                return SessionPlanList(plans=[
                    SessionPlan(
                        session_id=f"S{index}", date=f"2026-01-{index:02d}",
                        topic=f"日常 {index}", story_beat="朋友之间聊一件小事。",
                        outline_function="普通生活片段。", round_count=4,
                        session_type="daily_life", scene="傍晚回家路上",
                        user_intent="和朋友分享小事", continuity_hook="",
                    )
                    for index in range(1, 6)
                ], life_anchor=LifeAnchor(
                    identity="普通上班族", recurring_scenes=["通勤", "家中"],
                    interests=["做饭"], ongoing_threads=["学会几道家常饭"],
                ))

        plans = SessionPlanner(WrongPrefixLLM()).run(
            spec, min_rounds=4, max_rounds=6, session_count=5,
            total_session_count=10, id_prefix="MAYA", start_index=0,
        )

        self.assertEqual([plan.session_id for plan in plans.plans], [f"MAYA-S{i:02d}" for i in range(1, 6)])

    def test_legacy_saved_plans_still_load(self) -> None:
        plans = SessionPlanList.model_validate_json(
            (ROOT / "outputs" / "case_b" / "session_plans.json").read_text(encoding="utf-8")
        )

        self.assertGreater(len(plans.plans), 0)
        self.assertEqual(plans.plans[0].session_type, "daily_life")
        self.assertEqual(plans.plans[0].scene, "")

    def test_planner_only_pipeline_stops_before_writer(self) -> None:
        class FakePlanner:
            build_payload = staticmethod(SessionPlanner.build_payload)

            def run(self, _case, _min_rounds, _max_rounds, session_count,
                    total_session_count, id_prefix, start_index=0, prior_plans=None,
                    life_anchor=None):
                self.total_session_count = total_session_count
                anchor = life_anchor or LifeAnchor(
                    identity="普通上班族", recurring_scenes=["通勤", "家中"],
                    interests=["做饭"], ongoing_threads=["学会几道家常饭"],
                )
                return SessionPlanList(plans=[
                    SessionPlan(
                        session_id=f"{id_prefix}-S{index + 1:02d}",
                        date=f"2026-01-{index + 1:02d}",
                        topic=f"日常 {index + 1}",
                        story_beat="从一件具体小事开始，自然聊到一个小决定。",
                        outline_function="建立普通生活连续性。",
                        round_count=4,
                        session_type="daily_life",
                        scene="下班后的便利店门口",
                        user_intent="分享刚遇到的小事",
                        continuity_hook="",
                    )
                    for index in range(start_index, start_index + session_count)
                ], life_anchor=anchor)

        generation = SimpleNamespace(
            session_count=2, planner_batch_size=2, min_rounds=4, max_rounds=6,
            context_sessions=3, seed=42, max_retries=3, max_revision_cycles=3,
            run_eval=False, stop_after_planning=True,
        )
        pipeline = GenerationPipeline.__new__(GenerationPipeline)
        pipeline.config = SimpleNamespace(
            generation=generation, validation=ValidationConfig(),
            dataset_id_prefix="test",
        )
        pipeline.planner = FakePlanner()
        pipeline._llm_metadata = lambda _component: {}

        with tempfile.TemporaryDirectory() as temp_dir:
            artifact, qa = pipeline.run(ROOT / "cases" / "case_a.json", Path(temp_dir))
            result = (Path(temp_dir) / "logs" / "99_pipeline_result.json").read_text(encoding="utf-8")

        self.assertEqual(artifact.name, "session_plans.json")
        self.assertIsNone(qa)
        self.assertIn('"mode": "planner_only"', result)


if __name__ == "__main__":
    unittest.main()
