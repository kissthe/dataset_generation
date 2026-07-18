from __future__ import annotations

import unittest
from pathlib import Path

from src.components import SessionWriter
from src.models import CaseSpec, LifeAnchor, Session, SessionPlan, Turn


ROOT = Path(__file__).parents[1]


class _CapturingLLM:
    def __init__(self) -> None:
        self.payload = None

    def generate(self, _component, payload, _model):
        self.payload = payload
        return Session(
            session_id="M-S02", date="2026-01-02", topic="晚饭",
            turns=[
                Turn(
                    turn_id="M-S02_T01", round_id="M-S02_R01", speaker="user",
                    text="今晚懒得做饭。", image_id=[], image_dir="", image_caption=[],
                ),
                Turn(
                    turn_id="M-S02_T02", round_id="M-S02_R01", speaker="assistant",
                    text="那就别折腾，楼下那家面不是还行吗。", image_id=[], image_dir="", image_caption=[],
                ),
            ],
            summary="用户决定去楼下吃面。",
        )


class WriterChainTests(unittest.TestCase):
    def test_writer_receives_compact_long_term_story_context(self) -> None:
        llm = _CapturingLLM()
        writer = SessionWriter(llm)
        case = CaseSpec.model_validate({"name": "Maya", "core_emotional_event": "一次未完成的告别。"})
        plan = SessionPlan(
            session_id="M-S02", date="2026-01-02", topic="晚饭",
            story_beat="朋友之间聊晚饭。", outline_function="普通生活片段。",
            round_count=1, session_type="daily_life", scene="下班路上",
            user_intent="和朋友商量吃什么", continuity_hook="",
        )
        story_so_far = [{
            "session_id": "M-S01", "date": "2026-01-01",
            "topic": "加班", "summary": "用户昨天加班很晚。",
        }]

        writer.run(case, plan.model_dump(), [], story_so_far=story_so_far)

        self.assertEqual(llm.payload["story_so_far"], story_so_far)
        self.assertEqual(llm.payload["recent_history"], [])
        self.assertIn("immutable_facts", llm.payload)

    def test_writer_receives_the_implicit_life_anchor(self) -> None:
        llm = _CapturingLLM()
        writer = SessionWriter(llm)
        case = CaseSpec.model_validate({"name": "Maya", "core_emotional_event": "一次未完成的告别。"})
        anchor = LifeAnchor(
            identity="普通上班族", recurring_scenes=["通勤", "周末厨房"],
            interests=["做饭"], ongoing_threads=["逐渐学会几道家常饭"],
        )
        plan = SessionPlan(
            session_id="M-S02", date="2026-01-02", topic="晚饭",
            story_beat="朋友之间聊晚饭。", outline_function="推进做饭支线。",
            round_count=1, session_type="daily_life", scene="下班路上",
            user_intent="分享今天想吃什么", continuity_hook="",
            life_thread="逐渐学会几道家常饭", thread_progress="第一次挑一道简单菜",
            interaction_mode="share",
        )

        writer.run(case, plan.model_dump(), [], life_anchor=anchor)

        self.assertEqual(llm.payload["life_anchor"], anchor.model_dump())

    def test_writer_retries_dialogue_that_treats_a_deceased_person_as_alive(self) -> None:
        case = CaseSpec.model_validate({
            "name": "林澄",
            "core_emotional_event": "林澄错过奶奶去世前最后一次通话，留有遗憾。",
        })

        class RetryingLLM:
            def __init__(self):
                self.calls = 0

            def generate(self, _component, _payload, _model):
                self.calls += 1
                user_text = (
                    "我明天想去探望奶奶，问问她最近身体怎么样。"
                    if self.calls == 1 else "我明天想去墓园看看奶奶，顺路买束花。"
                )
                return Session(
                    session_id="L-S01", date="2026-01-01", topic="周末",
                    turns=[
                        Turn(turn_id="L-S01_T01", round_id="L-S01_R01", speaker="user",
                             text=user_text, image_id=[], image_dir="", image_caption=[]),
                        Turn(turn_id="L-S01_T02", round_id="L-S01_R01", speaker="assistant",
                             text="嗯，挑她以前喜欢的颜色吧。", image_id=[], image_dir="", image_caption=[]),
                    ],
                    summary="用户准备周末去墓园。",
                )

        llm = RetryingLLM()
        writer = SessionWriter(llm)
        session = writer.run(case, {
            "session_id": "L-S01", "date": "2026-01-01", "topic": "周末",
            "interaction_mode": "share",
        }, [])

        self.assertEqual(llm.calls, 2)
        self.assertIn("墓园", session.turns[0].text)

    def test_writer_prompt_requires_progression_and_friend_boundaries(self) -> None:
        prompt = (ROOT / "prompts" / "session_writer.txt").read_text(encoding="utf-8")

        self.assertIn("具体开场 → 来回推进 → 小变化或自然收束", prompt)
        self.assertIn("不得用“助手”“assistant”“AI”", prompt)
        self.assertIn("不是万能工具", prompt)
        self.assertIn("summary 用 1-2 句", prompt)
        self.assertIn("建议—照做—复述—成功", prompt)
        self.assertIn("禁止把悲伤强行转化成行动计划", prompt)


if __name__ == "__main__":
    unittest.main()
