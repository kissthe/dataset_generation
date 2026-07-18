from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.config import ValidationConfig
from src.models import (
    CaseSpec,
    Session,
    SessionPlan,
    Turn,
    VerificationResult,
)
from src.pipeline import GenerationPipeline


class _Writer:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.calls = 0

    def run(self, *_args, **_kwargs) -> Session:
        self.calls += 1
        return self.session


class _Verifier:
    def __init__(self) -> None:
        self.calls = 0

    def run(self, *_args) -> VerificationResult:
        self.calls += 1
        return VerificationResult(result="pass", issues=[])


class _MustNotRun:
    def run(self, *_args):
        raise AssertionError("disabled validation component was called")


def _session() -> Session:
    return Session(
        session_id="T-S01",
        topic="test",
        date="2026-01-01",
        turns=[
            Turn(
                turn_id="T-S01_T01", round_id="T-S01_R01", speaker="user",
                text="hello", image_id=[], image_dir="", image_caption=[],
            ),
            Turn(
                turn_id="T-S01_T02", round_id="T-S01_R01", speaker="assistant",
                text="hi", image_id=[], image_dir="", image_caption=[],
            ),
        ],
        summary="test",
    )


class ValidationStageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.case = CaseSpec.model_validate_json(
            (Path(__file__).parents[1] / "cases" / "case_a.json").read_text(encoding="utf-8")
        )
        cls.plan = SessionPlan(
            session_id="T-S01", date="2026-01-01", topic="test",
            story_beat="test", outline_function="test", round_count=1,
            session_type="daily_life", scene="test scene",
            user_intent="share", continuity_hook="",
        )

    def _pipeline(self, validation: ValidationConfig) -> GenerationPipeline:
        pipeline = GenerationPipeline.__new__(GenerationPipeline)
        pipeline.config = SimpleNamespace(validation=validation)
        pipeline.writer = _Writer(_session())
        pipeline.verifier = _Verifier()
        pipeline.reviser = _MustNotRun()
        pipeline.naturalness = _MustNotRun()
        pipeline._llm_metadata = lambda _component: {}
        return pipeline

    def test_default_path_stops_after_writer(self) -> None:
        pipeline = self._pipeline(ValidationConfig())
        with tempfile.TemporaryDirectory() as temp_dir:
            result = pipeline._generate_one_session(
                self.case, self.plan, [], SimpleNamespace(max_revision_cycles=3), Path(temp_dir)
            )
            names = {path.name for path in Path(temp_dir).glob("*.json")}

        self.assertEqual(result, pipeline.writer.session)
        self.assertEqual(pipeline.writer.calls, 1)
        self.assertEqual(pipeline.verifier.calls, 0)
        self.assertEqual(names, {"01_writer.json", "99_final_session.json"})

    def test_semantic_validation_can_be_enabled_independently(self) -> None:
        pipeline = self._pipeline(ValidationConfig(semantic=True))
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline._generate_one_session(
                self.case, self.plan, [], SimpleNamespace(max_revision_cycles=3), Path(temp_dir)
            )
            names = {path.name for path in Path(temp_dir).glob("*.json")}

        self.assertEqual(pipeline.verifier.calls, 1)
        self.assertIn("02_cycle_1_session_verifier.json", names)
        self.assertNotIn("90_naturalness_checker.json", names)


if __name__ == "__main__":
    unittest.main()
