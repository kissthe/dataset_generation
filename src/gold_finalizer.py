from __future__ import annotations

import hashlib
import random

from .models import CaseSpec, EvalCandidate, EvalSample, Session


class GoldFinalizer:
    """Apply deterministic integrity rules without making semantic judgments."""

    def __init__(self, seed: int) -> None:
        self.seed = seed

    def finalize(
        self,
        case: CaseSpec,
        sessions: list[Session],
        candidate: EvalCandidate,
        sample_number: int,
        sample_id: str | None = None,
    ) -> EvalSample:
        visible_turns = {}
        cutoff_found = False
        for session in sessions:
            for turn in session.turns:
                visible_turns[turn.turn_id] = turn.speaker
            if session.session_id == candidate.history_cutoff:
                cutoff_found = True
                break
        if not cutoff_found:
            raise ValueError(f"unknown history cutoff: {candidate.history_cutoff}")

        cue_ids = {cue.cue_id for cue in candidate.current_input.cue_options}
        gold = candidate.gold
        if gold.trigger_label == "triggered":
            if gold.trigger_cue_id not in cue_ids:
                raise ValueError("gold cue ID does not exist in cue_options")
            for turn_id in gold.evidence_turn_ids:
                if visible_turns.get(turn_id) != "user":
                    raise ValueError(f"evidence must reference a visible user turn: {turn_id}")

        final_sample_id = sample_id or f"{case.case_id}-E{sample_number:02d}"
        stable_seed = int.from_bytes(
            hashlib.sha256(f"{self.seed}:{final_sample_id}".encode("utf-8")).digest()[:8],
            "big",
        )
        options = list(candidate.current_input.cue_options)
        random.Random(stable_seed).shuffle(options)
        current_input = candidate.current_input.model_copy(update={"cue_options": options})
        return EvalSample(
            sample_id=final_sample_id,
            eval_user_id=case.character_profile.user_id,
            history_cutoff=candidate.history_cutoff,
            current_input=current_input,
            gold=gold,
        )
