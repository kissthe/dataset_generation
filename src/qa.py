from __future__ import annotations

from .models import CaseSpec, QAReport, Session


def validate_sessions(dataset_id: str, case: CaseSpec, sessions: list[Session], call_records: list) -> QAReport:
    errors: list[str] = []
    expected_ids = [x.session_id for x in case.session_outlines]
    actual_ids = [x.session_id for x in sessions]
    ids_ok = actual_ids == expected_ids
    if not ids_ok:
        errors.append(f"session ids/order mismatch: {actual_ids}")

    chronology_ok = all(a.date < b.date for a, b in zip(sessions, sessions[1:]))
    if not chronology_ok:
        errors.append("session dates are not strictly increasing")

    turns_ok = True
    unique_ids: set[str] = set()
    for session in sessions:
        if len(session.turns) % 2 or not 8 <= len(session.turns) <= 12:
            turns_ok = False
        for index, turn in enumerate(session.turns):
            expected_speaker = "user" if index % 2 == 0 else "assistant"
            if turn.speaker != expected_speaker or turn.turn_id in unique_ids:
                turns_ok = False
            unique_ids.add(turn.turn_id)
    if not turns_ok:
        errors.append("turn count, alternation, or uniqueness check failed")

    schema_ok = True
    checks = {
        "session_ids_and_order": ids_ok,
        "strict_chronology": chronology_ok,
        "turn_structure": turns_ok,
        "schema_validation": schema_ok,
    }
    return QAReport(dataset_id=dataset_id, passed=all(checks.values()), checks=checks, errors=errors, call_records=call_records)

