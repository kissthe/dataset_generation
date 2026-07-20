from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import asdict
from typing import Any

from .config import BlueprintConstraints
from .llm_client import LLMClient
from .models import (
    BlueprintEvalOutline, CaseSpec, DatasetBlueprint, EvalCandidate, EvalCandidateList,
    EvalGenerationResult, EvalSelection, EvidenceResolution, Gold, LifeAnchor, Session,
    SessionPlanBatch, SessionPlanList, SessionSlot, VerificationResult,
)


def _hard_fact_conflicts(case: CaseSpec, value: Any) -> list[str]:
    """Catch a small set of irreversible contradictions even when QA is disabled."""
    text = str(value)
    conflicts: list[str] = []
    for entity in case.deceased_entities():
        escaped = re.escape(entity)
        direct_interaction = re.search(
            rf"(?:{escaped}.{{0,24}}(?:下次|以后|明天|准备|计划|打算).{{0,24}}"
            rf"(?:探望|看望|拜访|见面|约好|打电话|通话|联系)"
            rf"|(?:下次|以后|明天|准备|计划|打算).{{0,24}}"
            rf"(?:探望|看望|拜访|见面|约好|打电话|通话|联系).{{0,24}}{escaped}"
            rf"|{escaped}.{{0,12}}(?:最近怎么样|身体怎么样|今天还好吗))",
            text,
        )
        implied_visit = re.search(
            r"(?:下次|准备|计划|打算).{0,18}(?:探望|看望|拜访)|(?:探望|看望|拜访)前"
            r"|我来看看你|约好我再去",
            text,
        )
        deceased_context = entity in text or ("最后一次通话" in text and "遗憾" in text)
        if direct_interaction or (deceased_context and implied_visit):
            conflicts.append(f"{entity}已经去世，但内容安排了后续现实互动")
    return conflicts


def canonical_session_ids(case: CaseSpec, session_count: int) -> list[str]:
    """Assign IDs in code; models may only copy these immutable values."""
    raw_prefix = case.case_id.rsplit("-", 1)[-1].upper()
    prefix = re.sub(r"[^A-Z0-9]+", "", raw_prefix)
    if not prefix:
        prefix = hashlib.sha256(case.case_id.encode("utf-8")).hexdigest()[:10].upper()
    return [f"{prefix}-S{index:02d}" for index in range(1, session_count + 1)]


def canonical_eval_ids(case: CaseSpec, eval_count: int) -> list[str]:
    return [f"{case.case_id}-E{index:02d}" for index in range(1, eval_count + 1)]


def blueprint_id_for(
    case: CaseSpec, session_ids: list[str], eval_count: int,
    constraints: BlueprintConstraints | None = None,
) -> str:
    identity_payload = {
        "planner_brief": case.planner_brief(),
        "immutable_facts": case.immutable_facts(),
        "session_ids": session_ids,
        "eval_count": eval_count,
    }
    if constraints is not None:
        identity_payload["blueprint_constraints"] = asdict(constraints)
    source = json.dumps(identity_payload, ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:10].upper()
    return f"{case.case_id}-BP-{digest}"


def blueprint_fingerprint(blueprint: DatasetBlueprint) -> str:
    """Detect any blueprint content change before reusing detailed plans."""
    source = json.dumps(
        blueprint.model_dump(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def plan_fingerprint(plans: SessionPlanList) -> str:
    """Identify an exact Plan independently of its source output directory."""
    source = json.dumps(plans.model_dump(), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def memory_role_targets(
    session_count: int, constraints: BlueprintConstraints | None = None
) -> dict[str, int]:
    """Keep most sessions ordinary while guaranteeing useful memory evidence."""
    if constraints is not None:
        non_none = {
            "encode_association": constraints.encode_association_count,
            "triggered_recall": constraints.triggered_recall_count,
            "memory_update": constraints.memory_update_count,
            "control": constraints.control_count,
        }
        return {"none": session_count - sum(non_none.values()), **non_none}
    targets = {
        "none": 0,
        "encode_association": 0,
        "triggered_recall": 0,
        "memory_update": 0,
        "control": 0,
    }
    if session_count <= 0:
        return targets
    if session_count == 1:
        targets["encode_association"] = 1
    elif session_count == 2:
        targets["encode_association"] = 1
        targets["triggered_recall"] = 1
    elif session_count <= 4:
        targets["encode_association"] = 1
        targets["triggered_recall"] = 1
        targets["none"] = session_count - 2
    elif session_count <= 7:
        targets["encode_association"] = 1
        targets["triggered_recall"] = 1
        targets["control"] = 1
        targets["none"] = session_count - 3
    else:
        targets["encode_association"] = 1
        targets["triggered_recall"] = 1
        targets["memory_update"] = 1
        targets["control"] = 1
        targets["none"] = session_count - 4
    return targets


def eval_label_targets(
    eval_count: int | None = None,
    constraints: BlueprintConstraints | None = None,
) -> dict[str, int]:
    """Distribute eval coverage deterministically; six becomes 2/2/2."""
    if constraints is not None:
        return {
            "triggered": constraints.eval_triggered_count,
            "insufficient_evidence": constraints.eval_insufficient_evidence_count,
            "not_triggered": constraints.eval_not_triggered_count,
        }
    eval_count = 6 if eval_count is None else eval_count
    labels = ["triggered", "insufficient_evidence", "not_triggered"]
    base, remainder = divmod(max(eval_count, 0), len(labels))
    return {
        label: base + (1 if index < remainder else 0)
        for index, label in enumerate(labels)
    }


def validate_blueprint_constraints(
    session_count: int, constraints: BlueprintConstraints
) -> list[str]:
    errors: list[str] = []
    role_counts = memory_role_targets(session_count, constraints)
    if any(value < 0 for value in role_counts.values()):
        errors.append(
            f"memory role 数量合计不能超过 Session 数量 {session_count}：{role_counts}"
        )
    if any(value < 0 for value in eval_label_targets(constraints=constraints).values()):
        errors.append("Eval label 数量不能为负数")
    if (
        constraints.triggered_recall_count or constraints.memory_update_count
    ) and constraints.encode_association_count < 1:
        errors.append("启用 triggered_recall 或 memory_update 时至少需要 1 个 encode_association")
    evidence_roles = (
        constraints.encode_association_count
        + constraints.triggered_recall_count
        + constraints.memory_update_count
    )
    if constraints.eval_triggered_count and evidence_roles < 1:
        errors.append("生成 triggered Eval 时至少需要 1 个历史证据角色")
    allowed_cues = {"object", "scene", "utterance", "sound", "smell", "taste"}
    cue_types = set(constraints.required_cue_types)
    if not cue_types:
        errors.append("required_cue_types 至少选择一种")
    if not cue_types <= allowed_cues:
        errors.append(f"存在不支持的 cue 类型：{sorted(cue_types - allowed_cues)}")
    return errors


def _program_role_schedule(
    session_ids: list[str], role_targets: dict[str, int]
) -> list[str]:
    """Place required memory roles at stable points with ordinary life between them."""
    count = len(session_ids)
    if not count:
        return []
    ordered_roles = [
        role
        for role in ("encode_association", "triggered_recall", "memory_update", "control")
        for _ in range(role_targets.get(role, 0))
    ]
    schedule = ["none"] * count
    fractions_by_count = {
        1: [0.35],
        2: [0.25, 1.0],
        3: [0.20, 0.58, 1.0],
        4: [0.20, 0.50, 0.80, 1.0],
    }
    fractions = fractions_by_count.get(len(ordered_roles))
    if fractions is None:
        fractions = [(index + 1) / len(ordered_roles) for index in range(len(ordered_roles))]
    previous = -1
    for role_index, (role, fraction) in enumerate(zip(ordered_roles, fractions)):
        remaining = len(ordered_roles) - role_index - 1
        preferred = math.ceil(fraction * count) - 1
        position = max(previous + 1, preferred)
        position = min(position, count - remaining - 1)
        schedule[position] = role
        previous = position
    return schedule


def compile_dataset_blueprint(
    raw: DatasetBlueprint,
    *,
    expected_blueprint_id: str,
    expected_session_ids: list[str],
    expected_eval_ids: list[str],
    role_targets: dict[str, int],
    label_targets: dict[str, int],
) -> tuple[DatasetBlueprint, list[str]]:
    """Compile LLM semantics into a deterministic, referentially valid blueprint.

    The model chooses ordinary-life details, memory meaning and cue wording. Code
    owns IDs, coverage counts, chronology, dependencies and negative-label rules.
    """
    notes: list[str] = []
    if raw.blueprint_id != expected_blueprint_id:
        notes.append("normalized blueprint_id")
    raw_roles = Counter(slot.memory_role for slot in raw.session_slots)
    expected_roles = {key: value for key, value in role_targets.items() if value}
    if dict(raw_roles) != expected_roles:
        notes.append(f"normalized memory_role distribution from {dict(raw_roles)} to {expected_roles}")
    raw_labels = Counter(outline.target_label for outline in raw.eval_outlines)
    expected_labels = {key: value for key, value in label_targets.items() if value}
    if dict(raw_labels) != expected_labels:
        notes.append(f"normalized eval label distribution from {dict(raw_labels)} to {expected_labels}")

    # One core emotional event is represented as one canonical memory. Later
    # memory change belongs in memory_update, not a second disconnected identity.
    source_memory = raw.emotion_memory_map[0]
    if len(raw.emotion_memory_map) > 1:
        notes.append("kept the first emotion memory and represented later change in memory_update")
    cue_priority = {"object": 0, "scene": 1, "utterance": 2, "sound": 3, "smell": 4, "taste": 5}
    ordered_cues = sorted(
        source_memory.cue_seeds,
        key=lambda cue: (cue_priority.get(cue.cue_type, 99), source_memory.cue_seeds.index(cue)),
    )
    canonical_cues = [
        cue.model_copy(update={"cue_id": f"C{index:02d}"})
        for index, cue in enumerate(ordered_cues, 1)
    ]
    memory = source_memory.model_copy(update={"memory_id": "M01", "cue_seeds": canonical_cues})
    if source_memory.memory_id != "M01" or any(
        cue.cue_id != canonical.cue_id
        for cue, canonical in zip(ordered_cues, canonical_cues)
    ):
        notes.append("canonicalized memory/cue IDs to M01 and C01..")

    role_templates: dict[str, SessionSlot] = {}
    for slot in raw.session_slots:
        role_templates.setdefault(slot.memory_role, slot)
    schedule = _program_role_schedule(expected_session_ids, role_targets)
    evidence_by_memory: list[str] = []
    slots: list[SessionSlot] = []
    evidence_roles = ["encode_association", "triggered_recall", "memory_update"]
    evidence_role_index = {role: index for index, role in enumerate(evidence_roles)}
    for session_id, role in zip(expected_session_ids, schedule):
        template = role_templates.get(role)
        if role == "none":
            slot = SessionSlot(
                session_id=session_id, memory_role=role, memory_id="none", cue_id="none",
                evidence_goal="", target_emotion="neutral",
                relative_to_past="not_applicable", depends_on_sessions=[],
            )
        else:
            if role == "control":
                cue = canonical_cues[0]
                slot = SessionSlot(
                    session_id=session_id, memory_role=role, memory_id="M01", cue_id=cue.cue_id,
                    evidence_goal=(
                        f"让{cue.cue_type}线索“{cue.canonical_form}”自然出现，但 user 不产生与"
                        "核心记忆相关的反应；若有情绪，明确来自另一件当下小事。"
                    ),
                    target_emotion="neutral", relative_to_past="not_applicable",
                    depends_on_sessions=[],
                )
            else:
                cue_index = min(evidence_role_index[role], len(canonical_cues) - 1)
                cue = canonical_cues[cue_index]
                if role == "encode_association":
                    goal = (
                        f"由 user 自然说明{cue.cue_type}线索“{cue.canonical_form}”与个人经历“"
                        f"{memory.event_summary}”的联系，并表达它为何带有{memory.emotion}意义。"
                    )
                    relative = "not_applicable"
                    dependencies: list[str] = []
                elif role == "triggered_recall":
                    goal = (
                        f"先让{cue.cue_type}线索“{cue.canonical_form}”在当下出现，再由 user 说明它"
                        f"为何连接到“{memory.event_summary}”，并呈现局部{memory.emotion}反应。"
                    )
                    relative = (
                        template.relative_to_past
                        if template and template.relative_to_past != "not_applicable" else "same"
                    )
                    dependencies = evidence_by_memory[-1:]
                else:
                    goal = (
                        f"让{cue.cue_type}线索“{cue.canonical_form}”再次连接到“{memory.event_summary}”，"
                        "并由 user 自然比较这次局部情绪与过去的差异，不强行写成彻底释怀。"
                    )
                    relative = (
                        template.relative_to_past
                        if template and template.relative_to_past != "not_applicable" else "mixed"
                    )
                    dependencies = evidence_by_memory[-2:] or evidence_by_memory[-1:]
                target_emotion = (
                    template.target_emotion
                    if template and template.target_emotion != "neutral" else memory.emotion
                )
                slot = SessionSlot(
                    session_id=session_id, memory_role=role, memory_id="M01", cue_id=cue.cue_id,
                    evidence_goal=goal, target_emotion=target_emotion,
                    relative_to_past=relative, depends_on_sessions=dependencies,
                )
                evidence_by_memory.append(session_id)
        slots.append(slot)

    evidence_slots = [
        slot for slot in slots
        if slot.memory_role in {"encode_association", "triggered_recall", "memory_update"}
    ]
    templates_by_label: dict[str, list[BlueprintEvalOutline]] = {}
    for outline in raw.eval_outlines:
        templates_by_label.setdefault(outline.target_label, []).append(outline)
    labels = [
        label
        for label in ("triggered", "insufficient_evidence", "not_triggered")
        for _ in range(label_targets.get(label, 0))
    ]
    evals: list[BlueprintEvalOutline] = []
    label_indexes: Counter = Counter()
    for outline_id, label in zip(expected_eval_ids, labels):
        label_index = label_indexes[label]
        label_indexes[label] += 1
        templates = templates_by_label.get(label, [])
        template = templates[label_index % len(templates)] if templates else None
        cutoff = expected_session_ids[-1]
        if label == "triggered":
            evidence_slot = evidence_slots[label_index % len(evidence_slots)]
            cue = next(cue for cue in canonical_cues if cue.cue_id == evidence_slot.cue_id)
            related_available = bool(cue.related_forms) and label_index % 2 == 1
            cue_text = cue.related_forms[0] if related_available else cue.canonical_form
            specificity = "related" if related_available else "exact"
            outline = BlueprintEvalOutline(
                outline_id=outline_id, target_label=label,
                target_emotion=evidence_slot.target_emotion, history_cutoff=cutoff,
                memory_id="M01", cue_id=cue.cue_id, cue_specificity=specificity,
                emotion_explicitness=("implicit" if label_index % 2 == 0 else "behavioral"),
                required_evidence_session_ids=[evidence_slot.session_id],
                current_input_goal=(
                    f"当前文本自然出现{cue.cue_type}线索“{cue_text}”并呈现"
                    f"{evidence_slot.target_emotion}反应，但不直接复述历史。"
                ),
                negative_reason="",
            )
        elif label == "insufficient_evidence":
            fallback_goal = (
                "当前输入出现一个可感知的新线索并伴随情绪或行为反应，但可见历史从未建立"
                "这个线索与用户个人记忆的联系。"
            )
            outline = BlueprintEvalOutline(
                outline_id=outline_id, target_label=label,
                target_emotion=(
                    template.target_emotion if template and template.target_emotion != "neutral"
                    else memory.emotion
                ),
                history_cutoff=cutoff, memory_id="none", cue_id="none",
                cue_specificity="unseen_control",
                emotion_explicitness=("implicit" if label_index % 2 == 0 else "behavioral"),
                required_evidence_session_ids=[],
                current_input_goal=(template.current_input_goal if template else fallback_goal),
                negative_reason="当前 cue 与反应存在，但可见历史缺少该 cue 的个人记忆关联证据。",
            )
        else:
            known_cue_control = label_index % 2 == 0
            if known_cue_control:
                cue = canonical_cues[0]
                outline = BlueprintEvalOutline(
                    outline_id=outline_id, target_label=label, target_emotion="neutral",
                    history_cutoff=cutoff, memory_id="M01", cue_id=cue.cue_id,
                    cue_specificity="exact", emotion_explicitness="none",
                    required_evidence_session_ids=[],
                    current_input_goal=(
                        f"已知{cue.cue_type}线索“{cue.canonical_form}”只作为普通背景出现，"
                        "用户没有相关情绪或行为反应。"
                    ),
                    negative_reason="虽然出现已知 cue，但当前没有可归因于该记忆的反应。",
                )
            else:
                fallback_goal = "用户表达一件当下小事造成的情绪，当前输入不包含个人记忆 cue。"
                outline = BlueprintEvalOutline(
                    outline_id=outline_id, target_label=label,
                    target_emotion=(
                        template.target_emotion if template and template.target_emotion != "neutral"
                        else "anxiety"
                    ),
                    history_cutoff=cutoff, memory_id="none", cue_id="none",
                    cue_specificity="none", emotion_explicitness="explicit",
                    required_evidence_session_ids=[],
                    current_input_goal=(template.current_input_goal if template else fallback_goal),
                    negative_reason="当前情绪有明确的现实来源，且没有有效记忆 cue。",
                )
        evals.append(outline)

    compiled = DatasetBlueprint(
        blueprint_id=expected_blueprint_id,
        life_anchor=raw.life_anchor,
        emotion_memory_map=[memory],
        session_slots=slots,
        eval_outlines=evals,
    )
    return compiled, notes


def validate_dataset_blueprint(
    blueprint: DatasetBlueprint,
    *,
    expected_blueprint_id: str,
    expected_session_ids: list[str],
    expected_eval_ids: list[str],
    role_targets: dict[str, int],
    label_targets: dict[str, int],
    required_cue_types: tuple[str, ...] = ("object", "scene", "utterance"),
) -> list[str]:
    """Validate global coverage and references before local planning begins."""
    errors: list[str] = []
    if blueprint.blueprint_id != expected_blueprint_id:
        errors.append(f"blueprint_id 必须是程序分配的 {expected_blueprint_id}")

    slot_ids = [slot.session_id for slot in blueprint.session_slots]
    if slot_ids != expected_session_ids:
        errors.append(f"session_slots 必须按顺序使用程序分配的 IDs：{expected_session_ids}")
    outline_ids = [outline.outline_id for outline in blueprint.eval_outlines]
    if outline_ids != expected_eval_ids:
        errors.append(f"eval_outlines 必须按顺序使用程序分配的 IDs：{expected_eval_ids}")

    actual_roles = Counter(slot.memory_role for slot in blueprint.session_slots)
    if dict(actual_roles) != {key: value for key, value in role_targets.items() if value}:
        errors.append(f"memory_role 数量必须是 {role_targets}，实际是 {dict(actual_roles)}")
    actual_labels = Counter(outline.target_label for outline in blueprint.eval_outlines)
    if dict(actual_labels) != {key: value for key, value in label_targets.items() if value}:
        errors.append(f"eval label 数量必须是 {label_targets}，实际是 {dict(actual_labels)}")

    memories = {memory.memory_id: memory for memory in blueprint.emotion_memory_map}
    if len(memories) != len(blueprint.emotion_memory_map):
        errors.append("emotion_memory_map 中的 memory_id 不能重复")
    cues = {}
    cue_types: set[str] = set()
    for memory in blueprint.emotion_memory_map:
        for cue in memory.cue_seeds:
            if cue.cue_id in cues:
                errors.append(f"cue_id 重复：{cue.cue_id}")
            cues[cue.cue_id] = (memory.memory_id, cue)
            cue_types.add(cue.cue_type)
    required_types = set(required_cue_types)
    if not required_types <= cue_types:
        errors.append(f"cue_seeds 缺少配置要求的类型：{sorted(required_types - cue_types)}")

    positions = {session_id: index for index, session_id in enumerate(expected_session_ids)}
    slots_by_id = {slot.session_id: slot for slot in blueprint.session_slots}
    earlier_memory_evidence: dict[str, list[str]] = {}
    for slot in blueprint.session_slots:
        if slot.memory_role == "none":
            if (slot.memory_id, slot.cue_id) != ("none", "none"):
                errors.append(f"{slot.session_id} 的 none 槽位必须使用 memory_id=cue_id=none")
        else:
            cue_owner = cues.get(slot.cue_id)
            if slot.memory_id not in memories or cue_owner is None or cue_owner[0] != slot.memory_id:
                errors.append(f"{slot.session_id} 引用了无效的 memory_id/cue_id 组合")
        for dependency in slot.depends_on_sessions:
            if dependency not in positions or positions[dependency] >= positions.get(slot.session_id, -1):
                errors.append(f"{slot.session_id} 只能依赖更早的有效 session：{dependency}")
        if slot.memory_role == "encode_association":
            earlier_memory_evidence.setdefault(slot.memory_id, []).append(slot.session_id)
        elif slot.memory_role in {"triggered_recall", "memory_update"}:
            established = earlier_memory_evidence.get(slot.memory_id, [])
            if not established:
                errors.append(f"{slot.session_id} 在对应个人记忆出现历史证据前就安排了 {slot.memory_role}")
            elif not set(established).intersection(slot.depends_on_sessions):
                errors.append(f"{slot.session_id} 的 depends_on_sessions 应包含更早的同一记忆证据 session")
            earlier_memory_evidence.setdefault(slot.memory_id, []).append(slot.session_id)

    for outline in blueprint.eval_outlines:
        if outline.history_cutoff not in positions:
            errors.append(f"{outline.outline_id} 的 history_cutoff 无效：{outline.history_cutoff}")
            cutoff_position = -1
        else:
            cutoff_position = positions[outline.history_cutoff]
        for evidence_id in outline.required_evidence_session_ids:
            if evidence_id not in positions or positions[evidence_id] > cutoff_position:
                errors.append(f"{outline.outline_id} 引用了 cutoff 后或不存在的证据 session：{evidence_id}")
        if outline.target_label == "triggered":
            cue_owner = cues.get(outline.cue_id)
            if outline.memory_id not in memories or cue_owner is None or cue_owner[0] != outline.memory_id:
                errors.append(f"{outline.outline_id} 的 triggered outline 必须引用有效记忆 cue")
            if not outline.required_evidence_session_ids:
                errors.append(f"{outline.outline_id} 的 triggered outline 必须预留历史证据 session")
            elif not any(
                (evidence_slot := slots_by_id.get(evidence_id)) is not None
                and evidence_slot.memory_id == outline.memory_id
                and evidence_slot.cue_id == outline.cue_id
                and evidence_slot.memory_role in {
                    "encode_association", "triggered_recall", "memory_update"
                }
                for evidence_id in outline.required_evidence_session_ids
            ):
                errors.append(f"{outline.outline_id} 的证据 session 必须实际建立同一 cue 的个人关联")
        elif outline.required_evidence_session_ids:
            errors.append(f"{outline.outline_id} 的非 triggered outline 不应预填 evidence session")
        if outline.target_label == "insufficient_evidence" and (
            outline.memory_id != "none" or outline.cue_id != "none"
        ):
            errors.append(f"{outline.outline_id} 的 insufficient_evidence 必须使用未建立关联的新 cue")
        if outline.target_label != "triggered" and not outline.negative_reason.strip():
            errors.append(f"{outline.outline_id} 必须说明负例成立原因")

    return errors


class DatasetBlueprintPlanner:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    @staticmethod
    def build_payload(
        case: CaseSpec, session_ids: list[str], eval_count: int | None = None,
        constraints: BlueprintConstraints | None = None,
        candidate_context: dict | None = None,
    ) -> dict:
        role_targets = memory_role_targets(len(session_ids), constraints)
        label_targets = eval_label_targets(eval_count, constraints)
        eval_count = sum(label_targets.values())
        role_schedule = _program_role_schedule(session_ids, role_targets)
        eval_ids = canonical_eval_ids(case, eval_count)
        label_schedule = [
            label
            for label in ("triggered", "insufficient_evidence", "not_triggered")
            for _ in range(label_targets.get(label, 0))
        ]
        payload = {
            "planner_brief": case.planner_brief(),
            "immutable_facts": case.immutable_facts(),
            "assigned_ids": {
                "blueprint_id": blueprint_id_for(case, session_ids, eval_count, constraints),
                "session_ids": session_ids,
                "eval_outline_ids": eval_ids,
            },
            "coverage_targets": {
                "memory_roles": role_targets,
                "eval_labels": label_targets,
                "required_cue_types": list(
                    constraints.required_cue_types if constraints
                    else ("object", "scene", "utterance")
                ),
                "session_role_schedule": [
                    {"session_id": session_id, "memory_role": role}
                    for session_id, role in zip(session_ids, role_schedule)
                ],
                "eval_label_schedule": [
                    {"outline_id": outline_id, "target_label": label}
                    for outline_id, label in zip(eval_ids, label_schedule)
                ],
            },
        }
        if candidate_context:
            payload["candidate_context"] = candidate_context
        return payload

    def run(
        self, case: CaseSpec, session_ids: list[str], eval_count: int | None = None,
        constraints: BlueprintConstraints | None = None,
        candidate_context: dict | None = None,
    ) -> DatasetBlueprint:
        payload = self.build_payload(
            case, session_ids, eval_count, constraints, candidate_context
        )
        assigned = payload["assigned_ids"]
        targets = payload["coverage_targets"]
        errors: list[str] = []
        self.last_raw_blueprint: DatasetBlueprint | None = None
        self.last_normalization_notes: list[str] = []
        for attempt in range(2):
            raw = self.llm.generate(
                "dataset_blueprint_planner", payload, DatasetBlueprint
            )
            self.last_raw_blueprint = raw
            generated, notes = compile_dataset_blueprint(
                raw,
                expected_blueprint_id=assigned["blueprint_id"],
                expected_session_ids=assigned["session_ids"],
                expected_eval_ids=assigned["eval_outline_ids"],
                role_targets=targets["memory_roles"],
                label_targets=targets["eval_labels"],
            )
            self.last_normalization_notes = notes
            errors = _hard_fact_conflicts(case, generated.model_dump())
            errors.extend(validate_dataset_blueprint(
                generated,
                expected_blueprint_id=assigned["blueprint_id"],
                expected_session_ids=assigned["session_ids"],
                expected_eval_ids=assigned["eval_outline_ids"],
                role_targets=targets["memory_roles"],
                label_targets=targets["eval_labels"],
                required_cue_types=tuple(targets["required_cue_types"]),
            ))
            if not errors:
                return generated
            payload = {
                **payload,
                "consistency_feedback": {
                    "attempt": attempt + 1,
                    "conflicts": errors,
                    "instruction": (
                        "程序会归一化 IDs、数量、依赖和负例空值；请只修正仍存在的语义冲突，"
                        "尤其是 immutable_facts、life_anchor 和配置要求的各类 cue 质量。"
                    ),
                },
            }
        raise ValueError(f"dataset blueprint invalid after retry: {errors}")


class SessionPlanner:
    def __init__(self, llm: LLMClient, component: str = "session_planner") -> None:
        self.llm = llm
        self.component = component

    @staticmethod
    def build_payload(case: CaseSpec, min_rounds: int, max_rounds: int,
                      session_count: int, total_session_count: int, id_prefix: str,
                      start_index: int = 0, prior_plans: list[dict] | None = None,
                      life_anchor: LifeAnchor | None = None,
                      blueprint: DatasetBlueprint | None = None,
                      session_slots: list[SessionSlot] | None = None,
                      candidate_context: dict | None = None) -> dict:
        assigned_slots = session_slots or []
        payload = {
            "planner_brief": case.planner_brief(),
            "immutable_facts": case.immutable_facts(),
            "established_life_anchor": life_anchor.model_dump() if life_anchor else None,
            "dataset_blueprint": blueprint.model_dump() if blueprint else None,
            "assigned_session_slots": [slot.model_dump() for slot in assigned_slots],
            "planning_window": {
                "batch_size": session_count,
                "total_session_count": total_session_count,
                "start_index": start_index,
                "id_prefix": id_prefix,
                "expected_session_ids": [slot.session_id for slot in assigned_slots],
            },
            "constraints": {"min_rounds": min_rounds, "max_rounds": max_rounds},
            "prior_plans": prior_plans or [],
        }
        if candidate_context:
            payload["candidate_context"] = candidate_context
        return payload

    def run(self, case: CaseSpec, min_rounds: int, max_rounds: int,
            session_count: int, total_session_count: int, id_prefix: str,
            start_index: int = 0, prior_plans: list[dict] | None = None,
            life_anchor: LifeAnchor | None = None,
            blueprint: DatasetBlueprint | None = None,
            session_slots: list[SessionSlot] | None = None,
            candidate_context: dict | None = None) -> SessionPlanList:
        payload = self.build_payload(
            case, min_rounds, max_rounds, session_count, total_session_count,
            id_prefix, start_index, prior_plans, life_anchor, blueprint, session_slots,
            candidate_context,
        )
        generated = None
        for attempt in range(2):
            generated = self.llm.generate(self.component, payload, SessionPlanBatch)
            current_batch = generated.plans[:session_count]
            conflicts = _hard_fact_conflicts(
                case, {"plans": [plan.model_dump() for plan in current_batch]}
            )
            if session_slots:
                slot_by_id = {slot.session_id: slot for slot in session_slots}
                for plan in current_batch:
                    slot = slot_by_id.get(plan.session_id)
                    if slot is None:
                        continue
                    for field in (
                        "memory_role", "memory_id", "cue_id", "evidence_goal",
                        "target_emotion", "relative_to_past", "depends_on_sessions",
                    ):
                        if getattr(plan, field) != getattr(slot, field):
                            conflicts.append(f"{plan.session_id} 改写了蓝图字段 {field}")
            if not conflicts:
                break
            payload = {
                **payload,
                "consistency_feedback": {
                    "attempt": attempt + 1,
                    "conflicts": conflicts,
                    "instruction": "重写冲突计划；immutable_facts 高于故事节拍，不得反转硬事实。",
                },
            }
        else:
            raise ValueError(f"planner violated immutable facts after retry: {conflicts}")
        assert generated is not None
        if session_slots:
            expected_ids = [slot.session_id for slot in session_slots]
        elif case.session_outlines:
            expected_ids = [outline.session_id for outline in case.session_outlines[:session_count]]
        else:
            expected_ids = [f"{id_prefix}-S{index + 1:02d}" for index in range(start_index, start_index + session_count)]

        if len(generated.plans) < session_count:
            raise ValueError(
                f"planner returned only {len(generated.plans)} plans for a batch of {session_count}"
            )

        exact_by_id = {plan.session_id: plan for plan in generated.plans}
        if all(session_id in exact_by_id for session_id in expected_ids):
            selected = [exact_by_id[session_id] for session_id in expected_ids]
        else:
            by_number = {}
            duplicate_numbers = set()
            for plan in generated.plans:
                match = re.search(r"S[-_]?0*(\d+)$", plan.session_id, flags=re.IGNORECASE)
                if not match:
                    continue
                number = int(match.group(1))
                if number in by_number:
                    duplicate_numbers.add(number)
                by_number[number] = plan
            expected_numbers = list(range(start_index + 1, start_index + session_count + 1))
            if not duplicate_numbers.intersection(expected_numbers) and all(
                number in by_number for number in expected_numbers
            ):
                selected = [by_number[number] for number in expected_numbers]
            else:
                selected = generated.plans[:session_count]

        normalized = [
            plan.model_copy(update={"session_id": expected_id})
            for plan, expected_id in zip(selected, expected_ids)
        ]
        if session_slots:
            slot_fields = (
                "memory_role", "memory_id", "cue_id", "evidence_goal", "target_emotion",
                "relative_to_past", "depends_on_sessions",
            )
            normalized = [
                plan.model_copy(update={field: getattr(slot, field) for field in slot_fields})
                for plan, slot in zip(normalized, session_slots)
            ]
        actual_ids = [plan.session_id for plan in selected]
        if len(generated.plans) != session_count or actual_ids != expected_ids:
            print(
                f"planner returned {len(generated.plans)} plans with non-canonical batch IDs; "
                f"normalized to {expected_ids}",
                flush=True,
            )
        return SessionPlanList(
            plans=normalized,
            life_anchor=life_anchor or getattr(generated, "life_anchor", None),
            blueprint_id=blueprint.blueprint_id if blueprint else "",
            blueprint_fingerprint=blueprint_fingerprint(blueprint) if blueprint else "",
        )


class SessionWriter:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def run(self, case: CaseSpec, plan: dict, recent_sessions: list[Session],
            story_so_far: list[dict] | None = None,
            life_anchor: LifeAnchor | None = None,
            blueprint: DatasetBlueprint | None = None) -> Session:
        payload = {
            "case_spec": case.model_dump(),
            "immutable_facts": case.immutable_facts(),
            "life_anchor": life_anchor.model_dump() if life_anchor else None,
            "dataset_blueprint": blueprint.model_dump() if blueprint else None,
            "session_plan": plan,
            "story_so_far": story_so_far or [],
            "recent_history": [s.model_dump() for s in recent_sessions],
        }
        session = None
        for attempt in range(2):
            session = self.llm.generate("session_writer", payload, Session)
            conflicts = _hard_fact_conflicts(case, session.model_dump())
            if not conflicts:
                return session
            payload = {
                **payload,
                "consistency_feedback": {
                    "attempt": attempt + 1,
                    "conflicts": conflicts,
                    "instruction": "忽略 SessionPlan 中与 immutable_facts 冲突的部分，并重写对话。",
                },
            }
        raise ValueError(f"writer violated immutable facts after retry: {conflicts}")


class SessionVerifier:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def run(self, case: CaseSpec, plan: dict, session: Session, recent_sessions: list[Session]) -> VerificationResult:
        return self.llm.generate("session_verifier", {
            "case_spec": case.model_dump(), "session_plan": plan,
            "session": session.model_dump(),
            "recent_history": [s.model_dump() for s in recent_sessions],
        }, VerificationResult)


class SessionReviser:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def run(self, case: CaseSpec, plan: dict, session: Session, issues: VerificationResult) -> Session:
        return self.llm.generate("session_reviser", {
            "case_spec": case.model_dump(), "session_plan": plan,
            "session": session.model_dump(), "issues": issues.model_dump(),
        }, Session)


class NaturalnessChecker:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def run(self, case: CaseSpec, session: Session) -> VerificationResult:
        return self.llm.generate("naturalness_checker", {
            "conversation_style": case.character_profile.conversation_style,
            "session": session.model_dump(),
        }, VerificationResult)


class EvalGenerator:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def run(
        self, case: CaseSpec, sessions: list[Session],
        outline: BlueprintEvalOutline, blueprint: DatasetBlueprint,
    ) -> EvalGenerationResult:
        visible = []
        cutoff_found = False
        for session in sessions:
            visible.append(session)
            if session.session_id == outline.history_cutoff:
                cutoff_found = True
                break
        if not cutoff_found:
            raise ValueError(f"eval history cutoff is not available: {outline.history_cutoff}")
        generated = self.llm.generate("eval_generator", {
            "case_spec": case.model_dump(),
            "dataset_blueprint": blueprint.model_dump(),
            "eval_outline": outline.model_dump(),
            "visible_sessions": [s.model_dump() for s in visible],
            "candidate_count": 3,
            "stage_boundary": {
                "resolver_enabled": False,
                "verifier_enabled": False,
                "finalizer_enabled": False,
                "instruction": "只生成候选当前输入，不解析 evidence_turn_ids，不输出最终 Gold。",
            },
        }, EvalGenerationResult)
        blueprint_cues = {
            cue.cue_id: cue
            for memory in blueprint.emotion_memory_map
            for cue in memory.cue_seeds
        }
        blueprint_cue = blueprint_cues.get(outline.cue_id)
        normalized = []
        for candidate in generated.candidates:
            current_input = candidate.current_input
            option_ids = [option.cue_id for option in current_input.cue_options]
            if (
                blueprint_cue is not None
                and outline.cue_id not in option_ids
                and current_input.cue_type == blueprint_cue.cue_type
                and current_input.cue_options
            ):
                # Models sometimes preserve the intended Blueprint cue but append
                # a candidate suffix (for example C02_candidate_01). The cue is
                # still the same semantic target, so lock its identifier back to
                # the canonical Blueprint ID before deterministic verification.
                matching_index = next(
                    (
                        index for index, option in enumerate(current_input.cue_options)
                        if option.cue_id.casefold().startswith(outline.cue_id.casefold())
                    ),
                    0,
                )
                cue_options = list(current_input.cue_options)
                cue_options[matching_index] = cue_options[matching_index].model_copy(
                    update={"cue_id": outline.cue_id}
                )
                current_input = current_input.model_copy(
                    update={"cue_options": cue_options}
                )
            normalized.append(candidate.model_copy(update={
                "current_input": current_input,
                "target_label": outline.target_label,
                "target_emotion": outline.target_emotion,
                "blueprint_cue_id": outline.cue_id,
                "history_cutoff": outline.history_cutoff,
            }))
        return EvalGenerationResult(outline_id=outline.outline_id, candidates=normalized)


def _visible_sessions(
    sessions: list[Session], history_cutoff: str
) -> list[Session]:
    visible: list[Session] = []
    for session in sessions:
        visible.append(session)
        if session.session_id == history_cutoff:
            return visible
    raise ValueError(f"eval history cutoff is not available: {history_cutoff}")


class EvidenceResolver:
    """Resolve Blueprint session-level evidence into exact historical user turns."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def run(
        self, sessions: list[Session], outline: BlueprintEvalOutline,
        blueprint: DatasetBlueprint,
    ) -> EvidenceResolution:
        if outline.target_label != "triggered":
            return EvidenceResolution(outline_id=outline.outline_id, evidence_turn_ids=[])

        visible = _visible_sessions(sessions, outline.history_cutoff)
        required_ids = set(outline.required_evidence_session_ids)
        evidence_sessions = [
            session for session in visible if session.session_id in required_ids
        ]
        if {session.session_id for session in evidence_sessions} != required_ids:
            raise ValueError(
                f"required evidence sessions are not visible for {outline.outline_id}"
            )
        allowed_turns = {
            turn.turn_id
            for session in evidence_sessions
            for turn in session.turns
            if turn.speaker == "user"
        }
        if not allowed_turns:
            raise ValueError(f"no user evidence turns are available for {outline.outline_id}")

        generated = self.llm.generate("eval_resolver", {
            "eval_outline": outline.model_dump(),
            "emotion_memory_map": [
                memory.model_dump()
                for memory in blueprint.emotion_memory_map
                if memory.memory_id == outline.memory_id
            ],
            "required_evidence_sessions": [
                session.model_dump() for session in evidence_sessions
            ],
            "instruction": (
                "只从给定 Session 的 user turns 中选择最小且联合充分的历史证据；"
                "不要引用 assistant turn，不要生成新 turn ID。"
            ),
        }, EvidenceResolution)
        evidence_ids = list(dict.fromkeys(generated.evidence_turn_ids))
        if not evidence_ids:
            raise ValueError(f"resolver returned no evidence for {outline.outline_id}")
        invalid = [turn_id for turn_id in evidence_ids if turn_id not in allowed_turns]
        if invalid:
            raise ValueError(
                f"resolver returned non-user or out-of-scope evidence turns: {invalid}"
            )
        return EvidenceResolution(
            outline_id=outline.outline_id,
            evidence_turn_ids=evidence_ids,
        )


def build_eval_candidate_list(
    generation: EvalGenerationResult,
    outline: BlueprintEvalOutline,
    resolution: EvidenceResolution,
) -> EvalCandidateList:
    evidence_ids = (
        resolution.evidence_turn_ids if outline.target_label == "triggered" else []
    )
    trigger_cue_id = outline.cue_id if outline.target_label == "triggered" else "none"
    return EvalCandidateList(candidates=[
        EvalCandidate(
            current_input=draft.current_input,
            history_cutoff=outline.history_cutoff,
            gold=Gold(
                trigger_label=outline.target_label,
                trigger_cue_id=trigger_cue_id,
                evidence_turn_ids=evidence_ids,
                current_emotion=outline.target_emotion,
            ),
        )
        for draft in generation.candidates
    ])


def validate_eval_candidate(
    candidate: EvalCandidate,
    outline: BlueprintEvalOutline,
    blueprint: DatasetBlueprint,
) -> list[str]:
    """Check deterministic Eval invariants before asking the semantic verifier."""
    errors: list[str] = []
    current = candidate.current_input
    if candidate.history_cutoff != outline.history_cutoff:
        errors.append("history_cutoff 与 Blueprint 不一致")
    if current.input_type != "text" or current.image_refs:
        errors.append("当前版本只允许纯文本 Eval，image_refs 必须为空")
    if not current.text.strip():
        errors.append("current_input.text 不能为空")
    option_ids = [option.cue_id for option in current.cue_options]
    if len(option_ids) != len(set(option_ids)):
        errors.append("cue_options 中的 cue_id 不能重复")

    blueprint_cues = {
        cue.cue_id: cue
        for memory in blueprint.emotion_memory_map
        for cue in memory.cue_seeds
    }
    if outline.target_label == "triggered":
        cue = blueprint_cues.get(outline.cue_id)
        if outline.cue_id not in option_ids:
            errors.append("triggered 候选必须在 cue_options 中包含 Blueprint cue_id")
        if cue and current.cue_type != cue.cue_type:
            errors.append("current_input.cue_type 与 Blueprint cue 类型不一致")
    elif outline.target_label == "insufficient_evidence":
        if not current.cue_options or current.cue_type == "none":
            errors.append("insufficient_evidence 必须包含一个可感知的新 cue")
        reused = sorted(set(option_ids).intersection(blueprint_cues))
        if reused:
            errors.append(f"insufficient_evidence 不得复用 Blueprint cue ID：{reused}")
    elif outline.cue_id != "none" and outline.cue_specificity in {"exact", "related"}:
        if outline.cue_id not in option_ids:
            errors.append("已知 cue 控制负例必须在 cue_options 中包含对应 Blueprint cue_id")
    return errors


class EvalVerifier:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def run(
        self, case: CaseSpec, sessions: list[Session], outline: BlueprintEvalOutline,
        candidates: EvalCandidateList, blueprint: DatasetBlueprint,
        precheck_issues: list[list[str]] | None = None,
    ) -> EvalSelection:
        visible = _visible_sessions(sessions, outline.history_cutoff)
        return self.llm.generate("eval_verifier", {
            "case_spec": case.model_dump(),
            "eval_outline": outline.model_dump(),
            "emotion_memory_map": [memory.model_dump() for memory in blueprint.emotion_memory_map],
            "visible_sessions": [s.model_dump() for s in visible],
            "candidates": candidates.model_dump(),
            "deterministic_precheck_issues": precheck_issues or [[], [], []],
            "selection_indexing": "zero_based: 0, 1, 2; reject_all 时必须为 -1",
        }, EvalSelection)
