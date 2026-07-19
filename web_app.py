from __future__ import annotations

import json
import html
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

try:
    import streamlit as st
except ModuleNotFoundError as exc:
    raise SystemExit(
        "未安装 Streamlit。请先运行 `pip install streamlit`，然后执行 "
        "`streamlit run web_app.py`。"
    ) from exc

from pydantic import ValidationError

from src.components import validate_blueprint_constraints
from src.config import BlueprintConstraints
from src.models import CaseSpec


ROOT = Path(__file__).resolve().parent
PROMPT_DIR = ROOT / "prompts"
DEFAULT_CASE = ROOT / "cases" / "case_a.json"
CONFIG_PATH = ROOT / "config.json"
WEB_OUTPUT_ROOT = ROOT / "outputs" / "web_runs"

FALLBACK_MODELS = {
    "OpenAI": [
        "gpt-5.4-nano", "gpt-5.4-mini", "gpt-5.4", "gpt-5.4-pro",
        "gpt-5.3-chat-latest", "gpt-5.2", "gpt-5.1", "gpt-4.1", "gpt-4o-mini",
    ],
    "DeepSeek": [
        "deepseek-v4-pro", "deepseek-v4-flash", "deepseek-v3.2",
        "deepseek-v3.2-think", "DeepSeek-V3.1-Think", "DeepSeek-R1",
    ],
    "Claude": [
        "claude-sonnet-5", "claude-opus-4-8", "claude-opus-4-8-think",
        "claude-sonnet-4-6", "claude-haiku-4-5", "claude-3-7-sonnet",
    ],
    "Qwen": [
        "qwen3.7-plus", "qwen3.7-max", "qwen3.6-plus", "qwen3.6-flash",
        "qwen3.5-plus", "qwen3.5-flash", "qwen3-max", "qwen3-coder-plus",
    ],
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_run_name(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip()).strip("-")
    return cleaned[:64]


def validate_spec(raw: str) -> tuple[CaseSpec | None, str | None]:
    try:
        return CaseSpec.model_validate_json(raw), None
    except (ValidationError, ValueError) as exc:
        return None, str(exc)


def prompt_files() -> list[Path]:
    return sorted(PROMPT_DIR.glob("*.txt"))


def provider_for_model(model_id: str) -> str:
    lowered = model_id.lower()
    if lowered.startswith("gpt-") or re.match(r"^o[134]-", lowered):
        return "OpenAI"
    if "deepseek" in lowered:
        return "DeepSeek"
    if "claude" in lowered:
        return "Claude"
    if "qwen" in lowered or "qwq" in lowered:
        return "Qwen"
    return "其他"


def group_models(model_ids: list[str]) -> dict[str, list[str]]:
    grouped = {name: [] for name in ["OpenAI", "DeepSeek", "Claude", "Qwen", "其他"]}
    for model_id in sorted(set(model_ids), key=str.lower):
        grouped[provider_for_model(model_id)].append(model_id)
    return grouped


def fetch_models(api_key: str, base_url: str) -> tuple[list[str], str | None]:
    if not api_key or not base_url:
        return [], "请先载入 API Key 并填写 Base URL。"
    script = r"""
$ErrorActionPreference = 'Stop'
$uri = $env:DG_BASE_URL.TrimEnd('/')
if (-not $uri.EndsWith('/v1')) { $uri += '/v1' }
$headers = @{ Authorization = "Bearer $env:DG_API_KEY" }
$response = Invoke-RestMethod -Uri ($uri + '/models') -Headers $headers -Method Get -TimeoutSec 60
@($response.data | ForEach-Object { $_.id }) | ConvertTo-Json -Compress
"""
    env = os.environ.copy()
    env["DG_API_KEY"] = api_key
    env["DG_BASE_URL"] = base_url
    try:
        completed = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=75,
            env=env,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [], f"模型列表请求失败：{exc}"
    if completed.returncode != 0:
        return [], "模型列表请求失败，请检查 API Key、Base URL 或网络。"
    try:
        payload = json.loads(completed.stdout.strip())
        return ([payload] if isinstance(payload, str) else payload), None
    except json.JSONDecodeError:
        return [], "模型列表响应无法解析。"


def store_api_key() -> None:
    value = st.session_state.get("api_key_draft", "").strip()
    if value:
        st.session_state["api_key_secret"] = value
        st.session_state["api_key_draft"] = ""


def clear_api_key() -> None:
    st.session_state["api_key_secret"] = ""
    st.session_state["api_key_draft"] = ""


def update_progress(line: str, total: int, completed: int) -> tuple[float, str]:
    blueprint_match = re.match(r"generating Blueprint candidate (\d+)/(\d+)", line)
    if blueprint_match:
        current, count = map(int, blueprint_match.groups())
        return 0.04 + 0.28 * ((current - 1) / max(count, 1)), f"正在生成 Blueprint 候选 {current}/{count}"
    if line.startswith("completed Blueprint candidate "):
        current, count = map(int, re.findall(r"\d+", line)[-2:])
        return 0.04 + 0.28 * (current / max(count, 1)), f"Blueprint 候选已完成 {current}/{count}"
    plan_match = re.match(r"generating Plan candidate (\d+)/(\d+)", line)
    if plan_match:
        current, count = map(int, plan_match.groups())
        return 0.36 + 0.58 * ((current - 1) / max(count, 1)), f"正在生成 Plan 候选 {current}/{count}"
    if line.startswith("completed Plan candidate "):
        current, count = map(int, re.findall(r"\d+", line)[-2:])
        return 0.36 + 0.58 * (current / max(count, 1)), f"Plan 候选已完成 {current}/{count}"
    if line.startswith(("blueprint candidates:", "plan candidates:")):
        return 0.98, "候选方案已就绪"
    if line.startswith("blueprint ready:") or line.startswith("resumed dataset blueprint"):
        return 0.06, "Dataset 蓝图已就绪"
    if line.startswith("planned ") or line.startswith("resumed ") and "saved plans" in line:
        return 0.12, "Session 计划已就绪"
    if line.startswith("generating eval "):
        return 0.95, f"正在生成 {line.removeprefix('generating ').removesuffix('...')} 候选"
    if line.startswith("completed eval "):
        return 0.97, "Eval 候选生成中"
    if line.startswith("completed Eval example "):
        return 0.985, "正式 Eval Examples 整理中"
    if line.startswith("generating "):
        session_id = line.removeprefix("generating ").removesuffix("...")
        value = 0.15 + 0.79 * (completed / max(total, 1))
        return value, f"正在生成 {session_id}"
    if line.startswith("completed "):
        value = 0.15 + 0.79 * (completed / max(total, 1))
        return value, f"已完成 {completed}/{total} 个 Session"
    if line.startswith("benchmark:") or line.startswith("plans:"):
        return 0.98, "正在汇总生成产物"
    return 0.03, "正在初始化生成任务"


def classify_log(rel_path: str) -> tuple[str, str, str]:
    """Map a log file's relative path to (group, label, icon).

    group:  用于分组的 key（init / planner / session_id / finalize）
    label:  人类可读的阶段名
    icon:   emoji 状态标记
    """
    name = Path(rel_path).name
    parent = Path(rel_path).parent.as_posix()

    if name == "00_original_case_spec.json":
        return "init", "CaseSpec 加载", "📦"
    if re.match(r"01_blueprint_candidate_\d+\.json", name):
        number = int(re.findall(r"\d+", name)[-1])
        return "blueprint-candidates", f"Blueprint 候选 {number} · 完成", "🧭"
    if re.match(r"02_plan_candidate_.+_\d+\.json", name):
        number = int(re.findall(r"\d+", name)[-1])
        return "plan-candidates", f"Plan 候选 {number} · 完成", "🗺️"
    if name == "99_blueprint_candidate_result.json":
        return "candidate-finalize", "Blueprint 候选准备完成", "✅"
    if name == "99_plan_candidate_result.json":
        return "candidate-finalize", "Plan 候选准备完成", "✅"
    if re.match(r"01_dataset_blueprint_(planner|resumed)\.json", name):
        tag = "全局规划" if "planner" in name else "复用已有蓝图"
        return "blueprint", f"DatasetBlueprintPlanner · {tag}", "🧭"
    if re.match(r"(?:01|02)_planner_(batch_\d+|resumed)\.json", name):
        m = re.search(r"batch_(\d+)", name)
        tag = f"批次 {m.group(1)}" if m else "复用已有计划"
        return "planner", f"SessionPlanner · {tag}", "🗺️"
    if name == "99_pipeline_result.json":
        return "finalize", "Pipeline 完成", "✅"
    if parent.startswith("sessions/"):
        session_id = Path(parent).name
        if name == "01_writer.json":
            return session_id, f"{session_id} · Writer", "✍️"
        if name == "99_final_session.json":
            return session_id, f"{session_id} · 完成", "✅"
        m = re.match(r"\d+_cycle_(\d+)_(.+)\.json", name)
        if m:
            cycle, kind = m.group(1), m.group(2)
            kind_label = {
                "structural_verifier": "结构校验",
                "session_verifier": "语义校验",
                "reviser": "修订",
                "naturalness_reviser": "自然度修订",
            }.get(kind, kind)
            return session_id, f"{session_id} · {kind_label} (Cycle {cycle})", "🔧"
        if name == "90_naturalness_checker.json":
            return session_id, f"{session_id} · 自然度检查", "🌿"
        if name == "91_naturalness_reviser.json":
            return session_id, f"{session_id} · 自然度修订", "🔧"
    if parent.startswith("evals/"):
        outline_id = Path(parent).name
        if name == "01_generator.json" or name.startswith("01_generator_retry_"):
            return "evals", f"{outline_id} · EvalGenerator", "🧪"
        if name == "02_evidence_resolver.json":
            return "evals", f"{outline_id} · EvidenceResolver", "🔎"
        if name.startswith("03_precheck_cycle_"):
            return "evals", f"{outline_id} · 硬约束预检", "🧱"
        if name.startswith("04_verifier_cycle_"):
            return "evals", f"{outline_id} · EvalVerifier", "✅"
        if name == "05_finalizer.json":
            return "evals", f"{outline_id} · GoldFinalizer", "🏁"
        if name == "98_eval_example_error.json":
            return "evals", f"{outline_id} · Eval 失败", "⚠️"
    return Path(rel_path).stem, rel_path, "📄"


def scan_pipeline_logs(logs_dir: Path, seen: set[str]) -> list[dict]:
    """Scan logs directory for newly appeared JSON artifact files."""
    found: list[dict] = []
    if not logs_dir.exists():
        return found
    for path in sorted(logs_dir.rglob("*.json")):
        rel = path.relative_to(logs_dir).as_posix()
        if rel in seen:
            continue
        seen.add(rel)
        group, label, icon = classify_log(rel)
        found.append({"path": path, "rel": rel, "group": group, "label": label, "icon": icon})
    return found


def render_live_stages(stages: list[dict]) -> str:
    """Build a compact markdown summary of completed pipeline stages."""
    if not stages:
        return "_等待产物生成…_"
    lines: list[str] = []
    current_group: str | None = None
    for stage in stages:
        if stage["group"] != current_group:
            if current_group is not None:
                lines.append("")
            current_group = stage["group"]
        lines.append(f"{stage['icon']} {stage['label']}")
    return "\n".join(lines)


def render_artifact(path: Path) -> None:
    """Render a log artifact file in a readable format."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        st.error(f"无法读取产物：{exc}")
        return
    component = payload.get("component", "pipeline")
    st.caption(f"组件：`{component}` · 路径：`{path.name}`")

    # 优先展示 output 字段
    output = payload.get("output")
    if output is None:
        st.json(payload, expanded=False)
        return

    # Dataset blueprint：展示全局记忆与覆盖安排
    if component == "dataset_blueprint_planner" and isinstance(output, dict):
        candidate_meta = output if "blueprint" in output else None
        if candidate_meta:
            st.write(
                f"**{candidate_meta.get('candidate_id', '')} · "
                f"{candidate_meta.get('title', '')}**"
            )
            st.caption(candidate_meta.get("summary", ""))
            output = candidate_meta["blueprint"]
        st.write(f"**Blueprint**：{output.get('blueprint_id', '')}")
        st.caption(
            f"{len(output.get('session_slots', []))} 个 session slots · "
            f"{len(output.get('eval_outlines', []))} 个 eval outlines"
        )
        for memory in output.get("emotion_memory_map", []):
            st.write(f"**{memory.get('memory_id', '')}** · {memory.get('event_summary', '')}")
            for cue in memory.get("cue_seeds", []):
                st.write(
                    f"- `{cue.get('cue_id', '')}` · {cue.get('cue_type', '')} · "
                    f"{cue.get('canonical_form', '')}"
                )
        if payload.get("normalization_notes"):
            st.info("程序归一化：" + "；".join(payload["normalization_notes"]))
        if payload.get("raw_output"):
            with st.expander("查看 LLM 原始蓝图输出"):
                st.json(payload["raw_output"], expanded=False)
        st.json(output.get("session_slots", []), expanded=False)
        return

    # Writer / final session：展示对话
    if component in {"session_writer", "session_reviser", "pipeline"} and isinstance(output, dict) and "turns" in output:
        st.write(f"**Session**：{output.get('session_id', '')} · {output.get('topic', '')}")
        for turn in output.get("turns", []):
            with st.chat_message(turn.get("speaker", "user")):
                st.caption(f"{turn.get('turn_id', '')} · {turn.get('round_id', '')}")
                st.write(turn.get("text", ""))
        return

    # Planner：展示计划列表
    if component in {"session_planner", "plan_candidate_planner"} and isinstance(output, dict):
        if "plan" in output:
            st.write(f"**{output.get('candidate_id', '')} · {output.get('title', '')}**")
            st.caption(output.get("summary", ""))
            output = output["plan"]
        if "plans" not in output:
            st.json(output, expanded=False)
            return
        for plan in output["plans"]:
            st.write(
                f"**{plan.get('session_id', '')}** · {plan.get('date', '')} · "
                f"`{plan.get('session_type', 'daily_life')}` · {plan.get('topic', '')}"
            )
            if plan.get("scene"):
                st.caption(f"场景：{plan['scene']}")
            if plan.get("user_intent"):
                st.caption(f"聊天动机：{plan['user_intent']}")
            st.write(plan.get("story_beat", ""))
            if plan.get("continuity_hook"):
                st.caption(f"后续钩子：{plan['continuity_hook']}")
            st.caption(f"rounds={plan.get('round_count', '')} · {plan.get('outline_function', '')}")
        return

    if component == "eval_generator" and isinstance(output, dict):
        st.write(f"**Eval outline**：{output.get('outline_id', '')}")
        for index, candidate in enumerate(output.get("candidates", []), 1):
            with st.expander(f"候选 {index} · {candidate.get('target_label', '')}", expanded=index == 1):
                current = candidate.get("current_input", {})
                st.write(current.get("text", ""))
                st.caption(
                    f"emotion={candidate.get('target_emotion', '')} · "
                    f"blueprint cue={candidate.get('blueprint_cue_id', 'none')} · "
                    f"cutoff={candidate.get('history_cutoff', '')}"
                )
                st.json(current.get("cue_options", []), expanded=False)
        return

    # Verifier：展示结论
    if component in {"session_verifier", "deterministic_structural_verifier", "naturalness_checker"} and isinstance(output, dict):
        result = output.get("result", "")
        st.write(f"**结论**：{'✅ pass' if result == 'pass' else '⚠️ revise'}")
        for issue in output.get("issues", []):
            st.write(f"- `{issue.get('turn_id', '')}` ({issue.get('type', '')})：{issue.get('description', '')}")
        return

    # 其他：展示 JSON
    st.json(output, expanded=False)


def render_eval_examples(examples: list[dict], *, key_prefix: str) -> None:
    if not examples:
        st.info("该运行没有完成正式 Eval Examples。")
        return
    sample_ids = [item.get("sample_id", f"Eval {index + 1}") for index, item in enumerate(examples)]
    selected_id = st.selectbox(
        "选择 Eval Example", sample_ids, key=f"{key_prefix}_eval_example"
    )
    sample = next(item for item in examples if item.get("sample_id") == selected_id)
    current = sample.get("current_input", {})
    gold = sample.get("gold", {})
    st.subheader(sample.get("sample_id", "Eval Example"))
    st.write(current.get("text", ""))
    st.caption(
        f"cutoff={sample.get('history_cutoff', '—')} · "
        f"input={current.get('input_type', 'text')} · cue type={current.get('cue_type', 'none')}"
    )
    cols = st.columns(3)
    cols[0].metric("Label", gold.get("trigger_label", "—"))
    cols[1].metric("Emotion", gold.get("current_emotion", "—"))
    cols[2].metric("Trigger cue", gold.get("trigger_cue_id", "none"))
    st.write("**Evidence turns**：" + (", ".join(gold.get("evidence_turn_ids", [])) or "无"))
    st.write("**Cue options**")
    st.json(current.get("cue_options", []), expanded=False)


def render_candidate_selection(
    output_dir: Path, plan_candidate_count: int
) -> tuple[str, str, str | None] | None:
    """Render sequential Blueprint → Plan review and return the next action."""
    manifest_path = output_dir / "planning_candidates.json"
    if not manifest_path.exists() or (output_dir / "benchmark.json").exists():
        return None
    manifest = read_json(manifest_path)
    if manifest.get("version", 1) != 2:
        st.warning("这是旧版 3×3 候选文件，不能按新流程继续；请用新的测试名称重新生成。")
        return None
    blueprint_candidates = manifest.get("blueprint_candidates", [])
    if not blueprint_candidates:
        st.warning("Blueprint 候选文件不完整，请重新生成。")
        return None

    key_suffix = re.sub(r"[^a-zA-Z0-9_-]+", "-", output_dir.name)
    blueprint_selection = manifest.get("blueprint_selection") or {}
    final_selection = manifest.get("selection") or {}
    bp_ids = [item["candidate_id"] for item in blueprint_candidates]
    default_bp = (
        final_selection.get("blueprint_candidate_id")
        or blueprint_selection.get("blueprint_candidate_id")
    )

    st.subheader("分步审阅规划方案")
    st.caption(
        "先选择一个 Blueprint；程序只针对这个 Blueprint 生成 Plan 候选。"
        "选择 Plan 后才会启动 Writer。"
    )
    steps = st.columns(3)
    steps[0].info("1 · 选择 Blueprint")
    steps[1].info("2 · 生成并选择 Plan")
    steps[2].info("3 · 继续生成对话")

    selected_bp_id = st.selectbox(
        "选择 Blueprint",
        bp_ids,
        index=bp_ids.index(default_bp) if default_bp in bp_ids else 0,
        format_func=lambda candidate_id: next(
            item["title"] for item in blueprint_candidates
            if item["candidate_id"] == candidate_id
        ),
        key=f"candidate_bp_{key_suffix}",
    )
    selected_bp = next(
        item for item in blueprint_candidates if item["candidate_id"] == selected_bp_id
    )
    blueprint = selected_bp["blueprint"]

    st.write(f"**{selected_bp['title']}**")
    st.caption(selected_bp.get("summary", ""))
    anchor = blueprint.get("life_anchor", {})
    anchor_cols = st.columns(2)
    with anchor_cols[0]:
        st.write(f"**人物生活锚点**：{anchor.get('identity', '—')}")
        st.write("**常见场景**：" + "、".join(anchor.get("recurring_scenes", [])))
    with anchor_cols[1]:
        st.write("**兴趣**：" + "、".join(anchor.get("interests", [])))
        st.write("**持续生活线**：" + "、".join(anchor.get("ongoing_threads", [])))
    role_counts: dict[str, int] = {}
    for slot in blueprint.get("session_slots", []):
        role = slot.get("memory_role", "none")
        role_counts[role] = role_counts.get(role, 0) + 1
    st.write(f"**Session 记忆角色分布**：{role_counts}")
    for memory in blueprint.get("emotion_memory_map", []):
        with st.expander(
            f"{memory.get('memory_id', 'Memory')} · {memory.get('event_summary', '')}",
            expanded=False,
        ):
            cue_rows = [{
                "类型": cue.get("cue_type", ""),
                "核心线索": cue.get("canonical_form", ""),
                "相近表达": "、".join(cue.get("related_forms", [])),
                "个人意义": cue.get("personal_meaning", ""),
            } for cue in memory.get("cue_seeds", [])]
            st.dataframe(cue_rows, use_container_width=True, hide_index=True)

    plan_candidates = [
        item for item in manifest.get("plan_candidates", [])
        if item.get("blueprint_candidate_id") == selected_bp_id
    ]
    plans_match_selection = (
        blueprint_selection.get("blueprint_candidate_id") == selected_bp_id
        and manifest.get("plan_candidate_count", 0) == plan_candidate_count
        and len(plan_candidates) == plan_candidate_count
    )
    if not plans_match_selection:
        if manifest.get("plan_candidates"):
            st.warning(
                "当前已保存的 Plan 属于另一个 Blueprint 或候选数量不同。"
                "继续后会为当前 Blueprint 重新生成 Plan 候选。"
            )
        action_cols = st.columns([1, 2])
        action_cols[0].download_button(
            "下载 Blueprint 候选",
            data=manifest_path.read_bytes(),
            file_name=manifest_path.name,
            mime="application/json",
            use_container_width=True,
        )
        generate_plans = action_cols[1].button(
            f"选定 {selected_bp_id}，生成 {plan_candidate_count} 个 Plan",
            type="primary",
            use_container_width=True,
            key=f"generate_plans_{key_suffix}",
        )
        return ("prepare_plans", selected_bp_id, None) if generate_plans else None

    plan_ids = [item["candidate_id"] for item in plan_candidates]
    default_plan = final_selection.get("plan_candidate_id")
    selected_plan_id = st.selectbox(
        f"选择 Plan（均基于 {selected_bp_id}）",
        plan_ids,
        index=plan_ids.index(default_plan) if default_plan in plan_ids else 0,
        format_func=lambda candidate_id: next(
            item["title"] for item in plan_candidates
            if item["candidate_id"] == candidate_id
        ),
        key=f"candidate_plan_{key_suffix}_{selected_bp_id}",
    )
    selected_plan = next(
        item for item in plan_candidates if item["candidate_id"] == selected_plan_id
    )
    plans = selected_plan["plan"].get("plans", [])
    plan_tab, coverage_tab = st.tabs(["Plan 内容", "Blueprint 落实情况"])
    with plan_tab:
        st.write(f"**{selected_plan['title']}**")
        st.caption(selected_plan.get("summary", ""))
        plan_rows = [{
            "Session": plan.get("session_id", ""),
            "日期": plan.get("date", ""),
            "主题": plan.get("topic", ""),
            "场景": plan.get("scene", ""),
            "故事推进": plan.get("story_beat", ""),
            "生活线": plan.get("life_thread", ""),
            "轮数": plan.get("round_count", ""),
        } for plan in plans]
        st.dataframe(plan_rows, use_container_width=True, hide_index=True, height=430)
        with st.expander("逐条查看完整 Plan"):
            for plan in plans:
                st.markdown(
                    f"**{plan.get('session_id', '')} · {plan.get('topic', '')}**  \n"
                    f"场景：{plan.get('scene', '')}  \n"
                    f"聊天动机：{plan.get('user_intent', '')}  \n"
                    f"故事节拍：{plan.get('story_beat', '')}  \n"
                    f"后续钩子：{plan.get('continuity_hook') or '无'}"
                )
                st.divider()
    with coverage_tab:
        st.success(f"{selected_plan['title']} 由 {selected_bp['title']} 直接生成")
        coverage_rows = [{
            "Session": plan.get("session_id", ""),
            "Plan 主题": plan.get("topic", ""),
            "记忆角色": plan.get("memory_role", "none"),
            "Cue": plan.get("cue_id", "none"),
            "证据目标": plan.get("evidence_goal", ""),
        } for plan in plans]
        st.dataframe(coverage_rows, use_container_width=True, hide_index=True, height=400)

    action_cols = st.columns([1, 1, 2])
    action_cols[0].download_button(
        "下载全部候选",
        data=manifest_path.read_bytes(),
        file_name=manifest_path.name,
        mime="application/json",
        use_container_width=True,
    )
    if final_selection:
        action_cols[1].caption(
            f"已锁定：{final_selection.get('blueprint_candidate_id')} → "
            f"{final_selection.get('plan_candidate_id')}"
        )
    confirmed = action_cols[2].button(
        "选定这个 Plan 并继续生成",
        type="primary",
        use_container_width=True,
        key=f"confirm_candidates_{key_suffix}",
    )
    if confirmed:
        return ("continue_selected", selected_bp_id, selected_plan_id)
    return None


def render_results(output_dir: Path) -> None:
    benchmark_path = output_dir / "benchmark.json"
    qa_path = output_dir / "qa_report.json"
    checkpoint_path = output_dir / "checkpoint_sessions.json"
    plan_path = output_dir / "session_plans.json"
    blueprint_path = output_dir / "dataset_blueprint.json"
    eval_candidates_path = output_dir / "eval_candidates.json"
    eval_examples_path = output_dir / "eval_examples.json"
    runtime_config_path = output_dir / ".runtime" / "config.json"
    runtime_config = read_json(runtime_config_path) if runtime_config_path.exists() else {}
    planner_only = runtime_config.get("generation", {}).get("stop_after_planning", False)

    if planner_only and plan_path.exists():
        plans = read_json(plan_path).get("plans", [])
        blueprint = read_json_optional(blueprint_path, {})
        st.subheader("故事线计划")
        type_counts: dict[str, int] = {}
        for plan in plans:
            session_type = plan.get("session_type", "daily_life")
            type_counts[session_type] = type_counts.get(session_type, 0) + 1
        metric_cols = st.columns(4)
        metric_cols[0].metric("Plans", len(plans))
        metric_cols[1].metric("日常", type_counts.get("daily_life", 0))
        metric_cols[2].metric("事件回声", type_counts.get("core_echo", 0))
        metric_cols[3].metric("直接相关", type_counts.get("core_event", 0))
        st.download_button(
            "下载 Session Plans",
            data=plan_path.read_bytes(),
            file_name=plan_path.name,
            mime="application/json",
        )
        if blueprint:
            role_counts: dict[str, int] = {}
            for slot in blueprint.get("session_slots", []):
                role = slot.get("memory_role", "none")
                role_counts[role] = role_counts.get(role, 0) + 1
            with st.expander("查看 Dataset 全局蓝图", expanded=True):
                st.write(f"**Blueprint ID**：`{blueprint.get('blueprint_id', '')}`")
                st.write(f"**记忆角色分布**：{role_counts}")
                for memory in blueprint.get("emotion_memory_map", []):
                    st.write(f"**{memory.get('memory_id', '')}** · {memory.get('event_summary', '')}")
                    st.write(
                        "线索：" + "；".join(
                            f"{cue.get('cue_type')} / {cue.get('canonical_form')}"
                            for cue in memory.get("cue_seeds", [])
                        )
                    )
        for plan in plans:
            label = (
                f"{plan.get('session_id', '')} · {plan.get('date', '')} · "
                f"{plan.get('topic', '')} · {plan.get('session_type', 'daily_life')}"
            )
            with st.expander(label):
                st.write(f"**场景**：{plan.get('scene', '')}")
                st.write(f"**聊天动机**：{plan.get('user_intent', '')}")
                st.write(f"**故事节拍**：{plan.get('story_beat', '')}")
                st.write(f"**叙事作用**：{plan.get('outline_function', '')}")
                st.write(
                    f"**记忆角色**：{plan.get('memory_role', 'none')} · "
                    f"{plan.get('memory_id', 'none')}/{plan.get('cue_id', 'none')}"
                )
                if plan.get("evidence_goal"):
                    st.write(f"**证据目标**：{plan['evidence_goal']}")
                st.write(f"**后续钩子**：{plan.get('continuity_hook') or '无'}")
                st.caption(f"rounds={plan.get('round_count', '')}")
        return

    if not benchmark_path.exists():
        if checkpoint_path.exists():
            checkpoint = read_json(checkpoint_path)
            st.info(f"该运行已有 {len(checkpoint)} 个 Session 检查点，可启用断点续跑。")
        return

    benchmark = read_json(benchmark_path)
    eval_payload = read_json_optional(eval_candidates_path, {})
    eval_results = eval_payload.get("results", []) if isinstance(eval_payload, dict) else []
    eval_candidate_count = sum(len(result.get("candidates", [])) for result in eval_results)
    eval_example_payload = read_json_optional(eval_examples_path, {})
    eval_examples = (
        eval_example_payload.get("eval_examples", [])
        if isinstance(eval_example_payload, dict) and eval_example_payload
        else benchmark.get("eval_samples", [])
    )
    qa = read_json(qa_path) if qa_path.exists() else None
    validation_config = runtime_config.get("validation")
    qa_enabled = validation_config.get("qa", False) if validation_config is not None else None
    dialogues = benchmark.get("dialogues", [])
    turn_count = sum(len(item.get("turns", [])) for item in dialogues)

    st.subheader("生成结果")
    metric_cols = st.columns(5)
    metric_cols[0].metric("Sessions", len(dialogues))
    metric_cols[1].metric("Turns", turn_count)
    metric_cols[2].metric("Eval 候选", eval_candidate_count)
    metric_cols[3].metric("Eval Examples", len(eval_examples))
    if qa:
        qa_status = "通过" if qa.get("passed") else "未通过"
    elif qa_enabled is False:
        qa_status = "已关闭"
    else:
        qa_status = "待检查"
    metric_cols[4].metric("QA", qa_status)

    download_cols = st.columns(4)
    download_cols[0].download_button(
        "下载 Benchmark",
        data=benchmark_path.read_bytes(),
        file_name=benchmark_path.name,
        mime="application/json",
        use_container_width=True,
    )
    if qa_path.exists():
        download_cols[1].download_button(
            "下载 QA 报告",
            data=qa_path.read_bytes(),
            file_name=qa_path.name,
            mime="application/json",
            use_container_width=True,
        )
    if eval_candidates_path.exists():
        download_cols[2].download_button(
            "下载 Eval 候选",
            data=eval_candidates_path.read_bytes(),
            file_name=eval_candidates_path.name,
            mime="application/json",
            use_container_width=True,
        )
    if eval_examples_path.exists():
        download_cols[3].download_button(
            "下载 Eval Examples",
            data=eval_examples_path.read_bytes(),
            file_name=eval_examples_path.name,
            mime="application/json",
            use_container_width=True,
        )

    result_tab, eval_tab, examples_tab, qa_tab, raw_tab = st.tabs([
        "对话预览", "Eval 候选", "Eval Examples", "QA 明细", "原始 JSON"
    ])
    with result_tab:
        for session in dialogues:
            label = f"{session['session_id']} · {session['date']} · {session['topic']}"
            with st.expander(label, expanded=session == dialogues[0]):
                for turn in session.get("turns", []):
                    with st.chat_message(turn["speaker"]):
                        st.caption(f"{turn['turn_id']} · {turn['round_id']}")
                        st.write(turn["text"])
    with eval_tab:
        if not eval_results:
            st.info("本次没有运行 EvalGenerator，或历史 Session 尚未完整生成。")
        for result in eval_results:
            with st.expander(result.get("outline_id", "Eval")):
                for index, candidate in enumerate(result.get("candidates", []), 1):
                    st.write(f"**候选 {index}**：{candidate.get('current_input', {}).get('text', '')}")
                    st.caption(
                        f"{candidate.get('target_label', '')} · "
                        f"{candidate.get('target_emotion', '')} · "
                        f"cue={candidate.get('blueprint_cue_id', 'none')}"
                    )
    with examples_tab:
        render_eval_examples(eval_examples, key_prefix="result")
    with qa_tab:
        if not qa:
            if qa_enabled is False:
                st.info("本次运行已关闭最终 QA；可在左侧“验证阶段”中重新开启。")
            else:
                st.warning("尚未生成 QA 报告。")
        else:
            for name, passed in qa.get("checks", {}).items():
                st.write(("✅" if passed else "❌") + f"  {name}")
            if qa.get("errors"):
                st.error("\n".join(qa["errors"]))
            st.caption(f"记录的模型调用：{len(qa.get('call_records', []))} 次")
    with raw_tab:
        st.json(benchmark, expanded=False)


def render_artifact_browser(output_dir: Path) -> None:
    """交互式产物浏览器：扫描 logs 目录，让用户选择查看任意阶段的产物。"""
    logs_dir = output_dir / "logs"
    if not logs_dir.exists():
        return
    artifacts: list[dict] = []
    seen: set[str] = set()
    for path in sorted(logs_dir.rglob("*.json")):
        rel = path.relative_to(logs_dir).as_posix()
        if rel in seen:
            continue
        seen.add(rel)
        group, label, icon = classify_log(rel)
        artifacts.append({"path": path, "rel": rel, "group": group, "label": label, "icon": icon})
    if not artifacts:
        return

    st.divider()
    st.subheader("产物浏览器")
    st.caption("选择任意阶段查看其完整产物（Writer 对话、Planner 计划、Verifier 结论等）。")

    options = [f"{a['icon']} {a['label']}  ·  {a['rel']}" for a in artifacts]
    selected = st.selectbox("选择阶段", options=options, index=len(options) - 1)
    idx = options.index(selected)
    render_artifact(artifacts[idx]["path"])


def read_json_optional(path: Path, default):
    """Read an optional JSON artifact without breaking the whole data browser."""
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def discover_generation_runs(output_root: Path) -> list[dict]:
    """Return all recognizable generation directories, newest first."""
    if not output_root.exists() or not output_root.is_dir():
        return []
    candidates = [output_root] if any((output_root / name).exists() for name in (
        "case_spec.json", "dataset_blueprint.json", "session_plans.json", "benchmark.json"
    )) else []
    candidates.extend(path for path in output_root.iterdir() if path.is_dir())
    runs: list[dict] = []
    for path in candidates:
        recognizable = any((path / name).exists() for name in (
            "case_spec.json", "dataset_blueprint.json", "session_plans.json",
            "checkpoint_sessions.json", "benchmark.json",
        ))
        if not recognizable:
            continue
        plans_payload = read_json_optional(path / "session_plans.json", {})
        plans = plans_payload.get("plans", []) if isinstance(plans_payload, dict) else []
        checkpoint = read_json_optional(path / "checkpoint_sessions.json", [])
        benchmark = read_json_optional(path / "benchmark.json", {})
        dialogues = benchmark.get("dialogues", []) if isinstance(benchmark, dict) else []
        completed = len(checkpoint) if isinstance(checkpoint, list) else len(dialogues)
        if (path / "benchmark.json").exists():
            status = "complete"
        elif completed:
            status = "partial"
        elif plans:
            status = "plans"
        elif (path / "dataset_blueprint.json").exists():
            status = "blueprint"
        else:
            status = "empty"
        mtimes = [item.stat().st_mtime for item in path.iterdir() if item.is_file()]
        runs.append({
            "path": path,
            "name": path.name,
            "modified": max(mtimes, default=path.stat().st_mtime),
            "status": status,
            "planned": len(plans),
            "completed": completed,
        })
    return sorted(runs, key=lambda item: item["modified"], reverse=True)


def load_generation_run(output_dir: Path) -> dict:
    """Load the canonical artifacts used by the evaluation page."""
    plans_payload = read_json_optional(output_dir / "session_plans.json", {})
    blueprint = read_json_optional(output_dir / "dataset_blueprint.json", {})
    eval_candidates = read_json_optional(output_dir / "eval_candidates.json", {})
    eval_examples_payload = read_json_optional(output_dir / "eval_examples.json", {})
    checkpoint = read_json_optional(output_dir / "checkpoint_sessions.json", [])
    benchmark = read_json_optional(output_dir / "benchmark.json", {})
    dialogues = checkpoint if isinstance(checkpoint, list) and checkpoint else benchmark.get("dialogues", [])
    return {
        "output_dir": output_dir,
        "case_spec": read_json_optional(output_dir / "case_spec.json", {}),
        "plans_payload": plans_payload if isinstance(plans_payload, dict) else {},
        "plans": plans_payload.get("plans", []) if isinstance(plans_payload, dict) else [],
        "blueprint": blueprint if isinstance(blueprint, dict) else {},
        "life_anchor": (
            blueprint.get("life_anchor") if isinstance(blueprint, dict) and blueprint
            else plans_payload.get("life_anchor") if isinstance(plans_payload, dict) else None
        ),
        "dialogues": dialogues if isinstance(dialogues, list) else [],
        "benchmark": benchmark if isinstance(benchmark, dict) else {},
        "eval_candidates": eval_candidates if isinstance(eval_candidates, dict) else {},
        "eval_examples": (
            eval_examples_payload.get("eval_examples", [])
            if isinstance(eval_examples_payload, dict) and eval_examples_payload
            else benchmark.get("eval_samples", []) if isinstance(benchmark, dict) else []
        ),
        "qa": read_json_optional(output_dir / "qa_report.json", None),
        "runtime_config": read_json_optional(output_dir / ".runtime" / "config.json", {}),
        "pipeline_result": read_json_optional(output_dir / "logs" / "99_pipeline_result.json", {}),
    }


def calculate_dialogue_metrics(dialogues: list[dict]) -> dict:
    turns = [turn for session in dialogues for turn in session.get("turns", [])]
    user_turns = [turn for turn in turns if turn.get("speaker") == "user"]
    assistant_turns = [turn for turn in turns if turn.get("speaker") == "assistant"]

    def average_length(items: list[dict]) -> float:
        return round(sum(len(str(item.get("text", ""))) for item in items) / len(items), 1) if items else 0.0

    question_count = sum(bool(re.search(r"[？?]", str(turn.get("text", "")))) for turn in assistant_turns)
    list_like_count = sum(bool(re.search(
        r"(?:^|\n)\s*(?:[-•]|\d+[）.)、])|首先|其次|最后|三段式|按.+规则",
        str(turn.get("text", "")),
    )) for turn in assistant_turns)
    compliance_count = sum(bool(re.search(
        r"按你说的|照你说的|我懂了.{0,12}我就|那我就按",
        str(turn.get("text", "")),
    )) for turn in user_turns)
    summaries = [str(session.get("summary", "")).strip() for session in dialogues]
    return {
        "sessions": len(dialogues),
        "turns": len(turns),
        "rounds": len(user_turns),
        "avg_rounds": round(len(user_turns) / len(dialogues), 1) if dialogues else 0.0,
        "user_avg_chars": average_length(user_turns),
        "assistant_avg_chars": average_length(assistant_turns),
        "assistant_question_rate": round(question_count / len(assistant_turns) * 100, 1) if assistant_turns else 0.0,
        "list_like_assistant_turns": list_like_count,
        "compliance_user_turns": compliance_count,
        "missing_summaries": sum(not summary for summary in summaries),
    }


def render_life_anchor(anchor: dict | None) -> None:
    if not anchor:
        st.info("该运行没有 life_anchor；旧版本 Plan 或仅有部分产物时可能出现这种情况。")
        return
    cols = st.columns([1, 1, 1])
    cols[0].write(f"**生活状态**\n\n{anchor.get('identity', '—')}")
    cols[1].write("**持续支线**")
    cols[1].write("\n".join(f"- {item}" for item in anchor.get("ongoing_threads", [])) or "—")
    cols[2].write("**兴趣与常见场景**")
    details = list(anchor.get("interests", [])) + list(anchor.get("recurring_scenes", []))
    cols[2].write("\n".join(f"- {item}" for item in details) or "—")


def render_dataset_blueprint(blueprint: dict) -> None:
    if not blueprint:
        st.info("该运行没有 dataset_blueprint.json；旧版本数据仍可在其他页签查看。")
        return
    slots = blueprint.get("session_slots", [])
    eval_outlines = blueprint.get("eval_outlines", [])
    memories = blueprint.get("emotion_memory_map", [])
    role_counts: dict[str, int] = {}
    for slot in slots:
        role = slot.get("memory_role", "none")
        role_counts[role] = role_counts.get(role, 0) + 1
    cols = st.columns(4)
    cols[0].metric("Memories", len(memories))
    cols[1].metric("Cue seeds", sum(len(item.get("cue_seeds", [])) for item in memories))
    cols[2].metric("Session slots", len(slots))
    cols[3].metric("Eval outlines", len(eval_outlines))
    st.caption(f"Blueprint ID：`{blueprint.get('blueprint_id', '')}` · 记忆角色分布：{role_counts}")

    memory_tab, slot_tab, eval_tab = st.tabs(["情绪记忆与线索", "Session 槽位", "Eval 覆盖"])
    with memory_tab:
        for memory in memories:
            legacy_emotion = memory.get("historical_emotion", {})
            emotion = memory.get("emotion", legacy_emotion.get("emotion", "—"))
            emotional_meaning = memory.get(
                "emotional_meaning", legacy_emotion.get("meaning", "—")
            )
            with st.expander(
                f"{memory.get('memory_id', '')} · {memory.get('event_summary', '')}",
                expanded=True,
            ):
                st.write(
                    f"**记忆情绪**：{emotion} · {emotional_meaning}"
                )
                for cue in memory.get("cue_seeds", []):
                    st.markdown(
                        f"**`{cue.get('cue_id', '')}` · {cue.get('cue_type', '')} · "
                        f"{cue.get('canonical_form', '')}**  \n"
                        f"个人意义：{cue.get('personal_meaning', '—')}  \n"
                        f"自然变体：{'；'.join(cue.get('related_forms', [])) or '—'}"
                    )
    with slot_tab:
        table = [
            "| Session | 记忆角色 | Memory / Cue | 情绪变化 | 依赖 |",
            "|---|---|---|---|---|",
        ]
        for slot in slots:
            row = [
                slot.get("session_id", "—"), slot.get("memory_role", "none"),
                f"{slot.get('memory_id', 'none')} / {slot.get('cue_id', 'none')}",
                f"{slot.get('target_emotion', 'neutral')} · "
                f"{slot.get('relative_to_past', 'not_applicable')}",
                ", ".join(slot.get("depends_on_sessions", [])) or "—",
            ]
            table.append("| " + " | ".join(str(cell).replace("|", "\\|") for cell in row) + " |")
        st.markdown("\n".join(table))
        if slots:
            slot_id = st.selectbox(
                "查看槽位目标", [slot.get("session_id", "") for slot in slots],
                key="library_blueprint_slot",
            )
            slot = next(item for item in slots if item.get("session_id") == slot_id)
            st.write(f"**证据目标**：{slot.get('evidence_goal') or '普通日常，不承担记忆证据'}")
            st.write(f"**依赖 Session**：{', '.join(slot.get('depends_on_sessions', [])) or '—'}")
    with eval_tab:
        for outline in eval_outlines:
            with st.expander(
                f"{outline.get('outline_id', '')} · {outline.get('target_label', '')} · "
                f"cutoff {outline.get('history_cutoff', '')}"
            ):
                st.write(f"**当前输入目标**：{outline.get('current_input_goal', '—')}")
                st.write(
                    f"**Cue**：{outline.get('memory_id', 'none')}/{outline.get('cue_id', 'none')} · "
                    f"{outline.get('cue_specificity', '—')} · {outline.get('emotion_explicitness', '—')}"
                )
                st.write(
                    f"**预期情绪**：{outline.get('target_emotion', 'neutral')} · "
                    f"**证据 Session**：{', '.join(outline.get('required_evidence_session_ids', [])) or '—'}"
                )
                if outline.get("negative_reason"):
                    st.write(f"**负例原因**：{outline['negative_reason']}")


def render_data_library(output_root: Path) -> None:
    """Independent page for reviewing every historical generation run."""
    st.markdown(
        """
        <div class="studio-hero">
          <h1>Generated Data Library</h1>
          <p>集中查看历史运行、故事线、完整对话、质量指标与原始产物</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.sidebar:
        st.divider()
        st.subheader("数据目录")
        st.text_input("输出根目录", key="output_root")
        st.button("刷新运行列表", use_container_width=True)
        st.caption(f"当前扫描：{output_root}")

    runs = discover_generation_runs(output_root)
    if not runs:
        st.warning("当前目录下没有找到生成记录。")
        return

    status_labels = {
        "complete": "完整", "partial": "部分完成", "plans": "仅 Plan",
        "blueprint": "仅蓝图", "empty": "初始化",
    }
    filter_cols = st.columns([1.4, 1, 1])
    query = filter_cols[0].text_input("搜索运行", placeholder="输入测试名称")
    selected_statuses = filter_cols[1].multiselect(
        "运行状态", options=list(status_labels), default=list(status_labels),
        format_func=lambda value: status_labels[value],
    )
    filter_cols[2].metric("历史运行", len(runs))
    visible_runs = [
        run for run in runs
        if run["status"] in selected_statuses and query.lower() in run["name"].lower()
    ]
    if not visible_runs:
        st.info("没有符合筛选条件的运行。")
        return

    by_path = {str(run["path"]): run for run in visible_runs}
    selected_path = st.selectbox(
        "选择生成记录",
        options=list(by_path),
        format_func=lambda value: (
            f"{by_path[value]['name']}  ·  {status_labels[by_path[value]['status']]}  ·  "
            f"{by_path[value]['completed']}/{by_path[value]['planned']} sessions  ·  "
            f"{datetime.fromtimestamp(by_path[value]['modified']):%Y-%m-%d %H:%M}"
        ),
    )
    selected_run = by_path[selected_path]
    bundle = load_generation_run(Path(selected_path))
    plans = bundle["plans"]
    dialogues = bundle["dialogues"]
    eval_results = bundle["eval_candidates"].get("results", [])
    eval_candidate_count = sum(len(item.get("candidates", [])) for item in eval_results)
    eval_examples = bundle["eval_examples"]
    metrics = calculate_dialogue_metrics(dialogues)

    st.caption(f"`{selected_path}`")
    metric_cols = st.columns(7)
    metric_cols[0].metric("状态", status_labels[selected_run["status"]])
    metric_cols[1].metric("Plans", len(plans))
    metric_cols[2].metric("Sessions", metrics["sessions"])
    metric_cols[3].metric("Rounds", metrics["rounds"])
    metric_cols[4].metric("QA", "通过" if bundle["qa"] and bundle["qa"].get("passed") else "无/未通过")
    metric_cols[5].metric("Eval 候选", eval_candidate_count)
    metric_cols[6].metric("Eval Examples", len(eval_examples))

    overview_tab, blueprint_tab, plan_tab, dialogue_tab, eval_tab, examples_tab, quality_tab, raw_tab = st.tabs([
        "运行概览", "Dataset 蓝图", "故事线 Plans", "Session 对话",
        "Eval 候选", "Eval Examples", "快速评估", "原始产物",
    ])
    with overview_tab:
        st.subheader("隐式生活锚点")
        render_life_anchor(bundle["life_anchor"])
        st.divider()
        spec_col, config_col = st.columns(2)
        with spec_col:
            st.subheader("CaseSpec")
            st.json(bundle["case_spec"], expanded=False)
        with config_col:
            st.subheader("本次运行配置")
            if bundle["runtime_config"]:
                generation = bundle["runtime_config"].get("generation", {})
                validation = bundle["runtime_config"].get("validation", {})
                st.write(f"**模型**：{bundle['runtime_config'].get('components', {}).get('session_writer', {}).get('model', '—')}")
                st.write(f"**Session 数量**：{generation.get('session_count', '—')}")
                st.write(f"**仅 Plan**：{generation.get('stop_after_planning', False)}")
                st.write(f"**EvalGenerator**：{generation.get('run_eval', False)}")
                st.write(
                    f"**正式 Eval Examples**：{generation.get('run_eval_examples', False)}"
                )
                st.write(
                    f"**Blueprint 约束**：{bundle['runtime_config'].get('blueprint_constraints', {})}"
                )
                st.write(f"**验证开关**：{validation}")
            else:
                st.info("该运行没有保存 runtime config。")

    with blueprint_tab:
        render_dataset_blueprint(bundle["blueprint"])

    with plan_tab:
        if not plans:
            st.info("该运行没有 Session Plans。")
        else:
            types = sorted({plan.get("session_type", "daily_life") for plan in plans})
            threads = sorted({plan.get("life_thread", "one_off") for plan in plans})
            roles = sorted({plan.get("memory_role", "none") for plan in plans})
            plan_filters = st.columns(3)
            chosen_types = plan_filters[0].multiselect("Session 类型", types, default=types)
            chosen_threads = plan_filters[1].multiselect("生活支线", threads, default=threads)
            chosen_roles = plan_filters[2].multiselect("记忆角色", roles, default=roles)
            filtered_plans = [
                plan for plan in plans
                if plan.get("session_type", "daily_life") in chosen_types
                and plan.get("life_thread", "one_off") in chosen_threads
                and plan.get("memory_role", "none") in chosen_roles
            ]
            table_lines = [
                "| ID | 日期 | 类型 | 记忆角色 | Cue | 互动 | 生活支线 | 主题 |",
                "|---|---|---|---|---|---|---|---|",
            ]
            for plan in filtered_plans:
                cells = [
                    plan.get("session_id"), plan.get("date"), plan.get("session_type"),
                    plan.get("memory_role", "none"), plan.get("cue_id", "none"),
                    plan.get("interaction_mode", "—"), plan.get("life_thread", "one_off"),
                    plan.get("topic"),
                ]
                escaped = [str(value or "—").replace("|", "\\|").replace("\n", " ") for value in cells]
                table_lines.append("| " + " | ".join(escaped) + " |")
            st.markdown("\n".join(table_lines))
            if filtered_plans:
                selected_plan_id = st.selectbox(
                    "查看 Plan 详情", [plan.get("session_id", "") for plan in filtered_plans],
                    key="library_plan_detail",
                )
                plan = next(item for item in filtered_plans if item.get("session_id") == selected_plan_id)
                st.write(f"**场景**：{plan.get('scene', '—')}")
                st.write(f"**聊天动机**：{plan.get('user_intent', '—')}")
                st.write(f"**故事节拍**：{plan.get('story_beat', '—')}")
                st.write(
                    f"**记忆槽位**：{plan.get('memory_role', 'none')} · "
                    f"{plan.get('memory_id', 'none')}/{plan.get('cue_id', 'none')}"
                )
                st.write(f"**证据目标**：{plan.get('evidence_goal') or '—'}")
                st.write(
                    f"**局部情绪**：{plan.get('target_emotion', 'neutral')} · "
                    f"{plan.get('relative_to_past', 'not_applicable')}"
                )
                st.write(f"**支线进展**：{plan.get('thread_progress') or '—'}")
                st.write(f"**后续钩子**：{plan.get('continuity_hook') or '—'}")

    with dialogue_tab:
        if not dialogues:
            st.info("该运行尚未生成 Session 对话。")
        else:
            plan_by_id = {plan.get("session_id"): plan for plan in plans}
            session_ids = [session.get("session_id", f"Session {index + 1}") for index, session in enumerate(dialogues)]
            selected_session_id = st.selectbox("选择 Session", session_ids, key="library_session_detail")
            session = next(item for item in dialogues if item.get("session_id") == selected_session_id)
            session_plan = plan_by_id.get(selected_session_id)
            header_cols = st.columns([3, 1])
            header_cols[0].subheader(f"{selected_session_id} · {session.get('topic', '')}")
            header_cols[1].download_button(
                "下载当前 Session",
                data=json.dumps(session, ensure_ascii=False, indent=2),
                file_name=f"{selected_session_id}.json", mime="application/json",
                use_container_width=True,
            )
            if session.get("summary"):
                st.info(f"摘要：{session['summary']}")
            if session_plan:
                st.caption(
                    f"{session_plan.get('session_type', 'daily_life')} · "
                    f"{session_plan.get('interaction_mode', '—')} · "
                    f"支线：{session_plan.get('life_thread', 'one_off')}"
                )
            for turn in session.get("turns", []):
                with st.chat_message(turn.get("speaker", "user")):
                    st.caption(f"{turn.get('turn_id', '')} · {turn.get('round_id', '')}")
                    st.write(turn.get("text", ""))

    with eval_tab:
        if not eval_results:
            st.info("该运行没有 EvalGenerator 候选。")
        else:
            outline_ids = [item.get("outline_id", "") for item in eval_results]
            selected_outline_id = st.selectbox(
                "选择 Eval outline", outline_ids, key="library_eval_outline"
            )
            result = next(
                item for item in eval_results if item.get("outline_id") == selected_outline_id
            )
            st.caption("Generator-only：尚未经过 Resolver、Verifier 或 Finalizer。")
            for index, candidate in enumerate(result.get("candidates", []), 1):
                with st.expander(f"候选 {index}", expanded=index == 1):
                    current = candidate.get("current_input", {})
                    st.write(current.get("text", ""))
                    st.caption(
                        f"label={candidate.get('target_label', '')} · "
                        f"emotion={candidate.get('target_emotion', '')} · "
                        f"blueprint cue={candidate.get('blueprint_cue_id', 'none')} · "
                        f"cutoff={candidate.get('history_cutoff', '')}"
                    )
                    st.write(f"**Cue 类型**：{current.get('cue_type', 'none')}")
                    st.json(current.get("cue_options", []), expanded=False)

    with examples_tab:
        render_eval_examples(eval_examples, key_prefix="library")

    with quality_tab:
        quality_cols = st.columns(4)
        quality_cols[0].metric("平均轮数", metrics["avg_rounds"])
        quality_cols[1].metric("User 平均字数", metrics["user_avg_chars"])
        quality_cols[2].metric("朋友平均字数", metrics["assistant_avg_chars"])
        quality_cols[3].metric("朋友提问率", f"{metrics['assistant_question_rate']}%")
        signal_cols = st.columns(3)
        signal_cols[0].metric("列表/流程式回复", metrics["list_like_assistant_turns"])
        signal_cols[1].metric("复述式配合回复", metrics["compliance_user_turns"])
        signal_cols[2].metric("缺少摘要", metrics["missing_summaries"])
        st.caption("这些是便于人工定位的启发式信号，不代表最终质量判定。")
        if plans:
            type_counts: dict[str, int] = {}
            mode_counts: dict[str, int] = {}
            thread_counts: dict[str, int] = {}
            role_counts: dict[str, int] = {}
            for plan in plans:
                for target, key, default in (
                    (type_counts, "session_type", "daily_life"),
                    (mode_counts, "interaction_mode", "—"),
                    (thread_counts, "life_thread", "one_off"),
                    (role_counts, "memory_role", "none"),
                ):
                    value = plan.get(key, default)
                    target[value] = target.get(value, 0) + 1
            distribution_cols = st.columns(4)
            distribution_cols[0].write("**Session 类型**")
            distribution_cols[0].json(type_counts)
            distribution_cols[1].write("**互动模式**")
            distribution_cols[1].json(mode_counts)
            distribution_cols[2].write("**生活支线**")
            distribution_cols[2].json(thread_counts)
            distribution_cols[3].write("**记忆角色**")
            distribution_cols[3].json(role_counts)

    with raw_tab:
        files = sorted(
            path for path in Path(selected_path).rglob("*")
            if path.is_file() and path.suffix.lower() in {".json", ".txt", ".md"}
        )
        if not files:
            st.info("没有可预览的文本产物。")
        else:
            relative_files = [path.relative_to(selected_path).as_posix() for path in files]
            chosen_file = st.selectbox("选择文件", relative_files, key="library_raw_file")
            file_path = Path(selected_path) / chosen_file
            file_bytes = file_path.read_bytes()
            st.download_button(
                "下载所选文件", data=file_bytes, file_name=file_path.name,
                mime="application/json" if file_path.suffix.lower() == ".json" else "text/plain",
            )
            if file_path.suffix.lower() == ".json":
                payload = read_json_optional(file_path, None)
                if payload is not None:
                    st.json(payload, expanded=False)
                else:
                    st.error("JSON 文件无法解析。")
            else:
                st.code(file_path.read_text(encoding="utf-8", errors="replace"), language="text")


def resolve_output_root(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def build_runtime_snapshot(
    output_dir: Path,
    model_name: str,
    writer_temperature: float,
    transport: str,
    prompt_drafts: dict[str, str],
    session_count: int,
    blueprint_constraints: dict,
    validation_config: dict[str, bool],
    stop_after_planning: bool,
    run_eval: bool,
    run_eval_examples: bool,
) -> Path:
    """Create an isolated config/prompt snapshot without touching project defaults."""
    runtime_dir = output_dir / ".runtime"
    runtime_prompt_dir = runtime_dir / "prompts"
    runtime_script_dir = runtime_dir / "scripts"
    runtime_prompt_dir.mkdir(parents=True, exist_ok=True)
    runtime_script_dir.mkdir(parents=True, exist_ok=True)

    runtime_config = read_json(CONFIG_PATH)
    runtime_config["transport"] = transport
    runtime_config["generation"]["session_count"] = session_count
    runtime_config["generation"].pop("eval_count", None)
    runtime_config["generation"]["stop_after_planning"] = stop_after_planning
    runtime_config["generation"]["run_eval"] = run_eval
    runtime_config["generation"]["run_eval_examples"] = run_eval_examples
    runtime_config["blueprint_constraints"] = blueprint_constraints
    runtime_config["validation"] = validation_config
    for component, component_config in runtime_config["components"].items():
        component_config["model"] = model_name.strip()
        if component in {"session_writer", "eval_generator"}:
            component_config["temperature"] = writer_temperature
    runtime_config_path = runtime_dir / "config.json"
    runtime_config_path.write_text(
        json.dumps(runtime_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for prompt_name, content in prompt_drafts.items():
        (runtime_prompt_dir / f"{prompt_name}.txt").write_text(content, encoding="utf-8")
    shutil.copy2(ROOT / "scripts" / "invoke_openai.ps1", runtime_script_dir / "invoke_openai.ps1")
    return runtime_config_path


def run_generation(
    spec_text: str,
    run_name: str,
    resume: bool,
    output_root_text: str,
    model_name: str,
    writer_temperature: float,
    transport: str,
    api_key_override: str,
    base_url_override: str,
    prompt_drafts: dict[str, str],
    session_count_override: int,
    blueprint_constraints: dict,
    validation_config: dict[str, bool],
    stop_after_planning: bool,
    run_eval: bool,
    run_eval_examples: bool,
    workflow_action: str = "prepare_blueprints",
    blueprint_candidate_count: int = 3,
    plan_candidate_count: int = 3,
    selected_blueprint_id: str | None = None,
    selected_plan_id: str | None = None,
    existing_output_dir: Path | None = None,
) -> None:
    spec, error = validate_spec(spec_text)
    if error or spec is None:
        st.error("Spec 校验失败，请先修正 JSON。")
        return

    safe_name = safe_run_name(run_name)
    if not safe_name:
        st.error("测试名称至少需要包含一个字母、数字、短横线或下划线。")
        return

    if not model_name.strip():
        st.error("模型名称不能为空。")
        return
    effective_session_count = (
        min(len(spec.session_outlines), session_count_override)
        if spec.session_outlines else session_count_override
    )
    constraints_model = BlueprintConstraints(
        **{
            **blueprint_constraints,
            "required_cue_types": tuple(blueprint_constraints["required_cue_types"]),
        }
    )
    constraint_errors = validate_blueprint_constraints(
        effective_session_count, constraints_model
    )
    if constraint_errors:
        st.error("Blueprint 约束无效：" + "；".join(constraint_errors))
        return
    try:
        output_dir = existing_output_dir or (
            resolve_output_root(output_root_text) / safe_name
        )
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        st.error(f"无法使用该保存路径：{exc}")
        return
    case_path = output_dir / "case_spec.json"
    if workflow_action == "prepare_blueprints" and not resume and any(
        (output_dir / name).exists() for name in (
            "planning_candidates.json", "dataset_blueprint.json", "session_plans.json"
        )
    ):
        st.error("该测试名称已有蓝图或 Plan。为避免误用旧结果，请换一个新的测试名称，或开启断点续跑。")
        return
    if resume and case_path.exists():
        previous, previous_error = validate_spec(case_path.read_text(encoding="utf-8"))
        if previous_error or previous is None or previous.model_dump() != spec.model_dump():
            st.error("同名运行的 Spec 与当前内容不同，不能安全续跑。请换一个新的测试名称。")
            return

    if workflow_action in {"prepare_plans", "continue_selected"}:
        runtime_config_path = output_dir / ".runtime" / "config.json"
        if not case_path.exists() or not runtime_config_path.exists():
            st.error("找不到候选生成时保存的 CaseSpec 或配置快照，无法安全继续。")
            return
        if not selected_blueprint_id:
            st.error("请先选择一个 Blueprint。")
            return
        if workflow_action == "continue_selected" and not selected_plan_id:
            st.error("请先选择一个 Blueprint 和一个 Plan。")
            return
    else:
        case_path.write_text(spec.model_dump_json(indent=2), encoding="utf-8")
        try:
            runtime_config_path = build_runtime_snapshot(
                output_dir, model_name, writer_temperature, transport, prompt_drafts,
                session_count_override, blueprint_constraints, validation_config,
                stop_after_planning, run_eval, run_eval_examples,
            )
        except OSError as exc:
            st.error(f"无法创建本次运行的配置快照：{exc}")
            return
    st.session_state["last_output"] = str(output_dir)

    # total 优先用滑块值；若 case 自带 outlines 则以 outlines 数量为准
    if spec.session_outlines:
        total = min(len(spec.session_outlines), session_count_override)
    else:
        total = session_count_override

    command = [
        sys.executable,
        "-u",
        str(ROOT / "run.py"),
        "--config",
        str(runtime_config_path),
        "--case",
        str(case_path),
        "--output",
        str(output_dir),
    ]
    if workflow_action == "prepare_blueprints":
        command.extend([
            "--prepare-blueprints", "--blueprint-count", str(blueprint_candidate_count)
        ])
    elif workflow_action == "prepare_plans":
        command.extend([
            "--prepare-plans",
            "--select-blueprint", selected_blueprint_id,
            "--plan-count", str(plan_candidate_count),
        ])
    elif workflow_action == "continue_selected":
        command.extend([
            "--select-blueprint", selected_blueprint_id,
            "--select-plan", selected_plan_id,
        ])
    child_env = os.environ.copy()
    child_env["PYTHONUTF8"] = "1"
    if api_key_override.strip():
        child_env["openai_api_key"] = api_key_override.strip()
        child_env["OPENAI_API_KEY"] = api_key_override.strip()
    if base_url_override.strip():
        child_env["base_url"] = base_url_override.strip()
        child_env["BASE_URL"] = base_url_override.strip()
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=child_env,
    )

    # 布局：进度条 + 状态 + 实时阶段面板 + 日志
    progress = st.progress(0.01, text="准备生成环境")
    status = st.empty()
    st.markdown("##### Pipeline 实时阶段")
    stage_placeholder = st.empty()
    log_box = st.empty()

    logs: list[str] = []
    completed = 0
    logs_dir = output_dir / "logs"
    seen_stages: set[str] = set()
    all_stages: list[dict] = []

    # 用后台线程非阻塞读取 stdout，主循环同时轮询日志目录
    stdout_queue: queue.Queue[str | None] = queue.Queue()

    def _reader(proc: subprocess.Popen, q: queue.Queue) -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            q.put(line)
        q.put(None)

    threading.Thread(target=_reader, args=(process, stdout_queue), daemon=True).start()

    while True:
        # 轮询日志目录，发现新产物立即更新阶段面板
        new_stages = scan_pipeline_logs(logs_dir, seen_stages)
        if new_stages:
            all_stages.extend(new_stages)
            stage_placeholder.markdown(render_live_stages(all_stages))

        try:
            raw_line = stdout_queue.get(timeout=0.5)
        except queue.Empty:
            if process.poll() is not None and stdout_queue.empty():
                break
            continue

        if raw_line is None:
            # 进程结束的哨兵，再轮询一次确保最后的日志文件被捕获
            final_stages = scan_pipeline_logs(logs_dir, seen_stages)
            if final_stages:
                all_stages.extend(final_stages)
                stage_placeholder.markdown(render_live_stages(all_stages))
            break

        line = raw_line.rstrip()
        if not line:
            continue
        logs.append(line)
        if line.startswith("retrying "):
            status.warning(line)
            log_box.code("\n".join(logs[-18:]), language="text")
            continue
        if line.startswith("completed ") and not line.startswith(
            ("completed eval ", "completed Eval example ")
        ):
            completed += 1
        value, label = update_progress(line, total, completed)
        progress.progress(min(value, 0.99), text=label)
        status.caption(line)
        log_box.code("\n".join(logs[-18:]), language="text")

    return_code = process.wait()
    if return_code == 0:
        progress.progress(1.0, text="生成完成")
        # 检查是否有 session 失败被跳过
        pipeline_result_path = output_dir / "logs" / "99_pipeline_result.json"
        failed_ids: list[str] = []
        eval_example_error = None
        if pipeline_result_path.exists():
            try:
                pr = json.loads(pipeline_result_path.read_text(encoding="utf-8"))
                failed_ids = pr.get("failed_session_ids", [])
                eval_example_error = pr.get("eval_example_error")
            except Exception:
                pass
        if failed_ids:
            status.warning(f"生成完成，但有 {len(failed_ids)} 个 session 失败被跳过：{', '.join(failed_ids)}")
        elif eval_example_error:
            status.warning(f"Session 与 Eval 候选已保存，但正式 Eval Examples 未完成：{eval_example_error}")
        elif workflow_action == "prepare_blueprints":
            status.success(
                f"{blueprint_candidate_count} 个 Blueprint 候选已生成，请先选择一个。"
            )
        elif workflow_action == "prepare_plans":
            status.success(
                f"已基于 {selected_blueprint_id} 生成 {plan_candidate_count} 个 Plan 候选。"
            )
        else:
            status.success("生成产物已保存。")
        if workflow_action in {"prepare_blueprints", "prepare_plans"}:
            render_candidate_selection(output_dir, plan_candidate_count)
        else:
            render_results(output_dir)
        render_artifact_browser(output_dir)
    else:
        progress.progress(min(0.99, 0.10 + 0.84 * completed / max(total, 1)), text="生成中断")
        status.error("生成任务未完成。日志已保留；修正问题后可使用同一名称断点续跑。")
        error_lines = [ln for ln in logs if ln.startswith("ERROR") or "Traceback" in ln or "Error" in ln]
        if error_lines:
            with st.expander("错误详情（最近 30 行）", expanded=True):
                st.code("\n".join(error_lines[-30:]), language="text")
        with st.expander("完整日志（最后 50 行）", expanded=False):
            st.code("\n".join(logs[-50:]), language="text")
        render_artifact_browser(output_dir)


st.set_page_config(
    page_title="Dialogue Data Studio",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        color-scheme: light !important;
        --studio-text: #172033;
        --studio-muted: #3f4d63;
        --studio-border: #d6deea;
        --studio-surface: #ffffff;
        --studio-canvas: #f4f7fb;
        --studio-primary: #2563eb;
    }
    html, body, .stApp, [data-testid="stAppViewContainer"],
    [data-testid="stMain"], .main {
        background: var(--studio-canvas) !important;
        color: var(--studio-text) !important;
    }
    [data-testid="stAppViewContainer"] {
        --text-color: var(--studio-text);
        --background-color: var(--studio-canvas);
        --secondary-background-color: var(--studio-surface);
        --primary-color: var(--studio-primary);
    }
    .main .block-container {
        max-width: 1480px;
        padding-top: 1.6rem;
        padding-bottom: 3rem;
    }
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5,
    .stApp p, .stApp label, .stApp span, .stApp small,
    .stApp [data-testid="stCaptionContainer"], .stApp [data-testid="stMetricValue"],
    .stApp [data-testid="stMetricLabel"] { color: #172033 !important; }
    .stApp [data-testid="stCaptionContainer"] { color: var(--studio-muted) !important; }
    [data-testid="stMain"] li,
    [data-testid="stMain"] strong,
    [data-testid="stMain"] em,
    [data-testid="stMain"] summary,
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] {
        color: var(--studio-text) !important;
    }
    [data-testid="stSidebar"] {
        background: #172136 !important;
        border-right: 1px solid #2c3952;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] small,
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color: #f4f7fb !important; }
    [data-testid="stSidebar"] div[data-testid="stExpander"] {
        background: #202c43 !important;
        border: 1px solid #35435e !important;
        border-radius: 10px !important;
        overflow: hidden;
    }
    [data-testid="stSidebar"] div[data-testid="stExpander"] details,
    [data-testid="stSidebar"] div[data-testid="stExpander"] details > summary {
        background: #202c43 !important;
    }
    [data-testid="stSidebar"] div[data-testid="stExpander"] details[open] > summary {
        background: #26354f !important;
        border-bottom: 1px solid #35435e !important;
    }
    [data-testid="stSidebar"] div[data-testid="stExpander"] details > summary:hover {
        background: #27354f !important;
    }
    [data-testid="stSidebar"] div[data-testid="stExpander"] details > summary p,
    [data-testid="stSidebar"] div[data-testid="stExpander"] details > summary span,
    [data-testid="stSidebar"] div[data-testid="stExpander"] details > summary svg {
        color: #f8fafc !important;
        fill: #f8fafc !important;
        -webkit-text-fill-color: #f8fafc !important;
    }
    [data-testid="stSidebar"] hr { border-color: #344159 !important; }
    .stApp input, .stApp textarea,
    .stApp [data-baseweb="select"] > div,
    .stApp [data-baseweb="base-input"] > div {
        background: #ffffff !important; color: #172033 !important;
        -webkit-text-fill-color: #172033 !important;
        border-color: #9eabc0 !important;
    }
    .stApp input::placeholder, .stApp textarea::placeholder {
        color: #68758a !important; -webkit-text-fill-color: #68758a !important; opacity: 1 !important;
    }
    .stApp [data-baseweb="select"] svg { fill: #172033 !important; }
    [data-baseweb="popover"], [data-baseweb="menu"],
    [role="listbox"], [role="option"] { background: #ffffff !important; color: #172033 !important; }
    [role="option"] span, [role="option"] div { color: #172033 !important; }
    [role="option"]:hover { background: #e8eef7 !important; }
    .stApp button {
        border-color: #8291aa !important;
        color: var(--studio-text) !important;
    }
    .stApp button[kind="secondary"],
    .stApp button[kind="tertiary"] {
        background: #ffffff !important;
    }
    .stApp button[kind="secondary"] p,
    .stApp button[kind="secondary"] span,
    .stApp button[kind="tertiary"] p,
    .stApp button[kind="tertiary"] span { color: var(--studio-text) !important; }
    .stApp button[aria-label^="Help for"],
    .stApp button[aria-label="Open"],
    .stApp button[aria-label="Show password"] {
        background: transparent !important;
        border-color: transparent !important;
        box-shadow: none !important;
    }
    .stApp button[kind="primary"] { background: #2563eb !important; border-color: #2563eb !important; }
    .stApp button[kind="primary"] p, .stApp button[kind="primary"] span { color: #ffffff !important; }
    .stApp button:disabled,
    .stApp button[kind="primary"]:disabled {
        background: #e2e8f0 !important;
        border-color: #c3cedd !important;
        opacity: 1 !important;
    }
    .stApp button:disabled p, .stApp button:disabled span {
        color: #526077 !important;
        -webkit-text-fill-color: #526077 !important;
    }
    [data-testid="stSidebar"] button[kind="secondary"],
    [data-testid="stSidebar"] button[kind="tertiary"] {
        background: #ffffff !important; border-color: #a8b4c7 !important;
    }
    [data-testid="stSidebar"] button[kind="secondary"] p,
    [data-testid="stSidebar"] button[kind="secondary"] span,
    [data-testid="stSidebar"] button[kind="tertiary"] p,
    [data-testid="stSidebar"] button[kind="tertiary"] span { color: #172033 !important; }
    [data-testid="stSidebar"] button[aria-label^="Help for"] svg {
        color: #cbd5e1 !important;
        fill: #cbd5e1 !important;
    }
    .studio-hero {
        padding: 1.35rem 1.5rem;
        border: 1px solid #d7e1ef;
        border-radius: 18px;
        background:
            radial-gradient(circle at 90% 10%, rgba(37, 99, 235, .12), transparent 32%),
            linear-gradient(120deg, #ffffff 15%, #f1f6ff 100%);
        box-shadow: 0 8px 24px rgba(30, 64, 175, .06);
        margin-bottom: 1.15rem;
    }
    .studio-eyebrow {
        color: #1d4ed8 !important;
        font-size: .72rem;
        font-weight: 750;
        letter-spacing: .11em;
        text-transform: uppercase;
        margin-bottom: .35rem;
    }
    .studio-hero h1 { font-size: 1.95rem; margin: 0 0 .3rem 0; color: #111827; }
    .studio-hero p { color: #526077 !important; margin: 0; max-width: 760px; }
    .status-pill {
        display: inline-block;
        padding: .25rem .62rem;
        margin: .15rem .35rem .05rem 0;
        border: 1px solid #c9d9f1;
        border-radius: 999px;
        background: #edf4ff;
        color: #244467 !important;
        font-size: .78rem;
        font-weight: 650;
    }
    .section-heading {
        display: flex;
        align-items: flex-start;
        gap: .75rem;
        margin: 1.4rem 0 .75rem;
    }
    .section-number {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 1.8rem;
        height: 1.8rem;
        flex: 0 0 1.8rem;
        border-radius: 8px;
        background: #dbeafe;
        color: #1d4ed8 !important;
        font-size: .78rem;
        font-weight: 800;
    }
    .section-heading strong { color: #172033 !important; font-size: 1.05rem; }
    .section-heading p { color: var(--studio-muted) !important; font-size: .82rem; margin: .08rem 0 0; }
    [data-baseweb="tab-list"] {
        gap: .35rem;
        border-bottom: 1px solid var(--studio-border);
    }
    [data-baseweb="tab"] {
        background: transparent !important;
        border-radius: 8px 8px 0 0;
    }
    [data-baseweb="tab"] p { color: #526077 !important; font-weight: 650; }
    [aria-selected="true"][data-baseweb="tab"] p { color: #1d4ed8 !important; }
    [data-testid="stTextArea"] textarea { line-height: 1.55; }
    [data-testid="stMetric"] {
        padding: .75rem .9rem;
        border-radius: 12px;
    }
    div[data-testid="stExpander"], [data-testid="stChatMessage"],
    [data-testid="stMetric"] { border-color: #d5deea !important; background: #ffffff !important; }
    [data-testid="stMain"] div[data-testid="stExpander"] {
        overflow: hidden;
        border: 1px solid #cbd5e1 !important;
    }
    [data-testid="stMain"] div[data-testid="stExpander"] details > summary {
        background: #f8fafc !important;
    }
    [data-testid="stMain"] div[data-testid="stExpander"] details[open] > summary {
        background: #eef3f8 !important;
        border-bottom: 1px solid #d7e0eb !important;
    }
    [data-testid="stMain"] div[data-testid="stExpander"] details > summary:hover {
        background: #e8eef6 !important;
    }
    [data-testid="stMain"] div[data-testid="stExpander"] details > summary p,
    [data-testid="stMain"] div[data-testid="stExpander"] details > summary span,
    [data-testid="stMain"] div[data-testid="stExpander"] details > summary svg {
        color: #172033 !important;
        fill: #172033 !important;
        -webkit-text-fill-color: #172033 !important;
    }
    [data-testid="stAlert"] { background: #ffffff !important; border: 1px solid #cbd5e1 !important; }
    [data-testid="stAlert"] p, [data-testid="stAlert"] span { color: #172033 !important; }
    [data-testid="stHeader"] {
        background: rgba(243, 246, 250, .96) !important;
        border-bottom: 1px solid #d5deea !important;
    }
    [data-testid="stToolbar"], [data-testid="stHeaderActionElements"] {
        color: #172033 !important;
    }
    [data-testid="stAppDeployButton"] button[kind="header"] {
        background: #ffffff !important;
        border: 1px solid #8291aa !important;
        box-shadow: 0 1px 2px rgba(15, 23, 42, .10) !important;
    }
    [data-testid="stAppDeployButton"] button[kind="header"] span,
    [data-testid="stAppDeployButton"] button[kind="header"] div {
        color: #172033 !important;
        -webkit-text-fill-color: #172033 !important;
    }
    [data-testid="stAppDeployButton"] button[kind="header"]:hover {
        background: #e8eef7 !important;
        border-color: #52647f !important;
    }
    [data-testid="stToolbar"] button[kind="header"] svg,
    [data-testid="stHeaderActionElements"] button svg { fill: #172033 !important; color: #172033 !important; }
    [data-testid="stMain"] [data-testid="stCodeBlock"] {
        background: #111827 !important;
        border: 1px solid #334155 !important;
        border-radius: 10px !important;
    }
    [data-testid="stMain"] [data-testid="stCodeBlock"] pre,
    [data-testid="stMain"] [data-testid="stCodeBlock"] code,
    [data-testid="stMain"] [data-testid="stCodeBlock"] span {
        background: transparent !important;
        color: #e5edf8 !important;
        -webkit-text-fill-color: #e5edf8 !important;
    }
    [data-testid="stMain"] [data-testid="stCodeBlock"] button {
        background: #253248 !important;
        border-color: #475569 !important;
    }
    [data-testid="stMain"] [data-testid="stCodeBlock"] button svg {
        color: #f8fafc !important;
        fill: #f8fafc !important;
    }
    [data-testid="stMain"] :not(pre) > code {
        padding: .08rem .32rem;
        border-radius: 5px;
        background: #e5edf7 !important;
        color: #1e3a5f !important;
        -webkit-text-fill-color: #1e3a5f !important;
    }
    [data-testid="stMain"] [data-testid="stCode"] {
        overflow: hidden;
        background: #111827 !important;
        border: 1px solid #334155 !important;
        border-radius: 10px !important;
    }
    [data-testid="stMain"] [data-testid="stCode"] pre,
    [data-testid="stMain"] [data-testid="stCode"] pre div,
    [data-testid="stMain"] [data-testid="stCode"] pre code,
    [data-testid="stMain"] [data-testid="stCode"] pre span {
        background: transparent !important;
        color: #f1f5f9 !important;
        -webkit-text-fill-color: #f1f5f9 !important;
    }
    [data-testid="stMain"] [data-testid="stCode"] button {
        background: #253248 !important;
        border-color: #475569 !important;
    }
    [data-testid="stMain"] [data-testid="stCode"] button svg {
        color: #f8fafc !important;
        fill: #f8fafc !important;
    }
    [data-testid="stMain"] [data-testid="stProgress"] p,
    [data-testid="stMain"] [data-testid="stStatusWidget"] p {
        color: #27364d !important;
    }
    @media (max-width: 900px) {
        .main .block-container { padding-left: 1rem; padding-right: 1rem; }
        .studio-hero { padding: 1.1rem; }
        .studio-hero h1 { font-size: 1.65rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

config = read_json(CONFIG_PATH)
default_spec = DEFAULT_CASE.read_text(encoding="utf-8")
defaults_by_prompt = {path.stem: path.read_text(encoding="utf-8") for path in prompt_files()}
if "spec_text" not in st.session_state:
    st.session_state["spec_text"] = default_spec
if "run_name" not in st.session_state:
    st.session_state["run_name"] = f"case-test-{datetime.now():%Y%m%d-%H%M%S}"
if "selected_model" not in st.session_state:
    st.session_state["selected_model"] = config["components"]["session_writer"]["model"]
if "model_provider" not in st.session_state:
    st.session_state["model_provider"] = provider_for_model(st.session_state["selected_model"])
if "model_catalog" not in st.session_state:
    st.session_state["model_catalog"] = [model for models in FALLBACK_MODELS.values() for model in models]
if "catalog_source" not in st.session_state:
    st.session_state["catalog_source"] = "内置候选"
if "writer_temperature" not in st.session_state:
    st.session_state["writer_temperature"] = float(config["components"]["session_writer"]["temperature"])
if "session_count_override" not in st.session_state:
    st.session_state["session_count_override"] = config["generation"]["session_count"]
if "blueprint_candidate_count" not in st.session_state:
    st.session_state["blueprint_candidate_count"] = 3
if "plan_candidate_count" not in st.session_state:
    st.session_state["plan_candidate_count"] = 3
blueprint_defaults = config.get("blueprint_constraints", {})
for state_key, config_key, fallback in (
    ("bp_encode_count", "encode_association_count", 1),
    ("bp_trigger_count", "triggered_recall_count", 1),
    ("bp_update_count", "memory_update_count", 1),
    ("bp_control_count", "control_count", 1),
    ("bp_eval_triggered", "eval_triggered_count", 2),
    ("bp_eval_insufficient", "eval_insufficient_evidence_count", 2),
    ("bp_eval_not_triggered", "eval_not_triggered_count", 2),
):
    if state_key not in st.session_state:
        st.session_state[state_key] = int(blueprint_defaults.get(config_key, fallback))
if "bp_required_cue_types" not in st.session_state:
    st.session_state["bp_required_cue_types"] = blueprint_defaults.get(
        "required_cue_types", ["object", "scene", "utterance"]
    )
if "stop_after_planning" not in st.session_state:
    st.session_state["stop_after_planning"] = config["generation"].get(
        "stop_after_planning", False
    )
if "run_eval_generator" not in st.session_state:
    st.session_state["run_eval_generator"] = config["generation"].get("run_eval", False)
if "run_eval_examples" not in st.session_state:
    st.session_state["run_eval_examples"] = config["generation"].get(
        "run_eval_examples", False
    )
if "transport" not in st.session_state:
    st.session_state["transport"] = config.get("transport", "openai_sdk")
validation_defaults = config.get("validation", {})
if "validate_structure" not in st.session_state:
    st.session_state["validate_structure"] = validation_defaults.get("structure", False)
if "validate_semantic" not in st.session_state:
    st.session_state["validate_semantic"] = validation_defaults.get("semantic", False)
if "validate_naturalness" not in st.session_state:
    st.session_state["validate_naturalness"] = validation_defaults.get("naturalness", False)
if "validate_qa" not in st.session_state:
    st.session_state["validate_qa"] = validation_defaults.get("qa", False)
if "api_key_secret" not in st.session_state:
    st.session_state["api_key_secret"] = st.session_state.pop("api_key_override", "")
if "api_key_draft" not in st.session_state:
    st.session_state["api_key_draft"] = ""
if "base_url_override" not in st.session_state:
    st.session_state["base_url_override"] = os.getenv("base_url") or os.getenv("BASE_URL") or ""
if "output_root" not in st.session_state:
    st.session_state["output_root"] = str(WEB_OUTPUT_ROOT)

app_page = st.sidebar.radio(
    "页面导航",
    options=["生成工作台", "数据浏览与评估"],
    key="app_page",
)
if app_page == "数据浏览与评估":
    render_data_library(resolve_output_root(st.session_state["output_root"]))
    st.stop()

model = st.session_state["selected_model"]
model_display = html.escape(str(model))
generation_mode = "候选审阅工作流"

st.markdown(
    f"""
    <div class="studio-hero">
      <div class="studio-eyebrow">Generation Workspace</div>
      <h1>Dialogue Data Studio</h1>
      <p>在一个清晰的工作流中完成案例配置、Prompt 审阅、长对话生成与结果检查。</p>
      <div style="margin-top:.75rem">
        <span class="status-pill">模型 · {model_display}</span>
        <span class="status-pill">{st.session_state['session_count_override']} Sessions</span>
        <span class="status-pill">{generation_mode}</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("生成设置")
    st.caption("按“运行 → 数据 → 阶段 → 检查”组织；所有设置只写入本次快照。")

    with st.expander("① 运行与保存", expanded=True):
        st.text_input(
            "测试名称", key="run_name",
            help="用于隔离输出目录；建议每次新实验使用新名称。",
        )
        resume = st.toggle(
            "断点续跑", value=True,
            help="复用同名运行已有的 Blueprint、Plans、Sessions 和 Eval candidates。",
        )
        st.text_input(
            "输出根目录", key="output_root",
            help="支持绝对路径或相对于项目根目录的路径。",
        )
        output_preview = (
            resolve_output_root(st.session_state["output_root"])
            / safe_run_name(st.session_state["run_name"])
        )
        st.caption(f"保存到：{output_preview}")

    with st.expander("② API 与模型", expanded=False):
        st.text_input(
            "API Key（一次性录入）", key="api_key_draft", type="password",
            placeholder="输入后点击“安全载入”",
            help="密钥只保存在当前页面会话内，不写入文件。",
        )
        key_cols = st.columns(2)
        key_cols[0].button("安全载入", on_click=store_api_key, use_container_width=True)
        key_cols[1].button("清除密钥", on_click=clear_api_key, use_container_width=True)
        if st.session_state["api_key_secret"]:
            st.success("API Key 已载入。")
        elif os.getenv("openai_api_key") or os.getenv("OPENAI_API_KEY"):
            st.info("使用环境变量中的 API Key。")
        else:
            st.warning("尚未配置 API Key。")
        st.text_input("Base URL", key="base_url_override", placeholder="https://example.com/v1")
        effective_key = (
            st.session_state["api_key_secret"]
            or os.getenv("openai_api_key") or os.getenv("OPENAI_API_KEY") or ""
        )
        refresh_models = st.button("从 API 刷新模型列表", use_container_width=True)
        if refresh_models:
            with st.spinner("正在读取模型目录..."):
                fetched_models, fetch_error = fetch_models(
                    effective_key, st.session_state["base_url_override"].strip()
                )
            if fetch_error:
                st.error(fetch_error)
            else:
                st.session_state["model_catalog"] = fetched_models
                st.session_state["catalog_source"] = f"API 返回 {len(fetched_models)} 个模型"
                st.success("模型列表已刷新。")
        st.selectbox(
            "模型厂商", options=["OpenAI", "DeepSeek", "Claude", "Qwen", "其他"],
            key="model_provider",
        )
        grouped_models = group_models(st.session_state["model_catalog"])
        provider = st.session_state["model_provider"]
        if provider == "其他":
            st.text_input("自定义模型 ID", key="custom_model_name", placeholder="输入模型 ID")
            if st.session_state.get("custom_model_name", "").strip():
                st.session_state["selected_model"] = st.session_state["custom_model_name"].strip()
        else:
            model_options = grouped_models.get(provider) or FALLBACK_MODELS[provider]
            if st.session_state["selected_model"] not in model_options:
                st.session_state["selected_model"] = model_options[0]
            st.selectbox("模型", options=model_options, key="selected_model")
        st.caption(f"模型目录：{st.session_state['catalog_source']}")
        st.slider(
            "Writer / Eval temperature", min_value=0.0, max_value=2.0,
            step=0.1, key="writer_temperature",
        )
        st.selectbox(
            "API 传输方式", options=["powershell", "openai_sdk"], key="transport",
            help="推荐 openai_sdk；仅在 Python 网络栈受代理或证书限制时尝试 powershell。",
        )

    with st.expander("③ 数据规模与 Blueprint", expanded=True):
        st.slider(
            "Session 数量", min_value=1, max_value=20, step=1,
            key="session_count_override",
        )
        st.caption("记忆角色数量；未分配的 Session 自动作为普通日常（none）。")
        role_cols = st.columns(2)
        role_cols[0].number_input(
            "建立关联", min_value=0, max_value=10, step=1, key="bp_encode_count"
        )
        role_cols[1].number_input(
            "触发回忆", min_value=0, max_value=10, step=1, key="bp_trigger_count"
        )
        role_cols[0].number_input(
            "记忆更新", min_value=0, max_value=10, step=1, key="bp_update_count"
        )
        role_cols[1].number_input(
            "控制场景", min_value=0, max_value=10, step=1, key="bp_control_count"
        )
        assigned_roles = sum(st.session_state[key] for key in (
            "bp_encode_count", "bp_trigger_count", "bp_update_count", "bp_control_count"
        ))
        ordinary_count = st.session_state["session_count_override"] - assigned_roles
        if ordinary_count < 0:
            st.error(f"已分配 {assigned_roles} 个记忆角色，超过 Session 总数。")
        else:
            st.info(f"本次将保留 {ordinary_count} 个普通日常 Session。")
        cue_labels = {
            "object": "物品 object", "scene": "场景 scene", "utterance": "话语 utterance",
            "sound": "声音 sound", "smell": "气味 smell", "taste": "味道 taste",
        }
        st.multiselect(
            "必须覆盖的 cue 类型", options=list(cue_labels), key="bp_required_cue_types",
            format_func=lambda value: cue_labels[value],
        )
        st.caption("Eval outline 标签数量")
        eval_cols = st.columns(3)
        eval_cols[0].number_input(
            "Triggered", min_value=0, max_value=10, step=1, key="bp_eval_triggered"
        )
        eval_cols[1].number_input(
            "Insufficient", min_value=0, max_value=10, step=1, key="bp_eval_insufficient"
        )
        eval_cols[2].number_input(
            "Not triggered", min_value=0, max_value=10, step=1,
            key="bp_eval_not_triggered",
        )
        eval_total = sum(st.session_state[key] for key in (
            "bp_eval_triggered", "bp_eval_insufficient", "bp_eval_not_triggered"
        ))
        st.caption(f"共 {eval_total} 条 Eval outlines；每条由 Generator 生成 3 个候选。")

    with st.expander("④ 流水线阶段", expanded=True):
        st.info(
            "先生成并选择 Blueprint，再只针对该 Blueprint 生成 Plan 候选；"
            "选定 Plan 后才会继续生成对话。"
        )
        candidate_cols = st.columns(2)
        candidate_cols[0].number_input(
            "Blueprint 候选数", min_value=1, max_value=8, step=1,
            key="blueprint_candidate_count",
        )
        candidate_cols[1].number_input(
            "Plan 候选数", min_value=1, max_value=8, step=1,
            key="plan_candidate_count",
        )
        st.caption(
            f"流程：生成 {st.session_state['blueprint_candidate_count']} 个 Blueprint → "
            f"选 1 个 → 生成 {st.session_state['plan_candidate_count']} 个 Plan → 选 1 个。"
        )
        st.toggle(
            "生成 Eval 候选", key="run_eval_generator",
            help="只运行 EvalGenerator，保留每条 outline 的 3 个草稿候选。",
        )
        st.toggle(
            "生成正式 Eval Examples", key="run_eval_examples",
            help=(
                "继续运行 EvidenceResolver、EvalVerifier 和确定性 GoldFinalizer，"
                "写入 eval_examples.json 与 benchmark.eval_samples。"
            ),
        )
        if st.session_state["run_eval_examples"] and not st.session_state["run_eval_generator"]:
            st.caption("正式 Eval Examples 会自动先运行 EvalGenerator。")
        st.caption("Eval 设置会在锁定 Blueprint + Plan 组合后生效。")

    with st.expander("⑤ 可选质量检查", expanded=False):
        st.caption("默认关闭；这些开关不影响 Blueprint 的确定性约束。")
        st.toggle("结构校验与修订", key="validate_structure")
        st.toggle("语义校验与修订", key="validate_semantic")
        st.toggle("自然度校验与修订", key="validate_naturalness")
        st.toggle("最终 QA 报告", key="validate_qa")

    api_ready = bool(
        st.session_state["api_key_secret"]
        or os.getenv("openai_api_key")
        or os.getenv("OPENAI_API_KEY")
    )
    url_ready = bool(
        st.session_state["base_url_override"].strip()
        or os.getenv("base_url")
        or os.getenv("BASE_URL")
    )
    st.divider()
    st.caption("运行就绪状态")
    st.write(("🟢" if api_ready else "🔴") + " API Key  " + ("已就绪" if api_ready else "未配置"))
    st.write(("🟢" if url_ready else "🔴") + " Base URL  " + ("已就绪" if url_ready else "未配置"))
    st.caption(f"当前模型 · {st.session_state['selected_model']}")
    run_clicked = st.button(
        f"生成 {st.session_state['blueprint_candidate_count']} 个 Blueprint 候选",
        type="primary",
        use_container_width=True,
        disabled=not (api_ready and url_ready),
    )

st.markdown(
    """
    <div class="section-heading">
      <span class="section-number">01</span>
      <div><strong>准备生成输入</strong><p>先校验案例信息，再按需审阅本次运行使用的 Prompt。</p></div>
    </div>
    """,
    unsafe_allow_html=True,
)
spec_tab, prompt_tab = st.tabs(["CaseSpec 案例配置", "Prompt 模板审阅"])
with spec_tab:
    st.caption("最简只需 name 和 core_emotional_event；其他人物信息可继续作为生成约束。")
    spec_text = st.text_area(
        "Spec JSON",
        key="spec_text",
        height=480,
        label_visibility="collapsed",
    )
    parsed_spec, spec_error = validate_spec(spec_text)
    if parsed_spec:
        info_cols = st.columns(4)
        info_cols[0].metric("角色", parsed_spec.character_profile.name)
        # session 数量：有 outlines 时取 outlines 数，否则取侧边栏滑块值
        session_count = len(parsed_spec.session_outlines) or st.session_state["session_count_override"]
        info_cols[1].metric("Sessions", session_count)
        info_cols[2].metric("身份", parsed_spec.character_profile.identity or "Planner 补全")
        info_cols[3].metric("Outlines", len(parsed_spec.session_outlines))
        st.success("CaseSpec 结构有效，可以开始生成。")
    else:
        with st.expander("Spec 校验错误", expanded=True):
            st.code(spec_error or "未知错误", language="text")

with prompt_tab:
    st.caption("选择组件并编辑本次运行使用的 Prompt；修改只作用于本次运行，不会覆盖项目默认文件。")
    prompt_names = list(defaults_by_prompt)
    selected_prompt_name = st.selectbox(
        "Prompt 组件",
        prompt_names,
        format_func=lambda name: name.replace("_", " ").title(),
    )
    editor_key = f"prompt_editor_{selected_prompt_name}"
    if editor_key not in st.session_state:
        st.session_state[editor_key] = defaults_by_prompt[selected_prompt_name]
    st.text_area(
        "Prompt 内容",
        key=editor_key,
        height=420,
        label_visibility="collapsed",
    )
    st.button(
        "恢复当前 Prompt 默认值",
        on_click=lambda key=editor_key, value=defaults_by_prompt[selected_prompt_name]: st.session_state.update({key: value}),
        use_container_width=True,
    )
    changed = st.session_state[editor_key] != defaults_by_prompt[selected_prompt_name]
    st.caption(
        f"{len(prompt_names)} 个 prompts · `{selected_prompt_name}.txt` · "
        + ("已修改" if changed else "默认版本")
    )

st.markdown(
    """
    <div class="section-heading">
      <span class="section-number">02</span>
      <div><strong>分步审阅与生成</strong><p>先确定 Blueprint，再审阅由它生成的多个 Plan，最后继续生成对话。</p></div>
    </div>
    """,
    unsafe_allow_html=True,
)

if run_clicked:
    prompt_drafts = {
        name: st.session_state.get(f"prompt_editor_{name}", default)
        for name, default in defaults_by_prompt.items()
    }
    run_generation(
        spec_text=spec_text,
        run_name=st.session_state["run_name"],
        resume=resume,
        output_root_text=st.session_state["output_root"],
        model_name=st.session_state["selected_model"],
        writer_temperature=st.session_state["writer_temperature"],
        transport=st.session_state["transport"],
        api_key_override=st.session_state["api_key_secret"],
        base_url_override=st.session_state["base_url_override"],
        prompt_drafts=prompt_drafts,
        session_count_override=st.session_state["session_count_override"],
        blueprint_constraints={
            "encode_association_count": st.session_state["bp_encode_count"],
            "triggered_recall_count": st.session_state["bp_trigger_count"],
            "memory_update_count": st.session_state["bp_update_count"],
            "control_count": st.session_state["bp_control_count"],
            "required_cue_types": st.session_state["bp_required_cue_types"],
            "eval_triggered_count": st.session_state["bp_eval_triggered"],
            "eval_insufficient_evidence_count": st.session_state["bp_eval_insufficient"],
            "eval_not_triggered_count": st.session_state["bp_eval_not_triggered"],
        },
        validation_config={
            "structure": st.session_state["validate_structure"],
            "semantic": st.session_state["validate_semantic"],
            "naturalness": st.session_state["validate_naturalness"],
            "qa": st.session_state["validate_qa"],
        },
        stop_after_planning=False,
        run_eval=st.session_state["run_eval_generator"],
        run_eval_examples=st.session_state["run_eval_examples"],
        workflow_action="prepare_blueprints",
        blueprint_candidate_count=st.session_state["blueprint_candidate_count"],
        plan_candidate_count=st.session_state["plan_candidate_count"],
    )
elif st.session_state.get("last_output"):
    last_output = Path(st.session_state["last_output"])
    next_action = render_candidate_selection(
        last_output, st.session_state["plan_candidate_count"]
    )
    if next_action:
        runtime_config = read_json(last_output / ".runtime" / "config.json")
        runtime_generation = runtime_config["generation"]
        runtime_validation = runtime_config.get("validation", {})
        saved_spec_text = (last_output / "case_spec.json").read_text(encoding="utf-8")
        workflow_action, selected_blueprint_id, selected_plan_id = next_action
        run_generation(
            spec_text=saved_spec_text,
            run_name=last_output.name,
            resume=True,
            output_root_text=str(last_output.parent),
            model_name=runtime_config["components"]["session_writer"]["model"],
            writer_temperature=float(
                runtime_config["components"]["session_writer"]["temperature"]
            ),
            transport=runtime_config.get("transport", "openai_sdk"),
            api_key_override=st.session_state["api_key_secret"],
            base_url_override=st.session_state["base_url_override"],
            prompt_drafts={},
            session_count_override=int(runtime_generation["session_count"]),
            blueprint_constraints=runtime_config["blueprint_constraints"],
            validation_config=runtime_validation,
            stop_after_planning=False,
            run_eval=bool(runtime_generation.get("run_eval", False)),
            run_eval_examples=bool(runtime_generation.get("run_eval_examples", False)),
            workflow_action=workflow_action,
            blueprint_candidate_count=st.session_state["blueprint_candidate_count"],
            plan_candidate_count=st.session_state["plan_candidate_count"],
            selected_blueprint_id=selected_blueprint_id,
            selected_plan_id=selected_plan_id,
            existing_output_dir=last_output,
        )
    else:
        render_results(last_output)
        render_artifact_browser(last_output)
else:
    st.info("完成 Spec 校验后，点击左侧“生成 Blueprint 候选”。候选将在这里展开。")
