from __future__ import annotations

import json
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
    if line.startswith("planned ") or line.startswith("resumed ") and "saved plans" in line:
        return 0.08, "Session 计划已就绪"
    if line.startswith("generating "):
        session_id = line.removeprefix("generating ").removesuffix("...")
        value = 0.10 + 0.84 * (completed / max(total, 1))
        return value, f"正在生成 {session_id}"
    if line.startswith("completed "):
        value = 0.10 + 0.84 * (completed / max(total, 1))
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
    if re.match(r"01_planner_(batch_\d+|resumed)\.json", name):
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

    # Writer / final session：展示对话
    if component in {"session_writer", "session_reviser", "pipeline"} and isinstance(output, dict) and "turns" in output:
        st.write(f"**Session**：{output.get('session_id', '')} · {output.get('topic', '')}")
        for turn in output.get("turns", []):
            with st.chat_message(turn.get("speaker", "user")):
                st.caption(f"{turn.get('turn_id', '')} · {turn.get('round_id', '')}")
                st.write(turn.get("text", ""))
        return

    # Planner：展示计划列表
    if component == "session_planner" and isinstance(output, dict) and "plans" in output:
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

    # Verifier：展示结论
    if component in {"session_verifier", "deterministic_structural_verifier", "naturalness_checker"} and isinstance(output, dict):
        result = output.get("result", "")
        st.write(f"**结论**：{'✅ pass' if result == 'pass' else '⚠️ revise'}")
        for issue in output.get("issues", []):
            st.write(f"- `{issue.get('turn_id', '')}` ({issue.get('type', '')})：{issue.get('description', '')}")
        return

    # 其他：展示 JSON
    st.json(output, expanded=False)


def render_results(output_dir: Path) -> None:
    benchmark_path = output_dir / "benchmark.json"
    qa_path = output_dir / "qa_report.json"
    checkpoint_path = output_dir / "checkpoint_sessions.json"
    plan_path = output_dir / "session_plans.json"
    runtime_config_path = output_dir / ".runtime" / "config.json"
    runtime_config = read_json(runtime_config_path) if runtime_config_path.exists() else {}
    planner_only = runtime_config.get("generation", {}).get("stop_after_planning", False)

    if planner_only and plan_path.exists():
        plans = read_json(plan_path).get("plans", [])
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
                st.write(f"**后续钩子**：{plan.get('continuity_hook') or '无'}")
                st.caption(f"rounds={plan.get('round_count', '')}")
        return

    if not benchmark_path.exists():
        if checkpoint_path.exists():
            checkpoint = read_json(checkpoint_path)
            st.info(f"该运行已有 {len(checkpoint)} 个 Session 检查点，可启用断点续跑。")
        return

    benchmark = read_json(benchmark_path)
    qa = read_json(qa_path) if qa_path.exists() else None
    validation_config = runtime_config.get("validation")
    qa_enabled = validation_config.get("qa", False) if validation_config is not None else None
    dialogues = benchmark.get("dialogues", [])
    turn_count = sum(len(item.get("turns", [])) for item in dialogues)

    st.subheader("生成结果")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Sessions", len(dialogues))
    metric_cols[1].metric("Turns", turn_count)
    metric_cols[2].metric("Eval samples", len(benchmark.get("eval_samples", [])))
    if qa:
        qa_status = "通过" if qa.get("passed") else "未通过"
    elif qa_enabled is False:
        qa_status = "已关闭"
    else:
        qa_status = "待检查"
    metric_cols[3].metric("QA", qa_status)

    download_cols = st.columns([1, 1, 3])
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

    result_tab, qa_tab, raw_tab = st.tabs(["对话预览", "QA 明细", "原始 JSON"])
    with result_tab:
        for session in dialogues:
            label = f"{session['session_id']} · {session['date']} · {session['topic']}"
            with st.expander(label, expanded=session == dialogues[0]):
                for turn in session.get("turns", []):
                    with st.chat_message(turn["speaker"]):
                        st.caption(f"{turn['turn_id']} · {turn['round_id']}")
                        st.write(turn["text"])
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
    candidates = [output_root] if (output_root / "case_spec.json").exists() else []
    candidates.extend(path for path in output_root.iterdir() if path.is_dir())
    runs: list[dict] = []
    for path in candidates:
        recognizable = any((path / name).exists() for name in (
            "case_spec.json", "session_plans.json", "checkpoint_sessions.json", "benchmark.json",
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
    checkpoint = read_json_optional(output_dir / "checkpoint_sessions.json", [])
    benchmark = read_json_optional(output_dir / "benchmark.json", {})
    dialogues = checkpoint if isinstance(checkpoint, list) and checkpoint else benchmark.get("dialogues", [])
    return {
        "output_dir": output_dir,
        "case_spec": read_json_optional(output_dir / "case_spec.json", {}),
        "plans_payload": plans_payload if isinstance(plans_payload, dict) else {},
        "plans": plans_payload.get("plans", []) if isinstance(plans_payload, dict) else [],
        "life_anchor": plans_payload.get("life_anchor") if isinstance(plans_payload, dict) else None,
        "dialogues": dialogues if isinstance(dialogues, list) else [],
        "benchmark": benchmark if isinstance(benchmark, dict) else {},
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

    status_labels = {"complete": "完整", "partial": "部分完成", "plans": "仅 Plan", "empty": "初始化"}
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
    metrics = calculate_dialogue_metrics(dialogues)

    st.caption(f"`{selected_path}`")
    metric_cols = st.columns(5)
    metric_cols[0].metric("状态", status_labels[selected_run["status"]])
    metric_cols[1].metric("Plans", len(plans))
    metric_cols[2].metric("Sessions", metrics["sessions"])
    metric_cols[3].metric("Rounds", metrics["rounds"])
    metric_cols[4].metric("QA", "通过" if bundle["qa"] and bundle["qa"].get("passed") else "无/未通过")

    overview_tab, plan_tab, dialogue_tab, quality_tab, raw_tab = st.tabs([
        "运行概览", "故事线 Plans", "Session 对话", "快速评估", "原始产物",
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
                st.write(f"**验证开关**：{validation}")
            else:
                st.info("该运行没有保存 runtime config。")

    with plan_tab:
        if not plans:
            st.info("该运行没有 Session Plans。")
        else:
            types = sorted({plan.get("session_type", "daily_life") for plan in plans})
            threads = sorted({plan.get("life_thread", "one_off") for plan in plans})
            plan_filters = st.columns(2)
            chosen_types = plan_filters[0].multiselect("Session 类型", types, default=types)
            chosen_threads = plan_filters[1].multiselect("生活支线", threads, default=threads)
            filtered_plans = [
                plan for plan in plans
                if plan.get("session_type", "daily_life") in chosen_types
                and plan.get("life_thread", "one_off") in chosen_threads
            ]
            table_lines = [
                "| ID | 日期 | 类型 | 互动 | 生活支线 | 主题 |",
                "|---|---|---|---|---|---|",
            ]
            for plan in filtered_plans:
                cells = [
                    plan.get("session_id"), plan.get("date"), plan.get("session_type"),
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
            for plan in plans:
                for target, key, default in (
                    (type_counts, "session_type", "daily_life"),
                    (mode_counts, "interaction_mode", "—"),
                    (thread_counts, "life_thread", "one_off"),
                ):
                    value = plan.get(key, default)
                    target[value] = target.get(value, 0) + 1
            distribution_cols = st.columns(3)
            distribution_cols[0].write("**Session 类型**")
            distribution_cols[0].json(type_counts)
            distribution_cols[1].write("**互动模式**")
            distribution_cols[1].json(mode_counts)
            distribution_cols[2].write("**生活支线**")
            distribution_cols[2].json(thread_counts)

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
    validation_config: dict[str, bool],
    stop_after_planning: bool,
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
    runtime_config["generation"]["stop_after_planning"] = stop_after_planning
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
    validation_config: dict[str, bool],
    stop_after_planning: bool,
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
    try:
        output_root = resolve_output_root(output_root_text)
        output_dir = output_root / safe_name
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        st.error(f"无法使用该保存路径：{exc}")
        return
    case_path = output_dir / "case_spec.json"
    if not resume and (output_dir / "session_plans.json").exists():
        st.error("该测试名称已有 Plan。为避免误用旧结果，请换一个新的测试名称，或开启断点续跑。")
        return
    if resume and case_path.exists():
        previous, previous_error = validate_spec(case_path.read_text(encoding="utf-8"))
        if previous_error or previous is None or previous.model_dump() != spec.model_dump():
            st.error("同名运行的 Spec 与当前内容不同，不能安全续跑。请换一个新的测试名称。")
            return

    case_path.write_text(spec.model_dump_json(indent=2), encoding="utf-8")
    try:
        runtime_config_path = build_runtime_snapshot(
            output_dir, model_name, writer_temperature, transport, prompt_drafts,
            session_count_override, validation_config, stop_after_planning,
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
        if line.startswith("completed "):
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
        if pipeline_result_path.exists():
            try:
                pr = json.loads(pipeline_result_path.read_text(encoding="utf-8"))
                failed_ids = pr.get("failed_session_ids", [])
            except Exception:
                pass
        if failed_ids:
            status.warning(f"生成完成，但有 {len(failed_ids)} 个 session 失败被跳过：{', '.join(failed_ids)}")
        else:
            status.success("生成产物已保存。")
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
    :root { color-scheme: light !important; }
    html, body, .stApp, [data-testid="stAppViewContainer"],
    [data-testid="stMain"], .main { background: #f3f6fa !important; color: #172033 !important; }
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5,
    .stApp p, .stApp label, .stApp span, .stApp small,
    .stApp [data-testid="stCaptionContainer"], .stApp [data-testid="stMetricValue"],
    .stApp [data-testid="stMetricLabel"] { color: #172033 !important; }
    [data-testid="stSidebar"] { background: #182235 !important; border-right: 1px solid #2e3a50; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] small,
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color: #f4f7fb !important; }
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
    .stApp button { border-color: #8291aa !important; }
    .stApp button[kind="primary"] { background: #2563eb !important; border-color: #2563eb !important; }
    .stApp button[kind="primary"] p, .stApp button[kind="primary"] span { color: #ffffff !important; }
    [data-testid="stSidebar"] button:not([kind="primary"]) {
        background: #ffffff !important; border-color: #a8b4c7 !important;
    }
    [data-testid="stSidebar"] button:not([kind="primary"]) p,
    [data-testid="stSidebar"] button:not([kind="primary"]) span { color: #172033 !important; }
    .studio-hero {
        padding: 1.15rem 1.35rem; border: 1px solid #dbe2ea; border-radius: 16px;
        background: linear-gradient(120deg, #ffffff 15%, #eef6ff 100%);
        margin-bottom: 1rem;
    }
    .studio-hero h1 { font-size: 1.85rem; margin: 0 0 .25rem 0; color: #111827; }
    .studio-hero p { color: #5b6472; margin: 0; }
    .status-pill {
        display: inline-block; padding: .2rem .55rem; margin-right: .35rem;
        border-radius: 999px; background: #e7f7ef; color: #17643b; font-size: .78rem;
    }
    div[data-testid="stExpander"], [data-testid="stChatMessage"],
    [data-testid="stMetric"] { border-color: #d5deea !important; background: #ffffff !important; }
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
if "stop_after_planning" not in st.session_state:
    st.session_state["stop_after_planning"] = True
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

st.markdown(
    f"""
    <div class="studio-hero">
      <h1>Dialogue Data Studio</h1>
      <p>单案例 Spec 调试、Prompt 审阅与长对话生成工作台</p>
      <div style="margin-top:.75rem">
        <span class="status-pill">模型 · {model}</span>
        <span class="status-pill">纯文本 Session</span>
        <span class="status-pill">验证阶段可选</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("运行控制")
    st.text_input("测试名称", key="run_name", help="用于隔离输出目录；建议每次新实验使用新名称。")
    resume = st.toggle("断点续跑", value=True, help="复用同名运行已有的 plans 和 checkpoint。")
    st.divider()
    st.subheader("API 配置")
    st.text_input(
        "API Key（一次性录入）",
        key="api_key_draft",
        type="password",
        placeholder="输入后点击“安全载入”",
        help="载入后立即清空输入框；密钥只保存在当前页面会话内，不写入文件。",
    )
    key_cols = st.columns(2)
    key_cols[0].button("安全载入", on_click=store_api_key, use_container_width=True)
    key_cols[1].button("清除密钥", on_click=clear_api_key, use_container_width=True)
    if st.session_state["api_key_secret"]:
        st.success("API Key 已安全载入，不显示内容。")
    elif os.getenv("openai_api_key") or os.getenv("OPENAI_API_KEY"):
        st.info("当前使用环境变量中的 API Key。")
    else:
        st.warning("尚未配置 API Key。")
    st.text_input(
        "Base URL",
        key="base_url_override",
        placeholder="https://example.com/v1",
    )
    effective_key = (
        st.session_state["api_key_secret"]
        or os.getenv("openai_api_key")
        or os.getenv("OPENAI_API_KEY")
        or ""
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
    st.divider()
    st.subheader("模型配置")
    st.selectbox(
        "模型厂商",
        options=["OpenAI", "DeepSeek", "Claude", "Qwen", "其他"],
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
        "Writer / Eval temperature",
        min_value=0.0,
        max_value=2.0,
        step=0.1,
        key="writer_temperature",
        help="Planner、Verifier 等组件继续使用其稳定性默认值。",
    )
    st.slider(
        "Session 数量",
        min_value=1,
        max_value=20,
        step=1,
        key="session_count_override",
        help="本次运行生成的 session 数量。仅当 CaseSpec 未提供 session_outlines 时生效。",
    )
    st.toggle(
        "仅生成故事线 Plan",
        key="stop_after_planning",
        help="开启后在 SessionPlanner 完成时停止，不调用 Writer 或任何验证阶段。",
    )
    st.selectbox(
        "API 传输方式",
        options=["powershell", "openai_sdk"],
        key="transport",
        help="当前 Windows 环境推荐 powershell；标准环境可使用 openai_sdk。",
    )
    st.divider()
    st.subheader("验证阶段（可选）")
    st.caption("默认全部关闭，便于直接调试 Planner / Writer；按需单独开启。")
    st.toggle(
        "结构校验与修订",
        key="validate_structure",
        help="检查 turn 数量、ID、round ID 和 speaker 顺序；失败时调用 Reviser。",
    )
    st.toggle(
        "语义校验与修订",
        key="validate_semantic",
        help="调用 SessionVerifier 检查事实泄露、人物冲突、重复等问题。",
    )
    st.toggle(
        "自然度校验与修订",
        key="validate_naturalness",
        help="调用 NaturalnessChecker，并在需要时做一次最小修订。",
    )
    st.toggle(
        "最终 QA 报告",
        key="validate_qa",
        help="对最终 sessions 做非 LLM 的顺序、时间和 turn 结构检查。",
    )
    st.divider()
    st.subheader("保存位置")
    st.text_input(
        "输出根目录",
        key="output_root",
        help="支持绝对路径或相对于项目根目录的路径；实际结果保存在该目录/测试名称下。",
    )
    output_preview = resolve_output_root(st.session_state["output_root"]) / safe_run_name(st.session_state["run_name"])
    st.caption(f"本次输出：{output_preview}")
    st.divider()
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
    st.caption("连接状态")
    st.write(("🟢" if api_ready else "🔴") + " API Key")
    st.write(("🟢" if url_ready else "🔴") + " Base URL")
    st.write(f"◈ {st.session_state['selected_model']}")
    st.divider()
    st.caption("设置仅作用于本次测试快照；项目默认配置保持不变。")
    run_clicked = st.button(
        "开始生成",
        type="primary",
        use_container_width=True,
        disabled=not (api_ready and url_ready),
    )

spec_col, prompt_col = st.columns([1.22, 0.78], gap="large")
with spec_col:
    st.subheader("CaseSpec")
    st.caption("最简只需 name 和 core_emotional_event；其他人物信息仍可作为可选约束。")
    spec_text = st.text_area(
        "Spec JSON",
        key="spec_text",
        height=560,
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
        st.success("Spec 结构有效，可以生成。")
    else:
        with st.expander("Spec 校验错误", expanded=True):
            st.code(spec_error or "未知错误", language="text")

with prompt_col:
    st.subheader("Prompt Library")
    st.caption("切换组件并编辑本次运行使用的 Prompt；不会覆盖项目默认文件。")
    prompt_names = list(defaults_by_prompt)
    selected_prompt_name = st.selectbox(
        "组件",
        prompt_names,
        format_func=lambda name: name.replace("_", " ").title(),
    )
    editor_key = f"prompt_editor_{selected_prompt_name}"
    if editor_key not in st.session_state:
        st.session_state[editor_key] = defaults_by_prompt[selected_prompt_name]
    st.text_area(
        "Prompt 内容",
        key=editor_key,
        height=560,
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

st.divider()
st.subheader("Generation Monitor")
st.caption("运行时会显示规划、逐 Session 生成，以及本次实际开启的验证阶段。")

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
        validation_config={
            "structure": st.session_state["validate_structure"],
            "semantic": st.session_state["validate_semantic"],
            "naturalness": st.session_state["validate_naturalness"],
            "qa": st.session_state["validate_qa"],
        },
        stop_after_planning=st.session_state["stop_after_planning"],
    )
elif st.session_state.get("last_output"):
    render_results(Path(st.session_state["last_output"]))
    render_artifact_browser(Path(st.session_state["last_output"]))
else:
    st.info("完成 Spec 校验后，点击左侧“开始生成”。结果将在这里展开。")
