from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
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
    if line.startswith("benchmark:"):
        return 0.98, "正在汇总 Benchmark 与 QA"
    return 0.03, "正在初始化生成任务"


def render_results(output_dir: Path) -> None:
    benchmark_path = output_dir / "benchmark.json"
    qa_path = output_dir / "qa_report.json"
    checkpoint_path = output_dir / "checkpoint_sessions.json"

    if not benchmark_path.exists():
        if checkpoint_path.exists():
            checkpoint = read_json(checkpoint_path)
            st.info(f"该运行已有 {len(checkpoint)} 个 Session 检查点，可启用断点续跑。")
        return

    benchmark = read_json(benchmark_path)
    qa = read_json(qa_path) if qa_path.exists() else None
    dialogues = benchmark.get("dialogues", [])
    turn_count = sum(len(item.get("turns", [])) for item in dialogues)

    st.subheader("生成结果")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Sessions", len(dialogues))
    metric_cols[1].metric("Turns", turn_count)
    metric_cols[2].metric("Eval samples", len(benchmark.get("eval_samples", [])))
    metric_cols[3].metric("QA", "通过" if qa and qa.get("passed") else "待检查")

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
                    avatar = "👤" if turn["speaker"] == "user" else "✦"
                    with st.chat_message(turn["speaker"], avatar=avatar):
                        st.caption(f"{turn['turn_id']} · {turn['round_id']}")
                        st.write(turn["text"])
    with qa_tab:
        if not qa:
            st.warning("尚未生成 QA 报告。")
        else:
            for name, passed in qa.get("checks", {}).items():
                st.write(("✅" if passed else "❌") + f"  {name}")
            if qa.get("errors"):
                st.error("\n".join(qa["errors"]))
            st.caption(f"记录的模型调用：{len(qa.get('call_records', []))} 次")
    with raw_tab:
        st.json(benchmark, expanded=False)


def resolve_output_root(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def build_runtime_snapshot(
    output_dir: Path,
    model_name: str,
    writer_temperature: float,
    transport: str,
    prompt_drafts: dict[str, str],
) -> Path:
    """Create an isolated config/prompt snapshot without touching project defaults."""
    runtime_dir = output_dir / ".runtime"
    runtime_prompt_dir = runtime_dir / "prompts"
    runtime_script_dir = runtime_dir / "scripts"
    runtime_prompt_dir.mkdir(parents=True, exist_ok=True)
    runtime_script_dir.mkdir(parents=True, exist_ok=True)

    runtime_config = read_json(CONFIG_PATH)
    runtime_config["transport"] = transport
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
    if not resume and (output_dir / "checkpoint_sessions.json").exists():
        st.error("该测试名称已有检查点。请开启断点续跑，或换一个新的测试名称。")
        return
    if resume and case_path.exists():
        previous, previous_error = validate_spec(case_path.read_text(encoding="utf-8"))
        if previous_error or previous is None or previous.model_dump() != spec.model_dump():
            st.error("同名运行的 Spec 与当前内容不同，不能安全续跑。请换一个新的测试名称。")
            return

    case_path.write_text(spec.model_dump_json(indent=2), encoding="utf-8")
    try:
        runtime_config_path = build_runtime_snapshot(
            output_dir, model_name, writer_temperature, transport, prompt_drafts
        )
    except OSError as exc:
        st.error(f"无法创建本次运行的配置快照：{exc}")
        return
    st.session_state["last_output"] = str(output_dir)

    progress = st.progress(0.01, text="准备生成环境")
    status = st.empty()
    log_box = st.empty()
    logs: list[str] = []
    completed = 0
    total = len(spec.session_outlines)

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

    assert process.stdout is not None
    for raw_line in process.stdout:
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
        status.success("Benchmark 与 QA 已生成。")
        render_results(output_dir)
    else:
        progress.progress(min(0.99, 0.10 + 0.84 * completed / max(total, 1)), text="生成中断")
        status.error("生成任务未完成。日志已保留；修正问题后可使用同一名称断点续跑。")


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
if "transport" not in st.session_state:
    st.session_state["transport"] = config.get("transport", "openai_sdk")
if "api_key_secret" not in st.session_state:
    st.session_state["api_key_secret"] = st.session_state.pop("api_key_override", "")
if "api_key_draft" not in st.session_state:
    st.session_state["api_key_draft"] = ""
if "base_url_override" not in st.session_state:
    st.session_state["base_url_override"] = os.getenv("base_url") or os.getenv("BASE_URL") or ""
if "output_root" not in st.session_state:
    st.session_state["output_root"] = str(WEB_OUTPUT_ROOT)

model = st.session_state["selected_model"]

st.markdown(
    f"""
    <div class="studio-hero">
      <h1>Dialogue Data Studio</h1>
      <p>单案例 Spec 调试、Prompt 审阅与长对话生成工作台</p>
      <div style="margin-top:.75rem">
        <span class="status-pill">模型 · {model}</span>
        <span class="status-pill">纯文本 Session</span>
        <span class="status-pill">结构化 QA</span>
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
    st.selectbox(
        "API 传输方式",
        options=["powershell", "openai_sdk"],
        key="transport",
        help="当前 Windows 环境推荐 powershell；标准环境可使用 openai_sdk。",
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
    st.caption("在这里粘贴或修改单个案例。运行前会使用项目 Pydantic 模型进行完整校验。")
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
        info_cols[1].metric("Sessions", len(parsed_spec.session_outlines))
        info_cols[2].metric("Eval outlines", len(parsed_spec.eval_outlines))
        info_cols[3].metric("精确 Cues", len(parsed_spec.cues.exact))
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
st.caption("运行时会显示规划、逐 Session 生成、修订与 QA 的实时状态。")

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
    )
elif st.session_state.get("last_output"):
    render_results(Path(st.session_state["last_output"]))
else:
    st.info("完成 Spec 校验后，点击左侧“开始生成”。结果将在这里展开。")
