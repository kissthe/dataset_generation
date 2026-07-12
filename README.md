# 长期对话数据生成器

本项目根据固定 `CaseSpec` 调用兼容 OpenAI API 的模型，生成按时间排列的长期用户—助手对话。当前版本完成纯文本 session 生成、校验、最小修订、自然度检查、结构 QA 和 benchmark JSON 输出；eval 组件的 prompt 已预留，本次测试按要求关闭 eval 生成。

## 项目结构

- `cases/`：角色、核心事件、cue、session/eval outlines，是事实唯一来源。
- `prompts/`：每个 LLM 组件独立的 prompt 控制文件。
- `src/`：配置、Pydantic 数据模型、统一 LLM 客户端、组件和流水线。
- `outputs/`：计划、生成检查点、benchmark 与 QA 结果。
- `config.json`：模型、temperature、seed、重试次数、上下文 session 数等配置。

Planner 通过 `planner_batch_size` 分批生成长计划，并把已生成计划传给下一批以维持严格时间顺序，避免兼容网关的单次输出长度限制。

## 运行方式

安装依赖并设置 `openai_api_key` 与 `base_url`（也兼容大写环境变量），然后运行：

```powershell
pip install -r requirements.txt
python run.py --case cases/case_a.json --output outputs/case_a
```

如果 `base_url` 只包含协议和主机名，程序会自动补全 OpenAI 兼容接口常用的 `/v1` 路径；已经带路径的地址保持不变。

`transport` 默认可设为 `openai_sdk`。本机若遇到 Python HTTP 客户端与代理/TLS 不兼容，可设为 `powershell`，通过 Windows 原生网络栈调用同一个 OpenAI 兼容接口；两种方式共享完全相同的 prompt、结构化输出和重试校验。

主要产物为 `outputs/case_a/benchmark.json`、`qa_report.json`、`session_plans.json` 和便于人工检查的 `session_plans.txt`。生成过程中每完成一个 session 都会更新 `checkpoint_sessions.json`，便于中断后检查。

再次使用同一输出目录运行时，程序会校验并复用已有 `session_plans.json` 和合法的 session 前缀 checkpoint，从未完成的 session 继续。

`EvalGenerator`、`EvalVerifier` 与非 LLM 的 `GoldFinalizer` 已提供独立接口。本轮 `run_eval=false`，所以测试 benchmark 的 `eval_samples` 是空数组；后续可在 pipeline 中启用这些组件生成正式 eval。
