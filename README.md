# 长期对话数据生成器

本项目根据固定 `CaseSpec` 调用兼容 OpenAI API 的模型，生成按时间排列的长期用户—助手对话。当前版本支持纯文本 session 生成，以及按需启用的结构校验、语义校验/修订、自然度检查和最终 QA；eval 组件的 prompt 已预留，本次测试按要求关闭 eval 生成。

## 项目结构

- `cases/`：角色、核心事件、cue、session/eval outlines，是事实唯一来源。
- `prompts/`：每个 LLM 组件独立的 prompt 控制文件。
- `src/`：配置、Pydantic 数据模型、统一 LLM 客户端、组件和流水线。
- `outputs/`：计划、生成检查点、benchmark 与 QA 结果。
- `config.json`：模型、temperature、seed、重试次数、上下文 session 数等配置。

Planner 通过 `planner_batch_size` 分批生成长计划，并把已生成计划传给下一批以维持严格时间顺序，避免兼容网关的单次输出长度限制。

## 最简 Spec 与故事线规划

新建案例时，最简输入只需要两个字段：

```json
{
  "name": "林澄",
  "core_emotional_event": "林澄错过奶奶去世前最后一次通话，留有遗憾。"
}
```

原有的嵌套 `character_profile` 格式继续兼容；`identity`、`daily_scenes`、`conversation_style`、`interests`、`persona_summary`、`traits` 和 `assistant_persona` 都是可选的人物种子。未提供时，Planner 会推断少量稳定的普通日常设定。系统只把与规划有关的 `planner_brief` 发送给 SessionPlanner，不再把 cues、eval 等后续阶段字段混入 Planner 输入。

新的 SessionPlan 除了 topic 和 story_beat，还显式包含 `session_type`、`scene`、`user_intent` 与 `continuity_hook`。默认规划要求至少 70% 的 session 完全属于无关日常，并用具体场景、聊天动机和弱连续钩子形成更自然的长期故事线。旧版 `session_plans.json` 仍可读取，缺失的新字段会在载入时使用兼容默认值。

调试 Planner 时，可在 Web 左侧开启“仅生成故事线 Plan”。该模式写出 `session_plans.json`、`session_plans.txt` 和 Planner 审计日志后立即停止，不调用 SessionWriter 或任何验证阶段。Web 会直接展示三类 session 的数量和每条计划的场景、聊天动机、故事节拍与后续钩子。

Web 侧边栏的“页面导航”可以切换到“数据浏览与评估”。该页面会扫描输出根目录下的全部历史运行，兼容完整 benchmark、仅 Plan 和只有 checkpoint 的中断运行；可以查看隐式生活锚点、筛选 Plans、逐 Session 阅读对话、检查快速质量指标，并预览或下载运行目录中的 JSON、TXT 与审计日志。

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

## 可选验证阶段

为了便于直接调试 SessionPlanner 和 SessionWriter，额外的产物验证阶段默认全部关闭。可在 `config.json` 的 `validation` 中分别开启：

- `structure`：确定性的 turn 数量、ID、round ID 和 speaker 顺序检查，失败时调用 Reviser；
- `semantic`：调用 SessionVerifier 做语义检查，失败时调用 Reviser；
- `naturalness`：调用 NaturalnessChecker，并按需做一次最小修订；
- `qa`：生成最终的非 LLM `qa_report.json`。

Web 页面左侧的“验证阶段（可选）”提供了对应的四个开关，设置只写入本次运行的配置快照，不会修改项目默认配置。即使全部关闭，LLM 输出仍必须通过 Pydantic 数据模型解析，以确保后续代码可以读取生成结果。

## 生成审计日志

每次运行会在输出目录的 `logs/` 下保存可审阅的阶段记录：`00_original_case_spec.json` 保存原始 spec、本次生成配置及验证开关；`01_planner_*.json` 保存 planner 的分批输入输出；`sessions/<session_id>/` 始终保存 writer 初稿和最终 session，并只为实际开启的验证阶段保存检查/修订日志；`99_pipeline_result.json` 保存最终结果与本次验证开关。日志不记录 API key。
