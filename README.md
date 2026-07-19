# 长期对话数据生成器

本项目根据固定 `CaseSpec` 调用兼容 OpenAI API 的模型，生成按时间排列的长期用户—朋友对话。当前版本支持纯文本 Session、Generator-only Eval 候选，以及可选的正式 Eval Examples 生成链路；Session 的结构校验、语义校验/修订、自然度检查和最终 QA 仍可分别开启。

## 项目结构

- `cases/`：角色名、核心情绪事件及可选人物种子，是人工输入的事实来源。
- `prompts/`：每个 LLM 组件独立的 prompt 控制文件。
- `src/`：配置、Pydantic 数据模型、统一 LLM 客户端、组件和流水线。
- `outputs/`：计划、生成检查点、benchmark 与 QA 结果。
- `config.json`：模型、temperature、seed、重试次数、上下文 Session 数和 Blueprint 确定性覆盖约束等配置。

Web 默认采用三阶段工作流：程序先生成若干 Dataset Blueprint 候选；用户选定一个 Blueprint 后，程序再依据该 Blueprint 的生活锚点、记忆槽位、cue 和证据目标生成若干完整 Plan 候选；用户选定 Plan 后才启动 Writer。Blueprint 与 Plan 的候选数量都可在 Web 侧边栏按次调整（1–8 个）。每个 Plan 都保存来源 Blueprint 的候选 ID 与指纹，因此不能与其他 Blueprint 交叉选择。命令行直接运行仍兼容原来的一次性完整流水线。

## 最简 Spec 与故事线规划

新建案例时，最简输入只需要两个字段：

```json
{
  "name": "林澄",
  "core_emotional_event": "林澄错过奶奶去世前最后一次通话，留有遗憾。"
}
```

原有的嵌套 `character_profile` 格式继续兼容；`identity`、`daily_scenes`、`conversation_style`、`interests`、`persona_summary`、`traits` 和 `assistant_persona` 都是可选的人物种子。未提供时，DatasetBlueprintPlanner 会推断少量稳定的普通日常设定。

不需要人工填写 `session_id`。程序根据 `case_id + 顺序号` 确定性生成完整 ID 列表（例如 `A-S01` 到 `A-S10`），LLM 只能复制并落实这些槽位，不能增删、重排或改写 ID。旧版 `session_outlines` 仍可作为话题种子读取，但其中的手写 ID 不再控制新蓝图。

默认 10 个 Session 的蓝图覆盖为：6 个普通日常、1 个 cue—个人记忆关联建立、1 个外部 cue 触发回忆、1 个记忆更新、1 个控制场景。蓝图至少生成 object、scene、utterance 三类用户特定 cue。默认 6 个 Eval outline 按 `triggered / insufficient_evidence / not_triggered = 2 / 2 / 2` 规划。这些数量与必须覆盖的 cue 类型位于 `config.json` 的 `blueprint_constraints`，也可以在 Web 的“数据规模与 Blueprint”区域按次调整。

精简后的 `EmotionMemory` 只保留 `memory_id`、`event_summary`、`emotion`、`emotional_meaning` 和 `cue_seeds`。旧版 `historical_emotion.intensity` 没有稳定的下游语义，且与 Session 的局部情绪变化重复，因此已经删除；`distinguishing_detail` 合并进 cue 的 `personal_meaning`。SessionSlot/SessionPlan 中重复的 `emotion_intensity`、`disclosure_level` 和反向索引 `supports_eval_outlines` 也已删除。旧 Blueprint 与 Plan 在读取时会自动迁移为新结构。

新的 SessionPlan 除了 topic 和 story_beat，还显式包含 `session_type`、`scene`、`user_intent`、`continuity_hook`，以及从蓝图锁定的 `memory_role`、`memory_id`、`cue_id`、`evidence_goal`、局部情绪和相对过去的变化。旧版 `session_plans.json` 仍可读取，缺失的新字段会在载入时使用兼容默认值。

Web 第一步写出只含 Blueprint 候选的 `planning_candidates.json` 后停止，不调用 SessionPlanner 或 SessionWriter。选择 Blueprint 后，第二步只针对该 Blueprint 生成 Plan 候选，并展示日期、场景、聊天动机、故事节拍、生活线、后续钩子及逐 Session 的蓝图落实情况；点击“选定这个 Plan 并继续生成”后才会写出正式的 `dataset_blueprint.json` 和 `session_plans.json`。

Web 侧边栏按“运行与保存 → API 与模型 → 数据规模与 Blueprint → 流水线阶段 → 可选质量检查”组织。“页面导航”可以切换到“数据浏览与评估”，扫描输出根目录下的全部历史运行，兼容完整 benchmark、仅蓝图、仅 Plan、Eval 候选和只有 checkpoint 的中断运行；“Dataset 蓝图”页签可查看情绪记忆、不同类型 cue、Session 槽位和 Eval 覆盖，Plans 页可按记忆角色筛选。

## 运行方式

安装依赖并设置 `openai_api_key` 与 `base_url`（也兼容大写环境变量），然后运行：

```powershell
pip install -r requirements.txt
python run.py --case cases/case_a.json --output outputs/case_a
```

命令行也可显式使用候选工作流：

```powershell
python run.py --case cases/case_a.json --output outputs/case_a --prepare-blueprints --blueprint-count 3
python run.py --case cases/case_a.json --output outputs/case_a --prepare-plans --select-blueprint BP-01 --plan-count 4
python run.py --case cases/case_a.json --output outputs/case_a --select-blueprint BP-01 --select-plan PLAN-02
```

如果 `base_url` 只包含协议和主机名，程序会自动补全 OpenAI 兼容接口常用的 `/v1` 路径；已经带路径的地址保持不变。

`transport` 默认使用 `openai_sdk`。本机若遇到 Python HTTP 客户端与代理/证书不兼容，可改用 `powershell`；PowerShell 连接在 TLS 握手、连接重置或超时时失败后，会自动回退到 SDK。两种方式共享完全相同的 prompt、结构化输出和重试校验，HTTP 认证或请求参数错误不会触发回退。

主要产物为 `dataset_blueprint.json`、`session_plans.json`、便于人工检查的 `session_plans.txt`、`benchmark.json`，以及按需生成的 `eval_candidates.json`、`eval_examples.json` 和 `qa_report.json`。生成过程中每完成一个 Session 都会更新 `checkpoint_sessions.json`；Eval 候选与正式样本也分别写入 checkpoint，便于中断后继续。

再次使用同一输出目录运行时，程序会先校验并复用 `dataset_blueprint.json`，再校验 SessionPlan 的蓝图字段未发生漂移，然后从合法的 session 前缀 checkpoint 继续。旧运行若没有蓝图，仍走兼容读取路径。

开启 `generation.run_eval`（或 Web 的“生成 Eval 候选”）后，`EvalGenerator` 会在全部历史 Session 完整时，为每条 Blueprint EvalOutline 生成恰好 3 个自然文本候选并写入 `eval_candidates.json`。它自身不生成 `evidence_turn_ids`、Gold、sample ID，也不选择最佳候选。

开启 `generation.run_eval_examples`（或 Web 的“生成正式 Eval Examples”）后，流水线会自动确保候选存在，再依次执行：EvidenceResolver 从 Blueprint 指定的历史 Session 中解析最小 user-turn 证据；EvalVerifier 检查标签边界、自然度和泄露并三选一；GoldFinalizer 确定性校验 cutoff、cue、speaker 和 ID，最终写入 `eval_examples.json` 与 `benchmark.eval_samples`。任何 outline 未通过时，正式 benchmark 不写入不完整的 Eval 前缀，但 checkpoint 会保留以便续跑。

## 可选验证阶段

为了便于直接调试 SessionPlanner 和 SessionWriter，额外的产物验证阶段默认全部关闭。可在 `config.json` 的 `validation` 中分别开启：

- `structure`：确定性的 turn 数量、ID、round ID 和 speaker 顺序检查，失败时调用 Reviser；
- `semantic`：调用 SessionVerifier 做语义检查，失败时调用 Reviser；
- `naturalness`：调用 NaturalnessChecker，并按需做一次最小修订；
- `qa`：生成最终的非 LLM `qa_report.json`。

Web 页面左侧的“验证阶段（可选）”提供了对应的四个开关，设置只写入本次运行的配置快照，不会修改项目默认配置。即使全部关闭，LLM 输出仍必须通过 Pydantic 数据模型解析，以确保后续代码可以读取生成结果。

## 生成审计日志

每次运行会在输出目录的 `logs/` 下保存可审阅的阶段记录：`00_original_case_spec.json` 保存原始 spec 与配置；`01_dataset_blueprint_*.json` 保存全局蓝图；`02_planner_*.json` 保存分批 SessionPlanner；`sessions/<session_id>/` 保存 Writer 和实际开启的验证阶段；`99_pipeline_result.json` 保存最终结果。日志不记录 API key。
