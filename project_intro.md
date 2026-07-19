## 1. 目标

实现一个可运行的数据生成程序，调用llm api来生成用户和agent的长期对话：

数据集简介：
数据集包含若干个角色
每个角色生成（这些参数都应可以调整）：
- 10 个按时间排列的历史 session；
- 每个 session 4–6 个 round，每个 round 包含一条 user turn 和一条 assistant turn；
- 6 个 eval sample；
- 一份自动 QA 结果；
- 一份最终 benchmark JSON。

---

## 2. 主任务定义

输入：

- 用户长期多 session 历史；
- 当前文本、图片或图文输入（本版本先实现文本，多模态功能将于后面加入）；
- 当前输入中的候选 cue。

模型需要输出：

```json
{
  "trigger_label": "triggered | not_triggered | insufficient_evidence",
  "trigger_cue_id": "cue_xxx | none",
  "evidence_turn_ids": ["..."],
  "current_emotion": "fear | anxiety | sadness | anger | comfort | nostalgia | happiness | neutral"
}
```

触发成立必须同时满足：

1. cue 出现在当前输入中；
2. 历史中存在 cue 与个人情绪事件的关联证据；
3. 该历史记忆能够解释用户当前的情绪或行为反应。

标签判断顺序：

1. 当前没有 cue，或没有情绪/行为反应：`not_triggered`；
2. 当前有 cue 和反应，但历史中缺少个人记忆证据：`insufficient_evidence`；
3. 当前有 cue、历史有证据，且历史能够解释当前反应：`triggered`；
4. 当前情绪明确来自其他现实事件：`not_triggered`。
（后续多模态版本可以需要考虑如果用户只发送一张图片，此时agent应该如何反应）

Gold 一致性要求：

- `triggered`：cue 不能为 `none`，evidence 不能为空；
- 其余两个标签：cue 必须为 `none`，evidence 必须为空；
- evidence 只能引用真实存在的 user turn；
- evidence 应形成最小、联合充分的证据集合。

---

## 3. 总体生成流程

```text
加载固定 CaseSpec（这个casespece，可以是人工写出的，也可以是机器生成的，作为后续生成情绪事件的依据）
    ↓
程序自动分配 session_id / eval outline_id
    ↓
DatasetBlueprintPlanner 一次性生成全局生活锚点、emotion memory map、cue seeds、session slots 和 eval outlines
    ↓
生成并校验 10 个 SessionPlan
    ↓
逐 Session 生成 user 与固定朋友的对话（JSON speaker 仍使用 assistant）
    ↓
自然度检查（检查对话风格、语气是否自然，比如只修改指定的两处表达；不得改变事件、时间、人物、cue）
    ↓
EvalGenerator 为每条 eval outline 生成 3 个当前输入候选
    ↓
输出 generator-only 的 eval_candidates.json
    ↓
Resolver 解析 evidence_turn_ids，Verifier 选优，Finalizer 输出正式 Gold
    ↓
输出最终数据和 QA 结果
```

所有模型输出均使用结构化 JSON（session plan也需要单独生成一份纯文本版本的方便检查）并通过数据模型校验。每个生成步骤失败时最多重试 3 次。

---

## 4. 核心数据对象

代码中需要定义以下数据对象，具体可使用 Pydantic 实现。

### 4.1 CaseSpec

保存一个角色的唯一事实源。最简输入只需 `name` 与 `core_emotional_event`；身份、日常场景、兴趣、说话风格和旧版 session outline 都是可选种子。Session/Eval 数量和 ID 不由 CaseSpec 手写，而由配置与程序确定。

### 4.2 DatasetBlueprint

位于 CaseSpec 之后、SessionPlanner 之前，只生成一次，包含：

- 稳定的 life anchor；
- emotion memory map，以及 object / scene / utterance 等用户特定 cue；
- 程序预先分配 ID 的全部 SessionSlot；
- 只描述覆盖目标的 EvalOutline。

`session_id` 不由人工输入，也不由 LLM 自由生成。程序先按 case 和顺序确定 ID，BlueprintPlanner 与后续 Planner 只能复制。蓝图阶段不生成最终 current input、真实 evidence turn ID 或 Gold。

精简后的 `EmotionMemory` 包含 `memory_id`、`event_summary`、`emotion`、`emotional_meaning` 和 `cue_seeds`。每个 cue 只保留 `cue_id`、类型、核心形式、相关形式和合并后的个人意义。历史情绪强度不单独量化：它对当前生成和标签判定没有稳定作用，记忆随时间的变化由 SessionSlot 的 `target_emotion` 与 `relative_to_past` 表达。

SessionSlot 只保留后续阶段实际消费的记忆角色、memory/cue 引用、证据目标、局部情绪、相对过去的变化和前置 Session 依赖。覆盖数量与必须出现的 cue 类型由 `blueprint_constraints` 确定，程序负责 ID、数量、顺序和引用一致性，LLM 负责具体生活与情绪语义。

### 4.3 SessionPlan

在 Blueprint 固定的槽位上补充日期、主题、场景、聊天动机、生活支线、story beat 和目标 round 数，并逐字继承该槽位的 memory role、memory/cue 引用、证据目标、局部情绪与依赖。

### 4.4 Session

包含：

- session ID、日期和主题；
- 按顺序排列的 turns；
- session 摘要；

每个 turn 包含 speaker、text、turn ID、round ID，以及可选的图片 caption 和图片生成 prompt（图片将在后续版本加入）。

### 4.5 EvalSample

包含：

- 当前输入类型、文本和图片描述（图片将于后续版本加入）；
- cue options；
- history cutoff；
- 四个正式 gold 字段。

`history cutoff` 用于控制模型在某条 eval 中能够看到哪些 session，尤其用于 Case B 的记忆更新测试。

---

# 5. Generator 设计
## 5.0 DatasetBlueprintPlanner

功能：一次性决定整组数据的全局覆盖，防止分批 SessionPlanner 各自规划后出现 10/10 普通日常或重复核心事件。

输入：精简 CaseSpec、不可变事实、程序分配的 ID、覆盖数量。

输出：`dataset_blueprint.json`。默认 10 个 Session 分配 6 个 none、1 个 encode_association、1 个 triggered_recall、1 个 memory_update、1 个 control；默认 6 个 eval outline 为三类标签各 2 个。

## 5.1 SessionPlanner

功能：在 DatasetBlueprint 的既定 SessionSlot 上补充日期、主题、具体场景、聊天动机、生活支线和故事节拍。

输入：CaseSpec + DatasetBlueprint + 本批 assigned SessionSlots + prior plans

职责：
- 不做宏观覆盖决策，不生成或修改 ID；
- 指定每个 session 的自然生活主题和局部节拍；
- 原样保留蓝图的 memory role、cue、证据目标与情绪变化。

不得改变核心事件

示例：
| A-S1 | 赶展览作业，讨论展签和排版 | 建立普通工作背景 |
| A-S2 | 回家整理厨房，提到奶奶常用的蓝色搪瓷杯 | 编码 cue 与人物关联，不提去世 |
....
| A-S9 | 用户吃到奶奶以前常给的桂花糖 | 以 nostalgia 为主 |
| A-S10 | 工作文件损坏，桌面有普通蓝杯子 | 情绪来自工作，不触发记忆 |

SessionWriter、SessionVerifier、SessionReviser

## 5.2 SessionWriter

功能：负责一次生成完整 session

## 5.3 SessionVerifier

功能：只做检查，不改文本。

检查：
assistant 是否知道了未披露信息；
人物身份是否冲突；
是否形成机械问答；
是否重复历史；
user 是否有可用 evidence turn；
session 是否像自然对话。

## 5.4 SessionReviser

功能：根据具体问题做最小修改。
例如 verifier 输出：

result: revise
issues:
  - turn: s4_t06
    type: assistant_overclaim
    description: assistant声称“你一定很后悔”，语气过度确定
  - turn: s4_t09
    type: repetition
    description: 重复了用户在s4_t05已经表达的事实

Reviser 只修改相关 turn，不重写整个 session。

## 5.5 EvalGenerator

功能：根据固定 eval outline 和 cutoff 内的真实历史，生成自然的当前文本与 cue options。多模态在后续版本加入。

要求：

根据覆盖目标决定题型；
严格使用 Blueprint 已确定的 history_cutoff；
读取 cutoff 内历史以避免与事实或已有证据矛盾；
生成当前文本输入；
生成 cue options；
一次生成 3 个候选 eval。

当前阶段只输出 `outline_id` 和 3 个候选。每个候选包含 `current_input`，以及由程序锁定的 `target_label`、`target_emotion`、`blueprint_cue_id`、`history_cutoff`。Generator 不输出 `evidence_turn_ids`、Gold、sample ID，不负责排序或选择候选。

输入：
CaseSpec
+ 已生成的 sessions
+ 本条 eval 的覆盖要求

覆盖要求由程序提前指定，例如：

target_label: triggered
target_emotion: sadness
modality: text/text_image（后续添加图片）
cue_specificity: exact
emotion_explicitness: implicit
history_cutoff: s10
Generator 不负责自由决定“想出什么题”，而是按照这个目标出题。

## 5.6 EvidenceResolver

从真实 user turns 中解析最小且联合充分的 `evidence_turn_ids`，不改写 Generator 候选。

## 5.7 EvalVerifier
Verifier 不生成新题，也不负责大幅重写，只完成两件事：

检查每个候选；
选择最佳候选，或者全部拒绝并要求 Generator 重试。

具体检查规则写入文件

## 5.8 GoldFinalizer
这个组件不需要 LLM，就是普通代码。

它负责：

检查 turn ID 是否存在；
检查 speaker 是否为 user；
检查 cue ID 是否存在于 options；
对 cue options 进行随机排序；
根据标签强制修正空值规则；
分配 sample ID；
删除调试字段；
输出最终 JSON。

它不判断语义，只保证最终数据结构正确。

---

# 10. 模型调用与配置

实现统一的 LLMClient，支持：

- 结构化输出；
- 按模块配置模型和 temperature；
- 从环境变量读取 API key；
- 调用失败重试；

模型名称、temperature、随机 seed、最大重试次数和上下文 session 数量均从配置读取，不得写死在业务逻辑中。

保存必要的调用元数据和错误信息，不保存模型隐藏推理。

---

# 11. 最终输出

每个案例最终输出一个 benchmark JSON，至少包含：

```json
{
  "dataset_id": "...",
  "character_profile": {},
  "dialogues": [],
  "eval_samples": []
}
```

Generator 候选单独保存在 `eval_candidates.json`。只有经过 Resolver、Verifier、Finalizer 的完整样本才写入 `eval_examples.json` 和 benchmark 的 `eval_samples`；正式样本还必须保存 `history_cutoff`，用于评测时截断可见历史。
