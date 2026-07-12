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
生成并校验 10 个 SessionPlan
    ↓
逐 session 生成 user 和 assistant 对话（SessionWriter、SessionVerifier、SessionReviser）
    ↓
自然度检查（检查对话风格、语气是否自然，比如只修改指定的两处表达；不得改变事件、时间、人物、cue）
    ↓
生成 6 个 eval sample
    ↓
解析具体 evidence_turn_ids
    ↓
执行 gold、cue、历史截断和泄露检查
    ↓
输出最终数据和 QA 结果
```

所有模型输出均使用结构化 JSON（session plan也需要单独生成一份纯文本版本的方便检查）并通过数据模型校验。每个生成步骤失败时最多重试 3 次。

---

## 4. 核心数据对象

代码中需要定义以下数据对象，具体可使用 Pydantic 实现。

### 4.1 CaseSpec

保存一个角色的唯一事实源：

- 角色身份、日常场景、兴趣和说话风格；
- 核心情绪事件；
- 10 个固定 session outline；
- 6 个固定 eval outline。

### 4.2 SessionPlan

轻量计划即可，至少包含：

- session ID、日期和主题；
- 本 session 的 story beat；
- 目标 round 数。

### 4.3 Session

包含：

- session ID、日期和主题；
- 按顺序排列的 turns；
- session 摘要；

每个 turn 包含 speaker、text、turn ID、round ID，以及可选的图片 caption 和图片生成 prompt（图片将在后续版本加入）。

### 4.4 EvalSample

包含：

- 当前输入类型、文本和图片描述（图片将于后续版本加入）；
- cue options；
- history cutoff；
- 四个正式 gold 字段。

`history cutoff` 用于控制模型在某条 eval 中能够看到哪些 session，尤其用于 Case B 的记忆更新测试。

---

# 5. Generator 设计
## 5.1 SessionPlanner

功能：将固定 CaseSpec 扩展成轻量 SessionPlan。

输入：CaseSpec

职责：
- 宏观控制
- 指定每个session 需要写入的主题；
- 保持固定故事功能不变。

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

功能：根据固定 eval outline 生成自然的当前输入、图片描述（多模态在后续版本加入）和 cue options。

要求：

根据覆盖目标决定题型；
选择 history_cutoff；
从历史中寻找可用 evidence；
生成当前文本或图文输入；
生成 cue options；
给出预期的 gold；
一次生成 3 个候选 eval。

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

## 5.6 EvalVerifier
Verifier 不生成新题，也不负责大幅重写，只完成两件事：

检查每个候选；
选择最佳候选，或者全部拒绝并要求 Generator 重试。

具体检查规则写入文件

## 5.7 GoldFinalizer
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

