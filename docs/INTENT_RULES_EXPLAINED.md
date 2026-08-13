# 意图识别规则理解

## 入口函数

`classify_message_with_llm()` 是对外使用的意图分类入口。

流程：

```text
先用规则分类
如果规则结果明确，直接返回
如果规则结果是 ambiguous，且启用了 LLM，则交给 LLM 复判
如果没有 LLM 或 LLM 调用失败，保持 ambiguous
```

对应代码：

```python
decision = classify_message_rule_based(user_message)
if decision.kind != MessageKind.AMBIGUOUS:
    return decision.kind
if not settings.llm.enabled:
    return MessageKind.AMBIGUOUS
try:
    return _classify_message_llm(user_message, settings, history)
except Exception:
    return MessageKind.AMBIGUOUS
```

设计目的：规则能确定的场景不浪费 LLM；规则不确定的场景再让 LLM 判断；失败时保守处理，不误触发旅行工具链。

## IntentDecision

`IntentDecision` 是规则分类的返回结果。

```python
@dataclass(frozen=True)
class IntentDecision:
    kind: MessageKind | None
    confidence: Literal["high", "uncertain"]
    reason: str
```

字段含义：

```text
kind        消息类型
confidence 判断置信度
reason      命中原因，方便调试
```

例如：

```python
IntentDecision(MessageKind.TRAVEL, "high", "strong_travel")
```

表示：这句话被高置信度判断为旅行需求，原因是命中了强旅行信号。

## 消息类型

`MessageKind` 定义了四类消息：

```python
class MessageKind(str, Enum):
    GREETING = "greeting"
    TRAVEL = "travel"
    AMBIGUOUS = "ambiguous"
    OUT_OF_SCOPE = "out_of_scope"
```

含义：

```text
GREETING      纯寒暄
TRAVEL        明确旅行需求，可以进入工具链
AMBIGUOUS     可能和旅行有关，但需要确认或 LLM 复判
OUT_OF_SCOPE  明确非旅行需求
```

## 规则分类流程

`classify_message_rule_based()` 按顺序判断：

```python
if not text:
    return OUT_OF_SCOPE
if is_pure_greeting(text):
    return GREETING
if _has_strong_non_travel_signal(text):
    return OUT_OF_SCOPE
if _has_strong_travel_signal(text):
    return TRAVEL
if _has_weak_travel_signal(text):
    return AMBIGUOUS
return OUT_OF_SCOPE
```

顺序很重要：

```text
空消息 -> 非旅行
纯问候 -> 寒暄
强非旅行信号 -> 非旅行
强旅行信号 -> 旅行
弱旅行信号 -> 模糊
都没有 -> 非旅行
```

## 强旅行信号

`_has_strong_travel_signal()` 用来判断一句话是否足够明确是旅行需求。

第一步会抽取结构化信息：

```python
extracted = extract_profile_rule_based(text)
```

`extracted` 是一个 `TravelProfile`，里面可能有：

```text
destination 目的地
days        天数
interests   兴趣
budget      预算
pace        节奏
```

强旅行判断规则：

```python
if extracted.destination and extracted.days:
    return True
```

同时有目的地和天数，直接认为是旅行需求。

例子：

```text
杭州三天
帮我规划上海两天
北京玩3天
```

---

```python
if (
    extracted.destination
    and _GO_PREFIX_RE.search(text)
    and not _looks_like_non_destination(extracted.destination)
):
    return True
```

有目的地，并且原文出现“想去 / 要去 / 准备去 / 打算去 / 去”，且目的地不像误抽结果。

例子：

```text
想去杭州
准备去成都
打算去西安看看
```

`_looks_like_non_destination()` 用来避免误判：

```text
想去吃火锅
```

这里“吃火锅”不应该被当成目的地。

---

```python
if extracted.destination and _STRONG_TRAVEL_RE.search(text):
    return True
```

有目的地，并且出现强旅行词。

例子：

```text
杭州旅游
成都攻略
上海自由行
北京路线
```

---

```python
if extracted.days and (_STRONG_TRAVEL_RE.search(text) or _PLAN_ACTION_RE.search(text)):
    return True
```

有天数，并且出现旅行词或规划动作。

例子：

```text
三天旅游
两天行程
帮我规划三天
安排一个两天路线
```

---

```python
if extracted.days and _DAY_ACTIVITY_RE.search(text):
    return True
```

有天数，并且文本像“玩几天 / 游几天 / 逛几天”。

例子：

```text
想玩三天
周末逛两天
```

---

```python
return bool(_GO_DEST_RE.search(text) and _DAY_RE.search(text))
```

最后兜底：同时满足“去某地玩”和“天数表达”，也算强旅行需求。

例子：

```text
我想去杭州玩三天
周末准备去上海逛逛
打算去成都看看两天
```

`bool()` 是把正则匹配结果转换成 `True` 或 `False`。

## 弱旅行信号

`_has_weak_travel_signal()` 用来判断一句话是否有旅行相关可能，但不够明确。

规则：

```python
if extracted.destination or extracted.days:
    return True
```

单独出现目的地或天数，算弱信号。

```text
杭州
三天
```

---

```python
if extracted.interests or extracted.budget_level or extracted.companions:
    return True
```

出现兴趣、预算、同行人，也算弱信号。

```text
喜欢美食
低预算
情侣
亲子
```

---

```python
if extracted.pace != "standard" or extracted.must_visit or extracted.avoid:
    return True
```

出现节奏、必去、避开偏好，也算弱信号。

```text
不要太累
轻松一点
尽量多玩
避开人多的地方
```

---

```python
if any(alias in text for alias in CITY_ALIASES):
    return True
```

文本包含城市别名，也算弱信号。

```text
上海
Beijing
杭州
```

---

```python
return any(hint in text for hint in _WEAK_TRAVEL_HINTS)
```

最后检查弱旅行关键词。

例子：

```text
美食
酒店
机票
景点
目的地
去哪
推荐
适合
出去走走
```

弱信号不会直接进入规划，而是返回 `AMBIGUOUS`。

## 目的地提取

目的地提取入口是 `workflow_rules.py` 里的 `_extract_destination()`。

整体顺序：

```text
匹配结构化字段 “目标位置 xxx”
查城市别名表
匹配自然语言正则
清洗候选目的地
过滤无效词
找不到则返回 None
```

### 结构化字段

```python
metadata_match = re.search(r"目标位置\s*([^\],，,\s]+)", text)
```

匹配：

```text
目标位置 杭州
目标位置上海
```

### 城市别名

```python
for alias, city in CITY_ALIASES.items():
    if alias in text:
        return city
```

`alias` 是用户可能输入的写法，`city` 是系统内部统一城市名。

例子：

```text
Hangzhou -> 杭州
Beijing  -> 北京
```

### 自然语言规则

```python
r"(?:规划|安排|做|制定|生成)\s*([^\s，,。！!？?；;]+?)(?:\d+\s*天|[一二两三四五六七]\s*天|行程|旅行|旅游|游玩|攻略|路线)"
```

匹配：

```text
规划苏州三天
安排南京行程
做青岛攻略
```

---

```python
r"([一-龥A-Za-z][一-龥A-Za-z·'’]{1,20}?)(?:\d+\s*天|[一二两三四五六七]\s*天)(?:行程|旅行|旅游|游玩|攻略|路线)?"
```

匹配：

```text
苏州三天
南京2天
Xi'an三天行程
```

---

```python
r"想去\s*([^\s，,。！!？?；;]+)"
```

匹配：

```text
想去南京
想去上海迪士尼
```

---

```python
r"去\s*([^\s，,。！!？?；;]{2,10}?)(?:玩|旅|游|出差|看看|度假)?"
```

匹配：

```text
去南京玩
去青岛看看
去重庆度假
```

## 目的地清洗

`_clean_destination()` 删除目的地末尾的标点、“的”和旅行描述词。

```python
cleaned = destination.strip(" ，,。！!？?；;的")
```

先去掉首尾空格、标点和“的”。

```python
while changed:
    for word in _DESTINATION_STOPWORDS:
        if cleaned.endswith(word):
            cleaned = cleaned[: -len(word)].strip(" 的")
            changed = True
```

只删除**结尾处**的停用词，不做全文替换。

例如：

```text
杭州三天       -> 杭州
上海旅游攻略   -> 上海
北京的行程     -> 北京
成都自由行     -> 成都
```

因为是循环，所以可以连续删尾巴：

```text
杭州旅游路线
-> 杭州旅游
-> 杭州
```

但不会删除开头或中间的词：

```text
旅游杭州
杭州旅游附近
```

如果结尾不是停用词，就不会被这段逻辑删除。

## 天数提取

天数提取函数是 `_extract_days()`。

支持：

```text
旅行天数 3
3天
三天
两天
七天
```

阿拉伯数字：

```python
match = re.search(r"(\d+)\s*天", text)
```

中文数字：

```python
{
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
}
```

例子：

```text
杭州3天  -> 3
上海两天 -> 2
北京一天 -> 1
```

## LLM 分类调用链

当规则结果是 `AMBIGUOUS`，并且配置里启用了 LLM，会进入 `_classify_message_llm()`。

核心代码：

```python
response = model.invoke(
    [
        SystemMessage(content=_INTENT_LLM_SYSTEM),
        HumanMessage(content=prompt),
    ]
)
```

完整系统提示词是：

```text
你是旅行规划 Agent 的意图分类器。只判断当前用户输入是否需要进入旅行规划工具链。

只输出 JSON 对象，不要输出 Markdown。格式：
{"kind":"greeting|travel|ambiguous|out_of_scope","confidence":"high|medium|low","reason":"简短原因"}

分类原则：
- greeting：纯寒暄。
- travel：用户明确表达出行、旅行、游玩、目的地探索、路线/攻略/住宿/餐饮安排等旅行相关需求。
- ambiguous：有旅行相关弱信号或可能是旅行场景，但当前表达不足以确定要进入规划工具链。
- out_of_scope：普通吃饭、预算管理、行业分析、编程、新闻、知识问答、情感闲聊等明确非出行需求。
- 不要因为城市名、酒店、预算、美食等单词单独出现就判 travel。
- 模糊但明显有“想出去/找地方玩/周末放松/情侣去哪”倾向时判 ambiguous。
- 如果不确定，选择 ambiguous。
```

这里的 `model` 来自 `runtime.py`：

```python
ChatOpenAI(
    model=settings.llm.model,
    api_key=settings.llm.api_key,
    base_url=settings.llm.base_url,
    temperature=settings.llm.temperature,
    timeout=settings.llm.timeout_seconds,
)
```

所以这条链路是：

```text
intent.py
-> _build_chat_model(settings)
-> ChatOpenAI
-> model.invoke(...)
-> OpenAI-compatible API
-> 大模型服务
```

LLM 期望返回的解析规则：

```python
content = response.content if isinstance(response.content, str) else str(response.content)
payload = json.loads(content)
if not isinstance(payload, dict):
    return MessageKind.AMBIGUOUS
try:
    return MessageKind(str(payload.get("kind", "")).lower())
except ValueError:
    return MessageKind.AMBIGUOUS
```

示例返回（`model.invoke` 的 `content`）：

```json
{"kind":"travel","confidence":"high","reason":"输入包含明确出行与行程规划意图"}
```

```json
{"kind":"ambiguous","confidence":"medium","reason":"有旅行相关弱信号但信息不足"}
```

```json
{"kind":"greeting","confidence":"high","reason":"纯问候"}
```

```json
{"kind":"out_of_scope","confidence":"high","reason":"非出行领域请求"}
```

如果不是合法 JSON、不是对象、或 `kind` 不是四个枚举之一，会回退为 `AMBIGUOUS`。

`SystemMessage` 和 `HumanMessage` 会被 LangChain 转成模型 API 能理解的消息格式。

概念上类似：

```json
{
  "model": "qwen-turbo",
  "messages": [
    {
      "role": "system",
      "content": "你是旅行规划 Agent 的意图分类器..."
    },
    {
      "role": "user",
      "content": "杭州三天"
    }
  ],
  "temperature": 0.2
}
```

真实请求还可能包含其他字段，比如 `stream`、`stop`、`extra_body` 等。

如果有历史上下文，`HumanMessage(content=prompt)` 里的内容可能是：

```text
最近对话：
user: 我想去上海
assistant: 计划玩几天？

当前用户输入：三天
```

`prompt` 的构造代码：

```python
recent = "\n".join(f"{role}: {content}" for role, content in history[-4:])
prompt = user_message if not recent else f"最近对话：\n{recent}\n\n当前用户输入：{user_message}"
```

没有历史时：

```python
user_message = "杭州三天"
history = []
```

得到：

```text
杭州三天
```

有历史时：

```python
history = [
    ("user", "我想去上海"),
    ("assistant", "计划玩几天？"),
]
user_message = "三天"
```

`recent` 是：

```text
user: 我想去上海
assistant: 计划玩几天？
```

最终 `prompt` 是：

```text
最近对话：
user: 我想去上海
assistant: 计划玩几天？

当前用户输入：三天
```

再比如最近 4 条历史：

```python
history = [
    ("user", "你好"),
    ("assistant", "你好，我可以帮你规划旅行。"),
    ("user", "我想去杭州"),
    ("assistant", "计划玩几天？"),
]
user_message = "两天，轻松一点"
```

最终 `prompt` 是：

```text
最近对话：
user: 你好
assistant: 你好，我可以帮你规划旅行。
user: 我想去杭州
assistant: 计划玩几天？

当前用户输入：两天，轻松一点
```

注意：这里最多取最近 4 条历史，只用于意图分类，不代表正式旅行规划只看最近 4 条。正式规划在 `runtime.py` 的 `_run_react()` 中使用 `MemoryFramework` 处理后的上下文。

## 纯对话 LLM 消息示例

当输入被判定为寒暄、无关话题或模糊意图时，会走 `runtime.py` 里的 `_run_conversation_only()`。

如果 `settings.llm.enabled` 为真，它会继续调用 `_run_conversation_chat()`：

```python
return _run_conversation_chat(user_message, ctx, history, settings, hint, kind)
```

这条路径只生成自然语言回复，不进入旅行规划工具链。

核心代码：

```python
messages: list[Any] = [SystemMessage(content=system)]
for role, content in history[-6:]:
    messages.append(HumanMessage(content=content) if role == "user" else AIMessage(content=content))
messages.append(HumanMessage(content=user_message))

response = model.invoke(messages)
```

假设当前输入是：

```text
周末想找个地方放空一下
```

用户长期偏好 `l3_hint` 是：

```text
常去/关注 杭州；通常 3 天；偏好 美食、自然；节奏 轻松
```

最近历史对话 `history` 是：

```python
[
    ("user", "你好"),
    ("assistant", "你好！我是旅行规划助手，可以帮你查景点、看天气、排行程。"),
    ("user", "我之前比较喜欢杭州这种城市"),
    ("assistant", "记住了，你比较关注杭州这类城市，也偏好轻松一点的旅行节奏。"),
]
```

那么 `_run_conversation_chat()` 组装出的 `messages` 近似是：

```python
[
    SystemMessage(
        content=(
            CONVERSATION_SYSTEM
            + "\n\n用户历史偏好（仅供参考，勿自动开规划）："
            + "常去/关注 杭州；通常 3 天；偏好 美食、自然；节奏 轻松"
        )
    ),
    HumanMessage(content="你好"),
    AIMessage(content="你好！我是旅行规划助手，可以帮你查景点、看天气、排行程。"),
    HumanMessage(content="我之前比较喜欢杭州这种城市"),
    AIMessage(content="记住了，你比较关注杭州这类城市，也偏好轻松一点的旅行节奏。"),
    HumanMessage(content="周末想找个地方放空一下"),
]
```

概念上，模型看到的是：

```text
System:
你是旅行规划助手。你正在处理寒暄、闲聊、无关话题或意图不明确的输入。
请自然、简短地回复用户。不要调用旅行规划工具。

用户历史偏好（仅供参考，勿自动开规划）：常去/关注 杭州；通常 3 天；偏好 美食、自然；节奏 轻松

User:
你好

AI:
你好！我是旅行规划助手，可以帮你查景点、看天气、排行程。

User:
我之前比较喜欢杭州这种城市

AI:
记住了，你比较关注杭州这类城市，也偏好轻松一点的旅行节奏。

User:
周末想找个地方放空一下
```

因为“周末想找个地方放空一下”还不足以直接进入规划工具链，LLM 可能返回类似：

```text
听起来你想要一个轻松一点的短途放松安排。你是想让我按旅行方向帮你推荐目的地和行程吗？如果是，可以告诉我出发城市、预算和大概玩几天。
```

最后函数包装成：

```python
AgentReply(
    text="听起来你想要一个轻松一点的短途放松安排。你是想让我按旅行方向帮你推荐目的地和行程吗？如果是，可以告诉我出发城市、预算和大概玩几天。",
    tool_trace=[],
    used_real_agent=True,
    clarification=True,
    profile=toolkit._profile_brief(ctx.profile),
)
```

这里的关键点是：

```text
tool_trace=[]
```

也就是说，这一轮没有调用 `search_poi`、`check_weather`、`plan_route`、`plan_and_critique` 等工具。LLM 只是负责回复或追问。

## model.invoke 做了什么

`model.invoke(...)` 是 LangChain 的同步调用入口。

源码位置：

```text
.venv/lib/python3.10/site-packages/langchain_core/language_models/chat_models.py
```

核心逻辑：

```python
return self.generate_prompt(
    [self._convert_input(input)],
    ...
).generations[0][0].message
```

简化理解：

```text
输入 messages
-> 转成 PromptValue
-> 调 generate_prompt
-> 得到 LLMResult
-> 取第一个 prompt 的第一个候选回复
-> 返回 AIMessage
```

所以在本项目里：

```python
response = model.invoke([...])
```

`response` 不是 `LLMResult`，而是 `AIMessage`。

真正文本在：

```python
response.content
```

## LanguageModelInput

`LanguageModelInput` 是 LangChain 允许传给模型的输入类型。

常见三种：

```text
str
PromptValue
消息列表
```

对应 `_convert_input()`：

```python
if isinstance(model_input, PromptValue):
    return model_input
if isinstance(model_input, str):
    return StringPromptValue(text=model_input)
if isinstance(model_input, Sequence):
    return ChatPromptValue(messages=convert_to_messages(model_input))
```

含义：

```text
PromptValue -> 原样返回
str         -> StringPromptValue
消息列表     -> ChatPromptValue
其他类型     -> ValueError
```

本项目传的是消息列表：

```python
[
    SystemMessage(content=_INTENT_LLM_SYSTEM),
    HumanMessage(content=prompt),
]
```

它会被转成：

```python
ChatPromptValue(
    messages=[
        SystemMessage(...),
        HumanMessage(...),
    ]
)
```

## RunnableConfig

`invoke()` 里有：

```python
config = ensure_config(config)
```

`config` 是 LangChain 的运行配置，不是模型参数。

常见字段：

```text
tags             调用标签，方便 trace / 过滤
metadata         额外元信息，比如 user_id、session_id
callbacks        回调，用于日志、监控、LangSmith tracing
run_name         本次运行名称
run_id           本次运行 ID
recursion_limit  Agent / Runnable 递归调用上限
max_concurrency  批量调用最大并发数
configurable     可配置 Runnable 的运行时参数
```

例子：

```python
model.invoke(
    messages,
    config={
        "tags": ["intent"],
        "metadata": {"scene": "intent_classify"},
        "run_name": "intent_classifier",
    },
)
```

本项目的 `_classify_message_llm()` 没传 `config`，所以使用默认配置。

模型名、API key、温度、超时这些是在创建 `ChatOpenAI` 时传入，不是通过 `RunnableConfig` 传入。

## LLMResult

`generate_prompt()` 返回 `LLMResult`，结构核心是：

```python
LLMResult(
    generations=[
        [
            ChatGeneration(
                message=AIMessage(content="...")
            )
        ]
    ],
    llm_output={...},
    run=[...],
)
```

`generations` 是二维列表：

```text
第一维：第几个输入 prompt
第二维：该 prompt 的第几个候选回复
```

例如批量处理三个输入：

```python
generations = [
    [ChatGeneration(message=AIMessage(content="第一个回答"))],
    [ChatGeneration(message=AIMessage(content="第二个回答"))],
    [ChatGeneration(message=AIMessage(content="第三个回答"))],
]
```

`invoke()` 只处理单个输入，所以取：

```python
generations[0][0].message
```

也就是第一个输入的第一个候选回复。

在当前意图分类场景里，`AIMessage.content` 可能是：

```json
{"kind":"travel","confidence":"high","reason":"包含目的地和天数"}
```

外层的 `LLMResult`、`ChatGeneration`、`AIMessage` 是 LangChain 的通用包装；里面的 JSON 才是本项目要求模型输出的业务分类结果。

## OpenAI-compatible API

`ChatOpenAI` 不只能调用 OpenAI，也可以调用兼容 OpenAI Chat Completions 格式的模型服务。

关键在：

```python
base_url=settings.llm.base_url
```

如果 `base_url` 指向 OpenAI，就调用 OpenAI。
如果指向 DashScope、DeepSeek、本地 vLLM 等兼容服务，就调用对应服务。

这不是所有模型天然统一，而是很多厂商主动兼容了 OpenAI 风格的接口。

常见差异：

```text
模型名不同
base_url 不同
鉴权方式可能不同
tool calling 支持程度不同
stream 格式可能不同
JSON mode / structured output 支持不同
reasoning 参数不同
```

所以工程上要记住：

```text
OpenAI-compatible 不等于 100% OpenAI-compatible
```

## MCP 和模型调用的区别

MCP 是 `Model Context Protocol`，用于标准化暴露工具、资源和上下文。

它不是大模型 API，也不是 `model.invoke()` 的必经路径。

本项目里模型调用链路是：

```text
Agent / intent.py
-> LangChain ChatOpenAI
-> OpenAI-compatible API
-> 大模型服务
```

MCP 工具调用链路是：

```text
Agent
-> MCP client
-> MCP server
-> toolkit.py 中的工具函数
```

可以这样区分：

```text
model.invoke(...)
  调用大模型，让模型理解、分类或生成

MCP tools/call
  调用外部工具，让系统查数据或执行动作
```

本项目的 MCP server 在：

```text
src/travel_agent/mcp_server.py
```

它把这些旅行工具暴露出去：

```text
search_poi
check_weather
plan_route
plan_and_critique
render_map
```

底层工具实现仍然在：

```text
src/travel_agent/agent/toolkit.py
```

一句话：

```text
LLM API 负责“调用大模型”，MCP 负责“调用工具”。
```

## 总结

这套规则的核心目标是：

```text
明确旅行需求 -> 进入规划工具链
可能旅行相关 -> 标记 ambiguous，必要时让 LLM 复判或追问
明确无关内容 -> 不触发工具链
```

目的地和天数提取靠规则完成，强旅行意图靠“目的地、天数、旅行词、去玩表达”的组合判断。
