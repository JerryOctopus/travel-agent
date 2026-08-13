"""用户消息意图判断：只有明确出行需求才进入规划工具链。

问候、模糊旅行倾向、明确无关问题统一走对话路径，不触发 search_poi / plan_and_critique。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

from travel_agent.schemas import TravelProfile
from travel_agent.workflow_rules import CITY_ALIASES, extract_profile_rule_based

if TYPE_CHECKING:
    from travel_agent.settings import Settings

_GREETING_RE = re.compile(
    r"^(你好|您好|嗨|哈喽|在吗|hello|hi|hey|早上好|下午好|晚上好)[!！。,，\s]*$",
    re.IGNORECASE,
)

_STRONG_TRAVEL_RE = re.compile(
    r"("
    r"旅行|旅游|出游|度假|出去玩|去玩|"
    r"一日游|两日游|二日游|三日游|周末游|自由行|自驾游|"
    r"行程|路线|攻略"
    r")",
    re.IGNORECASE,
)
_PLAN_ACTION_RE = re.compile(r"(规划|安排|制定|生成|做).{0,12}(行程|路线|攻略|旅行|旅游|游玩)?")
_DAY_RE = re.compile(r"(\d+\s*天|[一二两三四五六七]\s*天|一日|两日|二日|三日|周末)")
_GO_DEST_RE = re.compile(r"(想去|要去|准备去|打算去|去).{1,16}(玩|旅游|旅行|度假|看看|逛逛)")
_GO_PREFIX_RE = re.compile(r"(想去|要去|准备去|打算去|去)\s*")
_DAY_ACTIVITY_RE = re.compile(r"(玩|游|逛).{0,8}" + _DAY_RE.pattern)

_STRONG_NON_TRAVEL_PATTERNS = (
    re.compile(r"(奥运会|金牌榜|世界杯|比赛|新闻|股票|基金|代码|python|java|数学|算术)", re.IGNORECASE),
    re.compile(r"(行业分析|市场分析|竞品分析|财报|论文|报告)"),
    re.compile(r"(表格|excel|记账|报销|预算管理|怎么做预算)"),
    re.compile(r"(职业规划|人生规划|城市规划|项目规划|学习路线|技术路线)"),
    re.compile(r"(头像|文案|取名|起名|翻译|写代码|会写)"),
)

_WEAK_TRAVEL_HINTS = (
    "吃",
    "美食",
    "预算",
    "酒店",
    "民宿",
    "情侣",
    "亲子",
    "散步",
    "拍照",
    "放空",
    "出去走走",
    "景点",
    "目的地",
    "机票",
    "去哪",
    "去哪儿",
    "推荐",
    "适合",
    "本地特色",
)

_INTENT_LLM_SYSTEM = """你是旅行规划 Agent 的意图分类器。只判断当前用户输入是否需要进入旅行规划工具链。

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
"""


class MessageKind(str, Enum):
    GREETING = "greeting"  # 纯寒暄
    TRAVEL = "travel"  # 有出行/规划意图，可进入工具链
    AMBIGUOUS = "ambiguous"  # 可能与旅行相关，先轻量澄清
    OUT_OF_SCOPE = "out_of_scope"  # 明确无关话题


@dataclass(frozen=True)
class IntentDecision:
    kind: MessageKind | None
    confidence: Literal["high", "uncertain"]
    reason: str


def is_pure_greeting(user_message: str) -> bool:
    text = user_message.strip()
    if not text:
        return False
    return bool(_GREETING_RE.match(text))


def classify_message(user_message: str) -> MessageKind:
    """规则版分类：不确定时归为 ambiguous，保证离线路径能轻量澄清。"""
    decision = classify_message_rule_based(user_message)
    return decision.kind


def classify_message_rule_based(
    user_message: str,
    extracted_profile: TravelProfile | None = None,
) -> IntentDecision:
    """规则优先判断；弱旅行信号归为 ambiguous，留给 LLM 复判。"""
    text = user_message.strip() # 把消息前后空格、换行、\t、\n 这些首尾空白字符删掉
    if not text:
        return IntentDecision(MessageKind.OUT_OF_SCOPE, "high", "empty")
    if is_pure_greeting(text):
        return IntentDecision(MessageKind.GREETING, "high", "pure_greeting")
    if _has_strong_non_travel_signal(text):
        return IntentDecision(MessageKind.OUT_OF_SCOPE, "high", "strong_non_travel")
    extracted = extracted_profile or extract_profile_rule_based(text)
    if _has_strong_travel_signal(text, extracted):
        return IntentDecision(MessageKind.TRAVEL, "high", "strong_travel")
    if _has_weak_travel_signal(text, extracted):
        return IntentDecision(MessageKind.AMBIGUOUS, "uncertain", "weak_travel_signal")
    return IntentDecision(MessageKind.OUT_OF_SCOPE, "high", "no_travel_signal")


def classify_message_with_llm(
    user_message: str,
    settings: Settings,
    history: list[tuple[str, str]] | None = None,
    evaluation_trace: list[dict[str, Any]] | None = None,
) -> MessageKind:
    """混合分类：规则高置信直出；ambiguous 场景交给 LLM，失败仍保持 ambiguous。"""
    decision = classify_message_rule_based(user_message)
    if decision.kind != MessageKind.AMBIGUOUS:
        return decision.kind
    if not settings.llm.enabled:
        return MessageKind.AMBIGUOUS
    try:
        if evaluation_trace is None:
            return _classify_message_llm(user_message, settings, history)
        return _classify_message_llm(user_message, settings, history, evaluation_trace)
    except Exception:
        return MessageKind.AMBIGUOUS


def has_travel_intent(user_message: str) -> bool:
    """当前这句话是否应进入规划工具链。"""
    return classify_message(user_message) == MessageKind.TRAVEL


def _has_strong_travel_signal(text: str, extracted: TravelProfile) -> bool:
    if extracted.destination or extracted.days:
        if extracted.destination and extracted.days:
            return True
        if (
            extracted.destination
            and _GO_PREFIX_RE.search(text)
            and not _looks_like_non_destination(extracted.destination)
        ):
            return True
        if extracted.destination and _STRONG_TRAVEL_RE.search(text):
            return True
        if extracted.days and (_STRONG_TRAVEL_RE.search(text) or _PLAN_ACTION_RE.search(text)):
            return True
        if extracted.days and _DAY_ACTIVITY_RE.search(text):
            return True
    if _STRONG_TRAVEL_RE.search(text):
        return True
    if _PLAN_ACTION_RE.search(text) and (extracted.destination or extracted.days):
        return True
    return bool(_GO_DEST_RE.search(text) and _DAY_RE.search(text))


def _has_strong_non_travel_signal(text: str) -> bool:
    return any(pattern.search(text) for pattern in _STRONG_NON_TRAVEL_PATTERNS)


def _has_weak_travel_signal(text: str, extracted: TravelProfile) -> bool:
    if extracted.destination or extracted.days:
        return True
    if extracted.interests or extracted.budget_level or extracted.companions:
        return True
    if extracted.pace != "standard" or extracted.must_visit or extracted.avoid:
        return True
    if any(alias in text for alias in CITY_ALIASES):
        return True
    return any(hint in text for hint in _WEAK_TRAVEL_HINTS)


def _looks_like_non_destination(destination: str) -> bool:
    if destination.startswith(("哪", "哪里", "哪儿")):
        return True
    return any(
        word in destination
        for word in (
            "火锅",
            "烧烤",
            "奶茶",
            "咖啡",
            "餐厅",
            "饭店",
            "吃饭",
            "吃",
            "头像",
            "表格",
            "代码",
        )
    )


def _classify_message_llm(
    user_message: str,
    settings: Settings,
    history: list[tuple[str, str]] | None = None,
    evaluation_trace: list[dict[str, Any]] | None = None,
) -> MessageKind:
    from langchain_core.messages import HumanMessage, SystemMessage

    from travel_agent.agent.runtime import _build_chat_model

    history = history or []
    recent = "\n".join(f"{role}: {content}" for role, content in history[-4:])
    prompt = user_message if not recent else f"最近对话：\n{recent}\n\n当前用户输入：{user_message}"
    model = _build_chat_model(settings)
    callbacks = None
    if evaluation_trace is not None:
        from travel_agent.agent.evaluation_trace import EvaluationTraceCallback

        callbacks = [
            EvaluationTraceCallback(
                evaluation_trace,
                model=settings.llm.model,
                phase="intent",
            )
        ]
    response = model.invoke(
        [
            SystemMessage(content=_INTENT_LLM_SYSTEM),
            HumanMessage(content=prompt),
        ],
        config={"callbacks": callbacks} if callbacks else None,
    )
    content = response.content if isinstance(response.content, str) else str(response.content)
    payload: Any = json.loads(content)
    if not isinstance(payload, dict):
        return MessageKind.AMBIGUOUS
    try:
        return MessageKind(str(payload.get("kind", "")).lower())
    except ValueError:
        return MessageKind.AMBIGUOUS


def conversation_reply_text(
    kind: MessageKind,
    user_message: str,
    l3_hint: str | None = None,
) -> str:
    """离线兜底：按意图类别生成对话回复（不跑工具）。"""
    if kind == MessageKind.GREETING:
        base = "你好！我是旅行规划助手，可以帮你查景点、看天气、排行程。"
    elif kind == MessageKind.AMBIGUOUS:
        base = (
            "可以，我先确认一下：你是想让我帮你做旅行相关的目的地推荐或行程安排吗？"
            "如果是，告诉我大概想去哪、玩几天、偏好什么节奏。"
        )
    else:
        # 明确无关问题：先接住话头，再说明边界。
        snippet = user_message.strip()[:40]
        if len(user_message.strip()) > 40:
            snippet += "…"
        base = (
            f"收到你说的「{snippet}」。"
            "我主要擅长旅行规划（目的地、天数、偏好、路线安排），"
            "这类问题我可能帮不上忙。"
        )

    if l3_hint:
        return (
            f"{base}"
            f"我记得你之前的偏好：{l3_hint}。"
            "如果想规划行程，直接告诉我目的地、天数和偏好即可。"
        )
    return (
        f"{base}有出行计划时，告诉我目的地、天数，"
        "以及偏好（美食/自然/历史、轻松还是特种兵、预算与同行人等），我来帮你排。"
    )
