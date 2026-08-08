"""TravelProfile 的 Patch 更新语义：区分 SET / CLEAR / UNCHANGED。

- SET：本轮设置了新值；
- CLEAR：本轮明确取消/收回旧值；
- UNCHANGED：用「patch dict 中缺席」表达，本轮未提及的槽位保持原值。

这样多轮对话中「本轮没提目的地」不会覆盖已确认的杭州，
而「苏州不去了」才能真正清掉旧目的地。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from travel_agent.schemas import TravelProfile
from travel_agent.workflow_rules import CITY_ALIASES


class PatchOp(str, Enum):
    SET = "set"
    CLEAR = "clear"
    # UNCHANGED：不进入 patch dict


@dataclass(frozen=True)
class SlotPatch:
    op: PatchOp
    value: Any = None


SETTABLE_SLOTS = frozenset(
    {
        "destination",
        "days",
        "start_date",
        "budget_level",
        "budget_limit",
        "interests",
        "companions",
        "party_size",
        "pace",
        "hotel_area",
        "food_preference",
        "must_visit",
        "avoid",
        "transport_mode",
    }
)

LIST_SLOTS = frozenset({"interests", "food_preference", "must_visit", "avoid"})

_SCALAR_SLOTS = (
    "destination",
    "days",
    "start_date",
    "budget_level",
    "budget_limit",
    "companions",
    "party_size",
    "hotel_area",
)


def extracted_to_patches(extracted: TravelProfile) -> dict[str, SlotPatch]:
    """规则抽取结果 → patches：只 SET 非空值，None/默认值一律视为 UNCHANGED。"""
    patches: dict[str, SlotPatch] = {}
    for name in _SCALAR_SLOTS:
        value = getattr(extracted, name)
        if value:
            patches[name] = SlotPatch(PatchOp.SET, value)
    for name in LIST_SLOTS:
        value = getattr(extracted, name)
        if value:
            patches[name] = SlotPatch(PatchOp.SET, list(value))
    if extracted.pace != "standard":
        patches["pace"] = SlotPatch(PatchOp.SET, extracted.pace)
    if extracted.transport_mode != "public_transport":
        patches["transport_mode"] = SlotPatch(PatchOp.SET, extracted.transport_mode)
    return patches


# --------------------------------------------------------------------------- #
# 高精度 CLEAR 识别（宁缺勿滥，复杂表达交给 LLM 兜底）
# --------------------------------------------------------------------------- #
_CHINESE_CITY_ALIASES = {
    alias: city
    for alias, city in CITY_ALIASES.items()
    if re.search(r"[\u4e00-\u9fff]", alias)
}
_CITY_ALT = "|".join(sorted(_CHINESE_CITY_ALIASES, key=len, reverse=True))

_CITY_CANCEL_RE = re.compile(
    rf"(?:不(?:想|打算|要)?去\s*(?P<city_a>{_CITY_ALT})\s*了)"
    rf"|(?:(?P<city_b>{_CITY_ALT})\s*(?:就|也)?不(?:想|打算|要)?去(?:了|啦))"
)
_DEST_UNCERTAIN_RE = re.compile(r"目的地\s*(?:先)?(?:不定|不确定|再说|还没想好)")
_DAYS_UNCERTAIN_RE = re.compile(r"天数\s*(?:先)?(?:不定|不确定|再说)")
_DATE_UNCERTAIN_RE = re.compile(r"(?:出发)?日期\s*(?:先)?(?:不定|不确定|再说)|时间\s*(?:先)?(?:不定|再说)")


def extract_slot_clears(user_message: str) -> dict[str, SlotPatch]:
    """规则识别本轮对已确认槽位的明确取消；value 携带被取消的旧值（仅供诊断）。"""
    clears: dict[str, SlotPatch] = {}
    city_match = _CITY_CANCEL_RE.search(user_message)
    if _DEST_UNCERTAIN_RE.search(user_message):
        clears["destination"] = SlotPatch(PatchOp.CLEAR)
    elif city_match:
        alias = city_match.group("city_a") or city_match.group("city_b")
        clears["destination"] = SlotPatch(PatchOp.CLEAR, _CHINESE_CITY_ALIASES.get(alias, alias))
    if _DAYS_UNCERTAIN_RE.search(user_message):
        clears["days"] = SlotPatch(PatchOp.CLEAR)
    if _DATE_UNCERTAIN_RE.search(user_message):
        clears["start_date"] = SlotPatch(PatchOp.CLEAR)
    return clears


def merge_patches(
    base: dict[str, SlotPatch],
    overrides: dict[str, SlotPatch],
) -> dict[str, SlotPatch]:
    """合并两组 patches；同一槽位 SET 与 CLEAR 冲突时，SET 值等于被取消值则两者都丢弃。"""
    merged = dict(base)
    for name, patch in overrides.items():
        existing = merged.get(name)
        if (
            existing is not None
            and existing.op == PatchOp.SET
            and patch.op == PatchOp.CLEAR
        ):
            if patch.value and existing.value == patch.value:
                # 「苏州不去了」且规则恰好把苏州抽成新目的地：保持旧值，避免误写。
                merged.pop(name)
            # 值不同时 SET 新值本身就是替换动作，忽略 CLEAR。
            continue
        merged[name] = patch
    return merged


# --------------------------------------------------------------------------- #
# SlotPatch <-> JSON 友好 payload（供评测 trace 与 LLM 输出互转）
# --------------------------------------------------------------------------- #
def patches_to_payload(patches: dict[str, SlotPatch]) -> dict[str, dict[str, Any]]:
    return {
        name: {"op": patch.op.value, "value": patch.value}
        for name, patch in patches.items()
    }


def payload_to_patches(payload: dict[str, Any]) -> dict[str, SlotPatch]:
    result: dict[str, SlotPatch] = {}
    if not isinstance(payload, dict):
        return result
    for name, item in payload.items():
        if name not in SETTABLE_SLOTS or not isinstance(item, dict):
            continue
        try:
            op = PatchOp(str(item.get("op", "")).lower())
        except ValueError:
            continue
        result[name] = SlotPatch(op, item.get("value"))
    return result
