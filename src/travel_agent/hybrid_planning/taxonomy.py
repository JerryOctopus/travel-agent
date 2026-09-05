"""Versioned, deliberately small interest taxonomy shared by all extractors."""

from __future__ import annotations

from dataclasses import dataclass


INTEREST_TAXONOMY_VERSION = "travel-interest-v1"


@dataclass(frozen=True)
class InterestDefinition:
    label: str
    description: str
    aliases: tuple[str, ...]


# The first ten labels are the pre-existing production taxonomy.  The bounded
# extension covers recurring experiential language without introducing POI- or
# city-specific categories.
INTEREST_TAXONOMY: tuple[InterestDefinition, ...] = (
    InterestDefinition("nature", "自然、风景与滨水户外体验", ("自然风光", "自然", "风景", "风光", "海边", "园林")),
    InterestDefinition("food", "餐饮与地方美食", ("美食", "吃")),
    InterestDefinition("culture", "文化与人文体验", ("历史文化", "文化")),
    InterestDefinition("history", "历史遗存与历史景点", ("主要历史景点", "历史景点", "历史")),
    InterestDefinition("museum", "博物馆与展陈", ("博物馆",)),
    InterestDefinition("couple", "情侣体验", ("情侣",)),
    InterestDefinition("citywalk", "城市漫步与城市景观", ("经典城市景观", "城市景观", "经典景观", "城市漫步", "citywalk")),
    InterestDefinition("family", "亲子友好体验", ("亲子", "带孩子")),
    InterestDefinition("nightlife", "夜间娱乐与夜生活", ("夜生活",)),
    InterestDefinition("shopping", "购物体验", ("购物",)),
    InterestDefinition("local_life", "社区日常与本地生活氛围", ("老城区烟火气", "烟火气", "本地生活", "市井生活")),
    InterestDefinition("architecture", "建筑、街区风貌与设计", ("有年代感的建筑", "建筑摄影", "建筑")),
    InterestDefinition("industrial_heritage", "工业遗产与旧设施再利用", ("工业遗迹", "工业遗产")),
    InterestDefinition("urban_renewal", "城市更新与空间再生", ("城市更新", "空间再生")),
    InterestDefinition("photography", "以摄影为主要体验方式", ("适合建筑摄影", "建筑摄影", "摄影")),
    InterestDefinition("hands_on", "可参与的动手与互动体验", ("动手体验", "互动体验", "手作")),
    InterestDefinition("cafe", "咖啡馆与慢休闲", ("喝咖啡", "咖啡店", "咖啡")),
    InterestDefinition("commercialized", "高度商业化的游览体验", ("商业化景点", "太商业化", "商业化")),
)

INTEREST_DEFINITIONS = {item.label: item for item in INTEREST_TAXONOMY}
CANONICAL_INTERESTS = frozenset(INTEREST_DEFINITIONS)
INTEREST_ALIAS_PAIRS = tuple(
    (alias, item.label)
    for item in INTEREST_TAXONOMY
    for alias in item.aliases
)

# Compatibility map used by the pre-existing rule extractor while the hybrid
# feature is disabled.  In particular, a coffee shop remains the legacy broad
# ``food`` interest there; the richer normalizer may additionally use ``cafe``.
INTEREST_KEYWORDS = {
    "自然": "nature",
    "风景": "nature",
    "风光": "nature",
    "自然风光": "nature",
    "海边": "nature",
    "园林": "nature",
    "美食": "food",
    "吃": "food",
    "咖啡店": "food",
    "文化": "culture",
    "历史文化": "culture",
    "历史": "history",
    "历史景点": "history",
    "主要历史景点": "history",
    "博物馆": "museum",
    "情侣": "couple",
    "城市漫步": "citywalk",
    "城市景观": "citywalk",
    "经典城市景观": "citywalk",
    "经典景观": "citywalk",
    "citywalk": "citywalk",
    "亲子": "family",
    "夜生活": "nightlife",
    "购物": "shopping",
}

ALL_INTEREST_KEYWORDS = {
    alias: item.label
    for item in INTEREST_TAXONOMY
    for alias in item.aliases
}

# Generic retrieval language for the deterministic Intent -> Worker bridge.
# These are concepts, not aliases: they deliberately contain no city, POI, or
# evaluation-case names.  One descriptor phrase is one bounded provider query.
INTEREST_RETRIEVAL_DESCRIPTORS: dict[str, tuple[str, ...]] = {
    "nature": ("自然景观 户外 滨水",),
    "food": ("地方饮食 本地美食",),
    "culture": ("人文文化 公共文化空间",),
    "history": ("历史遗存 历史街区",),
    "museum": ("博物馆 展览 展陈",),
    "couple": ("情侣友好 约会体验",),
    "citywalk": ("城市漫步 街区步行",),
    "family": ("亲子友好 家庭体验",),
    "nightlife": ("夜间文化 夜生活",),
    "shopping": ("本地购物 特色市集",),
    "local_life": ("本地生活街区 社区生活",),
    "architecture": ("建筑 街区风貌 空间设计",),
    "industrial_heritage": ("工业遗产 旧工业空间 工业建筑再利用",),
    "urban_renewal": ("城市更新 空间再生 公共空间改造",),
    "photography": ("摄影友好 建筑与街景摄影",),
    "hands_on": ("手作 互动 动手体验",),
    "cafe": ("咖啡馆 慢休闲",),
    "commercialized": ("高度商业化 游客商业街",),
}


def taxonomy_prompt_payload() -> list[dict[str, str]]:
    """Return only controlled labels and meanings; aliases stay deterministic."""
    return [
        {"label": item.label, "description": item.description}
        for item in INTEREST_TAXONOMY
    ]
