"""Provider-grounded POI identity, verification, and coverage rules.

This module is deliberately independent of planner prompts.  Search, ranking,
planning, repair, critic, and rendering share the same entity invariants.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from datetime import date
from typing import Any, Iterable

from travel_agent.schemas import POI, TravelProfile


PLANNABLE_VERIFICATION_STATUSES = frozenset({"verified"})
REJECTED_VERIFICATION_STATUSES = frozenset(
    {"closed", "unsuitable", "wrong_entity", "evidence_insufficient"}
)
ACTIVITY_ENTITY_TYPES = frozenset(
    {"attraction", "museum", "cultural_venue", "park", "district", "venue"}
)
NON_ACTIVITY_ENTITY_TYPES = frozenset(
    {
        "beauty_service",
        "automotive_service",
        "commercial_service",
        "hotel",
        "parking",
        "restaurant",
        "retail",
        "ticket_office",
        "transport",
        "visitor_center",
    }
)

_ANNEX_NAME_MARKERS = (
    "票务中心",
    "票务服务",
    "售票处",
    "停车场",
    "游客中心",
    "服务中心",
    "文创商店",
    "纪念品商店",
    "纪念品",
    "旅游广场",
    "上客点",
    "下客点",
    "乘车点",
    "乘车服务点",
    "直通车",
    "连接线",
    "检票处",
    "出入口",
    "枢纽",
    "换乘中心",
)
_CLOSED_NAME_MARKERS = (
    "暂停开放",
    "暂不开放",
    "停止开放",
    "永久关闭",
    "已关闭",
    "暂停营业",
    "停业",
    "建设中",
    "在建",
    "尚未开放",
    "暂未开放",
)
_NON_VISITABLE_SUFFIXES = ("有限公司", "有限责任公司", "分公司", "办事处", "服务门店")

_AVOID_SPECIFIC_ALIASES: dict[str, frozenset[str]] = {
    "海边": frozenset({"海边", "海湾", "海滨", "沙滩", "海滩", "环岛", "seaside", "beach"}),
    "咖啡店": frozenset({"咖啡", "coffee", "cafe", "café"}),
    "园林": frozenset({"园林", "花园", "庭园", "庭院", "garden"}),
}
_AVOID_HARD_INTEREST_CATEGORIES: dict[str, frozenset[str]] = {
    "food": frozenset({"food"}),
    "history": frozenset({"culture", "museum"}),
    "culture": frozenset({"culture", "museum"}),
    "museum": frozenset({"museum"}),
    "nature": frozenset({"scenic"}),
    "shopping": frozenset({"shopping"}),
}


@dataclass(frozen=True)
class AvoidMatch:
    """Auditable deterministic evidence that a POI violates an active avoid rule."""

    term: str
    source: str
    method: str
    matched_value: str

    def to_audit_dict(self, poi: POI) -> dict[str, str]:
        return {
            "poi_id": poi.poi_id,
            "name": poi.name,
            "term": self.term,
            "source": self.source,
            "method": self.method,
            "matched_value": self.matched_value,
        }


@dataclass(frozen=True)
class _AvoidRule:
    term: str
    source: str
    entity_only: bool = False


def normalize_entity_name(value: object) -> str:
    """Normalize formatting only; do not drop semantic venue words."""
    text = str(value or "").strip().casefold()
    text = re.sub(r"[\s\-—–_·•:：,，。/\\()（）\[\]【】]+", "", text)
    return text


def normalize_candidate_requirement(value: object) -> str:
    """Remove request framing while preserving the named venue identity."""
    text = str(value or "").strip()
    return re.sub(r"(?:一日游|半日游|参观|游览|打卡)$", "", text).strip() or text


def canonical_names(poi: POI) -> set[str]:
    values = [poi.canonical_name or poi.name, *poi.aliases]
    return {normalized for value in values if (normalized := normalize_entity_name(value))}


def canonical_entity_match_evidence(
    poi: POI,
    required_name: str,
) -> dict[str, Any] | None:
    """Return auditable provider-backed identity evidence for a requirement.

    Formatting, city prefixes, scenic-area suffixes and museum branch suffixes
    are normalized generically.  Containment is accepted only for a specific
    name with compatible provider category/city evidence; broad terms such as
    ``湖`` or ``博物馆`` never match by substring alone.
    """
    required = normalize_entity_name(required_name)
    if not required or not is_verified_plannable_poi(poi):
        return None
    provider_id = normalize_entity_name(poi.source_poi_id or poi.poi_id)
    if provider_id and required in {provider_id, normalize_entity_name(poi.poi_id)}:
        return _match_evidence(poi, required_name, "provider_canonical_id", provider_id)

    names = [poi.canonical_name or poi.name, *poi.aliases]
    if _identity_name_disqualified(poi, required_name):
        return None
    normalized_names = [
        (str(value), normalize_entity_name(value))
        for value in names
        if normalize_entity_name(value)
    ]
    for original, normalized in normalized_names:
        if required == normalized and _specific_identity(required, poi.city):
            return _match_evidence(poi, required_name, "provider_canonical_name", original)

    if not _city_compatible(required, poi.city):
        return None
    required_key = canonical_identity_key(required_name, city=poi.city, entity_type=poi.entity_type)
    if not required_key or not _specific_identity(required, poi.city):
        return None
    for original, _normalized in normalized_names:
        provider_key = canonical_identity_key(
            original,
            city=poi.city,
            entity_type=poi.entity_type,
        )
        if not provider_key:
            continue
        if required_key == provider_key:
            return _match_evidence(poi, required_name, "canonical_name_variant", original)
        museum_requirement = any(
            marker in required
            for marker in ("博物馆", "博物院", "纪念馆", "美术馆", "展览馆")
        )
        if museum_requirement:
            required_admin_key = re.sub(
                r"(?:省|市|自治区|特别行政区)$", "", required_key
            )
            provider_admin_key = re.sub(
                r"(?:省|市|自治区|特别行政区)$", "", provider_key
            )
            if (
                len(required_admin_key) >= 2
                and required_admin_key == provider_admin_key
            ):
                return _match_evidence(
                    poi, required_name, "administrative_museum_variant", original
                )
        if (
            min(len(required_key), len(provider_key)) >= 2
            and not museum_requirement
            and (required_key in provider_key or provider_key in required_key)
            and _category_supports_name(poi, required_name)
        ):
            return _match_evidence(poi, required_name, "qualified_containment", original)
    return None


def canonical_identity_key(
    value: object,
    *,
    city: str | None = None,
    entity_type: str | None = None,
) -> str:
    """Build a generic identity key without city- or POI-specific aliases."""
    text = str(value or "").strip().casefold()
    text = re.sub(
        r"[（(][^）)]*(?:本馆|主馆|分馆|新馆|东馆|西馆|南馆|北馆|馆区)[^）)]*[）)]",
        "",
        text,
    )
    normalized = normalize_entity_name(text)
    city_name = normalize_entity_name(city)
    city_base = re.sub(r"(?:特别行政区|自治区|自治州|地区|盟|市|县)$", "", city_name)
    prefixes = sorted(
        {prefix for prefix in (city_name, city_base, f"{city_base}市" if city_base else "") if prefix},
        key=len,
        reverse=True,
    )
    for prefix in prefixes:
        if prefix and normalized.startswith(prefix) and len(normalized) > len(prefix) + 1:
            remainder = normalized[len(prefix):]
            # "甲城博物馆" is a specific city-named venue, not the generic
            # word "博物馆".  Keep the city component when stripping it would
            # leave only a venue-class suffix; otherwise city prefixes remain
            # formatting variants as before.
            if not re.fullmatch(
                r"(?:博物馆|博物院|纪念馆|美术馆|展览馆|公园|景区|风景区)",
                remainder,
            ):
                normalized = remainder
            break
    if entity_type in {"museum", "cultural_venue"} or "馆" in normalized:
        normalized = re.sub(
            r"(?:本馆|主馆|分馆|新馆|东馆|西馆|南馆|北馆|馆区)$",
            "",
            normalized,
        )
        normalized = re.sub(r"(?:博物馆|博物院|纪念馆|美术馆|展览馆)$", "", normalized)
    normalized = re.sub(
        r"(?:[一二三四五六七八九十百零〇壹贰叁肆伍陆柒捌玖\d]+号?"
        r"(?:坑|展厅)(?:大厅|遗址|展厅)?|展厅|大厅|遗址)$",
        "",
        normalized,
    )
    normalized = re.sub(
        r"(?:国家级)?(?:旅游景区|风景名胜区|风景区|名胜区|景区|旅游区)$",
        "",
        normalized,
    )
    return normalized


def _specific_identity(normalized: str, city: str | None) -> bool:
    value = normalized
    value = re.sub(
        r"(?:国家级)?(?:旅游景区|风景名胜区|风景区|名胜区|景区|旅游区|博物馆|博物院|纪念馆|美术馆|展览馆|公园)$",
        "",
        value,
    )
    # A city-qualified generic venue remains specific enough when the provider
    # city agrees (e.g. "某市博物馆" vs "某市市博物馆").
    city_base = re.sub(
        r"(?:特别行政区|自治区|自治州|地区|盟|市|县)$",
        "",
        normalize_entity_name(city),
    )
    return len(value) >= 2 or bool(city_base and normalized.startswith(city_base))


def _city_compatible(required: str, city: str | None) -> bool:
    explicit = re.match(r"^(.{2,10}?)(?:市|特别行政区|自治区|自治州)", required)
    if not explicit:
        return True
    expected = re.sub(
        r"(?:特别行政区|自治区|自治州|地区|盟|市|县)$",
        "",
        normalize_entity_name(explicit.group(1)),
    )
    actual = re.sub(
        r"(?:特别行政区|自治区|自治州|地区|盟|市|县)$",
        "",
        normalize_entity_name(city),
    )
    return not expected or not actual or expected == actual


def _category_supports_name(poi: POI, required_name: str) -> bool:
    required = normalize_entity_name(required_name)
    if any(marker in required for marker in ("博物馆", "博物院", "纪念馆", "美术馆")):
        return poi.entity_type in {"museum", "cultural_venue"} or poi.category in {"museum", "culture"}
    return poi.entity_type in ACTIVITY_ENTITY_TYPES


def _identity_name_disqualified(poi: POI, required_name: str) -> bool:
    provider = normalize_entity_name(poi.canonical_name or poi.name)
    required = normalize_entity_name(required_name)
    annex_markers = (*_ANNEX_NAME_MARKERS, "纪念品店", "文创店", "商店", "超市")
    if any(marker in provider and marker not in required for marker in annex_markers):
        return True
    museum_markers = ("博物馆", "博物院", "纪念馆", "美术馆", "展览馆")
    natural_markers = ("湖", "山", "河", "江", "海", "峡谷", "沙漠", "草原", "湿地")
    scenic_markers = ("风景名胜区", "风景区", "景区", "景点", "公园", "湿地")
    exact_alias = any(
        normalize_entity_name(alias) == required for alias in poi.aliases
    )
    if required.endswith(natural_markers) and not exact_alias:
        # A child whose provider name merely starts with the parent scenic
        # area's formal name is not evidence for visiting the parent itself.
        # Accept the main scenic-area suffix, but require explicit
        # child_covers_parent metadata when further subvenue text follows it.
        remainder = provider.partition(required)[2] if required in provider else ""
        if any(
            remainder.startswith(marker) and remainder != marker
            for marker in scenic_markers
        ):
            return True
    if (
        any(marker in provider for marker in museum_markers)
        and not any(marker in required for marker in museum_markers)
        and not exact_alias
    ):
        # A direct class suffix is often the provider's formal name for the
        # requested venue ("X" -> "X Museum").  A museum named after a
        # natural feature is a different entity, as is "X ... Museum" with
        # additional semantic qualifiers between the name and class suffix.
        direct_class_variant = any(
            provider == required + marker
            or (
                len(required) >= 3
                and provider.endswith(required + marker)
            )
            for marker in museum_markers
        )
        if required.endswith(natural_markers) or not direct_class_variant:
            return True
    if (
        required.endswith(natural_markers)
        and provider != required
        and not exact_alias
        and not any(marker in provider for marker in scenic_markers)
    ):
        # District names and commercial/cultural venues frequently begin with
        # a nearby natural landmark (e.g. "X District ... Hall").  Only an
        # explicit scenic-area relation may cover the landmark itself.
        return True
    return False


def _match_evidence(
    poi: POI,
    required_name: str,
    method: str,
    matched_value: str,
) -> dict[str, Any]:
    return {
        "requested_name": str(required_name),
        "original_name": poi.name,
        "canonical_name": poi.canonical_name or poi.name,
        "canonical_id": poi.source_poi_id or poi.poi_id,
        "match_method": method,
        "matched_value": matched_value,
        "provider_source": poi.source,
        "city": poi.city,
        "entity_type": poi.entity_type,
    }


def poi_covers_requirement(poi: POI, required_name: str) -> bool:
    """Match only canonical identity, provider aliases, or explicit parent coverage."""
    if canonical_entity_match_evidence(poi, required_name) is not None:
        return True
    required = normalize_entity_name(required_name)
    return bool(
        poi.parent_poi_id
        and poi.coverage_relation == "child_covers_parent"
        and required == normalize_entity_name(poi.parent_canonical_name)
    )


def _avoid_values(value: object) -> list[str]:
    if isinstance(value, str):
        raw = [value]
    elif isinstance(value, (set, frozenset)):
        raw = sorted(value, key=str)
    elif isinstance(value, (list, tuple)):
        raw = list(value)
    else:
        raw = []
    return [str(item).strip() for item in raw if str(item).strip()]


def _active_avoid_rules(profile: TravelProfile) -> list[_AvoidRule]:
    """Return trip-scoped rules first, followed by stable-profile fallbacks."""

    state = profile.constraint_state or {}
    rules: list[_AvoidRule] = []
    state_terms: set[str] = set()
    for field in ("removed", "avoid", "exclude"):
        field_terms: set[str] = set()
        for term in _avoid_values(state.get(field)):
            key = normalize_entity_name(term)
            if not key or key in field_terms:
                continue
            rules.append(_AvoidRule(term, field, entity_only=field == "removed"))
            field_terms.add(key)
            state_terms.add(key)
    profile_terms: set[str] = set()
    for term in _avoid_values(profile.avoid):
        key = normalize_entity_name(term)
        if not key or key in state_terms or key in profile_terms:
            continue
        rules.append(_AvoidRule(term, "profile", entity_only=False))
        profile_terms.add(key)
    return rules


def _explicit_must_visit_matches(poi: POI, profile: TravelProfile) -> bool:
    state = profile.constraint_state or {}
    requirements = [
        *_avoid_values(state.get("must_visit")),
        *_avoid_values(profile.must_visit),
    ]
    return any(poi_covers_requirement(poi, term) for term in requirements)


def _entity_avoid_match(poi: POI, rule: _AvoidRule) -> AvoidMatch | None:
    requested = normalize_entity_name(rule.term)
    provider_ids = {
        normalize_entity_name(poi.poi_id),
        normalize_entity_name(poi.source_poi_id),
    }
    if requested and requested in provider_ids:
        return AvoidMatch(rule.term, rule.source, "provider_id", poi.source_poi_id or poi.poi_id)

    evidence = canonical_entity_match_evidence(poi, rule.term)
    if evidence is not None:
        return AvoidMatch(
            rule.term,
            rule.source,
            # The canonical matcher has several internal evidence branches
            # (provider canonical name, alias, normalized variant). Keep the
            # public audit method stable while retaining the matched value.
            "canonical_identity",
            str(evidence.get("matched_value") or poi.canonical_name or poi.name),
        )

    requested_key = canonical_identity_key(
        rule.term,
        city=poi.city,
        entity_type=poi.entity_type,
    )
    for name in (poi.canonical_name or poi.name, poi.name, *poi.aliases):
        candidate_key = canonical_identity_key(
            name,
            city=poi.city,
            entity_type=poi.entity_type,
        )
        if requested_key and requested_key == candidate_key:
            return AvoidMatch(rule.term, rule.source, "canonical_identity", str(name))
    return None


def _semantic_avoid_match(poi: POI, rule: _AvoidRule) -> AvoidMatch | None:
    term = str(rule.term).strip()
    lowered_text = " ".join([poi.name, *poi.tags]).casefold()
    if term == "连续爬坡":
        mobility_marker = next(
            (
                marker
                for marker in ("登山", "爬山", "山路", "山道", "登高", "陡坡", "梯道")
                if marker in lowered_text
            ),
            None,
        )
        if mobility_marker is None and "步道" in lowered_text and "山" in lowered_text:
            mobility_marker = "山地步道"
        if mobility_marker is not None:
            return AvoidMatch(term, rule.source, "mobility_semantic", mobility_marker)
    if term == "长楼梯":
        stair_marker = next(
            (
                marker
                for marker in ("长楼梯", "长阶梯", "连续台阶", "天梯", "梯坎")
                if marker in lowered_text
            ),
            None,
        )
        if stair_marker is not None:
            return AvoidMatch(term, rule.source, "mobility_semantic", stair_marker)
    aliases = _AVOID_SPECIFIC_ALIASES.get(term)
    if aliases is not None:
        matched = next(
            (alias for alias in sorted(aliases) if alias.casefold() in lowered_text),
            None,
        )
        return (
            AvoidMatch(term, rule.source, "specific_alias", matched)
            if matched is not None
            else None
        )

    from travel_agent.workflow_rules import normalize_interest

    canonical = normalize_interest(term)
    categories = _AVOID_HARD_INTEREST_CATEGORIES.get(canonical)
    if categories is not None and poi.category in categories:
        return AvoidMatch(term, rule.source, "interest_category", poi.category)

    normalized_tags = {str(tag).strip().casefold() for tag in poi.tags if str(tag).strip()}
    if canonical and canonical.casefold() in normalized_tags:
        return AvoidMatch(term, rule.source, "provider_tag", canonical)

    normalized_term = normalize_entity_name(term)
    destination = normalize_entity_name(poi.city)
    if len(normalized_term) < 2 or normalized_term == destination:
        return None
    normalized_name = normalize_entity_name(poi.name)
    if normalized_term in normalized_name:
        return AvoidMatch(term, rule.source, "provider_name_term", poi.name)
    exact_tag = next(
        (tag for tag in poi.tags if normalize_entity_name(tag) == normalized_term),
        None,
    )
    if exact_tag is not None:
        return AvoidMatch(term, rule.source, "provider_tag", str(exact_tag))
    return None


def poi_avoid_match(poi: POI, profile: TravelProfile) -> AvoidMatch | None:
    """Return high-confidence avoid evidence, or ``None`` for ambiguous preferences.

    Trip-scoped ``removed/avoid/exclude`` rules always win. A stable-profile
    dislike is ignored when the same POI is an explicit must-visit for this
    trip, preventing old memory from erasing a current request.
    """

    must_visit = _explicit_must_visit_matches(poi, profile)
    for rule in _active_avoid_rules(profile):
        if rule.source == "profile" and must_visit:
            continue
        identity = _entity_avoid_match(poi, rule)
        if identity is not None:
            return identity
        if rule.entity_only:
            continue
        semantic = _semantic_avoid_match(poi, rule)
        if semantic is not None:
            return semantic
    return None


def default_entity_type(category: str) -> str:
    return {
        "food": "restaurant",
        "hotel": "hotel",
        "museum": "museum",
        "transport": "transport",
        "scenic": "attraction",
        "culture": "cultural_venue",
    }.get(str(category or "").strip(), "venue")


def normalize_poi_entity(poi: POI, profile: TravelProfile | None = None) -> POI:
    """Fill identity fields and deterministically apply verification evidence."""
    canonical_name = str(poi.canonical_name or poi.name).strip()
    entity_type = str(poi.entity_type or default_entity_type(poi.category)).strip()
    # Legacy constructors predate entity_type and therefore use ``venue``.
    # Category is structured source data, so using it to specialize that
    # default is not a name-keyword category inference.
    if entity_type == "venue":
        entity_type = default_entity_type(poi.category)
    status = str(poi.verification_status or "evidence_insufficient")
    reason = poi.verification_reason

    if status == "verified" and _is_closed(poi, profile):
        status = "closed"
        reason = reason or "工具开放信息显示目标日期不可用"
    elif status == "verified" and (
        any(marker in poi.name for marker in _ANNEX_NAME_MARKERS)
        or any(poi.name.endswith(suffix) for suffix in _NON_VISITABLE_SUFFIXES)
    ):
        status = "wrong_entity"
        reason = reason or "名称显示为附属设施或非游览实体，且无父子覆盖证据"
    elif (
        status == "verified"
        and entity_type not in ACTIVITY_ENTITY_TYPES
        and entity_type not in NON_ACTIVITY_ENTITY_TYPES
    ):
        status = "evidence_insufficient"
        reason = reason or f"工具实体类型 {entity_type or 'unknown'} 不足以证明其为景点"

    return replace(
        poi,
        canonical_name=canonical_name,
        entity_type=entity_type,
        source_poi_id=poi.source_poi_id or poi.poi_id,
        verification_status=status,  # type: ignore[arg-type]
        verification_reason=reason,
        aliases=list(dict.fromkeys(str(alias).strip() for alias in poi.aliases if str(alias).strip())),
    )


def is_verified_plannable_poi(poi: POI) -> bool:
    return (
        poi.verification_status in PLANNABLE_VERIFICATION_STATUSES
        and poi.entity_type in ACTIVITY_ENTITY_TYPES
        and poi.category not in {"food", "hotel", "transport", "service", "unknown"}
    )


def partition_verified_candidates(
    pois: Iterable[POI], profile: TravelProfile | None = None
) -> tuple[list[POI], list[POI]]:
    verified: list[POI] = []
    rejected: list[POI] = []
    for raw in pois:
        poi = normalize_poi_entity(raw, profile)
        if is_verified_plannable_poi(poi):
            verified.append(poi)
        else:
            if poi.verification_status == "verified" and poi.entity_type in NON_ACTIVITY_ENTITY_TYPES:
                poi = replace(
                    poi,
                    verification_status="wrong_entity",
                    verification_reason=(
                        poi.verification_reason
                        or f"工具实体类型 {poi.entity_type} 不是可游览场馆本体"
                    ),
                )
            rejected.append(poi)
    return verified, rejected


def _is_closed(poi: POI, profile: TravelProfile | None) -> bool:
    if any(marker in str(poi.name or "") for marker in _CLOSED_NAME_MARKERS):
        return True
    opening = re.sub(r"\s+", "", str(poi.opening_hours or ""))
    if not opening or profile is None:
        return False
    weekday = str((profile.constraint_state or {}).get("weekday") or "").strip()
    if not weekday and profile.start_date:
        try:
            weekday = "周" + "一二三四五六日"[date.fromisoformat(profile.start_date).weekday()]
        except (TypeError, ValueError):
            weekday = ""
    return bool(
        weekday
        and re.search(
            rf"(?:每)?{re.escape(weekday)}.{{0,12}}(?:全天)?(?:不开放|闭馆|休息|暂停营业|停止开放)",
            opening,
        )
    )
