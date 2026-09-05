"""Fail-closed normalization of natural-language interests into a taxonomy."""

from __future__ import annotations

import json
import hashlib
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from travel_agent.hybrid_planning.call_cache import HybridCallCache
from travel_agent.hybrid_planning.llm_types import (
    StructuredLLMClient,
    StructuredOutputError,
)
from travel_agent.hybrid_planning.taxonomy import (
    CANONICAL_INTERESTS,
    INTEREST_ALIAS_PAIRS,
    INTEREST_DEFINITIONS,
    INTEREST_TAXONOMY_VERSION,
    taxonomy_prompt_payload,
)


INTENT_NORMALIZER_SCHEMA_VERSION = "intent-normalizer-v2"
INTENT_NORMALIZER_PROMPT_VERSION = "intent-normalizer-prompt-v2"
_FULL_PLAN_TYPES = frozenset({"full_itinerary", "full_trip_plan"})
_NEGATION_MARKERS = ("不喜欢", "不想", "不要", "避开", "排斥", "拒绝", "不去", "别太", "不再")
_INTEREST_SEMANTIC_RE = re.compile(
    r"喜欢|偏好|感兴趣|氛围|体验|人文|气息|气质|像.{0,8}一样|烟火气|有年代感|工业遗迹|"
    r"城市更新|商业化|发呆|喝咖啡|摄影|动手|亲子|海边|悠闲|深度|"
    r"不喜欢|不想要|不要太|避开.{0,10}(?:类型|风格|氛围|体验)"
)
_FACT_QUERY_RE = re.compile(r"天气|气温|几点|开放时间|门票|多少钱|是什么|在哪里|怎么走|距离")
_MUST_VISIT_ONLY_RE = re.compile(r"^(?:我)?(?:想去|要去|必须去|必去)[^，。！？!?]+[。！？!?]?$" )
_CONSTRAINT_ONLY_RE = re.compile(
    r"预算|[0-9一二三四五六七八九十]+\s*(?:人|天)|截止|前返回|到达|出发|"
    r"最多换乘|固定活动|必须去|必去"
)
_FORBIDDEN_OUTPUT_KEYS = frozenset(
    {
        "hard_constraints",
        "hard_constraint_changes",
        "must_visit",
        "budget",
        "budget_limit",
        "budget_max_cny",
        "deadline",
        "return_deadline",
        "fixed_events",
        "itinerary",
    }
)


@dataclass(frozen=True)
class NormalizedInterest:
    label: str
    source_text: str
    confidence: float
    reason: str
    source_turn: str | None = None
    polarity: Literal["prefer", "avoid"] = "prefer"
    basis: Literal["canonical", "deterministic_alias", "llm"] = "deterministic_alias"


@dataclass(frozen=True)
class IntentNormalizationResult:
    raw_interests: list[str]
    normalized_interests: list[NormalizedInterest]
    unresolved: list[str]
    taxonomy_version: str = INTEREST_TAXONOMY_VERSION
    fallback_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_interests": list(self.raw_interests),
            "normalized_interests": [asdict(item) for item in self.normalized_interests],
            "unresolved": list(self.unresolved),
            "taxonomy_version": self.taxonomy_version,
            "fallback_reason": self.fallback_reason,
        }


@dataclass(frozen=True)
class IntentEligibility:
    eligible: bool
    reason_code: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class IntentNormalizer:
    """Interpret only soft interests; never emits profile hard constraints."""

    llm_client: StructuredLLMClient | None = None
    enabled: bool = False
    confidence_threshold: float = 0.65
    cache: HybridCallCache = field(default_factory=HybridCallCache, repr=False)
    request_scope: str = "standalone"

    def normalize(
        self,
        raw_interests: str | list[str],
        *,
        source_turn: str | None = None,
        request_type: Any = "full_itinerary",
        delivery_intent: Any | None = None,
        trace: Any | None = None,
    ) -> IntentNormalizationResult:
        raw = _clean_raw_interests(raw_interests)
        normalized: list[NormalizedInterest] = []
        unresolved: list[str] = []
        llm_inputs: list[str] = []
        eligibility: list[dict[str, Any]] = []
        for text in raw:
            matches = _deterministic_matches(text, source_turn)
            if matches:
                normalized.extend(matches)
                eligibility.append(
                    {"source_text": text, "eligible": False, "reason_code": "deterministic_match"}
                )
            else:
                decision = intent_eligibility(
                    text,
                    request_type=request_type,
                    delivery_intent=delivery_intent,
                )
                eligibility.append({"source_text": text, **decision.to_dict()})
                if decision.eligible:
                    llm_inputs.append(text)

        detail: dict[str, Any] = {
            "eligible": bool(llm_inputs),
            "called": False,
            "skipped_reason": None,
            "model": "",
            "prompt_version": INTENT_NORMALIZER_PROMPT_VERSION,
            "schema_version": INTENT_NORMALIZER_SCHEMA_VERSION,
            "taxonomy_version": INTEREST_TAXONOMY_VERSION,
            "source_turn": source_turn,
            "candidate_ids": [],
            "structured_output": None,
            "confidence": None,
            "fallback_reason": None,
            "latency_ms": None,
            "token_usage": {},
            "out_of_taxonomy_label_count": 0,
            "unknown_candidate_reference_count": 0,
            "attempted_hard_constraint_change": False,
            "eligibility_reasons": eligibility,
            "input_fingerprint": _fingerprint(llm_inputs),
            "cache_reason": None,
            "parse_status": None,
            "validation_status": None,
            "parse_reason_code": None,
            "raw_output_summary": {},
            "repair_attempted": False,
            "attempts": 0,
        }
        fallback_reason: str | None = None
        if not llm_inputs:
            if not raw:
                detail["skipped_reason"] = "empty_interest"
            elif all(item["reason_code"] == "deterministic_match" for item in eligibility):
                detail["skipped_reason"] = "deterministic_match"
            else:
                detail["skipped_reason"] = next(
                    item["reason_code"] for item in eligibility
                    if item["reason_code"] != "deterministic_match"
                )
        elif not self.enabled:
            unresolved.extend(llm_inputs)
            fallback_reason = "feature_disabled"
            detail["skipped_reason"] = "feature_disabled"
            detail["fallback_reason"] = fallback_reason
        elif self.llm_client is None:
            unresolved.extend(llm_inputs)
            fallback_reason = "llm_unavailable"
            detail["skipped_reason"] = "llm_unavailable"
            detail["fallback_reason"] = fallback_reason
        else:
            fingerprint = _fingerprint(llm_inputs)
            reservation, cached = self.cache.reserve(
                self.request_scope, "intent", fingerprint
            )
            if reservation == "in_progress":
                ready, cached = self.cache.wait(cached, timeout_seconds=65.0)
                reservation = "cache_hit" if ready else "dedup_wait_timeout"
            if reservation == "cache_hit" and isinstance(cached, IntentNormalizationResult):
                detail["cache_reason"] = "valid_cache_hit" if not cached.fallback_reason else "dedup_after_fallback"
                detail["skipped_reason"] = detail["cache_reason"]
                detail["fallback_reason"] = cached.fallback_reason
                detail["validation_status"] = "cached"
                _append_trace(trace, detail)
                return _merge_cached_intent(raw, normalized, cached)
            if reservation == "dedup_wait_timeout":
                unresolved.extend(llm_inputs)
                fallback_reason = "dedup_wait_timeout"
                detail["cache_reason"] = fallback_reason
                detail["fallback_reason"] = fallback_reason
                result = IntentNormalizationResult(
                    raw_interests=raw,
                    normalized_interests=_dedupe(normalized),
                    unresolved=list(dict.fromkeys(unresolved)),
                    fallback_reason=fallback_reason,
                )
                _append_trace(trace, detail)
                return result
            detail["called"] = True
            try:
                response = self.llm_client.complete_json(
                    system_prompt=_SYSTEM_PROMPT,
                    user_prompt=json.dumps(
                        {
                            "source_turn": source_turn,
                            "raw_interests": llm_inputs,
                            "taxonomy": taxonomy_prompt_payload(),
                        },
                        ensure_ascii=False,
                    ),
                    schema_version=INTENT_NORMALIZER_SCHEMA_VERSION,
                )
                detail["model"] = response.model
                detail["latency_ms"] = response.latency_ms
                detail["token_usage"] = dict(response.token_usage)
                detail["structured_output"] = response.payload
                detail["parse_status"] = response.parse_status
                detail["parse_reason_code"] = response.parse_reason_code
                detail["raw_output_summary"] = dict(response.raw_output_summary)
                detail["repair_attempted"] = response.repair_attempted
                detail["attempts"] = response.attempts
                attempted_hard_change = bool(
                    isinstance(response.payload, dict)
                    and _FORBIDDEN_OUTPUT_KEYS.intersection(response.payload)
                )
                detail["attempted_hard_constraint_change"] = attempted_hard_change
                if attempted_hard_change:
                    accepted, rejected, low_confidence, model_unresolved = [], 0, set(), set()
                else:
                    accepted, rejected, low_confidence, model_unresolved = self._validate_llm_output(
                        response.payload, llm_inputs, source_turn
                    )
                normalized.extend(accepted)
                detail["out_of_taxonomy_label_count"] = rejected
                confidence_values = [item.confidence for item in accepted]
                detail["confidence"] = (
                    round(sum(confidence_values) / len(confidence_values), 4)
                    if confidence_values else None
                )
                resolved_sources = {
                    owner
                    for item in accepted
                    if (owner := _traceable_source(item.source_text, llm_inputs)) is not None
                }
                unresolved.extend(
                    text for text in llm_inputs
                    if text in model_unresolved
                    or text not in resolved_sources
                    or text in low_confidence
                )
                if attempted_hard_change:
                    fallback_reason = "attempted_hard_constraint_change"
                    detail["validation_status"] = "schema_invalid"
                    detail["parse_reason_code"] = fallback_reason
                elif rejected:
                    fallback_reason = "taxonomy_invalid"
                    detail["parse_status"] = "taxonomy_invalid"
                    detail["validation_status"] = "taxonomy_invalid"
                    detail["parse_reason_code"] = "taxonomy_label_out_of_range"
                    normalized = [item for item in normalized if item.basis != "llm"]
                    unresolved.extend(llm_inputs)
                elif low_confidence and not accepted:
                    fallback_reason = "low_confidence"
                    detail["parse_status"] = "low_confidence"
                    detail["validation_status"] = "low_confidence"
                    detail["parse_reason_code"] = "all_items_below_threshold"
                elif low_confidence:
                    detail["validation_status"] = "valid_with_unresolved"
                    detail["parse_reason_code"] = "some_items_below_threshold"
                elif not accepted and set(llm_inputs).issubset(model_unresolved):
                    detail["validation_status"] = "valid_with_unresolved"
                    detail["parse_reason_code"] = "model_reported_unresolved"
                elif not accepted:
                    fallback_reason = "schema_invalid"
                    detail["parse_status"] = "schema_invalid"
                    detail["validation_status"] = "schema_invalid"
                    detail["parse_reason_code"] = "no_valid_items"
                else:
                    detail["validation_status"] = "valid"
            except StructuredOutputError as exc:
                unresolved.extend(llm_inputs)
                fallback_reason = exc.reason_code
                detail["parse_status"] = exc.parse_status
                detail["validation_status"] = exc.parse_status
                detail["parse_reason_code"] = exc.reason_code
                detail["raw_output_summary"] = dict(exc.raw_output_summary)
                detail["latency_ms"] = exc.latency_ms
                detail["token_usage"] = dict(exc.token_usage)
                detail["repair_attempted"] = exc.repair_attempted
                detail["attempts"] = exc.attempts
            except Exception as exc:  # injected clients and validator failures
                unresolved.extend(llm_inputs)
                fallback_reason, status, reason = _validation_failure(exc)
                detail["parse_status"] = status
                detail["validation_status"] = status
                detail["parse_reason_code"] = reason
            detail["fallback_reason"] = fallback_reason

        result = IntentNormalizationResult(
            raw_interests=raw,
            normalized_interests=_dedupe(normalized),
            unresolved=list(dict.fromkeys(unresolved)),
            fallback_reason=fallback_reason,
        )
        if llm_inputs and self.enabled and self.llm_client is not None:
            self.cache.publish(self.request_scope, "intent", _fingerprint(llm_inputs), result)
        _append_trace(trace, detail)
        return result

    def _validate_llm_output(
        self,
        payload: dict[str, Any],
        requested: list[str],
        source_turn: str | None,
    ) -> tuple[list[NormalizedInterest], int, set[str], set[str]]:
        if not isinstance(payload, dict):
            raise ValueError("schema:not_object")
        if "normalized_interests" not in payload or "unresolved" not in payload:
            raise ValueError("schema:missing_field")
        if not isinstance(payload["normalized_interests"], list) or not isinstance(payload["unresolved"], list):
            raise TypeError("schema:type_mismatch")
        accepted: list[NormalizedInterest] = []
        rejected = 0
        low_confidence: set[str] = set()
        model_unresolved: set[str] = set()
        for unresolved_item in payload["unresolved"]:
            if not isinstance(unresolved_item, str):
                raise TypeError("schema:unresolved_item_type_mismatch")
            owning_source = _traceable_source(unresolved_item.strip(), requested)
            if owning_source is None:
                raise ValueError("schema:unresolved_source_mismatch")
            model_unresolved.add(owning_source)
        for raw_item in payload["normalized_interests"]:
            if not isinstance(raw_item, dict):
                raise TypeError("schema:item_type_mismatch")
            required = {"label", "source_text", "confidence", "polarity"}
            if not required.issubset(raw_item):
                raise ValueError("schema:missing_item_field")
            if not isinstance(raw_item["label"], str) or not isinstance(raw_item["source_text"], str):
                raise TypeError("schema:string_type_mismatch")
            label = raw_item["label"].strip().lower()
            source_text = raw_item["source_text"].strip()
            if label not in CANONICAL_INTERESTS:
                rejected += 1
                continue
            owning_source = _traceable_source(source_text, requested)
            if owning_source is None:
                raise ValueError("schema:source_text_mismatch")
            raw_confidence = raw_item["confidence"]
            if isinstance(raw_confidence, bool) or not isinstance(raw_confidence, (int, float)):
                raise TypeError("schema:confidence_type_mismatch")
            confidence = float(raw_confidence)
            if not 0.0 <= confidence <= 1.0:
                raise ValueError("schema:confidence_out_of_range")
            if confidence < self.confidence_threshold:
                low_confidence.add(owning_source)
                continue
            polarity = raw_item["polarity"]
            if not isinstance(polarity, str):
                raise TypeError("schema:polarity_type_mismatch")
            polarity = polarity.lower()
            if polarity not in {"prefer", "avoid"}:
                raise ValueError("schema:polarity_enum_mismatch")
            accepted.append(
                NormalizedInterest(
                    label=label,
                    source_text=source_text,
                    confidence=confidence,
                    reason=str(raw_item.get("reason") or INTEREST_DEFINITIONS[label].description),
                    source_turn=source_turn,
                    polarity=polarity,  # type: ignore[arg-type]
                    basis="llm",
                )
            )
        return accepted, rejected, low_confidence, model_unresolved


def _clean_raw_interests(value: str | list[str]) -> list[str]:
    items = [value] if isinstance(value, str) else list(value)
    return list(dict.fromkeys(text for item in items if (text := str(item).strip())))


def intent_eligibility(
    text: str,
    *,
    request_type: Any = "full_itinerary",
    delivery_intent: Any | None = None,
) -> IntentEligibility:
    normalized = str(text or "").strip()
    task = str(getattr(request_type, "value", request_type)).lower()
    delivery = str(getattr(delivery_intent, "value", delivery_intent or "")).lower()
    if not normalized:
        return IntentEligibility(False, "empty_interest")
    if task not in _FULL_PLAN_TYPES:
        return IntentEligibility(False, "request_type_not_eligible")
    if delivery == "state_update_only":
        return IntentEligibility(False, "rebuild_pending_without_final_request")
    if _FACT_QUERY_RE.search(normalized) and not _INTEREST_SEMANTIC_RE.search(normalized):
        return IntentEligibility(False, "fact_query")
    if _MUST_VISIT_ONLY_RE.fullmatch(normalized):
        return IntentEligibility(False, "must_visit_only")
    if _CONSTRAINT_ONLY_RE.search(normalized) and not _INTEREST_SEMANTIC_RE.search(normalized):
        return IntentEligibility(False, "constraints_only")
    if not _INTEREST_SEMANTIC_RE.search(normalized):
        return IntentEligibility(False, "no_interest_semantics")
    if _CONSTRAINT_ONLY_RE.search(normalized) and not re.search(
        r"喜欢|偏好|氛围|体验|人文|气息|气质|像.{0,8}一样|商业化|烟火气|有年代感|工业遗迹|城市更新|发呆|咖啡",
        normalized,
    ):
        return IntentEligibility(False, "constraints_only")
    return IntentEligibility(True, "llm_required")


def _traceable_source(source_text: str, requested: list[str]) -> str | None:
    if not source_text:
        return None
    for original in requested:
        if source_text == original or source_text in original:
            return original
    return None


def _fingerprint(items: list[str]) -> str:
    encoded = json.dumps(
        {"raw": [re.sub(r"\s+", " ", item.strip()).lower() for item in items],
         "taxonomy": INTEREST_TAXONOMY_VERSION,
         "schema": INTENT_NORMALIZER_SCHEMA_VERSION},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def _merge_cached_intent(
    raw: list[str],
    deterministic: list[NormalizedInterest],
    cached: IntentNormalizationResult,
) -> IntentNormalizationResult:
    return IntentNormalizationResult(
        raw_interests=raw,
        normalized_interests=_dedupe([*deterministic, *cached.normalized_interests]),
        unresolved=list(cached.unresolved),
        fallback_reason=cached.fallback_reason,
    )


def _deterministic_matches(text: str, source_turn: str | None) -> list[NormalizedInterest]:
    lowered = text.lower()
    matches: list[NormalizedInterest] = []
    # Canonical labels are an exact fast path.  Natural-language aliases are
    # scanned longest-first so a compound alias wins over its substring.
    if lowered in CANONICAL_INTERESTS:
        return [
            NormalizedInterest(
                label=lowered,
                source_text=text,
                confidence=1.0,
                reason="输入已经是受控 taxonomy 标签",
                source_turn=source_turn,
                basis="canonical",
            )
        ]
    occupied: list[tuple[int, int, str]] = []
    for alias, label in sorted(INTEREST_ALIAS_PAIRS, key=lambda item: len(item[0]), reverse=True):
        for match in re.finditer(re.escape(alias), text, re.IGNORECASE):
            if any(start <= match.start() and match.end() <= end and other == label for start, end, other in occupied):
                continue
            prefix = text[max(0, match.start() - 8):match.start()]
            polarity: Literal["prefer", "avoid"] = (
                "avoid" if any(marker in prefix for marker in _NEGATION_MARKERS) else "prefer"
            )
            matches.append(
                NormalizedInterest(
                    label=label,
                    source_text=text,
                    confidence=1.0,
                    reason=INTEREST_DEFINITIONS[label].description,
                    source_turn=source_turn,
                    polarity=polarity,
                    basis="deterministic_alias",
                )
            )
            occupied.append((match.start(), match.end(), label))
    return _dedupe(matches)


def _dedupe(items: list[NormalizedInterest]) -> list[NormalizedInterest]:
    seen: set[tuple[str, str, str]] = set()
    result: list[NormalizedInterest] = []
    for item in items:
        key = (item.label, item.source_text, item.polarity)
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def _validation_failure(exc: Exception) -> tuple[str, str, str]:
    name = type(exc).__name__.lower()
    if "timeout" in name or "timed out" in str(exc).lower():
        return "timeout", "timeout", "timeout"
    if isinstance(exc, (ValueError, TypeError, KeyError)):
        reason = str(exc)
        code = reason.removeprefix("schema:") if reason.startswith("schema:") else "schema_invalid"
        return "schema_invalid", "schema_invalid", code
    return "provider_error", "provider_error", "provider_error"


def _append_trace(trace: Any | None, detail: dict[str, Any]) -> None:
    if trace is None:
        try:
            from travel_agent.orchestration.multi_agent.trace import current_trace

            trace = current_trace()
        except Exception:
            trace = None
    if trace is not None:
        trace.append(
            "hybrid_intent_normalizer",
            agent="intent_normalizer",
            status="completed" if not detail.get("fallback_reason") else "fallback",
            duration_ms=detail.get("latency_ms"),
            detail=detail,
        )


_SYSTEM_PROMPT = """你是旅行软兴趣归一化器。只输出一个 JSON object，不要 markdown。
label 只能逐字选择输入 taxonomy 中的标签。source_text 必须逐字复制输入原文中的连续片段，
不得改写；正确区分 prefer 与 avoid。无法可靠归类时放入 unresolved，不要猜。
不得输出或修改预算、日期、人数、必去点、deadline、fixed event 或交通禁令。
固定结构：{\"normalized_interests\":[{\"label\":\"taxonomy_label\",\"source_text\":\"输入原文片段\",
\"confidence\":0.0,\"polarity\":\"prefer\",\"reason\":\"简短依据\"}],\"unresolved\":[\"原文\"]}。"""
