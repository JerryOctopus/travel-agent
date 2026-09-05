"""Provider-neutral empty-result recovery with structured attribution."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
import socket
from typing import Any, Callable

from travel_agent.providers import ProviderRateLimitError


@dataclass
class RecoveryAttempt:
    stage: str
    provider: str
    query: list[str]
    outcome: str
    detail: str = ""


@dataclass
class RecoveryResult:
    items: list[Any] = field(default_factory=list)
    attempts: list[RecoveryAttempt] = field(default_factory=list)
    unmet_constraints: list[str] = field(default_factory=list)

    @property
    def recovered(self) -> bool:
        return bool(self.items) and len(self.attempts) > 1

    @property
    def failure_kind(self) -> str | None:
        return None if self.items else (self.attempts[-1].outcome if self.attempts else "no_supply")


def _failure_kind(exc: Exception) -> str:
    text = f"{type(exc).__name__}: {exc}".lower()
    if isinstance(exc, (TimeoutError, socket.timeout)) or "timeout" in text or "timed out" in text:
        return "provider_timeout"
    if isinstance(exc, (TypeError, ValueError, KeyError)):
        return "schema_error"
    return "provider_error"


def _normalized_terms(values: list[str]) -> list[str]:
    cleaned: list[str] = []
    for value in values:
        term = re.sub(r"(?:附近|周边|景区|公园|博物馆|博物院)$", "", str(value).strip())
        if term and term not in cleaned:
            cleaned.append(term)
    return cleaned


def recover_search(
    provider: Any,
    *,
    city: str,
    query_tags: list[str] | None,
    category: str | None,
    max_results: int,
    hard_constraints: dict[str, Any] | None = None,
) -> RecoveryResult:
    """Run a bounded recovery chain without weakening any hard constraint."""
    original = [str(item) for item in (query_tags or []) if str(item).strip()]
    normalized = _normalized_terms(original)
    strategies: list[tuple[str, Any, list[str]]] = [
        ("exact", provider, original),
    ]
    if normalized and normalized != original:
        strategies.append(("normalized_name", provider, normalized))
    # Broaden discovery terms only; hard constraints remain attached and are
    # evaluated downstream. An empty query means city/nearby supply, not that
    # accessibility, dietary, budget, parking, or mobility constraints vanished.
    strategies.append(("broaden_keywords", provider, []))
    primary = getattr(provider, "primary", None)
    fallback = getattr(provider, "fallback", None)
    if primary is not None and fallback is not None:
        strategies = [(stage, primary, terms) for stage, _, terms in strategies]
        strategies.append(("alternate_provider", fallback, normalized or original))

    result = RecoveryResult()
    seen: set[tuple[str, tuple[str, ...], str]] = set()
    for stage, active, terms in strategies:
        key = (type(active).__name__, tuple(terms), stage)
        if key in seen:
            continue
        seen.add(key)
        try:
            items = active.search_pois(
                city=city,
                query_tags=terms or None,
                category=category,
                max_results=max_results,
            )
            if not isinstance(items, list):
                raise TypeError("provider search_pois must return a list")
        except ProviderRateLimitError:
            # A real-evaluation quota failure is not an empty-result condition
            # and must remain visible so the whole run can be marked invalid.
            raise
        except Exception as exc:  # noqa: BLE001 - attribution is the recovery contract
            result.attempts.append(RecoveryAttempt(stage, type(active).__name__, terms, _failure_kind(exc), str(exc)))
            continue
        result.attempts.append(RecoveryAttempt(stage, type(active).__name__, terms, "success" if items else "empty"))
        if items:
            result.items = items
            return result
    if result.attempts and all(item.outcome == "empty" for item in result.attempts):
        result.attempts.append(RecoveryAttempt("supply_assessment", type(provider).__name__, original, "no_supply"))
    result.unmet_constraints = sorted(str(key) for key, value in (hard_constraints or {}).items() if value not in (None, "", [], {}, False))
    return result
