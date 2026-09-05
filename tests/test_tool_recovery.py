from __future__ import annotations

import pytest

from travel_agent.providers import ProviderRateLimitError
from travel_agent.tool_recovery import recover_search


class SequenceProvider:
    def __init__(self, values):
        self.values = list(values)
        self.calls = []

    def search_pois(self, **kwargs):
        self.calls.append(kwargs)
        value = self.values.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


def test_recovery_normalizes_name_then_preserves_hard_constraints() -> None:
    provider = SequenceProvider([[], ["candidate"]])
    result = recover_search(
        provider, city="宁波", query_tags=["海港博物馆"], category=None,
        max_results=5, hard_constraints={"accessibility_priority": True, "budget_max_cny": 5000},
    )
    assert result.items == ["candidate"]
    assert [item.stage for item in result.attempts] == ["exact", "normalized_name"]


def test_recovery_attributes_timeout_schema_and_real_no_supply() -> None:
    provider = SequenceProvider([TimeoutError("slow"), {"bad": "shape"}, []])
    result = recover_search(provider, city="洛阳", query_tags=["古迹博物馆"], category=None, max_results=5)
    assert [item.outcome for item in result.attempts] == [
        "provider_timeout", "schema_error", "empty"
    ]


def test_empty_chain_is_honest_and_does_not_invent_candidates() -> None:
    provider = SequenceProvider([[], []])
    result = recover_search(provider, city="烟台", query_tags=["海边"], category=None, max_results=5)
    assert result.items == []
    assert result.failure_kind == "no_supply"


def test_recovery_never_hides_provider_rate_limit_as_empty_supply() -> None:
    provider = SequenceProvider([
        ProviderRateLimitError("AMap rate limit: USER_DAILY_QUERY_OVER_LIMIT (10044)")
    ])

    with pytest.raises(ProviderRateLimitError, match="10044"):
        recover_search(
            provider,
            city="杭州",
            query_tags=["景点"],
            category="scenic",
            max_results=5,
        )
