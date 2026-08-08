from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from travel_agent.providers import TravelToolProvider
from travel_agent.schemas import POI, RouteInfo, TransportMode, WeatherInfo


@dataclass
class FaultInjectingProvider:
    """Deterministic provider wrapper used only by evaluation cases."""

    delegate: TravelToolProvider
    specification: dict[str, Any]
    _attempts: dict[str, int] = field(default_factory=dict)
    _events: list[dict[str, str]] = field(default_factory=list)

    def _mode(self, operation: str) -> str | None:
        self._attempts[operation] = self._attempts.get(operation, 0) + 1
        operations = self.specification.get("operations")
        if isinstance(operations, dict):
            value = operations.get(operation)
            if isinstance(value, dict):
                if self._attempts[operation] > 1 and not value.get("persistent", False):
                    return None
                mode = value.get("mode")
                if mode:
                    self._events.append({"operation": operation, "mode": str(mode)})
                return mode
            return str(value) if value else None
        if self.specification.get("operation") == operation:
            if self._attempts[operation] > 1 and not self.specification.get("persistent", False):
                return None
            mode = self.specification.get("mode")
            if mode:
                self._events.append({"operation": operation, "mode": str(mode)})
            return mode
        return None

    def consume_fault_events(self, operation: str) -> list[dict[str, str]]:
        matched = [event for event in self._events if event["operation"] == operation]
        self._events = [event for event in self._events if event["operation"] != operation]
        return matched

    def search_pois(
        self,
        city: str,
        query_tags: list[str] | None = None,
        category: str | None = None,
        max_results: int = 20,
    ) -> list[POI]:
        mode = self._mode("search_pois")
        if mode == "error" or mode == "timeout":
            raise TimeoutError("injected search_pois failure")
        if mode == "empty":
            return []
        results = self.delegate.search_pois(city, query_tags, category, max_results)
        if mode == "partial":
            return results[:1]
        return results

    def get_weather(self, city: str) -> WeatherInfo:
        mode = self._mode("get_weather")
        if mode == "error" or mode == "timeout":
            raise TimeoutError("injected get_weather failure")
        if mode == "empty":
            return WeatherInfo(city=city, condition="unknown", temperature_c=25, source="injected")
        return self.delegate.get_weather(city)

    def estimate_route(
        self,
        origin: POI,
        destination: POI,
        mode: TransportMode = "public_transport",
    ) -> RouteInfo:
        fault_mode = self._mode("estimate_route")
        if fault_mode in {"error", "timeout", "unavailable"}:
            raise TimeoutError("injected estimate_route failure")
        return self.delegate.estimate_route(origin, destination, mode)
