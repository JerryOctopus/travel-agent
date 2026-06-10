from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import urlencode
from urllib.request import urlopen

from travel_agent.config import ToolProviderConfig, load_tool_provider_config
from travel_agent.data_loader import load_seed_pois
from travel_agent.schemas import POI, RouteInfo, TransportMode, WeatherInfo
from travel_agent.tools import estimate_route_minutes, search_poi


class TravelToolProvider(Protocol):
    def search_pois(
        self,
        city: str,
        query_tags: list[str] | None = None,
        category: str | None = None,
        max_results: int = 20,
    ) -> list[POI]:
        ...

    def get_weather(self, city: str) -> WeatherInfo:
        ...

    def estimate_route(
        self,
        origin: POI,
        destination: POI,
        mode: TransportMode = "public_transport",
    ) -> RouteInfo:
        ...


@dataclass
class LocalToolProvider:
    pois: list[POI]

    def search_pois(
        self,
        city: str,
        query_tags: list[str] | None = None,
        category: str | None = None,
        max_results: int = 20,
    ) -> list[POI]:
        return search_poi(
            pois=self.pois,
            city=city,
            query_tags=query_tags,
            category=category,
            max_results=max_results,
        )

    def get_weather(self, city: str) -> WeatherInfo:
        from travel_agent.weather import get_weather

        return get_weather(city)

    def estimate_route(
        self,
        origin: POI,
        destination: POI,
        mode: TransportMode = "public_transport",
    ) -> RouteInfo:
        distance_km, duration_min = estimate_route_minutes(origin, destination, mode)
        return RouteInfo(
            origin_poi_id=origin.poi_id,
            destination_poi_id=destination.poi_id,
            distance_km=distance_km,
            duration_min=duration_min,
            mode=mode,
            source="haversine_estimate",
        )


@dataclass
class AmapToolProvider:
    api_key: str
    base_url: str = "https://restapi.amap.com"
    timeout_seconds: int = 5

    def search_pois(
        self,
        city: str,
        query_tags: list[str] | None = None,
        category: str | None = None,
        max_results: int = 20,
    ) -> list[POI]:
        keywords = _build_keywords(city, query_tags, category)
        payload = self._get_json(
            "/v3/place/text",
            {
                "key": self.api_key,
                "keywords": keywords,
                "city": city,
                "offset": str(max_results),
                "page": "1",
                "extensions": "all",
            },
        )
        if payload.get("status") != "1":
            return []
        pois = payload.get("pois", [])
        return [
            _amap_poi_to_schema(item, city=city, rank=index)
            for index, item in enumerate(pois)
            if item.get("location")
        ]

    def get_weather(self, city: str) -> WeatherInfo:
        adcode = self._resolve_city_adcode(city)
        payload = self._get_json(
            "/v3/weather/weatherInfo",
            {
                "key": self.api_key,
                "city": adcode or city,
                "extensions": "base",
            },
        )
        lives = payload.get("lives") or []
        if payload.get("status") != "1" or not lives:
            return WeatherInfo(city=city, condition="unknown", temperature_c=25, source="amap")
        live = lives[0]
        temperature = _safe_int(live.get("temperature"), default=25)
        return WeatherInfo(
            city=city,
            condition=str(live.get("weather") or "unknown"),
            temperature_c=temperature,
            source="amap",
        )

    def estimate_route(
        self,
        origin: POI,
        destination: POI,
        mode: TransportMode = "public_transport",
    ) -> RouteInfo:
        path, params = _amap_route_request(origin, destination, mode)
        payload = self._get_json(path, {"key": self.api_key, **params})
        distance_m, duration_seconds = _parse_amap_route_payload(payload, mode)
        if distance_m <= 0 or duration_seconds <= 0:
            distance_km, duration_min = estimate_route_minutes(origin, destination, mode)
            return RouteInfo(
                origin_poi_id=origin.poi_id,
                destination_poi_id=destination.poi_id,
                distance_km=distance_km,
                duration_min=duration_min,
                mode=mode,
                source="amap_fallback_estimate",
            )
        return RouteInfo(
            origin_poi_id=origin.poi_id,
            destination_poi_id=destination.poi_id,
            distance_km=round(distance_m / 1000, 2),
            duration_min=max(1, round(duration_seconds / 60)),
            mode=mode,
            source="amap",
        )

    def _resolve_city_adcode(self, city: str) -> str | None:
        payload = self._get_json(
            "/v3/geocode/geo",
            {
                "key": self.api_key,
                "address": city,
                "city": city,
            },
        )
        geocodes = payload.get("geocodes") or []
        if payload.get("status") != "1" or not geocodes:
            return None
        return geocodes[0].get("adcode")

    def _get_json(self, path: str, params: dict[str, str]) -> dict:
        url = f"{self.base_url}{path}?{urlencode(params)}"
        with urlopen(url, timeout=self.timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))


@dataclass
class FallbackToolProvider:
    primary: TravelToolProvider
    fallback: TravelToolProvider

    def search_pois(
        self,
        city: str,
        query_tags: list[str] | None = None,
        category: str | None = None,
        max_results: int = 20,
    ) -> list[POI]:
        try:
            results = self.primary.search_pois(city, query_tags, category, max_results)
        except Exception:
            results = []
        if results:
            return results
        return self.fallback.search_pois(city, query_tags, category, max_results)

    def get_weather(self, city: str) -> WeatherInfo:
        try:
            weather = self.primary.get_weather(city)
        except Exception:
            weather = None
        if weather and weather.condition != "unknown":
            return weather
        return self.fallback.get_weather(city)

    def estimate_route(
        self,
        origin: POI,
        destination: POI,
        mode: TransportMode = "public_transport",
    ) -> RouteInfo:
        try:
            route = self.primary.estimate_route(origin, destination, mode)
        except Exception:
            route = None
        if route and route.duration_min > 0:
            return route
        return self.fallback.estimate_route(origin, destination, mode)


def build_tool_provider(poi_path) -> TravelToolProvider:
    """构建工具数据源。

    优先读取新版 ``settings``（``config.toml`` 的 ``[amap].web_key``）；为兼容旧用例，
    若新配置未启用高德，再回退到旧的环境变量配置 ``load_tool_provider_config``。
    任何情况下都以本地 seed 作为兜底。
    """
    local = LocalToolProvider(load_seed_pois(poi_path))

    try:
        from travel_agent.settings import get_settings

        amap_settings = get_settings().amap
    except Exception:
        amap_settings = None

    if amap_settings is not None and amap_settings.rest_enabled:
        amap = AmapToolProvider(
            api_key=amap_settings.web_key or "",
            base_url=amap_settings.base_url,
            timeout_seconds=amap_settings.timeout_seconds,
        )
        return FallbackToolProvider(primary=amap, fallback=local)

    config = load_tool_provider_config()
    if config.amap_enabled:
        amap = AmapToolProvider(
            api_key=config.amap_api_key or "",
            base_url=config.amap_base_url,
            timeout_seconds=config.timeout_seconds,
        )
        return FallbackToolProvider(primary=amap, fallback=local)
    return local


def _build_keywords(
    city: str,
    query_tags: list[str] | None,
    category: str | None,
) -> str:
    tags = query_tags or []
    mapped = [_TAG_TO_KEYWORD.get(tag, tag) for tag in tags]
    if category:
        mapped.append(_CATEGORY_TO_KEYWORD.get(category, category))
    return " ".join(mapped) or f"{city} 景点"


def _amap_poi_to_schema(item: dict, city: str, rank: int) -> POI:
    lng, lat = _parse_location(item.get("location", "0,0"))
    type_text = str(item.get("type") or "")
    category = _map_amap_category(type_text)
    biz_ext = item.get("biz_ext") if isinstance(item.get("biz_ext"), dict) else {}
    rating = _safe_float(biz_ext.get("rating"), default=4.0)
    return POI(
        poi_id=f"amap_{item.get('id') or item.get('name') or rank}",
        name=str(item.get("name") or "未知地点"),
        city=city,
        category=category,
        lat=lat,
        lng=lng,
        rating=rating,
        popularity=max(0.2, round(1 - rank * 0.03, 2)),
        tags=_tags_for_category(category),
        estimated_duration_min=_duration_for_category(category),
        price_level="mid",
        indoor=category in {"museum", "shopping", "food"},
        opening_hours=None,
        source="amap",
    )


def _parse_location(location: str) -> tuple[float, float]:
    lng_text, lat_text = location.split(",", 1)
    return float(lng_text), float(lat_text)


def _amap_route_request(
    origin: POI,
    destination: POI,
    mode: TransportMode,
) -> tuple[str, dict[str, str]]:
    params = {
        "origin": f"{origin.lng},{origin.lat}",
        "destination": f"{destination.lng},{destination.lat}",
    }
    if mode == "walk":
        return "/v3/direction/walking", params
    if mode == "public_transport":
        return (
            "/v3/direction/transit/integrated",
            {
                **params,
                "city": origin.city,
                "cityd": destination.city,
            },
        )
    return "/v3/direction/driving", params


def _parse_amap_route_payload(payload: dict, mode: TransportMode) -> tuple[float, float]:
    if payload.get("status") != "1":
        return 0.0, 0.0
    route = payload.get("route") or {}
    if mode == "public_transport":
        transits = route.get("transits") or []
        if not transits:
            return 0.0, 0.0
        transit = transits[0]
        return (
            _safe_float(transit.get("distance"), default=0.0),
            _safe_float(transit.get("duration"), default=0.0),
        )
    paths = route.get("paths") or []
    if not paths:
        return 0.0, 0.0
    path = paths[0]
    return (
        _safe_float(path.get("distance"), default=0.0),
        _safe_float(path.get("duration"), default=0.0),
    )


def _safe_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _map_amap_category(type_text: str) -> str:
    if "餐饮" in type_text:
        return "food"
    if "博物馆" in type_text or "展览" in type_text:
        return "museum"
    if "购物" in type_text:
        return "shopping"
    if "风景" in type_text or "旅游" in type_text:
        return "scenic"
    return "scenic"


def _tags_for_category(category: str) -> list[str]:
    return {
        "food": ["food", "local"],
        "museum": ["history", "culture", "rainy-day"],
        "shopping": ["shopping", "citywalk"],
        "scenic": ["classic", "sightseeing"],
    }.get(category, ["classic"])


def _duration_for_category(category: str) -> int:
    return {
        "food": 60,
        "museum": 120,
        "shopping": 90,
        "scenic": 120,
    }.get(category, 90)


_TAG_TO_KEYWORD = {
    "nature": "自然风光",
    "food": "美食",
    "history": "历史文化",
    "culture": "文化",
    "shopping": "购物",
    "citywalk": "城市漫步",
}

_CATEGORY_TO_KEYWORD = {
    "food": "餐饮",
    "museum": "博物馆",
    "shopping": "购物",
    "scenic": "景点",
}
