from __future__ import annotations

from collections import OrderedDict
import copy
import hashlib
import json
import re
import threading
import time
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import urlencode
from urllib.request import urlopen

from travel_agent.config import ToolProviderConfig, load_tool_provider_config
from travel_agent.data_loader import load_seed_pois
from travel_agent.schemas import POI, RouteInfo, TransportMode, WeatherInfo
from travel_agent.route_evidence import normalize_route_evidence
from travel_agent.tools import estimate_route_minutes, search_poi


_AMAP_PLACE_CACHE_TTL_SECONDS = 3600.0
_AMAP_PLACE_CACHE_MAX_ENTRIES = 512
_AMAP_PLACE_CACHE: OrderedDict[tuple, tuple[float, dict]] = OrderedDict()
_AMAP_PLACE_CACHE_LOCK = threading.RLock()
_AMAP_ROUTE_CACHE_TTL_SECONDS = 300.0
_AMAP_ROUTE_CACHE_MAX_ENTRIES = 2048
_AMAP_ROUTE_CACHE: OrderedDict[tuple, tuple[float, RouteInfo]] = OrderedDict()
_AMAP_ROUTE_CACHE_LOCK = threading.RLock()
_AMAP_MIN_REQUEST_INTERVAL_SECONDS = 0.12
_AMAP_REQUEST_LOCK = threading.RLock()
_AMAP_LAST_REQUEST_AT = 0.0
_AMAP_TRANSIENT_LIMIT_RETRY_DELAYS = (0.35, 0.8)


class TravelToolProvider(Protocol):
    def search_pois(
        self,
        city: str,
        query_tags: list[str] | None = None,
        category: str | None = None,
        max_results: int = 20,
    ) -> list[POI]:
        ...

    def search_pois_nearby(
        self,
        city: str,
        anchor: POI,
        query_tags: list[str] | None = None,
        category: str | None = None,
        radius_m: int = 5000,
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


class ProviderRateLimitError(RuntimeError):
    """A provider quota/rate limit that must invalidate a real evaluation."""


def _raise_for_provider_limit(payload: dict) -> None:
    if str(payload.get("status")) == "1":
        return
    info = str(payload.get("info") or "")
    infocode = str(payload.get("infocode") or "")
    if "LIMIT" in info.upper() or infocode in {"10003", "10004", "10044"}:
        raise ProviderRateLimitError(
            f"AMap rate limit: {info or 'UNKNOWN_LIMIT'} ({infocode or 'unknown'})"
        )


def _is_transient_amap_limit(payload: dict) -> bool:
    if str(payload.get("status")) == "1":
        return False
    info = str(payload.get("info") or "").upper()
    infocode = str(payload.get("infocode") or "")
    is_limit = "LIMIT" in info or infocode in {"10003", "10004", "10021", "10044"}
    return bool(is_limit and "DAILY" not in info)


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

    def search_pois_nearby(
        self,
        city: str,
        anchor: POI,
        query_tags: list[str] | None = None,
        category: str | None = None,
        radius_m: int = 5000,
        max_results: int = 20,
    ) -> list[POI]:
        candidates = self.search_pois(
            city=city,
            query_tags=query_tags,
            category=category,
            max_results=max(len(self.pois), max_results),
        )
        nearby: list[tuple[float, POI]] = []
        for poi in candidates:
            distance_km, _ = estimate_route_minutes(anchor, poi, "walk")
            if distance_km * 1000 <= max(0, int(radius_m)):
                nearby.append((distance_km, poi))
        nearby.sort(key=lambda item: item[0])
        return [poi for _, poi in nearby[:max_results]]

    def estimate_route(
        self,
        origin: POI,
        destination: POI,
        mode: TransportMode = "public_transport",
    ) -> RouteInfo:
        distance_km, duration_min = estimate_route_minutes(origin, destination, mode)
        return normalize_route_evidence(RouteInfo(
            origin_poi_id=origin.poi_id,
            destination_poi_id=destination.poi_id,
            distance_km=distance_km,
            duration_min=duration_min,
            mode=mode,
            source="haversine_estimate",
            walking_distance_km=distance_km if mode == "walk" else None,
        ), provider_explicit=False)


@dataclass
class AmapToolProvider:
    api_key: str
    base_url: str = "https://restapi.amap.com"
    timeout_seconds: int = 5

    @classmethod
    def clear_place_cache(cls) -> None:
        with _AMAP_PLACE_CACHE_LOCK:
            _AMAP_PLACE_CACHE.clear()

    @classmethod
    def clear_route_cache(cls) -> None:
        with _AMAP_ROUTE_CACHE_LOCK:
            _AMAP_ROUTE_CACHE.clear()

    def _route_cache_key(
        self,
        origin: POI,
        destination: POI,
        mode: TransportMode,
    ) -> tuple | None:
        if "_get_json" in self.__dict__:
            return None
        return (
            hashlib.sha256(self.api_key.encode()).hexdigest()[:16],
            self.base_url.rstrip("/"),
            id(type(self)._get_json),
            origin.source_poi_id or origin.poi_id,
            round(origin.lng, 6),
            round(origin.lat, 6),
            destination.source_poi_id or destination.poi_id,
            round(destination.lng, 6),
            round(destination.lat, 6),
            mode,
        )

    def _cached_route(self, key: tuple | None) -> RouteInfo | None:
        if key is None:
            return None
        now = time.monotonic()
        with _AMAP_ROUTE_CACHE_LOCK:
            cached = _AMAP_ROUTE_CACHE.get(key)
            if cached is not None and now - cached[0] <= _AMAP_ROUTE_CACHE_TTL_SECONDS:
                _AMAP_ROUTE_CACHE.move_to_end(key)
                return copy.deepcopy(cached[1])
            if cached is not None:
                _AMAP_ROUTE_CACHE.pop(key, None)
        return None

    def _store_route(self, key: tuple | None, route: RouteInfo) -> None:
        if key is None:
            return
        with _AMAP_ROUTE_CACHE_LOCK:
            _AMAP_ROUTE_CACHE[key] = (time.monotonic(), copy.deepcopy(route))
            _AMAP_ROUTE_CACHE.move_to_end(key)
            while len(_AMAP_ROUTE_CACHE) > _AMAP_ROUTE_CACHE_MAX_ENTRIES:
                _AMAP_ROUTE_CACHE.popitem(last=False)

    def _get_cached_place_json(
        self,
        params: dict[str, str],
        path: str = "/v5/place/text",
    ) -> dict:
        # Instance-level monkeypatches are test/injection hooks and must remain
        # fully observable rather than being shadowed by process cache state.
        if "_get_json" in self.__dict__:
            return self._get_json(path, params)
        safe_params = tuple(sorted((key, value) for key, value in params.items() if key != "key"))
        cache_key = (
            hashlib.sha256(self.api_key.encode()).hexdigest()[:16],
            self.base_url.rstrip("/"),
            id(type(self)._get_json),
            path,
            safe_params,
        )
        now = time.monotonic()
        with _AMAP_PLACE_CACHE_LOCK:
            cached = _AMAP_PLACE_CACHE.get(cache_key)
            if cached is not None and now - cached[0] <= _AMAP_PLACE_CACHE_TTL_SECONDS:
                _AMAP_PLACE_CACHE.move_to_end(cache_key)
                return copy.deepcopy(cached[1])
            if cached is not None:
                _AMAP_PLACE_CACHE.pop(cache_key, None)
            payload = self._get_json(path, params)
            # Never cache provider failures, especially quota/rate limits.
            if str(payload.get("status")) == "1":
                _AMAP_PLACE_CACHE[cache_key] = (now, copy.deepcopy(payload))
                _AMAP_PLACE_CACHE.move_to_end(cache_key)
                while len(_AMAP_PLACE_CACHE) > _AMAP_PLACE_CACHE_MAX_ENTRIES:
                    _AMAP_PLACE_CACHE.popitem(last=False)
            return payload

    def search_pois(
        self,
        city: str,
        query_tags: list[str] | None = None,
        category: str | None = None,
        max_results: int = 20,
    ) -> list[POI]:
        keywords = _build_keywords(city, query_tags, category)
        normalized_category = _normalize_amap_search_category(category)
        allow_cross_city = bool(
            _has_named_venue_query(query_tags)
            and normalized_category not in {"food", "hotel", "shopping"}
        )
        params = {
            "key": self.api_key,
            "keywords": keywords,
            "region": city,
            # A named venue may legitimately sit in a neighbouring city
            # (for example a fixed day trip).  Generic interests remain
            # city-scoped; explicit venue queries may search nationwide.
            "city_limit": "false" if allow_cross_city else "true",
            "page_size": str(min(25, max(1, int(max_results)))),
            "page_num": "1",
            "show_fields": "business",
        }
        type_code = _AMAP_CATEGORY_TYPE_CODE.get(normalized_category or "")
        if type_code:
            params["types"] = type_code
        payload = self._get_cached_place_json(params)
        _raise_for_provider_limit(payload)
        if payload.get("status") != "1":
            return []
        pois = payload.get("pois", [])
        results = [
            _amap_poi_to_schema(item, city=city, rank=index)
            for index, item in enumerate(pois)
            if item.get("location")
        ]
        # Provider taxonomy is evidence; a keyword hit in the wrong taxonomy
        # (for example an office building returned for a food query) must not
        # become a restaurant/hotel/activity candidate.
        if normalized_category:
            results = [
                item for item in results if item.category == normalized_category
            ]
        return results

    def search_pois_nearby(
        self,
        city: str,
        anchor: POI,
        query_tags: list[str] | None = None,
        category: str | None = None,
        radius_m: int = 5000,
        max_results: int = 20,
    ) -> list[POI]:
        """Search provider-classified POIs around an evidenced anchor."""
        normalized_category = _normalize_amap_search_category(category)
        terms = [str(term).strip() for term in (query_tags or []) if str(term).strip()]
        params = {
            "key": self.api_key,
            "location": f"{anchor.lng:.6f},{anchor.lat:.6f}",
            "radius": str(min(50000, max(1, int(radius_m)))),
            "sortrule": "distance",
            "region": city,
            "city_limit": "true",
            "page_size": str(min(25, max(1, int(max_results)))),
            "page_num": "1",
            "show_fields": "business",
        }
        if terms:
            # Place Search 2.0 around accepts one free-text keyword.  Keep the
            # hard taxonomy in ``types`` and use the joined phrase only as a
            # relevance hint.
            params["keywords"] = " ".join(terms)[:80]
        type_code = _AMAP_CATEGORY_TYPE_CODE.get(normalized_category or "")
        if type_code:
            params["types"] = type_code
        payload = self._get_cached_place_json(params, path="/v5/place/around")
        _raise_for_provider_limit(payload)
        if payload.get("status") != "1":
            return []
        results = [
            _amap_poi_to_schema(item, city=city, rank=index)
            for index, item in enumerate(payload.get("pois") or [])
            if item.get("location")
        ]
        if normalized_category:
            results = [
                item for item in results if item.category == normalized_category
            ]
        return results

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
        _raise_for_provider_limit(payload)
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
        cache_key = self._route_cache_key(origin, destination, mode)
        cached = self._cached_route(cache_key)
        if cached is not None:
            return cached
        path, params = _amap_route_request(origin, destination, mode)
        request_params = {"key": self.api_key, **params}
        distance_m = duration_seconds = 0.0
        walking_distance_m: float | None = None
        # AMap occasionally returns status=1 with an empty path/transit list.
        # One bounded retry recovers that transient response while preserving
        # the explicit fallback status if both responses remain empty.
        for attempt in range(2):
            try:
                payload = self._get_json(path, request_params)
                _raise_for_provider_limit(payload)
            except ProviderRateLimitError:
                raise
            except Exception:  # noqa: BLE001 - retry once, then let fallback own it
                if attempt:
                    raise
                continue
            distance_m, duration_seconds, walking_distance_m = _parse_amap_route_payload(
                payload, mode
            )
            if distance_m > 0 and duration_seconds > 0:
                break
        if distance_m <= 0 or duration_seconds <= 0:
            distance_km, duration_min = estimate_route_minutes(origin, destination, mode)
            return normalize_route_evidence(RouteInfo(
                origin_poi_id=origin.poi_id,
                destination_poi_id=destination.poi_id,
                distance_km=distance_km,
                duration_min=duration_min,
                mode=mode,
                source="amap_fallback_estimate",
                walking_distance_km=distance_km if mode == "walk" else None,
            ), provider_explicit=False)
        route = normalize_route_evidence(RouteInfo(
            origin_poi_id=origin.poi_id,
            destination_poi_id=destination.poi_id,
            distance_km=round(distance_m / 1000, 2),
            duration_min=max(1, round(duration_seconds / 60)),
            mode=mode,
            source="amap",
            walking_distance_km=(
                round(walking_distance_m / 1000, 2)
                if walking_distance_m is not None
                else (round(distance_m / 1000, 2) if mode == "walk" else None)
            ),
        ), provider_explicit=True)
        self._store_route(cache_key, route)
        return route

    def _resolve_city_adcode(self, city: str) -> str | None:
        payload = self._get_json(
            "/v3/geocode/geo",
            {
                "key": self.api_key,
                "address": city,
                "city": city,
            },
        )
        _raise_for_provider_limit(payload)
        geocodes = payload.get("geocodes") or []
        if payload.get("status") != "1" or not geocodes:
            return None
        return geocodes[0].get("adcode")

    def _get_json(self, path: str, params: dict[str, str]) -> dict:
        global _AMAP_LAST_REQUEST_AT

        url = f"{self.base_url}{path}?{urlencode(params)}"
        for attempt in range(len(_AMAP_TRANSIENT_LIMIT_RETRY_DELAYS) + 1):
            with _AMAP_REQUEST_LOCK:
                wait_seconds = max(
                    0.0,
                    _AMAP_MIN_REQUEST_INTERVAL_SECONDS
                    - (time.monotonic() - _AMAP_LAST_REQUEST_AT),
                )
                if wait_seconds:
                    time.sleep(wait_seconds)
                with urlopen(url, timeout=self.timeout_seconds) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                _AMAP_LAST_REQUEST_AT = time.monotonic()
            if not _is_transient_amap_limit(payload):
                return payload
            if attempt < len(_AMAP_TRANSIENT_LIMIT_RETRY_DELAYS):
                time.sleep(_AMAP_TRANSIENT_LIMIT_RETRY_DELAYS[attempt])
        return payload


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
        except ProviderRateLimitError:
            raise
        except Exception:
            results = []
        if results:
            return results
        return self.fallback.search_pois(city, query_tags, category, max_results)

    def get_weather(self, city: str) -> WeatherInfo:
        try:
            weather = self.primary.get_weather(city)
        except ProviderRateLimitError:
            raise
        except Exception:
            weather = None
        if weather and weather.condition != "unknown":
            return weather
        return self.fallback.get_weather(city)

    def search_pois_nearby(
        self,
        city: str,
        anchor: POI,
        query_tags: list[str] | None = None,
        category: str | None = None,
        radius_m: int = 5000,
        max_results: int = 20,
    ) -> list[POI]:
        args = {
            "city": city,
            "anchor": anchor,
            "query_tags": query_tags,
            "category": category,
            "radius_m": radius_m,
            "max_results": max_results,
        }
        try:
            nearby_search = getattr(self.primary, "search_pois_nearby", None)
            results = nearby_search(**args) if callable(nearby_search) else []
        except ProviderRateLimitError:
            raise
        except Exception:
            results = []
        if results:
            return results
        fallback_search = getattr(self.fallback, "search_pois_nearby", None)
        return fallback_search(**args) if callable(fallback_search) else []

    def estimate_route(
        self,
        origin: POI,
        destination: POI,
        mode: TransportMode = "public_transport",
    ) -> RouteInfo:
        try:
            route = self.primary.estimate_route(origin, destination, mode)
        except ProviderRateLimitError:
            raise
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
    tags = [
        str(item).strip()
        for item in (query_tags or [])
        if str(item).strip() and not _is_non_venue_query_constraint(item)
    ]
    # AMap v5 accepts multiple OR keywords separated by ``|``.  Keep explicit
    # venue names first so a fixed/must-visit anchor is not displaced by broad
    # interests.  ``types`` independently constrains the provider taxonomy.
    named = [tag for tag in tags if tag not in _TAG_TO_KEYWORD]
    generic = [_TAG_TO_KEYWORD[tag] for tag in tags if tag in _TAG_TO_KEYWORD]
    keywords = list(dict.fromkeys([*named, *generic]))

    normalized_category = _normalize_amap_search_category(category)
    # For activities, the category keyword widens sparse preference queries
    # while the taxonomy filter rejects wrong entity types.  Area-scoped food,
    # hotel and shopping queries must not be widened to every venue in the city.
    if normalized_category and normalized_category not in {"food", "hotel", "shopping"}:
        category_keyword = _CATEGORY_TO_KEYWORD.get(normalized_category)
        if category_keyword and category_keyword not in keywords:
            keywords.append(category_keyword)

    if keywords:
        return "|".join(keywords[:5])
    if category:
        return _CATEGORY_TO_KEYWORD.get(normalized_category or "", str(category).strip())
    return "景点"


def _amap_poi_to_schema(item: dict, city: str, rank: int) -> POI:
    lng, lat = _parse_location(item.get("location", "0,0"))
    name = str(item.get("name") or "未知地点")
    type_text = str(item.get("type") or "")
    category = _map_amap_category(type_text)
    entity_type = _map_amap_entity_type(type_text)
    business = item.get("business") if isinstance(item.get("business"), dict) else {}
    biz_ext = item.get("biz_ext") if isinstance(item.get("biz_ext"), dict) else {}
    rating = _safe_float(
        business.get("rating") or biz_ext.get("rating"), default=4.0
    )
    average_cost = _safe_optional_float(
        business.get("cost") or biz_ext.get("cost")
    )
    opening_hours = _optional_amap_text(
        business.get("opentime_week")
        or business.get("opentime_today")
        or biz_ext.get("opentime2")
        or biz_ext.get("open_time")
    )
    actual_city = _optional_amap_text(item.get("cityname")) or city
    district = _optional_amap_text(item.get("adname"))
    street_address = _optional_amap_text(item.get("address"))
    if district and street_address and district not in street_address:
        address = district + street_address
    else:
        address = street_address or district
    return POI(
        poi_id=f"amap_{item.get('id') or name or rank}",
        name=name,
        city=actual_city,
        category=category,
        lat=lat,
        lng=lng,
        rating=rating,
        popularity=max(0.2, round(1 - rank * 0.03, 2)),
        tags=_tags_for_category(category),
        estimated_duration_min=_duration_for_category(category, name=name, type_text=type_text),
        # v3 place/text does not expose a trustworthy price tier.  Preserve
        # that uncertainty instead of inventing "mid", which would wrongly
        # exclude a real venue from a user-requested low/high search.
        price_level="unknown",
        indoor=category in {"museum", "shopping", "food"},
        opening_hours=opening_hours,
        address=address,
        average_cost=average_cost,
        parking_type=_optional_amap_text(item.get("parking_type")),
        source="amap",
        canonical_name=name,
        entity_type=entity_type,
        source_poi_id=str(item.get("id") or name or rank),
        verification_status=(
            "evidence_insufficient"
            if entity_type == "unknown"
            else ("wrong_entity" if entity_type in {
                "beauty_service", "automotive_service", "commercial_service",
                "parking", "retail", "ticket_office", "visitor_center",
            } else "verified")
        ),
        verification_reason=(
            f"高德类型未映射为可游览实体：{type_text or 'unknown'}"
            if entity_type == "unknown"
            else None
        ),
        aliases=_amap_aliases(item.get("alias")),
        parent_poi_id=_optional_amap_text(item.get("parent")),
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


def _parse_amap_route_payload(
    payload: dict, mode: TransportMode
) -> tuple[float, float, float | None]:
    if payload.get("status") != "1":
        return 0.0, 0.0, None
    route = payload.get("route") or {}
    if mode == "public_transport":
        transits = route.get("transits") or []
        if not transits:
            return 0.0, 0.0, None
        transit = transits[0]
        return (
            _safe_float(transit.get("distance"), default=0.0),
            _safe_float(transit.get("duration"), default=0.0),
            _safe_float(transit.get("walking_distance"), default=0.0),
        )
    paths = route.get("paths") or []
    if not paths:
        return 0.0, 0.0, None
    path = paths[0]
    return (
        _safe_float(path.get("distance"), default=0.0),
        _safe_float(path.get("duration"), default=0.0),
        _safe_float(path.get("distance"), default=0.0) if mode == "walk" else None,
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


def _safe_optional_float(value) -> float | None:
    try:
        return float(value) if value not in (None, "", [], {}) else None
    except (TypeError, ValueError):
        return None


def _optional_amap_text(value) -> str | None:
    """Normalize optional AMap text fields without inventing data."""
    if value is None or isinstance(value, (list, dict)):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"[]", "null", "none"}:
        return None
    return text


def _map_amap_category(type_text: str) -> str:
    if any(term in type_text for term in ("公交车站", "地铁站", "交通设施", "停车场")):
        return "transport"
    if "住宿" in type_text or "酒店" in type_text or "宾馆" in type_text:
        return "hotel"
    if "餐饮" in type_text:
        return "food"
    if "博物馆" in type_text or "展览" in type_text:
        return "museum"
    if "购物" in type_text:
        return "shopping"
    if "风景" in type_text or "旅游" in type_text:
        return "scenic"
    return "unknown"


def _map_amap_entity_type(type_text: str) -> str:
    """Map provider taxonomy, never venue-name keywords, to entity identity."""
    text = str(type_text or "")
    if any(term in text for term in ("美容美发店", "美容美发", "美容服务")):
        return "beauty_service"
    if any(term in text for term in ("汽车美容", "汽车养护", "汽车维修")):
        return "automotive_service"
    if "停车场" in text:
        return "parking"
    if any(term in text for term in ("售票", "票务")):
        return "ticket_office"
    if "游客中心" in text:
        return "visitor_center"
    if "交通设施" in text or "车站" in text or "地铁站" in text:
        return "transport"
    if "住宿" in text or "酒店" in text or "宾馆" in text:
        return "hotel"
    if "餐饮" in text:
        return "restaurant"
    if any(term in text for term in ("特色商业街", "商业街", "步行街")):
        return "district"
    if "购物" in text or "商店" in text:
        return "retail"
    if "博物馆" in text or "展览" in text:
        return "museum"
    if "风景" in text or "旅游景点" in text or "公园广场" in text:
        return "attraction"
    return "unknown"


def _amap_aliases(value: object) -> list[str]:
    if isinstance(value, list):
        raw = value
    elif value in (None, "", [], {}):
        raw = []
    else:
        raw = re.split(r"[;；,，|/]", str(value))
    return list(dict.fromkeys(str(item).strip() for item in raw if str(item).strip()))


def _tags_for_category(category: str) -> list[str]:
    return {
        "food": ["food", "local"],
        "museum": ["history", "culture", "rainy-day"],
        "shopping": ["shopping", "citywalk"],
        "scenic": ["classic", "sightseeing"],
        "hotel": ["hotel", "accommodation"],
        "transport": ["transport"],
        "unknown": [],
    }.get(category, ["classic"])


def _duration_for_category(category: str, *, name: str = "", type_text: str = "") -> int:
    text = f"{name} {type_text}"
    if category == "food":
        return 60
    if category == "hotel":
        return 0
    if category == "transport":
        return 0
    if category == "shopping":
        if any(term in text for term in ("步行街", "夜市", "街区", "商业街")):
            return 90
        return 75
    if category == "museum":
        if any(term in text for term in ("国家", "省", "大型", "故宫", "博物院")):
            return 120
        return 90
    if category == "scenic":
        if any(term in text for term in ("风景名胜区", "国家森林公园", "国家级景点", "度假区")):
            return 150
        if any(term in text for term in ("山", "湖", "古镇", "古城", "峡", "园林")):
            return 120
        if any(term in text for term in ("公园", "广场", "街", "码头", "观景台")):
            return 90
        return 105
    return 90


_TAG_TO_KEYWORD = {
    "nature": "自然风光",
    "food": "美食",
    "local": "本地美食",
    "history": "历史文化",
    "culture": "文化",
    "classic": "经典景点",
    "sightseeing": "景点",
    "shopping": "购物",
    "citywalk": "城市漫步",
    "family": "亲子游乐",
    "museum": "博物馆",
    "night": "夜市美食",
    "餐厅": "美食",
    "hotel": "酒店",
    "酒店": "酒店",
    "住宿": "酒店",
    "公园": "公园",
    "好玩": "景点",
    "海边": "海边",
    "海岛": "海岛",
    "沙滩": "沙滩",
    "夜景": "夜景",
    "亲子": "亲子游乐",
    "历史": "历史文化",
    "文化": "文化",
    "自然": "自然风光",
}


def _has_named_venue_query(query_tags: list[str] | None) -> bool:
    return any(
        str(tag).strip()
        and not _is_non_venue_query_constraint(tag)
        and str(tag).strip() not in _TAG_TO_KEYWORD
        for tag in (query_tags or [])
    )


_NON_VENUE_QUERY_CONSTRAINT_MARKERS = (
    "老人",
    "老年",
    "长辈",
    "轮椅",
    "无障碍",
    "少步行",
    "步行少",
    "行动不便",
    "体力有限",
    "不爬坡",
    "轻松节奏",
)


def _is_non_venue_query_constraint(value: object) -> bool:
    """Keep user mobility constraints out of provider venue-name keywords."""
    text = re.sub(r"\s+", "", str(value or "").strip())
    return bool(text) and any(marker in text for marker in _NON_VENUE_QUERY_CONSTRAINT_MARKERS)

_CATEGORY_TO_KEYWORD = {
    "food": "餐饮",
    "hotel": "酒店",
    "museum": "博物馆",
    "shopping": "购物",
    "scenic": "景点",
}

_AMAP_CATEGORY_TYPE_CODE = {
    "food": "050000",
    "hotel": "100000",
    "scenic": "110000",
    "museum": "140100",
    "shopping": "060000",
}


def _normalize_amap_search_category(category: str | None) -> str | None:
    """Map user/model category wording to the provider taxonomy when known.

    Unknown values are usually named venues or free-form interests.  They stay
    in the keyword query but must not be used as an impossible exact taxonomy
    filter (for example ``category='鼓浪屿'``).
    """
    text = str(category or "").strip().lower()
    if not text:
        return None
    if text in _AMAP_CATEGORY_TYPE_CODE:
        return text
    if any(marker in text for marker in ("餐厅", "餐饮", "饭店", "美食")):
        return "food"
    if any(marker in text for marker in ("酒店", "住宿", "宾馆", "旅馆")):
        return "hotel"
    if any(marker in text for marker in ("博物馆", "博物院", "展览馆", "展馆")):
        return "museum"
    if any(marker in text for marker in ("购物", "商场", "商业街", "步行街")):
        return "shopping"
    if any(
        marker in text
        for marker in ("景点", "风景", "名胜", "自然", "户外", "公园", "海边")
    ):
        return "scenic"
    return None
