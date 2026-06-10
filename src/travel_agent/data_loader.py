from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from travel_agent.schemas import POI


def load_seed_pois(path: Path | str) -> list[POI]:
    raw_items = json.loads(Path(path).read_text(encoding="utf-8"))
    return [_poi_from_dict(item) for item in raw_items]


def _poi_from_dict(item: dict[str, Any]) -> POI:
    return POI(
        poi_id=item["poi_id"],
        name=item["name"],
        city=item["city"],
        category=item["category"],
        lat=float(item["lat"]),
        lng=float(item["lng"]),
        rating=float(item["rating"]),
        popularity=float(item["popularity"]),
        tags=list(item["tags"]),
        estimated_duration_min=int(item["estimated_duration_min"]),
        price_level=item["price_level"],
        indoor=bool(item.get("indoor", False)),
        opening_hours=item.get("opening_hours"),
        source=item.get("source", "seed"),
    )
