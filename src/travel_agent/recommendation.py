from __future__ import annotations

from travel_agent.critic import poi_matches_interest, requested_interests
from travel_agent.schemas import POI, ScoredPOI, TravelProfile


def score_pois(pois: list[POI], profile: TravelProfile) -> list[ScoredPOI]:
    """Score candidate POIs as planning inputs, not as final recommendations."""
    scored = [_score_one(poi, profile) for poi in pois]
    ranked = sorted(scored, key=lambda item: item.score, reverse=True)
    return diversify_ranked_pois(ranked)


def diversify_ranked_pois(ranked: list[ScoredPOI]) -> list[ScoredPOI]:
    """轻量多样性重排，避免前几名被单一 category 占满。"""
    selected: list[ScoredPOI] = []
    remaining = list(ranked)
    category_counts: dict[str, int] = {}

    while remaining:
        best_index = 0
        best_adjusted = -1.0
        for index, item in enumerate(remaining):
            category_penalty = 0.08 * category_counts.get(item.poi.category, 0)
            adjusted = item.score - category_penalty
            if adjusted > best_adjusted:
                best_adjusted = adjusted
                best_index = index
        item = remaining.pop(best_index)
        selected.append(item)
        category_counts[item.poi.category] = category_counts.get(item.poi.category, 0) + 1
    return selected


def _score_one(poi: POI, profile: TravelProfile) -> ScoredPOI:
    reasons: list[str] = []
    interest_match = _profile_interest_match(poi, profile)
    if interest_match > 0:
        reasons.append("匹配用户兴趣")

    popularity = max(0.0, min(1.0, poi.popularity))
    rating = max(0.0, min(1.0, poi.rating / 5.0))
    pace_fit = _pace_fit(poi, profile) #按出行节奏匹配时长阈值
    budget_fit = _budget_fit(poi, profile)
    constraint_boost = _constraint_boost(poi, profile) # 命中 must_visit 给 1.0；命中 avoid 给 0.0；否则 0.5

    score = (
        0.30 * interest_match
        + 0.20 * popularity
        + 0.15 * rating
        + 0.15 * pace_fit
        + 0.10 * budget_fit
        + 0.10 * constraint_boost
    )

    if pace_fit >= 0.8:
        reasons.append("符合旅行节奏")
    if poi.popularity >= 0.85:
        reasons.append("城市热门亮点")
    if budget_fit >= 0.8 and profile.budget_level:
        reasons.append("符合预算偏好")
    if constraint_boost >= 1.0:
        reasons.append("满足强约束")

    return ScoredPOI(poi=poi, score=round(score, 4), reasons=reasons)


def _interest_match(poi: POI, interests: list[str]) -> float:
    if not interests:
        return 0.5
    overlap = set(poi.tags).intersection(interests)
    return len(overlap) / max(1, len(set(interests)))


def _profile_interest_match(poi: POI, profile: TravelProfile) -> float:
    interests = requested_interests(profile)
    if not interests:
        return 0.5
    matched = sum(1 for interest in interests if poi_matches_interest(poi, interest))
    return matched / len(interests)


def _pace_fit(poi: POI, profile: TravelProfile) -> float:
    if profile.pace == "relaxed" and poi.estimated_duration_min <= 150:
        return 1.0
    if profile.pace == "intensive" and poi.estimated_duration_min <= 240:
        return 1.0
    if profile.pace == "standard" and poi.estimated_duration_min <= 180:
        return 1.0
    return 0.5


def _budget_fit(poi: POI, profile: TravelProfile) -> float:
    if not profile.budget_level:
        return 0.7
    if profile.budget_level == "low":
        return {"free": 1.0, "low": 1.0, "mid": 0.5, "high": 0.1}.get(poi.price_level, 0.5)
    if profile.budget_level == "mid":
        return {"free": 0.8, "low": 0.9, "mid": 1.0, "high": 0.4}.get(poi.price_level, 0.6)
    return {"free": 0.7, "low": 0.7, "mid": 0.9, "high": 1.0}.get(poi.price_level, 0.7)


def _constraint_boost(poi: POI, profile: TravelProfile) -> float:
    if profile.avoid and any(term in poi.name for term in profile.avoid):
        return 0.0
    if profile.must_visit and any(term in poi.name for term in profile.must_visit):
        return 1.0
    return 0.5
