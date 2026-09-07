from travel_agent.agent.session import build_session
from travel_agent.agent import toolkit
from travel_agent.agent.runtime import _wants_restaurants, run_production_turn
from travel_agent.agent.render import build_supplement_cards
from travel_agent.providers import LocalToolProvider
from travel_agent.schemas import POI, TravelProfile
from travel_agent.settings import LLMSettings, Settings


def _poi(name: str, category: str = "food", price_level: str = "mid") -> POI:
    return POI(
        poi_id=name,
        name=name,
        city="杭州",
        category=category,
        lat=30.25,
        lng=120.15,
        rating=4.6,
        popularity=0.8,
        tags=["food", "本帮菜"] if category == "food" else ["nature"],
        estimated_duration_min=90,
        price_level=price_level,
    )


def test_plannable_entity_dedupe_prefers_exact_required_venue() -> None:
    profile = TravelProfile(destination="成都", must_visit=["三星堆博物馆"])
    variants = [
        _poi("三星堆博物馆陈列馆", "museum"),
        _poi("三星堆博物馆", "museum"),
        _poi("三星堆博物馆-青铜馆", "museum"),
    ]

    result = toolkit._dedupe_plannable_entities(variants, profile)

    assert [poi.name for poi in result] == ["三星堆博物馆"]


def test_plannable_entity_dedupe_prefers_formal_parent_over_numbered_pit() -> None:
    profile = TravelProfile(destination="测试城", must_visit=["青铜遗址"])
    parent = POI(
        **{
            **_poi("古王朝青铜遗址博物馆", "museum").__dict__,
            "source_poi_id": "provider-parent",
        }
    )
    child = POI(
        **{
            **_poi("古王朝青铜遗址一号坑", "museum").__dict__,
            "parent_poi_id": "provider-parent",
        }
    )

    result = toolkit._dedupe_plannable_entities([child, parent], profile)

    assert [poi.name for poi in result] == ["古王朝青铜遗址博物馆"]


def test_unavailable_and_infrastructure_pois_are_not_plannable() -> None:
    assert toolkit._is_unavailable_or_infrastructure_poi(
        _poi("西湖风景区(暂停开放)", "scenic")
    )
    assert toolkit._is_unavailable_or_infrastructure_poi(
        _poi("观光电瓶车上客点", "scenic")
    )
    assert toolkit._is_unavailable_or_infrastructure_poi(
        _poi("兵马俑直通车乘车点", "scenic")
    )
    assert toolkit._is_unavailable_or_infrastructure_poi(
        _poi("历史景区连接线", "scenic")
    )
    assert toolkit._is_unavailable_or_infrastructure_poi(
        _poi("历史景区文创店", "scenic")
    )
    assert toolkit._is_unavailable_or_infrastructure_poi(
        _poi("广州塔旅游文化发展有限公司", "scenic")
    )


def test_poi_closed_on_requested_weekday_is_not_plannable() -> None:
    museum = POI(
        **{
            **_poi("天津博物馆", "museum").__dict__,
            "opening_hours": "周二至周日 09:00-16:30开放；周一 全天不开放；说明:周一闭馆",
        }
    )
    monday = TravelProfile(destination="天津", constraint_state={"weekday": "周一"})
    tuesday = TravelProfile(destination="天津", constraint_state={"weekday": "周二"})

    assert toolkit._is_unavailable_or_infrastructure_poi(museum, monday)
    assert not toolkit._is_unavailable_or_infrastructure_poi(museum, tuesday)


def test_poi_using_quan_tian_guan_bi_wording_is_not_plannable() -> None:
    museum = POI(
        **{
            **_poi("自然博物馆", "museum").__dict__,
            "opening_hours": "1月至9月 周一 全天关闭；周二至周日 09:00-16:30",
        }
    )
    monday = TravelProfile(destination="天津", constraint_state={"weekday": "周一"})

    assert toolkit._is_unavailable_or_infrastructure_poi(museum, monday)


def test_poi_outside_explicit_opening_season_is_not_plannable() -> None:
    summer_only = POI(
        **{
            **_poi("夏季主题景区", "scenic").__dict__,
            "opening_hours": "7月至8月 周一至周日 09:30-20:30",
        }
    )
    november = TravelProfile(destination="厦门", start_date="2026-11-06")
    july = TravelProfile(destination="厦门", start_date="2026-07-06")

    assert toolkit._is_unavailable_or_infrastructure_poi(summer_only, november)
    assert not toolkit._is_unavailable_or_infrastructure_poi(summer_only, july)


def test_multiple_opening_seasons_are_alternatives() -> None:
    all_year_ranges = POI(
        **{
            **_poi("杭州动物园", "scenic").__dict__,
            "opening_hours": "10月8日至次年4月3日07:00-17:00，4月4日至10月7日07:00-17:30",
        }
    )
    november = TravelProfile(destination="杭州", start_date="2026-11-06")

    assert not toolkit._is_unavailable_or_infrastructure_poi(all_year_ranges, november)


def test_dated_closed_exact_record_does_not_displace_open_venue_alias() -> None:
    closed_exact = POI(
        **{
            **_poi("古城墙", "scenic").__dict__,
            "opening_hours": "2.15日-2.23日 08:00-23:00；2.24日-3.4日 08:00-22:00",
        }
    )
    open_alias = POI(
        **{
            **_poi("古城墙历史文化景区", "scenic").__dict__,
            "poi_id": "open-alias",
            "opening_hours": "周一至周日 08:00-22:00",
        }
    )
    winter = TravelProfile(
        destination="测试城",
        days=3,
        start_date="2026-12-02",
        must_visit=["古城墙"],
    )

    eligible = [
        poi for poi in (closed_exact, open_alias)
        if not toolkit._is_unavailable_or_infrastructure_poi(poi, winter)
    ]
    deduped = toolkit._dedupe_plannable_entities(eligible, winter)

    assert [poi.poi_id for poi in deduped] == ["open-alias"]


def test_cross_city_candidate_requires_explicit_venue_constraint() -> None:
    required = POI(
        poi_id="sxd",
        name="三星堆博物馆",
        city="德阳市",
        category="museum",
        lat=31.0,
        lng=104.2,
        rating=4.9,
        popularity=1.0,
        tags=["history"],
        estimated_duration_min=120,
        price_level="mid",
    )
    optional = POI(**{**required.__dict__, "poi_id": "lake", "name": "金雁湖"})
    profile = TravelProfile(destination="成都", must_visit=["三星堆博物馆"])

    assert toolkit._poi_city_allowed_for_plan(required, profile)
    assert not toolkit._poi_city_allowed_for_plan(optional, profile)


def test_cross_city_fixed_event_venue_is_allowed_without_becoming_must_visit() -> None:
    venue = POI(
        poi_id="event-venue",
        name="三星文化博物馆",
        city="邻城市",
        category="museum",
        lat=31.0,
        lng=104.2,
        rating=4.9,
        popularity=1.0,
        tags=["history"],
        estimated_duration_min=120,
        price_level="mid",
    )
    profile = TravelProfile(
        destination="主城",
        must_visit=[],
        constraint_state={
            "fixed_events": [{
                "day": 1,
                "start": "14:00",
                "end": "16:00",
                "location": "三星文化博物馆",
            }],
        },
    )

    assert toolkit._poi_city_allowed_for_plan(venue, profile)


def test_restaurant_hotel_budget_tools() -> None:
    ctx = build_session(persist=False)
    ctx.provider = LocalToolProvider([_poi("西湖餐厅"), _poi("西湖酒店", "hotel"), _poi("西湖", "scenic")])
    ctx.profile = TravelProfile(
        destination="杭州",
        days=3,
        budget_level="mid",
        hotel_area="西湖",
        companions="情侣",
        constraint_state={"return_deadline": "18:30", "wheelchair_user": True},
    )

    restaurants = toolkit.search_restaurant(ctx, cuisine="本帮菜")
    hotels = toolkit.search_hotel(ctx)
    budget = toolkit.estimate_budget(ctx)
    constraints = toolkit.build_constraints(ctx)

    assert restaurants["isError"] is False
    assert restaurants["count"] == 1
    assert hotels["isError"] is False
    assert hotels["hotels"][0]["area"] == "西湖"
    assert hotels["hotels"][0]["source"] == "seed"
    assert budget["budget"]["companions"] == 2
    assert budget["budget"]["total_high"] > budget["budget"]["total_low"]
    assert constraints["constraints"]["destination"] == "杭州"
    assert constraints["constraints"]["return_deadline"] == "18:30"
    assert constraints["constraints"]["wheelchair_user"] is True


def test_coffee_interest_requests_restaurant_evidence() -> None:
    ctx = build_session(persist=False)
    ctx.profile = TravelProfile(
        destination="厦门",
        days=3,
        interests=["nature", "food"],
        constraint_state={"interests": ["海边", "咖啡店"]},
    )

    assert _wants_restaurants("喜欢海边和咖啡店", ctx) is True


def test_empty_structured_mirror_does_not_erase_compact_must_visit() -> None:
    ctx = build_session(persist=False)
    ctx.profile = TravelProfile(
        destination="成都",
        days=3,
        must_visit=["三星堆博物馆"],
        constraint_state={"must_visit": []},
    )

    result = toolkit.build_constraints(ctx)

    assert result["constraints"]["must_visit"] == ["三星堆博物馆"]


def test_restaurant_hotel_budget_cards() -> None:
    ctx = build_session(persist=False)
    ctx.provider = LocalToolProvider([
        _poi("西湖餐厅"), _poi("西湖酒店", "hotel"), _poi("西湖", "scenic")
    ])
    ctx.profile = TravelProfile(destination="杭州", days=2, budget_level="mid", hotel_area="西湖")

    toolkit.search_restaurant(ctx, cuisine="本帮菜")
    toolkit.search_hotel(ctx)
    toolkit.estimate_budget(ctx)

    cards = build_supplement_cards(
        restaurants=ctx.store.latest("restaurants"),
        hotels=ctx.store.latest("hotels"),
        budget=ctx.store.latest("budget"),
    )

    assert [card["type"] for card in cards] == ["restaurants", "hotels", "budget"]
    assert cards[0]["items"][0]["name"] == "西湖餐厅"
    assert cards[1]["items"][0]["area"] == "西湖"
    assert cards[2]["total_high"] > cards[2]["total_low"]


def test_fallback_includes_restaurant_hotel_budget_when_requested() -> None:
    ctx = build_session(persist=False)
    ctx.provider = LocalToolProvider(
        [
            _poi("西湖餐厅", "food"),
            _poi("西湖", "scenic"),
            _poi("杭州博物馆", "museum"),
            _poi("河坊街", "shopping"),
        ]
    )
    settings = Settings(llm=LLMSettings(provider="rule"))

    reply = run_production_turn(
        "帮我规划杭州两天，想吃本帮菜，住西湖附近，预算中等",
        ctx=ctx,
        settings=settings,
    )

    assert "search_restaurant" in reply.tool_trace
    assert "search_hotel" in reply.tool_trace
    assert "estimate_budget" in reply.tool_trace
    assert {"restaurants", "hotels", "budget"}.issubset({card["type"] for card in reply.cards})
