from travel_agent.weather import get_weather


def test_get_weather_returns_mock_city_weather() -> None:
    weather = get_weather("上海")

    assert weather.city == "上海"
    assert weather.condition == "rain"
    assert weather.source == "mock"


def test_get_weather_handles_unknown_city() -> None:
    weather = get_weather("不存在")

    assert weather.city == "不存在"
    assert weather.condition == "unknown"
