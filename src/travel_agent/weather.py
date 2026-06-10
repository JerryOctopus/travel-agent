from __future__ import annotations

from travel_agent.schemas import WeatherInfo


MOCK_WEATHER = {
    "北京": WeatherInfo(city="北京", condition="sunny", temperature_c=27),
    "杭州": WeatherInfo(city="杭州", condition="cloudy", temperature_c=26),
    "上海": WeatherInfo(city="上海", condition="rain", temperature_c=25),
    "成都": WeatherInfo(city="成都", condition="cloudy", temperature_c=24),
    "西安": WeatherInfo(city="西安", condition="sunny", temperature_c=29),
}


def get_weather(city: str) -> WeatherInfo:
    return MOCK_WEATHER.get(
        city,
        WeatherInfo(city=city, condition="unknown", temperature_c=25),
    )
