from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from travel_agent.config import LLMConfig
from travel_agent.llm_extractor import (
    RuleBasedTravelProfileExtractor,
    travel_profile_from_payload,
)
from travel_agent.schemas import TravelProfile


SYSTEM_PROMPT = """你是旅行规划 Agent 的需求抽取模块。
只把用户明确表达的信息抽取成 JSON，不要编造缺失字段。

字段要求：
- destination: 城市中文名；未知则 null
- days: 旅行天数；未知则 null
- start_date: 出发日期；未知则 null
- budget_level: low / mid / high / null
- interests: 英文标签数组，例如 history, food, nature, culture, museum, citywalk, family, nightlife, shopping, couple
- companions: couple / family / elderly / solo / null
- pace: relaxed / standard / intensive
- hotel_area: 住宿区域；未知则 null
- food_preference: 英文或中文标签数组
- must_visit: 用户明确说必须去的地点数组
- avoid: 用户明确说要避开的地点或偏好数组
- transport_mode: walk / public_transport / taxi / drive

只输出 JSON 对象，不要输出 Markdown。"""


@dataclass(frozen=True)
class OpenAICompatibleTravelProfileExtractor:
    config: LLMConfig
    fallback: RuleBasedTravelProfileExtractor = RuleBasedTravelProfileExtractor()

    def extract(self, user_message: str) -> TravelProfile:
        if not self.config.enabled:
            return self.fallback.extract(user_message)
        try:
            payload = self._call_llm(user_message)
            return travel_profile_from_payload(payload)
        except (OSError, ValueError, KeyError, urllib.error.URLError):
            return self.fallback.extract(user_message)

    def _call_llm(self, user_message: str) -> dict[str, Any]:
        body = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        request = urllib.request.Request(
            url=f"{self.config.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
            raw = json.loads(response.read().decode("utf-8"))
        content = raw["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise ValueError("LLM response is not a JSON object")
        return parsed
