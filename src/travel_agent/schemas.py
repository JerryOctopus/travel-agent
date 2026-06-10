from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


BudgetLevel = Literal["low", "mid", "high"]
Pace = Literal["relaxed", "standard", "intensive"]
TransportMode = Literal["walk", "public_transport", "taxi", "drive"]


@dataclass(frozen=True)
class POI:
    poi_id: str  # POI唯一标识
    name: str  # 名称
    city: str  # 城市
    category: str  # 类别
    lat: float  # 纬度
    lng: float  # 经度
    rating: float  # 评分
    popularity: float  # 热度
    tags: list[str]  # 标签
    estimated_duration_min: int  # 预计游览时长（分钟）
    price_level: str  # 价格等级
    indoor: bool = False  # 是否室内
    opening_hours: str | None = None  # 开放时间
    source: str = "seed"  # 数据来源


@dataclass
class TravelProfile:
    destination: str | None = None  # 目的地
    days: int | None = None  # 旅行天数
    start_date: str | None = None  # 出发日期
    budget_level: BudgetLevel | None = None  # 预算等级
    interests: list[str] = field(default_factory=list)  # 兴趣偏好
    companions: str | None = None  # 同行人
    pace: Pace = "standard"  # 旅行节奏
    hotel_area: str | None = None  # 住宿区域
    food_preference: list[str] = field(default_factory=list)  # 饮食偏好
    must_visit: list[str] = field(default_factory=list)  # 必去地点
    avoid: list[str] = field(default_factory=list)  # 避开地点或偏好
    transport_mode: TransportMode = "public_transport"  # 交通方式

    def missing_required_fields(self) -> list[str]:
        missing = []
        if not self.destination:
            missing.append("destination")
        if not self.days:
            missing.append("days")
        return missing


@dataclass(frozen=True)
class ScoredPOI:
    poi: POI  # POI
    score: float  # 候选分数
    reasons: list[str]  # 推荐理由


@dataclass(frozen=True)
class ItineraryStop:
    poi: POI  # POI
    start_time: str  # 开始时间
    duration_min: int  # 停留时长（分钟）
    note: str  # 行程说明
    route_from_previous: RouteInfo | None = None  # 与上一站之间的路线信息


@dataclass(frozen=True)
class ItineraryDay:
    day_index: int  # 第几天
    theme: str  # 当天主题
    stops: list[ItineraryStop]  # 行程停靠点


@dataclass(frozen=True)
class Itinerary:
    city: str  # 城市
    days: list[ItineraryDay]  # 每日行程
    summary: str  # 行程摘要


@dataclass(frozen=True)
class CriticIssue:
    code: str  # 问题编码
    message: str  # 面向用户或日志的中文说明
    severity: Literal["info", "warning", "error"] = "warning"  # 严重程度


@dataclass(frozen=True)
class CriticResult:
    passed: bool  # 是否通过检查
    issues: list[CriticIssue]  # 检查发现的问题


@dataclass(frozen=True)
class KnowledgeChunk:
    chunk_id: str  # 知识片段 ID
    city: str  # 城市
    text: str  # 文本内容
    score: float  # 检索分数
    source: str  # 来源


@dataclass(frozen=True)
class WeatherInfo:
    city: str  # 城市
    condition: str  # 天气情况
    temperature_c: int  # 摄氏温度
    source: str = "mock"  # 来源


@dataclass(frozen=True)
class RouteInfo:
    origin_poi_id: str  # 起点 POI ID
    destination_poi_id: str  # 终点 POI ID
    distance_km: float  # 路线距离（公里）
    duration_min: int  # 预计通勤时间（分钟）
    mode: TransportMode  # 交通方式
    source: str = "haversine_estimate"  # 来源


@dataclass(frozen=True)
class WorkflowResult:
    profile: TravelProfile  # 出行画像
    ranked_pois: list[ScoredPOI]  # 排序后的POI候选
    itinerary: Itinerary | None  # 行程
    critic_result: CriticResult | None = None  # 行程约束检查结果
    knowledge_chunks: list[KnowledgeChunk] = field(default_factory=list)  # RAG知识片段
    weather: WeatherInfo | None = None  # 天气信息
    revised: bool = False  # 是否经过自动修正
    revision_notes: list[str] = field(default_factory=list)  # 修正说明
    clarification_question: str | None = None  # 澄清问题


@dataclass(frozen=True)
class ChatMessage:
    role: Literal["user", "assistant"]  # 消息角色
    content: str  # 消息内容


@dataclass(frozen=True)
class DialogueTurn:
    user_message: str  # 用户输入
    assistant_message: str  # 助手回复
    result: WorkflowResult  # 本轮 workflow 结果
