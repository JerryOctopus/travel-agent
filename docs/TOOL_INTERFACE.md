# 工具接口设计

工具层应该稳定、单一、结构化。Agent 负责判断什么时候调用工具；工具只负责
执行一个明确操作，并返回 JSON 风格的结构化数据。

## 工具设计原则

- 每个工具只做一件事；
- 返回结构化对象，不返回大段自由文本；
- 包含足够的元数据，方便规划和评估；
- 第一版支持 local/mock provider；
- provider 实现可替换，方便后续接真实 API。

## 数据模型

### POI

```json
{
  "poi_id": "hangzhou_west_lake",
  "name": "西湖",
  "city": "杭州",
  "category": "scenic",
  "lat": 30.259,
  "lng": 120.149,
  "rating": 4.8,
  "popularity": 0.95,
  "tags": ["nature", "classic", "couple-friendly"],
  "estimated_duration_min": 120,
  "price_level": "free",
  "indoor": false,
  "opening_hours": "all_day",
  "source": "seed"
}
```

字段说明：

- `poi_id`：POI 唯一标识；
- `name`：地点名称；
- `city`：城市；
- `category`：地点类型，例如景点、博物馆、餐厅、酒店；
- `lat` / `lng`：经纬度；
- `rating`：评分；
- `popularity`：热度，归一化到 0-1；
- `tags`：兴趣标签；
- `estimated_duration_min`：建议停留时长；
- `price_level`：价格档位；
- `indoor`：是否室内；
- `opening_hours`：开放时间；
- `source`：数据来源。

### TravelProfile

```json
{
  "destination": "杭州",
  "days": 3,
  "start_date": null,
  "budget_level": "mid",
  "interests": ["nature", "food"],
  "companions": "couple",
  "pace": "relaxed",
  "hotel_area": "西湖",
  "food_preference": ["local"],
  "must_visit": [],
  "avoid": [],
  "transport_mode": "public_transport"
}
```

## MVP 工具

### search_poi

用途：为行程规划检索候选 POI。

输入：

```json
{
  "city": "杭州",
  "query": "自然风光和本地美食",
  "category": "scenic",
  "max_results": 20
}
```

输出：

```json
{
  "items": [],
  "source": "seed",
  "provider": "local"
}
```

### get_weather

用途：获取天气信息，供规划、回复和 critic 检查使用。

输入：

```json
{
  "city": "杭州",
  "date": "2026-06-10"
}
```

输出：

```json
{
  "city": "杭州",
  "date": "2026-06-10",
  "condition": "rain",
  "temperature_c": 26,
  "source": "mock"
}
```

### estimate_route

用途：估算两个 POI 之间的交通距离和时间。

输入：

```json
{
  "origin": {"lat": 30.259, "lng": 120.149},
  "destination": {"lat": 30.25, "lng": 120.17},
  "mode": "public_transport"
}
```

输出：

```json
{
  "distance_km": 2.1,
  "duration_min": 18,
  "mode": "public_transport",
  "source": "haversine_estimate"
}
```

### retrieve_destination_knowledge

用途：检索目的地攻略知识，为 RAG 提供上下文。

输入：

```json
{
  "city": "杭州",
  "query": "情侣在西湖附近适合怎么玩"
}
```

输出：

```json
{
  "chunks": [
    {
      "chunk_id": "hangzhou_guide_001",
      "text": "检索到的一小段攻略内容。",
      "score": 0.82,
      "source": "local_markdown"
    }
  ]
}
```

## Provider 抽象

当前 workflow 不再直接依赖固定数据源，而是通过 `TravelToolProvider` 调用工具：

```text
workflow
  -> TravelToolProvider
  -> search_pois / get_weather
```

已实现 provider：

- `LocalToolProvider`：读取 seed POI，并使用 mock weather；
- `AmapToolProvider`：调用高德 POI 与天气 API；
- `FallbackToolProvider`：真实 API 失败时自动回退到本地 provider。

高德 provider 通过环境变量启用：

```bash
export TRAVEL_AGENT_TOOL_PROVIDER=amap
export TRAVEL_AGENT_AMAP_API_KEY=你的高德Key
```

本地攻略不作为 POI 主数据源，只作为 `retrieve_destination_knowledge` 的补充上下文。

## Provider 升级路径

```text
本地 seed 数据
  -> 高德地图 / 商业 API
  -> OpenTripMap / OpenStreetMap / Open-Meteo
  -> provider 抽象与 fallback 机制
```
