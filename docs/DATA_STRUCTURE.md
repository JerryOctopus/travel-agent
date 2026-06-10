# 数据结构说明

本文档说明当前 Travel Agent MVP 的核心数据结构。对应代码位于
`src/travel_agent/schemas.py`。

当前数据结构采用 Python dataclass 定义，程序内部继续使用英文属性名；每个
属性都在代码里补充了中文注释，便于阅读字段含义，也方便后续整理前端展示、
结构化日志或 LLM prompt 字段说明。

## 数据流总览

```text
用户输入
  -> TravelProfile 出行画像
  -> POI 候选召回
  -> ScoredPOI 候选打分结果
  -> ItineraryStop 单个行程点
  -> ItineraryDay 单日行程
  -> Itinerary 完整行程
  -> WorkflowResult 工作流结果
```

## 枚举类型

### BudgetLevel

预算等级，用在 `TravelProfile.budget_level`。

| 值 | 中文含义 |
| --- | --- |
| `low` | 低预算 |
| `mid` | 中等预算 |
| `high` | 高预算 |

### Pace

旅行节奏，用在 `TravelProfile.pace`。

| 值 | 中文含义 |
| --- | --- |
| `relaxed` | 轻松 |
| `standard` | 标准 |
| `intensive` | 紧凑 |

### TransportMode

交通方式，用在 `TravelProfile.transport_mode`。

| 值 | 中文含义 |
| --- | --- |
| `walk` | 步行 |
| `public_transport` | 公共交通 |
| `taxi` | 打车 |
| `drive` | 自驾 |

## POI

`POI` 表示一个可被召回、打分并放进行程的地点候选。

| 属性名 | 中文名 | 类型 | 说明 |
| --- | --- | --- | --- |
| `poi_id` | POI唯一标识 | `str` | 地点的稳定 ID。 |
| `name` | 名称 | `str` | 地点名称。 |
| `city` | 城市 | `str` | 地点所在城市。 |
| `category` | 类别 | `str` | 地点类别，例如景点、美食、文化等。 |
| `lat` | 纬度 | `float` | 地理坐标纬度。 |
| `lng` | 经度 | `float` | 地理坐标经度。 |
| `rating` | 评分 | `float` | 地点评分，当前 seed 数据按 5 分制理解。 |
| `popularity` | 热度 | `float` | 热门程度，当前推荐逻辑按 0 到 1 归一化值使用。 |
| `tags` | 标签 | `list[str]` | 用于兴趣匹配的标签集合。 |
| `estimated_duration_min` | 预计游览时长（分钟） | `int` | planner 估算停留时长的基础字段。 |
| `price_level` | 价格等级 | `str` | 地点消费水平。 |
| `indoor` | 是否室内 | `bool` | 是否为室内地点，默认 `False`。 |
| `opening_hours` | 开放时间 | `str \| None` | 开放时间文本，未知时为 `None`。 |
| `source` | 数据来源 | `str` | 数据来源，当前默认 `seed`。 |

## TravelProfile

`TravelProfile` 是从用户输入中抽取出的结构化出行画像，也是后续召回、排序和
规划的主要约束来源。

| 属性名 | 中文名 | 类型 | 说明 |
| --- | --- | --- | --- |
| `destination` | 目的地 | `str \| None` | 旅行目的地。当前 MVP 中缺失时会追问。 |
| `days` | 旅行天数 | `int \| None` | 计划游玩天数。当前 MVP 中缺失时会追问。 |
| `start_date` | 出发日期 | `str \| None` | 出发日期，当前预留字段。 |
| `budget_level` | 预算等级 | `BudgetLevel \| None` | 低、中、高预算偏好。 |
| `interests` | 兴趣偏好 | `list[str]` | 用户兴趣标签，例如 `nature`、`food`。 |
| `companions` | 同行人 | `str \| None` | 同行关系，例如情侣、亲子、独自旅行等。 |
| `pace` | 旅行节奏 | `Pace` | 默认 `standard`。影响每天安排的 POI 数量和时长偏好。 |
| `hotel_area` | 住宿区域 | `str \| None` | 酒店或住宿区域，当前预留字段。 |
| `food_preference` | 饮食偏好 | `list[str]` | 饮食口味或限制，当前预留字段。 |
| `must_visit` | 必去地点 | `list[str]` | 用户明确要求必须包含的地点。 |
| `avoid` | 避开地点或偏好 | `list[str]` | 用户不想去的地点、类别或限制。 |
| `transport_mode` | 交通方式 | `TransportMode` | 默认 `public_transport`。 |

`TravelProfile.missing_required_fields()` 用于检查最小必填信息。目前必填字段是：

- `destination`
- `days`

## ScoredPOI

`ScoredPOI` 是推荐模块的输出，表示一个带分数和理由的 POI 候选。

| 属性名 | 中文名 | 类型 | 说明 |
| --- | --- | --- | --- |
| `poi` | POI | `POI` | 原始 POI 对象。 |
| `score` | 候选分数 | `float` | 偏好感知打分结果。 |
| `reasons` | 推荐理由 | `list[str]` | 命中兴趣、节奏、热度等原因。 |

当前分数由兴趣匹配、热度、评分和节奏适配加权得到。它不是最终推荐文案，而是
planner 的候选输入。

## ItineraryStop

`ItineraryStop` 表示一天行程中的一个具体停靠点。

| 属性名 | 中文名 | 类型 | 说明 |
| --- | --- | --- | --- |
| `poi` | POI | `POI` | 被安排进该时间段的地点。 |
| `start_time` | 开始时间 | `str` | 计划开始时间，例如 `09:30`。 |
| `duration_min` | 停留时长（分钟） | `int` | 预计停留时长。 |
| `note` | 行程说明 | `str` | 安排该地点的说明。 |

## ItineraryDay

`ItineraryDay` 表示单日行程。

| 属性名 | 中文名 | 类型 | 说明 |
| --- | --- | --- | --- |
| `day_index` | 第几天 | `int` | 从 1 开始的天序号。 |
| `theme` | 当天主题 | `str` | 根据当天 POI 类别生成的主题。 |
| `stops` | 行程停靠点 | `list[ItineraryStop]` | 当天安排的地点列表。 |

## Itinerary

`Itinerary` 表示完整行程。

| 属性名 | 中文名 | 类型 | 说明 |
| --- | --- | --- | --- |
| `city` | 城市 | `str` | 行程城市。 |
| `days` | 每日行程 | `list[ItineraryDay]` | 按天组织的行程内容。 |
| `summary` | 行程摘要 | `str` | 面向用户的一句话行程摘要。 |

## WorkflowResult

`WorkflowResult` 是 `run_mvp_workflow()` 的统一返回结构。

| 属性名 | 中文名 | 类型 | 说明 |
| --- | --- | --- | --- |
| `profile` | 出行画像 | `TravelProfile` | 从用户输入抽取出的结构化画像。 |
| `ranked_pois` | 排序后的POI候选 | `list[ScoredPOI]` | 召回并打分后的候选地点。 |
| `itinerary` | 行程 | `Itinerary \| None` | 完整行程；信息不足时为 `None`。 |
| `clarification_question` | 澄清问题 | `str \| None` | 缺少目的地或天数时返回的追问。 |

## 当前 MVP 中的结构关系

当前规则版 workflow 的结构关系如下：

1. `extract_profile_rule_based()` 从中文用户输入中抽取 `TravelProfile`。
2. 如果 `destination` 或 `days` 缺失，直接返回带 `clarification_question` 的
   `WorkflowResult`。
3. 信息完整时，从本地 `data/seed/pois.json` 加载 `POI` 数据。
4. `search_poi()` 按城市和兴趣标签召回候选 POI。
5. `score_pois()` 输出按分数排序的 `ScoredPOI` 列表。
6. `build_simple_itinerary()` 把高分候选组装成 `Itinerary`。
7. `run_mvp_workflow()` 返回包含画像、候选、行程或追问的 `WorkflowResult`。

## 后续扩展建议

- 将 `category`、`price_level`、`companions` 等字符串字段升级为更明确的
  `Literal` 或枚举类型。
- 为 `start_date` 使用 `date` 类型，而不是普通字符串。
- 增加 route、weather、critic result 等结构，为真实工具调用和约束检查服务。
- 增加序列化层，把 dataclass 转成适合 API 返回的 JSON，同时带上中文展示名。
