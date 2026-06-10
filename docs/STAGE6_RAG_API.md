# 阶段 6：真实工具 API 与外部数据增强

## 当前状态

阶段 6 已从“本地 RAG 实验”升级为“真实工具 provider 优先”的实现：

```text
workflow
  -> TravelToolProvider
  -> AmapToolProvider（可选真实 API）
  -> LocalToolProvider（稳定 fallback）
```

本地攻略仍然保留，但定位已经改变：

```text
本地目的地攻略 Markdown
  -> keyword overlap 检索
  -> knowledge_chunks
  -> workflow
  -> 中文回复中的“出行上下文”
```

它的作用是补充城市背景、区域建议、雨天策略，而不是替代真实 POI API。

同时加入了 mock weather：

```text
city
  -> get_weather
  -> WeatherInfo
  -> workflow / response
```

## 已实现文件

```text
data/guides/*.md
src/travel_agent/rag.py
src/travel_agent/weather.py
src/travel_agent/providers.py
```

## 当前能力

- 支持 `TravelToolProvider` 抽象；
- 支持 `LocalToolProvider` 作为本地 fallback；
- 支持 `AmapToolProvider` 调用高德 POI、高德天气和高德路线请求骨架；
- 支持 API 失败后自动 fallback 到本地数据；
- 支持本地 haversine 路线 fallback；
- 支持 planner 挂载 `RouteInfo`，critic 检查路线约束；
- response 展示相邻 POI 通勤时间和距离；
- 支持北京、杭州、上海、成都、西安的本地攻略检索；
- 支持根据用户 query 返回 top-k 攻略片段；
- 支持 mock weather；
- workflow 输出 `knowledge_chunks` 和 `weather`；
- response 展示天气和攻略依据。

## 高德 API 配置

默认不启用外部 API。需要真实工具调用时，在环境变量中配置：

```bash
export TRAVEL_AGENT_TOOL_PROVIDER=amap
export TRAVEL_AGENT_AMAP_API_KEY=你的高德Key
```

可选配置：

```bash
export TRAVEL_AGENT_AMAP_BASE_URL=https://restapi.amap.com
export TRAVEL_AGENT_TOOL_TIMEOUT_SECONDS=5
```

不要把 API key 写进代码、README 或测试文件。

## 仍未完成

- 真实 key 下的高德路线端到端验证；
- 距离矩阵或批量路线优化；
- BM25 / embedding 检索；
- rerank 模型；
- 外部数据缓存和失败重试。

## 后续升级建议

优先级：

1. 工具结果缓存，覆盖 POI、天气和路线请求；
2. 本地 keyword overlap -> BM25；
3. BM25 -> embedding retrieval；
4. OpenTripMap / OSM provider，作为高德以外的数据源。
