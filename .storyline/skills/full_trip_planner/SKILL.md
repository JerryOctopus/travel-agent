---
title: 完整行程规划
description: 从需求澄清到地图渲染的完整旅行规划流程
---

# 完整行程规划技能

当用户希望一次性得到可执行行程时，按以下顺序自主调用工具：

1. `update_travel_profile`：抽取目的地、天数、偏好、预算、节奏；
2. 若缺必要字段，调用 `request_travel_info` 并停止；
3. `search_poi` + `check_weather` 采集真实数据；
4. `recommend_candidates` 多目标打分；
5. `plan_and_critique` 产出经 critic 修正的行程（必须使用，不可手写行程）；
6. `render_itinerary` 与 `render_map` 供前端展示。

输出时说明 critic 是否通过、有无自动修正项。
