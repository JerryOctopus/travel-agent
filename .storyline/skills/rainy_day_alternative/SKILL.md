---
title: 雨天备选方案
description: 雨天优先室内 POI，调整户外行程
---

# 雨天备选技能

当 `check_weather` 返回雨天或用户担心下雨时：

1. 优先推荐 `indoor=true` 或 category 为 museum/shopping/food 的 POI；
2. 减少单日户外 scenic 点位数量；
3. 在 `plan_and_critique` 之后向用户说明哪些户外点可改为室内备选；
4. 若用户明确接受室内方案，可再次 `search_poi`（category=museum 或 interests 含 culture）后重跑 `recommend_candidates` → `plan_and_critique`。
