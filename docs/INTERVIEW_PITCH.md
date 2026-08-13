# 面试讲述稿（3 分钟版）

## 一句话

我用 LangGraph 做了一个**真 ReAct 旅行规划 Agent**：外层 LLM 自主决定调哪些工具、何时追问；内层用**确定性 planner-critic-reviser 子图**保证行程约束可评估，而不是让模型手写行程。

## 三条卖点（带数字，见 `docs/EVALUATION.md`）

1. **可控规划**：`plan_and_critique` 子图把合成用例违规项从 5 降到 0（下降 100%），保留原始稿 vs 修正稿对比。
2. **真 Agent**：有 key 时 `create_react_agent` 自主 tool calling（POI / 天气 / 路线 / 规划 / 渲染）；无 key 时同一套 toolkit 离线兜底，可复现评估。
3. **工程完整度**：WebSocket 聊天 + 高德地图 + A2UI 卡片；三层记忆；可选分层编排；MCP 暴露工具；意图门控避免「你好就规划」。

## 演示路径（浏览器 2 分钟）

1. 打开 `http://localhost:8000`，发「你好」→ 只寒暄，不跑工具。
2. 「帮我规划杭州三天」→ 引导偏好；补「美食+轻松」→ 右侧出现卡片与地图。
3. 指出工具轨迹：`search_poi` → `recommend_candidates` → `plan_and_critique` → `render_*`。

## 常被问

**Q：为什么不让 LLM 直接写行程？**
A：约束是否满足必须可控、可回归。critic 检查通勤、雨天、偏好覆盖等，reviser 自动修正，指标可量化。

**Q：和套壳 POI 推荐有什么区别？**
A：工具拆细，LLM 自主编排；规划质量由子图保证，不是一次性生成文本。

**Q：没有 LLM key 能跑吗？**
A：能。离线兜底走同一套工具链，评估脚本可复现。

## 命令备忘

```bash
PYTHONPATH=src .venv/bin/python -m travel_agent.server
.venv/bin/python scripts/eval_agent.py --write-report
.venv/bin/python scripts/validate_api_keys.py
```
