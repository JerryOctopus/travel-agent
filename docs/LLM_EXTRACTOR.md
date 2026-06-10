# LLM Structured Output 设计

## 目标

让大模型只负责需求抽取，不直接生成行程。

```text
用户自然语言
  -> LLM structured output
  -> TravelProfile
  -> 召回 / 排序 / planner / critic / reviser
```

这样可以避免 LLM 一次性编行程，保证系统可控、可测、可解释。

## 当前实现

代码位置：

```text
src/travel_agent/llm_extractor.py
```

当前 provider：

- `RuleBasedTravelProfileExtractor`：默认规则抽取，本地稳定可运行；
- `FakeLLMTravelProfileExtractor`：测试用 fake provider，模拟 LLM JSON 输出；
- `OpenAICompatibleTravelProfileExtractor`：真实 OpenAI-compatible provider，无 key 时 fallback；
- `travel_profile_from_payload`：把模型输出的 dict 安全转换成 `TravelProfile`。

## 目标 JSON Schema

真实 LLM provider 应输出类似：

```json
{
  "destination": "北京",
  "days": 2,
  "start_date": null,
  "budget_level": "mid",
  "interests": ["history", "food"],
  "companions": null,
  "pace": "relaxed",
  "hotel_area": null,
  "food_preference": ["local"],
  "must_visit": ["故宫"],
  "avoid": [],
  "transport_mode": "public_transport"
}
```

## Provider 接口

真实 provider 只需要实现：

```python
class SomeLLMTravelProfileExtractor:
    def extract(self, user_message: str) -> TravelProfile:
        ...
```

workflow 不关心底层是 OpenAI、DeepSeek、Qwen，还是规则抽取。

## 环境变量

配置示例见项目根目录：

```text
.env.example
```

支持：

```text
TRAVEL_AGENT_LLM_PROVIDER=deepseek
TRAVEL_AGENT_LLM_API_KEY=你的 key
TRAVEL_AGENT_LLM_BASE_URL=https://api.deepseek.com/v1
TRAVEL_AGENT_LLM_MODEL=deepseek-chat
TRAVEL_AGENT_LLM_TIMEOUT_SECONDS=20
```

千问 / 阿里云百炼国内北京地域：

```text
TRAVEL_AGENT_LLM_PROVIDER=qwen
TRAVEL_AGENT_LLM_API_KEY=你的百炼 API Key
TRAVEL_AGENT_LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
TRAVEL_AGENT_LLM_MODEL=qwen-plus
```

使用 demo：

```bash
PYTHONPATH=src python3 demo.py --llm --show-provider "帮我规划北京两天，喜欢历史和美食，不要太累"
```

如果没有 `TRAVEL_AGENT_LLM_API_KEY`，会自动 fallback 到规则抽取。

## Prompt 原则

系统提示词应该强调：

- 只抽取用户明确表达的信息；
- 不要补编目的地、天数、预算；
- 不确定的字段返回 `null` 或空数组；
- `pace` 只能是 `relaxed`、`standard`、`intensive`；
- `budget_level` 只能是 `low`、`mid`、`high`；
- 兴趣标签尽量归一化到英文 tag，例如 `history`、`food`、`nature`。

## 后续接入顺序

1. 先接一个本地 fake provider，已完成；
2. 再接真实 LLM provider，但只在有 API key 时启用，已完成；
3. 保留规则抽取作为 fallback，已完成；
4. 增加 LLM 输出解析失败、字段非法、JSON 缺失时的测试；
5. 对比 rule extractor 与 LLM extractor 在同一批 case 上的 slot fill 质量。
