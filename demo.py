from __future__ import annotations

import argparse

from travel_agent.extractor_factory import build_extractor, provider_label
from travel_agent.response import render_markdown_response
from travel_agent.workflow import run_mvp_workflow


def main() -> None:
    parser = argparse.ArgumentParser(description="Travel Agent local demo.")
    parser.add_argument("message", nargs="*", help="用户旅行需求")
    parser.add_argument(
        "--llm",
        action="store_true",
        help="使用 OpenAI-compatible LLM extractor；无 key 时自动 fallback。",
    )
    parser.add_argument(
        "--show-provider",
        action="store_true",
        help="输出当前 extractor provider。",
    )
    args = parser.parse_args()

    user_message = " ".join(args.message).strip()
    if not user_message:
        user_message = "帮我规划杭州三天情侣旅行，喜欢自然和美食，轻松一点，预算中等"
    extractor = build_extractor(args.llm)
    if args.show_provider:
        print(provider_label(extractor))
    result = run_mvp_workflow(user_message, extractor=extractor)
    print(render_markdown_response(result))


if __name__ == "__main__":
    main()
