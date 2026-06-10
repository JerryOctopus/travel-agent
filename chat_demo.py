from __future__ import annotations

import argparse

from travel_agent.dialogue import DialogueSession
from travel_agent.extractor_factory import build_extractor, provider_label


def main() -> None:
    parser = argparse.ArgumentParser(description="Travel Agent multi-turn CLI demo.")
    parser.add_argument("--llm", action="store_true", help="使用 LLM extractor；无 key 时 fallback。")
    parser.add_argument("--show-provider", action="store_true", help="显示 extractor provider。")
    args = parser.parse_args()

    extractor = build_extractor(args.llm)
    if args.show_provider:
        print(provider_label(extractor))

    session = DialogueSession(extractor=extractor)
    print("Travel Agent 多轮 Demo。输入 /exit 退出，/reset 重置会话。")
    while True:
        try:
            user_message = input("你：").strip()
        except EOFError:
            print("")
            break
        if not user_message:
            continue
        if user_message == "/exit":
            break
        if user_message == "/reset":
            session.reset()
            print("已重置会话。")
            continue
        print(f"Agent：{session.send(user_message)}")


if __name__ == "__main__":
    main()
