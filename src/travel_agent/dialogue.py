from __future__ import annotations

from dataclasses import dataclass, field

from travel_agent.llm_extractor import RuleBasedTravelProfileExtractor, TravelProfileExtractor
from travel_agent.response import render_markdown_response
from travel_agent.schemas import ChatMessage, DialogueTurn, TravelProfile, WorkflowResult
from travel_agent.workflow import run_mvp_workflow


@dataclass
class DialogueSession:
    extractor: TravelProfileExtractor = field(default_factory=RuleBasedTravelProfileExtractor)
    profile: TravelProfile | None = None
    last_result: WorkflowResult | None = None
    messages: list[ChatMessage] = field(default_factory=list)
    turns: list[DialogueTurn] = field(default_factory=list)

    def send(self, user_message: str) -> str:
        result = run_mvp_workflow(
            user_message=user_message,
            existing_profile=self.profile,
            extractor=self.extractor,
        )
        assistant_message = render_markdown_response(result)
        self.profile = result.profile
        self.last_result = result
        self.messages.append(ChatMessage(role="user", content=user_message))
        self.messages.append(ChatMessage(role="assistant", content=assistant_message))
        self.turns.append(
            DialogueTurn(
                user_message=user_message,
                assistant_message=assistant_message,
                result=result,
            )
        )
        return assistant_message

    def reset(self) -> None:
        self.profile = None
        self.last_result = None
        self.messages.clear()
        self.turns.clear()
