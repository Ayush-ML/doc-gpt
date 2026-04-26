# This a a Script That Defines the State of the Agent using LangGraph
# It defines a Class called Agent state, which acts as an intermediary for The Agent and The User
# It also acts as a way for the Agent to Receive Data
# It handles The AgentLoop and How the agent moves from node to node
# Imported Libraries

from typing import Annotated, Literal, TypedDict
from operator import add
from dataclasses import dataclass

@dataclass
class AgentState(TypedDict):
    # -- Session Data

    session_id: str
    user_id: str

    # Step Handler

    current_step: Literal[1, 2, 3, 4]
    max_step: int
    retries: dict[int, int]

    # Session Handling

    messages: Annotated[list[dict], add]

    # Tools

    end_response_reason: str # End Response Tool
    requested_next: Literal["forward", "back"]
    requested_target_step: int | None 

    # Preporcessing Node

    all_skills: dict[str, str]
    clinical_profile: str
    semantic_search: list

    # Skills 

    used_skills: list[str]
    skill_contents: list[str]

    # Gatekeeper

    gatekeeper_decision: bool
    gatekeeper_reason: str