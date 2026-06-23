# This is a Script That handles the fourth and final node of the agent
# This Node is known as the report node
# It is responsible for Organizing the Analysis of all other nodes and Finalizing Them into a Report
# Imported Libraries

from agent.steps.prompts import STEP_4_PROMPT
from agent.main.state import AgentState
from agent.main.router import get_agent
from agent.utils import strip_end_response, parse_end_response
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Create the Run Function that handles the Node

def run(state: AgentState) -> dict:
    # Load Necessary Data from AgentState

    messages = state['messages']
    clinical_profile = state['clinical_profile']
    skill_contents = state['skill_contents']
    semantic_search = state['semantic_search']

    # Load The Agent

    agent = get_agent()

    # Build Context for Agent

    user_message = f"Message History: {messages}, Selected Skill Contents: {skill_contents}, Clinical Profile of the user: {clinical_profile}, Semantic Search Results for the Users Query: {semantic_search}"
    messages_list = [
        SystemMessage(content=STEP_4_PROMPT),
        HumanMessage(content=user_message)
    ]

    # Step 4 is synthesis only — no tool calls
    response = agent.invoke(messages_list)
    response_content = response.content if hasattr(response, 'content') else str(response)
    if isinstance(response_content, list):
        response_content = " ".join(
            block.get("text", "") for block in response_content
            if isinstance(block, dict) and "text" in block
        )

    reason, next_dir, target = parse_end_response(response=response_content)
    response = strip_end_response(response=response_content)

    # Return and Write Everything back to AgentState using LangGraph

    return {
        "current_step": 4,
        "retries": {
        **state['retries'],
        state['current_step']: state['retries'].get(state['current_step'], 0) + 1
        },
        "max_reached_step": max(
            state['max_reached_step'],
            state['current_step']
        ),
        "requested_next": next_dir,
        "requested_target_step": target,
        "end_response_reason": reason,
        "messages": [AIMessage(content=response)] # Append The models response to Messages
    }
