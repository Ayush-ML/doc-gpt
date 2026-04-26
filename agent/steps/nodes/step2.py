# This is a Script that handles the Second Node of The Agent
# This Node is responsible for a predictive analysis and diagnosis of the user using ML models
# Imported Libraries

from agent.tools.ml_classifier import classifier
from agent.tools.semantic_search import semantic_search
from agent.tools.drug_lookup import drug_lookup
from agent.main.router import get_agent
from agent.main.state import AgentState
from agent.steps.prompts import STEP_2_PROMPT
from agent.utils import strip_end_response, parse_end_response

# Create The Run function that is passed to LangGraph and handles the flow of the Second Node

def run(state: AgentState) -> dict:
    # Load Necessary Data from AgentState

    messages = state['messages']
    clinical_profile = state['clinical_profile']
    sem_search = state['semantic_search']
    skill_contents = state['skill_contents']

    # Load Agent

    agent = get_agent()
    agent = agent.bind_tools([classifier, drug_lookup, semantic_search])

    # Design The Models Context

    user_message = f"Message History: {messages}, Selected Skill Contents: {skill_contents}, Clinical Profile of the user: {clinical_profile}, Semantic Search Results for the Users Query: {sem_search}"
    context = [
        {"role": "system", "content": STEP_2_PROMPT},
        {"role": "user", "content": user_message}
    ]

    # Get Models Response

    response = (agent.invoke(context)).content
    reason, next_dir, target = parse_end_response(response=response) # Parse it to get details from The End Response Tag
    response = strip_end_response(response=response) # strip The End Response Tag

    # Return and Write Everything to AgentState using LangGraph

    return {
        "retries": {
        **state['retries'],
        state['current_step']: state['retries'].get(state['current_step'], 0) + 1
        },
        "max_reached_step": max(
            state['max_reached_step'],
            state['current_step']
        ),
        "end_response_reason": reason,
        "requested_next": next_dir,
        "requested_target_step": target,
        "messages": [{"role": "agent", "message": response}] # Append The models response to Messages
    }