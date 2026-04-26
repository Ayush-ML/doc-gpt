# this is a Script that handles the Third Node of The Agent
# this node is the verification Node
# It focuses on Verifying the data that is given by The first and second Node
# Imported Libraries

from agent.tools.drug_lookup import drug_lookup
from agent.tools.pubmed import pubmed
from agent.tools.semantic_search import semantic_search
from agent.tools.web_search import web_search
from agent.steps.prompts import STEP_3_PROMPT
from agent.main.state import AgentState
from agent.main.router import get_agent
from agent.utils import strip_end_response, parse_end_response

# Create The Run function that handles the Third Node

def run(state: AgentState) -> dict:
    # Load Necessary Data from AgentState

    messages = state['messages']
    skill_contents = state['skill_contents']
    sem_search = state['semantic_search']
    clinical_profile = state['clinical_profile']

    # Load The Agent

    agent = get_agent()
    agent = agent.bind_tools([drug_lookup, pubmed, semantic_search, web_search])

    # Design the models context

    user_message = f"Message History: {messages}, Selected Skill Contents: {skill_contents}, Clinical Profile of the user: {clinical_profile}, Semantic Search Results for the Users Query: {sem_search}"
    context = [
        {"role": "system", "content": STEP_3_PROMPT},
        {"role": "user", "content": user_message}
    ]

    # Get Agents Response

    response = (agent.invoke(context)).content
    reason, next_dir, target = parse_end_response(response=response) # Parse it to get details from The End Response Tag
    response = strip_end_response(response=response) # strip The End Response Tag

    # Return and Write Everything back to AgentState using LangGraph

    return {
        "requested_next": next_dir,
        "requested_target_step": target,
        "end_response_reason": reason,
        "messages": [{"role": "agent", "message": response}] # Append The models response to Messages
    }