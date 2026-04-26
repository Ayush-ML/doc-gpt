# This contains the gatekeeper node which acts as a sub node or gate between 2 Nodes
# It identifies whether the agent should really continue to the next node or continue researching current node
# Imported Libraries

from agent.main.state import AgentState
from agent.main.router import get_gatekeeper
from agent.steps.prompts import GATEKEEPER_PROMPT

# Create the Function that handles what happens in the Node

def run(state: AgentState) -> dict:
    # Load all data from State
    current_step = state['current_step']
    end_response_reason = state['end_response_reason']
    requested_next = state['requested_next']
    messages = state['messages']

    if requested_next == "back": # Always Approve Backward Moves
        return {
            "gatekeeper_decision": True,
            "gatekeeper_reason": "Backward request always approved.",
        }
    
    agent_response = messages[-1]['message'] # Get Step Response ofr current Step
    
    user_message = f"Current Step: {current_step}, Agent's Reason for Ending Response: {end_response_reason}, Agents's Entire Response: {agent_response}" # Build Gatekeeper Content

    gatekeeper = get_gatekeeper()

    messages = [
        {"role": "system", "content": GATEKEEPER_PROMPT},
        {"role": "user", "content": user_message},
    ]

    response = gatekeeper.invoke(messages)
    response = response.content
    return {
        "gatekeeper_decision": response['approved'],
        "gatekeeper_reason": response['reason']
    }
    


