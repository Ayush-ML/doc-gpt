# This is a Script that handles the Routing Logic of the Agent
# It decides which node runs next after the gatekeeper
# Imported Libraries

from agent.main.state import AgentState
from agent.config import MAX_RETRIES

def route_after_gatekeeper(state: AgentState) -> str:
    # Load Ncessary Data from AgentState

    current_step = state['current_step']
    approved = state['gatekeeper_decision']
    requested_next = state['requested_next']
    requested_target = state['requested_target_step']
    retries = state['retries']
    max_reached = state['max_reached_step']

    # Always Approve Back Requests no matter the Target

    if requested_next == "back" and requested_target is not None:
        return f"step{requested_target}"

    # Handle Gatekeeper Rejection

    if not approved:
        attempts = retries.get(current_step, 0)
        if attempts >= MAX_RETRIES:
            if current_step == 4:
                return "profile_updater"
            return f"step{next_step}"
        return f"step{current_step}"

    # Handle Gatekeeper Approval
    if approved:
        next_step = current_step + 1

        # One step forward at a time
        if next_step > max_reached + 1:
            return f"step{current_step}"

        # If step 4 completed, go to updation nodes
        if current_step == 4:
            return "profile_updater"

        return f"step{next_step}"