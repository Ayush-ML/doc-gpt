# profile_updater.py
# Runs after Step 4 — updates USER.md with new clinical information
# No gatekeeper — always runs, always writes

from agent.main.state import AgentState
from agent.main.router import get_agent
from agent.config import PROFILE_UPDATER_PROMPT


def run(state: AgentState) -> dict:

    messages = state['messages']
    clinical_profile = state['clinical_profile']
    user_id = state['user_id']

    agent = get_agent()

    user_message = f"Current Clinical Profile:\n{clinical_profile}\n\nFull Session:\n{messages}"
    context = [
        {"role": "system", "content": PROFILE_UPDATER_PROMPT},
        {"role": "user", "content": user_message}
    ]

    updated_profile = (agent.invoke(context)).content

    # write updated profile back to disk
    profile_path = f"agent\users\{user_id}\USER.md"
    with open(profile_path, "w") as file:
        file.write(updated_profile)

    return {
        "clinical_profile": updated_profile
    }