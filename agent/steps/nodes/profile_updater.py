# Runs after Step 4 — updates USER.md with new clinical information
# No gatekeeper — always runs, always writes
# Imported Libraries

from agent.main.state import AgentState
from agent.main.router import get_agent
from agent.steps.prompts import PROFILE_UPDATER_PROMPT
from agent.config import user_profile, ACTIVE_USER
from langchain_core.messages import HumanMessage


def run(state: AgentState) -> dict:

    messages = state['messages']
    clinical_profile = state['clinical_profile']
    user_id = state['user_id']

    agent = get_agent()

    session_text = "\n".join([
    f"{'User' if isinstance(m, HumanMessage) else 'Agent'}: {m.content}"
    for m in messages
    ])

    user_message = f"Current Clinical Profile:\n{clinical_profile}\n\nFull Session:\n{session_text}"
    context = [
        {"role": "system", "content": PROFILE_UPDATER_PROMPT},
        {"role": "user", "content": user_message}
    ]

    updated_profile = (agent.invoke(context)).content

    # write updated profile back to disk
    profile_path = user_profile(ACTIVE_USER)
    with open(profile_path, "w") as file:
        file.write(updated_profile)

    return {
        "clinical_profile": updated_profile
    }