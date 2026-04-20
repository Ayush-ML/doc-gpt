# This is Script that writes the Preprocessing Node as a Function
# This Node is responsible for loading the Semantic Search Results, USER.md file, all of Session Histroy and The Skills and their names
# Imported Libraries

import json
from agent.main.state import AgentState
from agent.utils import unknown
from agent.config import INDEX
from agent.main.router import get_agent


# The function that is passed to LangGraph and is responsible for the Node

def run(state: AgentState) -> dict:
    user_message = state['messages'][0]
    user_id = state['user_id']

    with open(INDEX, "r") as file:
        all_skils = json.load(file)

    with open(f"agent\users\{user_id}\USER.md", "r") as file:
        clinical_profile = file.read()

    
