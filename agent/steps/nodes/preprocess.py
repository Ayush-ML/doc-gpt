# This is Script that writes the Preprocessing Node as a Function
# This Node is responsible for loading the Semantic Search Results, USER.md file, all of Session Histroy and The Skills and their names
# Imported Libraries

import json
from agent.main.state import AgentState
from agent.config import INDEX
from agent.memory.chroma import search


# The function that is passed to LangGraph and is responsible for the Node

def run(state: AgentState) -> dict:
    user_message = state['messages'][0]['message'] # Load User Message
    user_id = state['user_id'] # User ID
    all_skills = {}

    with open(INDEX, "r") as file: # Load Skills Index
        for line in file.read():
            entry = json.loads(line.strip())
            all_skills.update(entry)

    with open(f"agent\users\{user_id}\USER.md", "r") as file: # Load Clinical Profile
        clinical_profile = file.read()

    try:
        semantic_search = search(user_query=user_message) # Load Semantic Search Results
    except:
        semantic_search = [] # Return Empty if Collection does not exsist or throws an error for whatever reason

    return {
        "semantic_search": semantic_search, # Write to AgentState using LangGraph
        "clinical_profile": clinical_profile,
        "all_skills": all_skills
    }


