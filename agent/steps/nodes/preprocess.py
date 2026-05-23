# This is Script that writes the Preprocessing Node as a Function
# This Node is responsible for loading the Semantic Search Results, USER.md file, all of Session Histroy and The Skills and their names
# Imported Libraries

import json
from agent.main.state import AgentState
from agent.config import user_skills_index, user_profile, ACTIVE_USER
from agent.memory.chroma import search


# The function that is passed to LangGraph and is responsible for the Node

def run(state: AgentState) -> dict:
    user_message = state['messages'][0].content # Load User Message
    all_skills = {}

    with open(user_skills_index(ACTIVE_USER), "r") as file: # Load Skills Index
        for line in file.readlines():
            entry = json.loads(line.strip())
            all_skills.update(entry)

    with open(user_profile(ACTIVE_USER), "r") as file: # Load Clinical Profile
        clinical_profile = file.read()

    try:
        semantic_search = search(user_query=user_message) # Load Semantic Search Results
    except Exception as e:
        semantic_search = [] # Return Empty if Collection does not exsist or throws an error for whatever reason

    return {
        "semantic_search": semantic_search, # Write to AgentState using LangGraph
        "clinical_profile": clinical_profile,
        "all_skills": all_skills
    }


