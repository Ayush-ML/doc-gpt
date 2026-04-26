# Skill Updater - Uses the Agent to create a new skill.md file
# The agent creates the file based on what they have learned based on the current diagnosis
# Runs after profile_updater, without the gatekeeper
# Imported Libraries

import json
from pathlib import Path
from agent.main.state import AgentState
from agent.config import SKILLS, INDEX
from agent.main.router import get_agent
from agent.steps.prompts import SKILL_WRITER_PROMPT
from agent.memory.chroma import write_memory, session_exists

# Create A function that will parse the models response and convert to json

def _parse_response(response: str) -> dict | None:
    try:
        clean = response.strip()
        if clean.startswith("```"):
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        return json.loads(clean.strip())
    except Exception:
        return None

# Create The function that will handle The Node

def run(state: AgentState) -> dict:
    # Load Necessary data from AgentState

    clinical_profile = state['clinical_profile']
    messages = state['messages']
    all_skills = state['all_skills']
    session_id = state['session_id']

    # Create The Agent

    agent = get_agent()

    # Build Agent Context

    user_message = f"User's Clinical Profile: {clinical_profile}, All Skill titles and summaries that currently exsist: {all_skills}, Current Session Chat History: {messages}"
    context = [
        {"role": "system", "content": SKILL_WRITER_PROMPT},
        {"role": "user", "content": user_message}
    ]

    # Get Response from Agent
    response = (agent.invoke(context)).content

    # Parse Skills and make File Containing Skills

    skill = _parse_response(response=response)
    if skill:
        title = skill["title"]
        summary = skill["summary"]
        content = skill["content"]
        
        skill_path = Path(SKILLS) / f"{title}.md"
        skill_path.parent.mkdir(parents=True, exist_ok=True) # Create File
        skill_path.write_text(content) # Write Skill Content to the File

        # Append the title and Summary to skill index

        with open(INDEX, "a") as file:
            file.write(json.dumps({title: summary}) + "\n")

    # Write Memory to ChromaDB
    if not session_exists(session_id=session_id):
        write_memory(session_id=session_id, messages=messages)

    return {}
        
    