# Skill Updater - Uses the Agent to create a new skill.md file
# The agent creates the file based on what they have learned based on the current diagnosis
# Runs after profile_updater, without the gatekeeper
# Imported Libraries

import json
from pathlib import Path
from agent.main.state import AgentState
from agent.config import user_skills_dir, user_skills_index, ACTIVE_USER
from agent.main.router import get_agent
from agent.utils import sanitize_filename
from agent.steps.prompts import SKILL_WRITER_PROMPT, SESSION_TITLE_PROMPT
from agent.memory.chroma import write_memory, session_exists
from langchain_core.messages import HumanMessage
from datetime import datetime

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

    # Create The Agent

    agent = get_agent()

    # Build Agent Context

    session_text = "\n".join([
    f"{'User' if isinstance(m, HumanMessage) else 'Agent'}: {m.content}"
    for m in messages
    ])
    user_message = f"User's Clinical Profile: {clinical_profile}, All Skill titles and summaries that currently exsist: {all_skills}, Current Session Chat History: {session_text}"
    context = [
        {"role": "system", "content": SKILL_WRITER_PROMPT},
        {"role": "user", "content": user_message}
    ]

    # Get Response from Agent
    response = (agent.invoke(context)).content
    if isinstance(response, list):
        response_content = " ".join(
            block.get("text", "") for block in response
            if isinstance(block, dict) and "text" in block
        )
    else:
        response_content = response

    # Parse Skills and make File Containing Skills

    skill = _parse_response(response=response_content)
    if skill:
        title = skill["title"]
        summary = skill["summary"]
        content = skill["content"]
        
        skill_path = Path(user_skills_dir(ACTIVE_USER)) / f"{sanitize_filename(title)}.md"
        skill_path.parent.mkdir(parents=True, exist_ok=True) # Create File
        skill_path.write_text(content) # Write Skill Content to the File

        # Append the title and Summary to skill index

        with open(user_skills_index(ACTIVE_USER), "a") as file:
            file.write(json.dumps({"title": title, "summary": summary}) + "\n")

    return {}
        
    