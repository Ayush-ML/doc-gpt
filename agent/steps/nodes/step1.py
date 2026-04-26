# This is a Script that handles the Step 1 Node 
# which is responsible for the Models Analysis of the Users Symptoms
# Imported Libraries

from agent.config import STEP_1_PHASE_A, STEP_1_PHASE_B
from agent.main.state import AgentState
from agent.utils import parse_end_response, strip_end_response
from agent.main.router import get_agent
from agent.tools.web_search import web_search
from agent.tools.semantic_search import semantic_search
from agent.tools.pubmed import pubmed

# Create a function that handles Phase A of Step 1
# Phase A is responsible for Skill Selection

def run(state: AgentState) -> dict:
    # Load Data From AgentState

    messages = state['messages']
    all_skills = state['all_skills']
    clinical_profile = state['clinical_profile']
    semantic_search = state['semantic_search']
    skill_contents = []

    # Load Agent

    agent = get_agent()
    agent_a = agent.bind_tools([web_search])

    phase_a_user_message = f"Message History: {messages}, Skill Titles and Their Summaries: {all_skills}, Clinical Profile of the user: {clinical_profile}, Semantic Search Results for the Users Query: {semantic_search}"

    phase_a_context = [
        {"role": "system", "content": STEP_1_PHASE_A},
        {"role": "user", "content": phase_a_user_message}
        ] # Build Context

    phase_a_response = agent_a.invoke(phase_a_context) # Get Response

    skills = [
    line.strip() 
    for line in phase_a_response.content.strip().splitlines()
    if line.strip()
    ] # Get Selected Skills


    for skill in skills:
        with open(f"agent\skills\{skill}.md", "r") as file:
            skill_contents.append(file.read()) # Get all Skill Contexts

    agent_b = agent.bind_tools([web_search, pubmed, semantic_search])

    phase_b_user_message = f"Message History: {messages}, Selected Skill Contents: {skill_contents}, Clinical Profile of the user: {clinical_profile}, Semantic Search Results for the Users Query: {semantic_search}"

    phase_b_context = [
        {"role": "system", "content": STEP_1_PHASE_B},
        {"role": "user", "content": phase_b_user_message}
        ] # Build Context

    phase_b_response = (agent.invoke(phase_b_context)).content

    phase_b_response = strip_end_response(phase_b_response)

    reason, next_dir, target = parse_end_response(response=phase_b_response)

    return {
        "used_skills": skills,
        "skill_contents": skill_contents,
        "end_response_reason": reason,
        "requested_next": next_dir,
        "requested_target_step": target,
        "messages": [{"role": "agent", "message": phase_b_response}]
    }

    



    