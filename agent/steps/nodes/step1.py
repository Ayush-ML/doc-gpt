# This is a Script that handles the Step 1 Node 
# which is responsible for the Models Analysis of the Users Symptoms
# Imported Libraries

from agent.steps.prompts import STEP_1_PHASE_A, PHASE_B_PROMPT
from agent.main.state import AgentState
from agent.utils import parse_end_response, strip_end_response, sanitize_filename
from agent.main.router import get_agent
from agent.tools.web_search import web_search
from agent.tools.pubmed import pubmed
from agent.tools.semantic_search import semantic_search
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from pathlib import Path
from agent.config import user_skills_dir, ACTIVE_USER


# Create a function that handles Phase A of Step 1
# Phase A is responsible for Skill Selection

def run(state: AgentState) -> dict:
    # Load Data From AgentState

    messages = state['messages']
    all_skills = state['all_skills']
    clinical_profile = state['clinical_profile']
    sem_search = state['semantic_search']
    skill_contents = []
    skills = []

    # Load Agent

    agent = get_agent()
    files = [f for f in Path(user_skills_dir(ACTIVE_USER)).iterdir() if f.is_file()]
    if len(files) >= 1:
        agent_a = agent

        phase_a_user_message = f"Message History: {messages}, Skill Titles and Their Summaries: {all_skills}, Clinical Profile of the user: {clinical_profile}, Semantic Search Results for the Users Query: {sem_search}"

        phase_a_context = [
        SystemMessage(content=STEP_1_PHASE_A),
        HumanMessage(content=phase_a_user_message)
        ] # Build Context

        phase_a_response = agent_a.invoke(phase_a_context) # Get Response
        phase_a_response = phase_a_response.content if hasattr(phase_a_response, 'content') else str(phase_a_response)
        if isinstance(phase_a_response, list):
                phase_a_response = " ".join(
                    block.get("text", "") for block in phase_a_response 
                    if isinstance(block, dict) and "text" in block
                )
        if not phase_a_response == "None":

            skills = [
            line.strip() 
            for line in phase_a_response.splitlines()
            if line.strip()
            ] # Get Selected Skills

            for skill in skills:
                skill_path = Path(user_skills_dir(ACTIVE_USER)) / f"{sanitize_filename(skill)}.md"
                try:
                    with open(skill_path, "r", encoding="utf-8") as file:
                        skill_contents.append(file.read())
                except FileNotFoundError:
                    # skip missing skill files
                    continue # Get all Skill Contexts

    agent_b = agent.bind_tools([web_search, pubmed, semantic_search])

    phase_b_user_message = f"Message History: {messages}, Selected Skill Contents: {skill_contents}, Clinical Profile of the user: {clinical_profile}, Semantic Search Results for the Users Query: {sem_search}"

    phase_b_context =  [
    SystemMessage(content=PHASE_B_PROMPT),
    HumanMessage(content=phase_b_user_message)
    ] # Build Context

    # Agentic loop to execute tool calls
    while True:
        response = agent_b.invoke(phase_b_context)
        phase_b_context.append(response)
        
        # Check if response has tool calls
        if hasattr(response, 'tool_calls') and response.tool_calls:
            # Execute each tool call
            for tool_call in response.tool_calls:
                tool_name = tool_call['type']
                tool_input = tool_call.get('args', {})
                
                try:
                    # Execute the appropriate tool
                    if tool_name == 'web_search':
                        tool_output = web_search.invoke(tool_input)
                    elif tool_name == 'pubmed':
                        tool_output = pubmed.invoke(tool_input)
                    elif tool_name == 'semantic_search':
                        tool_output = semantic_search.invoke(tool_input)
                    else:
                        tool_output = f"Unknown tool: {tool_name}"
                except Exception as e:
                    tool_output = f"Tool execution error: {str(e)}"
                
                # Add tool result to messages
                tool_message = ToolMessage(
                    content=str(tool_output),
                    tool_call_id=tool_call.get('id', tool_name)
                )
                phase_b_context.append(tool_message)
        else:
            # No more tool calls, we have the final response
            phase_b_response = response.content if hasattr(response, 'content') else str(response)
            if isinstance(phase_b_response, list):
                phase_b_response = " ".join(
                    block.get("text", "") for block in phase_b_response 
                    if isinstance(block, dict) and "text" in block
                )
            break

    reason, next_dir, target = parse_end_response(response=phase_b_response)

    phase_b_response = strip_end_response(phase_b_response)

    return {
        "current_step": 1,
        "retries": {
        **state['retries'],
        state['current_step']: state['retries'].get(state['current_step'], 0) + 1
        },
        "max_reached_step": max(
            state['max_reached_step'],
            state['current_step']
        ),
        "used_skills": skills,
        "skill_contents": skill_contents,
        "end_response_reason": reason,
        "requested_next": next_dir,
        "requested_target_step": target,
        "messages": [AIMessage(content=phase_b_response)]
    }

    



    