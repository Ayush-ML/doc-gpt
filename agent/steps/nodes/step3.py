# this is a Script that handles the Third Node of The Agent
# this node is the verification Node
# It focuses on Verifying the data that is given by The first and second Node
# Imported Libraries

from agent.tools.drug_lookup import drug_lookup
from agent.tools.pubmed import pubmed
from agent.tools.semantic_search import semantic_search
from agent.tools.web_search import web_search
from agent.steps.prompts import STEP_3_PROMPT
from agent.main.state import AgentState
from agent.main.router import get_agent
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from agent.utils import strip_end_response, parse_end_response

# Create The Run function that handles the Third Node

def run(state: AgentState) -> dict:
    # Load Necessary Data from AgentState

    messages = state['messages']
    skill_contents = state['skill_contents']
    sem_search = state['semantic_search']
    clinical_profile = state['clinical_profile']

    # Load The Agent

    agent = get_agent()
    agent = agent.bind_tools([drug_lookup, pubmed, semantic_search, web_search])

    # Design the models context

    user_message = f"Message History: {messages}, Selected Skill Contents: {skill_contents}, Clinical Profile of the user: {clinical_profile}, Semantic Search Results for the Users Query: {sem_search}"
    messages_list = [
        SystemMessage(content=STEP_3_PROMPT),
        HumanMessage(content=user_message)
    ]

    # Agentic loop to execute tool calls and gather verification data
    while True:
        response = agent.invoke(messages_list)
        messages_list.append(response)
        
        # Check if response has tool calls
        if hasattr(response, 'tool_calls') and response.tool_calls:
            # Execute each tool call
            for tool_call in response.tool_calls:
                tool_name = tool_call['name']
                tool_input = tool_call.get('args', {})
                
                try:
                    # Execute the appropriate tool
                    if tool_name == 'drug_lookup':
                        tool_output = drug_lookup.invoke(tool_input)
                    elif tool_name == 'pubmed':
                        tool_output = pubmed.invoke(tool_input)
                    elif tool_name == 'semantic_search':
                        tool_output = semantic_search.invoke(tool_input)
                    elif tool_name == 'web_search':
                        tool_output = web_search.invoke(tool_input)
                    else:
                        tool_output = f"Unknown tool: {tool_name}"
                except Exception as e:
                    tool_output = f"Tool execution error: {str(e)}"
                
                # Add tool result to messages
                tool_message = ToolMessage(
                    content=str(tool_output),
                    tool_call_id=tool_call.get('id', tool_name)
                )
                messages_list.append(tool_message)
        else:
            # No more tool calls, we have the final response
            response_content = response.content if hasattr(response, 'content') else str(response)
            if isinstance(response_content, list):
                response_content = " ".join(
                    block.get("text", "") for block in response_content
                    if isinstance(block, dict) and "text" in block
                )
            break
    reason, next_dir, target = parse_end_response(response=response_content)
    response = strip_end_response(response=response_content)
    print(f"Step 3 Response: {response}")

    # Return and Write Everything back to AgentState using LangGraph

    return {
        "current_step": 3,
        "retries": {
        **state['retries'],
        state['current_step']: state['retries'].get(state['current_step'], 0) + 1
        },
        "max_reached_step": max(
            state['max_reached_step'],
            state['current_step']
        ),
        "requested_next": next_dir,
        "requested_target_step": target,
        "end_response_reason": reason,
        "messages": [AIMessage(content=response)] # Append The models response to Messages
    }