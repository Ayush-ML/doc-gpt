# This is the Script that coneects all nodes and tools together
# It handles the flow of the agent loop as it goes from node to node
# It also handles the tool execution of the agent
# As well as edges such as the gatekeeper, profile_updater and skill_writer
# Imported Libraries

from agent.tools.drug_lookup import drug_lookup
from agent.tools.ml_classifier import classifier
from agent.tools.pubmed import pubmed
from agent.tools.semantic_search import semantic_search
from agent.tools.web_search import web_search
from agent.main.state import AgentState
from agent.steps.nodes import profile_updater
from agent.steps.nodes import skill_updater
from agent.steps.nodes import step1
from agent.steps.nodes import step2
from agent.steps.nodes import step3
from agent.steps.nodes import step4
from agent.steps.nodes import gatekeeper
from agent.steps.nodes import preprocess
from agent.config import user_checkpoints, ACTIVE_USER
from agent.main.edges import route_after_gatekeeper
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver

# Register My Tools as Nodes

tools = [drug_lookup, pubmed, classifier, semantic_search, web_search]
tool_node = ToolNode(tools=tools)

# Create The graph builder Function

def _build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # Register All Nodes

    graph.add_node("preprocess", preprocess.run)
    graph.add_node("step1", step1.run)
    graph.add_node("step2", step2.run)
    graph.add_node("step3", step3.run)
    graph.add_node("step4", step4.run)
    graph.add_node("gatekeeper", gatekeeper.run)
    graph.add_node("tools", tool_node)
    graph.add_node("profile_updater", profile_updater.run)
    graph.add_node("skill_writer", skill_updater.run)

    # Set My Graph Entry Point

    graph.set_entry_point("preprocess")
    graph.add_edge("preprocess", "step1")

    # After each step check, check if model made a tool call
    # If yes, then move to tool node
    # if no, route to gatekeeper
    for step in ["step1", "step2", "step3", "step4"]:
        graph.add_conditional_edges(
            step,
            tools_condition,
            {
                "tools": "tools",
                "__end__": "gatekeeper",
            }
        )

    # Handle the execution of the tool node

    graph.add_conditional_edges(
        "tools",
        lambda state: f"step{state['current_step']}",
        {
            "step1": "step1",
            "step2": "step2",
            "step3": "step3",
            "step4": "step4",
        }
    )

    # Add a Conditional edge so that it can only go the next step after gatekeeper

    graph.add_conditional_edges(
        "gatekeeper",
        route_after_gatekeeper,
        {
            "step1": "step1",
            "step2": "step2",
            "step3": "step3",
            "step4": "step4",
            "profile_updater": "profile_updater",
        }
     )

    # Add the profile updater and skill writer nodes before ending

    graph.add_edge("profile_updater", "skill_writer")
    graph.add_edge("skill_writer", END)

    return graph # Finally Return the graph

# Create The Function that handles the Compiling of the graph
# Only called once then reused

def get_graph() -> CompiledStateGraph:
    checkpointer = SqliteSaver.from_conn_string(str(user_checkpoints(ACTIVE_USER))) # Register Checkpoint
    graph = _build_graph() # Build Graph
    return graph.compile(checkpointer=checkpointer) # Compile and return