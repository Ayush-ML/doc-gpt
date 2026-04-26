# This is a Script that creates the Web Search Tool for The Agent
# This give the Agent Access to Info not in its training
# Imported Libraries

from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchResults
from agent.config import N_RESULTS

# Initialize DuckDuckGo with Parameters

_search = None

def _get_search(): # Lazy Initialization
    global _search
    if _search is None:
        _search = DuckDuckGoSearchResults(num_results=N_RESULTS, output_format='list')
    return _search

# Create the function that will be passed to the model

@tool
def web_search(query: str) -> list[dict]:
    """
    Search the web for general clinical information about symptoms, conditions, and treatments.

    Use this when you need to:
    - Look up general information about a symptom or condition
    - Find clinical guidelines or diagnostic criteria
    - Research treatment options or risk factors

    Do NOT use this for peer reviewed literature — use pubmed_search for that.
    Do NOT use this for drug information — use drug_lookup for that.

    Args:
        query: specific search query, be as precise as possible

    Returns:
        list of results each containing title, url, and snippet
    """
    try:
        result = _get_search().invoke(query)
    except Exception as e:
        result = [{"title": "Search failed", "url": "", "snippet": str(e)}]

    return result
    
