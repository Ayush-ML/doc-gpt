# This is a Script that contains the Public Medical Information of Pubmed tool
# This tool gives the model acces to Public Medical Information
# Imported Libraries

from langchain_core.tools import tool
from langchain_community.utilities.pubmed import PubMedAPIWrapper
from langchain_community.tools.pubmed.tool import PubmedQueryRun
from agent.config import EMAIL
from agent.config import N_RESULTS

_pubmed = None
_wrapper = None

# Initialize PubMed and its params properly using lazy initialization

def _get_pubmed() -> PubmedQueryRun | None:
    global _pubmed, _wrapper
    if _pubmed is None and _wrapper is None:
        _wrapper = PubMedAPIWrapper(top_k_results=N_RESULTS, email=EMAIL)
        _pubmed = PubmedQueryRun(api_wrapper=_wrapper)
    return _pubmed

# The function that the model calls as a tool

@tool
def pubmed(query: str) -> str:
    """
    Search PubMed for peer reviewed clinical literature.

    Use this when you need to:
    - Verify a clinical claim with medical evidence
    - Find diagnostic criteria from authoritative sources
    - Research evidence based treatment guidelines

    Do NOT use this for general information — use web_search for that.
    Do NOT use this for drug information — use drug_lookup for that.

    Args:
        query: clinical search query, use medical terminology

    Returns:
        list of articles each containing title, abstract, authors, year and url
    """
    try:
        results = _get_pubmed().invoke(query)
    except Exception as e:
        results = f"PubMed failed to Fetch results due to error: {e}"

    return results
