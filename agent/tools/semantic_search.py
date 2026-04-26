# This is a Script that contains the Semantic Search tool for The Agent
# It uses ChromaDB's RAG system in order to retrieve relevant Data from the Collection
# Imported Libraries

from langchain_core.tools import tool
from agent.memory.chroma import _get_chroma
from agent.config import N_RESULTS

# Create The funtion that will be passed to the model

@tool
def semantic_search(keywords: list[str]) -> list[dict]:
    """
    Search the patient's full history using specific keywords.

    Use this when you need to:
    - Find if a specific symptom was mentioned in past sessions
    - Look up previous test results or diagnoses
    - Check if a condition was discussed before

    Do NOT use this for general medical information — use web_search for that.
    Do NOT use this for literature — use pubmed for that.

    Args:
        keywords: list of specific medical terms to search for

    Returns:
        list of relevant past session chunks matching the keywords
    """
    if not keywords:
        return []
    
    collection = _get_chroma()
    
    keyword_list = []
    for keyword in keywords:
        keyword_list.append({"$contains": keyword})

    results = collection.query(query_texts=[""], where_document={
        "$or": keyword_list
    }, n_results=N_RESULTS)

    hits = []
    for i, doc in enumerate(results["documents"][0]):
        hits.append({
            "text": doc,
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        })

    return hits

