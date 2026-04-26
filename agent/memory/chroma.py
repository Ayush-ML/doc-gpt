# This is a Script that handles all the Memory of The Sessions betweem User and Agent
# It uses ChromaDB to store data as Embedding and to fetch it back
# Imported Libraries

import chromadb
from datetime import datetime
from agent.config import HISTORY

_client = None
_collection = None

# Create a Function to make a Persistent Client and Collection (done as a func so that it is not re ran every single import)

def _get_chroma() -> chromadb.Collection:
    if _client is None and _collection is None:
        _client = chromadb.PersistentClient(path=HISTORY)
        _collection = _client.get_or_create_collection(name="agent")
    
    return _collection


# Create a Function that writes The Session upon end to ChormaDb

def write_memory(session_id: str, messages: list[dict]) -> None:
     collection = _get_chroma()
     ids = [f"{session_id}_{i}" for i in range(len(messages))]
     documents = [m['message'] for m in messages]
     metadatas = [
        {
            "session_id": session_id,
            "role": m["role"],
            "index": i,
            "time": datetime.now()
        }
        for i, m in enumerate(messages)
    ]
     collection.add(ids=ids, documents=documents, metadatas=metadatas)

# Create a Function in order to Retrieve Relevant Results

def search(user_query: str, n_results: int = 5) -> list:
    collection = _get_chroma()
    results = collection.query(n_results=n_results, query_texts=[user_query])
    
    hits = []
    for i, doc in enumerate(results["documents"][0]):
        hits.append({
            "text": doc,
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        })

    return hits

# Create a Function used to Check if the session has already been embedded before Embedding

def session_exists(session_id: str) -> bool:
    collection = _get_chroma()
    results = collection.get(where={"session_id": session_id})
    return len(results["documents"]) > 0




     


