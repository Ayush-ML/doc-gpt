# This is a Script that handles all the Memory of The Sessions betweem User and Agent
# It uses ChromaDB to store data as Embedding and to fetch it back
# Imported Libraries

import chromadb
from datetime import datetime
from agent.config import HISTORY

client = chromadb.PersistentClient(path=HISTORY)
collection = client.get_or_create_collection(name="agent")

# Create a Function that writes The Session upon end to ChormaDb

def write_memory(session_id: str, messages: list[dict]) -> None:
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
    results = collection.get(where={"session_id": session_id})
    return len(results["documents"]) > 0




     


