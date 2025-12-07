import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import json
import chromadb
from sentence_transformers import SentenceTransformer
import requests

# Setup
client = chromadb.PersistentClient(path="./asl_chroma_db")

print("Loading embedding model...")
embedding_model = SentenceTransformer('BAAI/bge-large-en-v1.5')
print("Model loaded.")

def _get_retrieved_context(user_query, user_filters):
    """
    HELPER FUNCTION: Handles the retrieval logic for both Top-1 and Top-3 searches.
    """
    final_where_clause = {}
    if user_filters:
        filter_conditions = []
        for key, value in user_filters.items():
            collection_name = f"asl_metadata_{key.lower()}"
            if collection_name in [c.name for c in client.list_collections()]:
                meta_collection = client.get_collection(name=collection_name)
                results = meta_collection.query(query_embeddings=[embedding_model.encode(value)], n_results=3)
                similar_terms = results['ids'][0]
                if similar_terms:
                    filter_conditions.append({key: {"$in": similar_terms}})
        if filter_conditions:
            final_where_clause = {"$and": filter_conditions}

    main_collection = client.get_collection(name="asl_signs")
    query_embedding = embedding_model.encode(user_query)
    
    if final_where_clause:
        filtered_results = main_collection.query(query_embeddings=[query_embedding], n_results=7, where=final_where_clause)
    else:
        filtered_results = {'ids': [[]], 'documents': [[]]}
    
    semantic_results = main_collection.query(query_embeddings=[query_embedding], n_results=7)
    
    combined_results = {}
    for doc_id, document in zip(filtered_results['ids'][0], filtered_results['documents'][0]):
        combined_results[doc_id] = document
    for doc_id, document in zip(semantic_results['ids'][0], semantic_results['documents'][0]):
        combined_results[doc_id] = document
        
    # Return the context string (Top 10 results)
    return "\n".join(list(combined_results.values())[:10])


def advanced_find_sign(user_query, user_filters):
    """
    Standard Top-1 Search.
    Uses 'asl-rag-model' which has the strict single-word system prompt.
    """
    retrieved_context = _get_retrieved_context(user_query, user_filters)
    
    final_prompt = f"""
    --- User's Description ---
    {user_query}

    --- Context from Database ---
    {retrieved_context}
    --- End Context ---
    """
    try:
        # Uses the Single-Word Model
        response = requests.post("http://localhost:11434/api/generate", json={"model": "asl-rag-model", "prompt": final_prompt, "stream": False})
        response.raise_for_status()
        return response.json()['response'].strip()
    except requests.exceptions.RequestException as e:
        return f"Error communicating with Ollama: {e}"


def advanced_find_sign_top_3(user_query, user_filters):
    """
    New Top-3 Search.
    Uses 'asl-rag-model-top3' which has the comma-list system prompt.
    """
    retrieved_context = _get_retrieved_context(user_query, user_filters)
    
    final_prompt = f"""
    --- User's Description ---
    {user_query}

    --- Context from Database ---
    {retrieved_context}
    --- End Context ---
    """
    try:
        # Uses the Top-3 Model
        response = requests.post("http://localhost:11434/api/generate", json={"model": "asl-rag-model-top3", "prompt": final_prompt, "stream": False})
        response.raise_for_status()
        return response.json()['response'].strip()
    except requests.exceptions.RequestException as e:
        return f"Error communicating with Ollama: {e}"