#1759 signs
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import json
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

def build_database(json_file_path):
    """
    This function's purpose is to build the persistent ChromaDB database.
    It should only be run once, or when the source data changes.
    """
    # Initialize the client with reset enabled for the build process.
    client = chromadb.PersistentClient(path="./asl_chroma_db", settings=Settings(allow_reset=True))
    
    print("Initializing database build...")
    
    # resets the database every time this script is run
    client.reset()

    print("Loading embedding model...")
    embedding_model = SentenceTransformer('BAAI/bge-large-en-v1.5')
    print("Model loaded.")

    with open(json_file_path, 'r', encoding='utf-8') as f:
        signs_data = json.load(f)

    # --- Build Main Signs Collection ---
    print("Building 'asl_signs' collection...")
    main_collection = client.create_collection(name="asl_signs")
    ids_list, documents_list, metadatas_list = [], [], []
    
    for word, details in signs_data.items():
        metadata = {key: str(value) for key, value in details.items() if key != "Video File"}
        description = f"The sign for '{word}' is described as: " + ". ".join([f"{k}: {v}" for k, v in metadata.items()]) + "."
        
        ids_list.append(word)
        documents_list.append(description)
        metadatas_list.append(metadata)

    embeddings_list = embedding_model.encode(documents_list)
    main_collection.add(ids=ids_list, embeddings=embeddings_list, documents=documents_list, metadatas=metadatas_list)
    print(f"Main collection setup complete with {main_collection.count()} signs.")
    
    # --- Build Searchable Metadata Collections ---
    metadata_fields = ["Handshape", "Location", "Orientation", "Movement"]
    for field in metadata_fields:
        print(f"Building 'asl_metadata_{field.lower()}' collection...")
        collection_name = f"asl_metadata_{field.lower()}"
        meta_collection = client.create_collection(name=collection_name)
        
        unique_values = set(details[field] for details in signs_data.values() if details.get(field))
        
        if unique_values:
            field_embeddings = embedding_model.encode(list(unique_values))
            meta_collection.add(
                ids=list(unique_values),
                documents=list(unique_values),
                embeddings=field_embeddings
            )
        print(f"'{field}' metadata collection setup complete with {meta_collection.count()} unique values.")
    
    print("\nDatabase build process complete.")

if __name__ == "__main__":
    build_database('all_yes_descriptions.json')
