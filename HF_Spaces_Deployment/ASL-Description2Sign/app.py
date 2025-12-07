import gradio as gr
import chromadb
import json
import sys
import os
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download, snapshot_download

import warnings
warnings.filterwarnings("ignore")

import logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("chromadb").setLevel(logging.ERROR)

if sys.platform.startswith('linux'):
    try:
        __import__('pysqlite3')
        sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    except ImportError:
        pass
import sqlite3

MODEL_REPO_ID = "SudheendraP/ASL-Desc2Sign-Model-Weights"

print("⏳ Downloading Qwen Model...")
qwen_path = hf_hub_download(
    repo_id=MODEL_REPO_ID, 
    filename="qwen2-7b-instruct-q4_k_m.gguf"
)

print("⏳ Connecting to Local Database...")
client = chromadb.PersistentClient(path="./asl_chroma_db")

print("⏳ Downloading BAAI Model...")
baai_path = snapshot_download(repo_id=MODEL_REPO_ID)
baai_model_path = os.path.join(baai_path, "baai_model")

print("⏳ Loading Models into RAM...")
embed_model = SentenceTransformer(baai_model_path)

llm = Llama(
    model_path=qwen_path,
    n_ctx=4096,
    n_threads=4, 
    verbose=False
)

SYSTEM_PROMPT = """
You are a highly analytical ASL expert. 
Your task is to find the 3 best signs from the provided context that match the user's attributes.

Step 1: Analyze the user's attributes (Handshape, Location, etc).
Step 2: Compare them against each sign provided in the context.
Step 3: Select the top 3 matches based STRICTLY on the text descriptions.

Output STRICT JSON only. Do not output reasoning.
Format:
[
  {"sign": "WORD1"},
  {"sign": "WORD2"},
  {"sign": "WORD3"}
]
"""

#RETRIEVAL LOGIC
def get_retrieved_context(synthetic_description, user_filters):
    final_where_clause = {}
    
    # A. Fuzzy Filter Matching logic
    if user_filters:
        filter_conditions = []
        existing_collections = [c.name for c in client.list_collections()]
        
        for key, value in user_filters.items():
            if not value: continue 
            
            collection_name = f"asl_metadata_{key.lower()}"
            if collection_name in existing_collections:
                meta_collection = client.get_collection(name=collection_name)
                # Find closest metadata term
                results = meta_collection.query(
                    query_embeddings=[embed_model.encode(value).tolist()], 
                    n_results=1
                )
                if results['ids'][0]:
                    similar_term = results['ids'][0][0]
                    filter_conditions.append({key: {"$in": [similar_term]}})
        
        if filter_conditions:
            if len(filter_conditions) > 1:
                final_where_clause = {"$and": filter_conditions}
            else:
                final_where_clause = filter_conditions[0]

    # B. Main Search (Hybrid)
    main_collection = client.get_collection(name="asl_signs")
    query_embedding = embed_model.encode(synthetic_description).tolist()
    
    # Query 1: Filtered (Strict)
    if final_where_clause:
        filtered_results = main_collection.query(
            query_embeddings=[query_embedding], 
            n_results=7, 
            where=final_where_clause,
            include=["documents", "metadatas", "distances"]
        )
    else:
        filtered_results = {'ids': [[]], 'documents': [[]], 'metadatas': [[]], 'distances': [[]]}
    
    # Query 2: Semantic (Loose)
    semantic_results = main_collection.query(
        query_embeddings=[query_embedding], 
        n_results=7,
        include=["documents", "metadatas", "distances"]
    )
    
    # C. Combine, Deduplicate, and Score
    combined_context = {}
    video_map = {}
    score_map = {} 
    
    def process_batch(res_obj):
        if not res_obj['ids']: return
        for i in range(len(res_obj['ids'][0])):
            doc_id = res_obj['ids'][0][i]
            doc_text = res_obj['documents'][0][i]
            meta = res_obj['metadatas'][0][i]
            
            # Math: Convert Cosine Distance to % Score
            if 'distances' in res_obj and res_obj['distances']:
                dist = res_obj['distances'][0][i]
                math_score = max(0, min(100, int((1 - dist) * 100)))
            else:
                math_score = 0
            
            # Keep highest score if duplicate
            if doc_id not in score_map or math_score > score_map[doc_id]:
                combined_context[doc_id] = doc_text
                score_map[doc_id] = math_score
                if "Video_URL" in meta:
                    video_map[doc_id] = meta["Video_URL"]

    process_batch(filtered_results)
    process_batch(semantic_results)
    
    # Return context text (blind) + maps
    # We sort candidates by score to give LLM the best context first
    sorted_ids = sorted(score_map, key=score_map.get, reverse=True)[:10]
    
    final_texts = []
    for sign_id in sorted_ids:
        text = combined_context[sign_id]
        final_texts.append(f"Sign '{sign_id}': {text}")
        
    return "\n".join(final_texts), video_map, score_map

#MAIN GRADIO FUNCTION
def search_sign(handshape, location, orientation, movement):
    # 1. Synthesize Description
    parts = []
    if handshape: parts.append(f"Handshape: {handshape}")
    if location: parts.append(f"Location: {location}")
    if orientation: parts.append(f"Orientation: {orientation}")
    if movement: parts.append(f"Movement: {movement}")
    
    if not parts: return [None]*6
    synthetic_description = ". ".join(parts) + "."
    
    # 2. Prepare Filters for Logic
    user_filters = {}
    if handshape: user_filters["Handshape"] = handshape
    if location: user_filters["Location"] = location
    if orientation: user_filters["Orientation"] = orientation
    if movement: user_filters["Movement"] = movement

    # 3. Retrieve
    context_text, video_map, score_map = get_retrieved_context(synthetic_description, user_filters)

    # 4. Inference (Blind)
    prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nUser Attributes: {synthetic_description}\n\nContext:\n{context_text}<|im_end|>\n<|im_start|>assistant\n"

    output = llm(
        prompt, max_tokens=256, stop=["<|im_end|>"], temperature=0.0, echo=False
    )
    
    # 5. Parse & Merge Scores
    try:
        json_raw = output['choices'][0]['text']
        json_clean = json_raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(json_clean)
        
        outputs = []
        for item in data[:3]:
            sign = item.get('sign', 'Unknown')
            
            # LOOK UP THE SCORE
            conf = score_map.get(sign, 0)
            
            # Simple Display
            text_html = f"## {sign}\n**Match Score:** {conf}%"
            vid_url = video_map.get(sign)
            
            outputs.append(text_html)
            outputs.append(vid_url)
            
        while len(outputs) < 6:
            outputs.extend(["No Match Found", None])
            
        return outputs

    except Exception as e:
        return [f"Error: {e}", None, None, None, None, None]

# UI LAYOUT
with gr.Blocks(title="ASL Dictionary") as demo:
    gr.Markdown("# ASL Description To Sign")
    gr.Markdown("Enter the attributes of the sign you are looking for, or click an example below.")
    
    with gr.Row():
        hs_inp = gr.Textbox(label="Handshape", placeholder="e.g. Flat, Fist")
        loc_inp = gr.Textbox(label="Location", placeholder="e.g. Chest, Chin")
    with gr.Row():
        ori_inp = gr.Textbox(label="Orientation", placeholder="e.g. Palm up")
        mov_inp = gr.Textbox(label="Movement", placeholder="e.g. Circle")

    btn = gr.Button("Search", variant="primary")
    
    gr.Markdown("### Top 3 Candidates")
    
    with gr.Row():
        with gr.Column():
            t1 = gr.Markdown()
            v1 = gr.Video(label="Match 1", height=300)
        with gr.Column():
            t2 = gr.Markdown()
            v2 = gr.Video(label="Match 2", height=300)
        with gr.Column():
            t3 = gr.Markdown()
            v3 = gr.Video(label="Match 3", height=300)

    btn.click(
        search_sign, 
        inputs=[hs_inp, loc_inp, ori_inp, mov_inp], 
        outputs=[t1, v1, t2, v2, t3, v3]
    )

    gr.Examples(
        examples=[
            ["Both hands in bent 'B' handshapes", "In front of the body at chest level", "Palms facing downward", "Hands start a few inches apart and move toward each other until fingertips touch"],
            ["Non-dominant hand flat 'B', dominant hand 'I'", "In front of the body at chest level", "Non-dominant palm up; Dominant palm down", "Dominant hand moves in a squiggly line down the open palm"],
            ["'Y' handshape (thumb and pinky extended)", "Chin", "Palm facing inward", "The hand touches the chin"],
            ["'H' handshape (index and middle fingers extended)", "In front of the body at head level", "Palm facing down", "The hand moves upward to indicate height"]
        ],
        inputs=[hs_inp, loc_inp, ori_inp, mov_inp],
        label="Click an example to test:"
    )

demo.launch(server_name="0.0.0.0", server_port=7860)