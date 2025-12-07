---
title: ASL Description2Sign
emoji: 🤟
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
license: mit
---

# 🤟 ASL RAG: Semantic Sign Search

This application allows users to find American Sign Language (ASL) signs by describing their physical attributes (Handshape, Orientation, Location, Movement). Unlike traditional dictionaries that require you to know the English gloss (the word), this tool lets you search by **describing what you see**.

## 🚀 How It Works

This is a **Retrieval-Augmented Generation (RAG)** system running entirely locally on CPU.

1. **Vector Search (Retrieval):** Your description is converted into a mathematical vector using `BAAI/bge-large-en-v1.5`. We search a custom ChromaDB database of 1,700+ signs (from the ASL Citizen dataset) to find the closest semantic matches.

2. **Hybrid Filtering:** The system combines "fuzzy" semantic search with strict metadata filters (Handshape, Location, etc) to narrow down candidates.

3. **LLM Reasoning (Ranking):** A quantized `Qwen2-7B-Instruct` model (running locally via `llama.cpp`) acts as a judge. It reads the retrieved descriptions, compares them to your input, and assigns a confidence score.

4. **Video Reference:** The top matches are displayed with their confidence scores and a real-world video demonstration streamed from the ASL Citizen dataset.

## 🛠️ Tech Stack

* **LLM:** Qwen2-7B-Instruct (GGUF / 4-bit Quantized)

* **Embedding Model:** BAAI/bge-large-en-v1.5

* **Vector Database:** ChromaDB (Persistent Local Storage)

* **Interface:** Gradio

* **Deployment:** Hugging Face Spaces (CPU Basic)

## 🎯 Who is this for?

* **ASL Learners:** Who see a sign but don't know the English word for it.

* **Researchers:** Exploring how LLMs can "understand" sign language phonology through text descriptions.

*Note: This app runs on a free CPU tier, so inference may take 20-30 seconds. Please be patient!*