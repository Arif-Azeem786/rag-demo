# RAG Demo (FAISS + Sentence Transformers)

A simple Retrieval Augmented Generation (RAG) system using:
- Sentence Transformers for embeddings
- FAISS for vector search
- A small set of text documents
- Optional OpenAI/HuggingFace LLM for answer generation

This project demonstrates how RAG works end-to-end.

---

## Features
- Convert text docs → embeddings  
- Build FAISS vector index  
- Retrieve top-k relevant chunks  
- Pass context → LLM to generate final answer  

---

## Install
```bash
pip install -r requirements.txt
