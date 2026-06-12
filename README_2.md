# Domain-Specific RAG Chatbot 🐍

A Retrieval-Augmented Generation (RAG) chatbot that answers questions from a Python Programming Guide corpus — no API key needed.

## Corpus Description

**Python Programming Guide** — 15 chapters covering:
- Variables & Data Types
- Control Flow (loops, conditions)
- Functions & Decorators
- OOP (Classes, Inheritance)
- File Handling
- Modules & Packages
- Error Handling
- Best Practices (PEP 8)

~25 pages of curated Python programming content, chunked into ~500-token segments.

## Tech Stack

| Component | Tool |
|---|---|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (free, local) |
| Vector Store | ChromaDB (in-memory) |
| UI | Streamlit |
| Chunking | Custom sliding window (500 tokens, 50 overlap) |
| Retrieval | Top-5 cosine similarity search |

## Demo Q&A

**Q1 (In scope):** What are the basic data types in Python?
> Python has several basic data types: int, float, str, bool, NoneType, list, tuple, dict, and set. Type conversion functions like int(), float(), str() convert between types.

**Q2 (In scope):** How do decorators work in Python?
> Decorators modify function behavior using the @ syntax. Common built-in decorators include @staticmethod, @classmethod, and @property.

**Q3 (Out of scope):** What is the best recipe for chocolate cake?
> ❌ Out of scope: This chatbot only answers questions about Python programming.

## Live Demo

👉 [Open Chatbot on Streamlit Cloud](https://your-app.streamlit.app)

## How to Run Locally

```bash
git clone https://github.com/Mr-360-17/rag-chatbot.git
cd rag-chatbot
pip install -r requirements.txt
streamlit run app.py
```

## Reflection

The key RAG insight: retrieval quality matters more than generation quality. A good embedding model + well-chunked corpus answers most questions accurately even without a powerful LLM. The out-of-scope detection uses keyword matching against the corpus domain — a simple but effective guard for production chatbots.
