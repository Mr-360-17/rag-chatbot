import streamlit as st
import os
import re
import numpy as np
from pathlib import Path

st.set_page_config(
    page_title="Python Guide RAG Chatbot",
    page_icon="🐍",
    layout="wide"
)

@st.cache_resource(show_spinner="Loading AI models...")
def load_rag_pipeline():
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer('all-MiniLM-L6-v2')

    corpus_path = Path(__file__).parent / "corpus.txt"
    text = corpus_path.read_text(encoding='utf-8')
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    embeddings = model.encode(chunks, show_progress_bar=False)

    return model, chunks, embeddings


def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)


def retrieve(query, model, chunks, embeddings, top_k=5):
    query_emb = model.encode([query])[0]
    scores = [cosine_similarity(query_emb, emb) for emb in embeddings]
    top_indices = np.argsort(scores)[::-1][:top_k]
    top_chunks = [chunks[i] for i in top_indices]
    top_scores = [scores[i] for i in top_indices]
    return top_chunks, top_scores


def generate_answer(query, context_chunks):
    query_lower = query.lower()
    python_keywords = [
        'python', 'variable', 'function', 'class', 'list', 'dict',
        'loop', 'for', 'while', 'if', 'import', 'module', 'string',
        'integer', 'float', 'bool', 'tuple', 'set', 'exception',
        'error', 'file', 'object', 'method', 'decorator', 'lambda',
        'generator', 'iterator', 'package', 'pip', 'install', 'type',
        'operator', 'indentation', 'syntax', 'keyword', 'scope',
        'inheritance', 'oop', 'pep', 'slice', 'index', 'append',
        'sort', 'range', 'print', 'input', 'return', 'yield', 'def',
        'break', 'continue', 'pass', 'global', 'local', 'map',
        'filter', 'zip', 'enumerate', 'format', 'open', 'read',
        'write', 'csv', 'json', 'os', 'sys', 'math', 'data', 'type',
        'int', 'str', 'none', 'true', 'false', 'and', 'or', 'not'
    ]

    in_scope = any(kw in query_lower for kw in python_keywords)

    if not in_scope:
        return (
            "❌ **Out of scope:** This chatbot only answers questions about "
            "Python programming. Try asking about Python variables, functions, "
            "loops, classes, modules, etc."
        ), []

    sentences = []
    for chunk in context_chunks[:3]:
        for sent in re.split(r'(?<=[.!?])\s+', chunk):
            sent = sent.strip()
            if len(sent) > 40:
                sentences.append(sent)

    query_words = set(query_lower.split()) - {
        'what', 'how', 'is', 'are', 'the', 'a', 'an', 'in', 'of',
        'to', 'do', 'does', 'can', 'i', 'python', 'me', 'tell',
        'explain', 'difference', 'between', 'and', 'or'
    }

    scored = []
    for sent in sentences:
        sent_lower = sent.lower()
        score = sum(1 for w in query_words if w in sent_lower)
        if score > 0:
            scored.append((score, sent))

    scored.sort(key=lambda x: -x[0])
    top_sentences = [s for _, s in scored[:5]]

    if not top_sentences:
        top_sentences = sentences[:4]

    if not top_sentences:
        return (
            "⚠️ **Not found in corpus:** I couldn't find a specific answer. "
            "Try rephrasing or ask about a specific Python concept."
        ), context_chunks[:3]

    answer = " ".join(top_sentences)
    answer = re.sub(r'\s+', ' ', answer).strip()
    return answer, context_chunks[:3]


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🐍 Python Guide RAG Chatbot")
st.markdown("Ask any question about **Python programming** and get answers from the corpus with cited sources.")

model, chunks, embeddings = load_rag_pipeline()
st.success(f"✅ Corpus loaded — {len(chunks)} chunks indexed")

with st.sidebar:
    st.header("📚 About")
    st.markdown("""
    **Corpus:** Python Programming Guide  
    **Chapters:** 15 topics  
    **Embedding:** all-MiniLM-L6-v2  
    **Vector Search:** Cosine similarity (numpy)  
    **Chunks:** ~500 tokens each  
    """)

    st.header("💡 Sample Questions")
    sample_questions = [
        "What are Python data types?",
        "How do I define a function?",
        "What is a list comprehension?",
        "How does inheritance work?",
        "What is the difference between list and tuple?",
        "How do I handle exceptions?",
        "What are decorators?",
        "How do I read a file?",
    ]
    for q in sample_questions:
        if st.button(q, key=q, use_container_width=True):
            st.session_state['query'] = q

    st.header("⚙️ Settings")
    top_k = st.slider("Top-K chunks", 1, 10, 5)

st.markdown("---")

query = st.text_input(
    "Ask a question about Python:",
    value=st.session_state.get('query', ''),
    placeholder="e.g. What is a lambda function?",
)

col1, col2 = st.columns([1, 5])
with col1:
    ask_btn = st.button("🔍 Ask", type="primary", use_container_width=True)
with col2:
    if st.button("🗑️ Clear", use_container_width=True):
        st.session_state['query'] = ''
        st.rerun()

if ask_btn and query.strip():
    with st.spinner("Retrieving relevant chunks..."):
        docs, scores = retrieve(query, model, chunks, embeddings, top_k=top_k)

    with st.spinner("Generating answer..."):
        answer, cited_chunks = generate_answer(query, docs)

    st.markdown("### 💬 Answer")
    st.markdown(answer)

    if cited_chunks:
        st.markdown("### 📄 Source Chunks")
        for i, (chunk, score) in enumerate(zip(cited_chunks, scores[:3]), 1):
            with st.expander(f"Source {i} — similarity: {score:.3f}"):
                st.text(chunk[:600] + "..." if len(chunk) > 600 else chunk)

    with st.expander("🔎 All Retrieved Chunks"):
        for i, (doc, score) in enumerate(zip(docs, scores), 1):
            st.markdown(f"**Chunk {i}** (similarity: {score:.3f})")
            st.text(doc[:400] + "..." if len(doc) > 400 else doc)
            st.markdown("---")

elif ask_btn:
    st.warning("Please enter a question!")

st.markdown("---")
st.markdown("### 🎯 Demo Q&A")

demo_qa = [
    {
        "q": "What are the basic data types in Python?",
        "a": "✅ **In scope** — Python has int, float, str, bool, NoneType, list, tuple, dict, and set as basic data types. Type conversion functions like int(), float(), str() convert between types.",
        "type": "success"
    },
    {
        "q": "How do decorators work in Python?",
        "a": "✅ **In scope** — Decorators modify function behavior using the @ syntax. Common built-in decorators include @staticmethod, @classmethod, and @property.",
        "type": "success"
    },
    {
        "q": "What is the best recipe for chocolate cake?",
        "a": "❌ **Out of scope** — This chatbot only answers questions about Python programming. Your question is outside the corpus.",
        "type": "error"
    }
]

for item in demo_qa:
    with st.expander(f"Q: {item['q']}"):
        if item['type'] == 'success':
            st.success(item['a'])
        else:
            st.error(item['a'])
