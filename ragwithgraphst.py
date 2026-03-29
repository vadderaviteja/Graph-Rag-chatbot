"""
Streamlit UI for Graph RAG Chatbot
===================================
Run:
    streamlit run app.py
"""

import os
import sys
import glob
import pickle
import re
import tempfile

import streamlit as st
import pandas as pd
import networkx as nx

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# ── Hidden config (users never see these) ────────────────────────────────────
load_dotenv()
GROQ_API_KEY   = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
GROQ_MODEL     = "llama-3.3-70b-versatile"
EMBED_MODEL    = "all-MiniLM-L6-v2"
FAISS_INDEX    = "./faiss_index"
GRAPH_FILE     = "./knowledge_graph.pkl"
CHUNK_SIZE     = 500
CHUNK_OVERLAP  = 50
TOP_K_VECTOR   = 4
TOP_K_GRAPH    = 3
MAX_GRAPH_HOPS = 2

os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

/* App background */
.stApp { background: #0f0f13; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #16161d !important;
    border-right: 1px solid #2a2a35;
}
[data-testid="stSidebar"] * { color: #c8c8d4 !important; }

/* Main area */
.main .block-container {
    padding: 2rem 2.5rem;
    max-width: 900px;
}

/* Title */
.app-title {
    font-size: 1.6rem;
    font-weight: 600;
    color: #e8e8f0;
    letter-spacing: -0.02em;
    margin-bottom: 0.2rem;
}
.app-subtitle {
    font-size: 0.85rem;
    color: #6b6b80;
    margin-bottom: 2rem;
    font-family: 'DM Mono', monospace;
}

/* Upload zone */
.upload-label {
    font-size: 0.78rem;
    font-weight: 500;
    color: #9090a8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.5rem;
}

/* Chat messages */
.msg-user {
    background: #1e1e2a;
    border: 1px solid #2e2e3d;
    border-radius: 12px 12px 4px 12px;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
    margin-left: 3rem;
    color: #e0e0ee;
    font-size: 0.95rem;
    line-height: 1.6;
}
.msg-bot {
    background: #13131a;
    border: 1px solid #24243a;
    border-left: 3px solid #5b5bf0;
    border-radius: 4px 12px 12px 12px;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
    margin-right: 3rem;
    color: #d8d8ec;
    font-size: 0.95rem;
    line-height: 1.7;
}
.msg-label-user {
    font-size: 0.7rem;
    color: #5a5a70;
    font-family: 'DM Mono', monospace;
    text-align: right;
    margin-right: 0.3rem;
    margin-bottom: 0.2rem;
}
.msg-label-bot {
    font-size: 0.7rem;
    color: #4a4a7a;
    font-family: 'DM Mono', monospace;
    margin-left: 0.3rem;
    margin-bottom: 0.2rem;
}

/* Source pill */
.source-pill {
    display: inline-block;
    background: #1c1c28;
    border: 1px solid #2e2e44;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.72rem;
    color: #7070a0;
    font-family: 'DM Mono', monospace;
    margin: 0.3rem 0.2rem 0 0;
}
.source-tag {
    font-size: 0.72rem;
    color: #50507a;
    margin-top: 0.6rem;
}

/* Stats bar */
.stat-box {
    background: #16161d;
    border: 1px solid #24243a;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    text-align: center;
}
.stat-num {
    font-size: 1.4rem;
    font-weight: 600;
    color: #8080f0;
    font-family: 'DM Mono', monospace;
}
.stat-lbl {
    font-size: 0.72rem;
    color: #5a5a78;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* Input box */
.stTextInput input {
    background: #16161d !important;
    border: 1px solid #2e2e44 !important;
    border-radius: 10px !important;
    color: #e0e0f0 !important;
    font-family: 'DM Sans', sans-serif !important;
    padding: 0.75rem 1rem !important;
}
.stTextInput input:focus {
    border-color: #5b5bf0 !important;
    box-shadow: 0 0 0 2px #5b5bf020 !important;
}

/* Buttons */
.stButton > button {
    background: #5b5bf0 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    padding: 0.5rem 1.4rem !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* Secondary button */
.stButton.secondary > button {
    background: #1e1e2a !important;
    border: 1px solid #2e2e44 !important;
    color: #9090b8 !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #16161d;
    border: 1.5px dashed #2e2e44;
    border-radius: 10px;
    padding: 1rem;
}
[data-testid="stFileUploader"]:hover {
    border-color: #5b5bf0;
}

/* Divider */
hr { border-color: #1e1e2a; }

/* Spinner */
.stSpinner { color: #5b5bf0; }

/* Welcome state */
.welcome-box {
    background: #13131a;
    border: 1px dashed #2a2a3a;
    border-radius: 12px;
    padding: 3rem 2rem;
    text-align: center;
    margin-top: 2rem;
}
.welcome-icon { font-size: 2.5rem; margin-bottom: 1rem; }
.welcome-title { color: #6060a0; font-size: 1rem; font-weight: 500; }
.welcome-sub { color: #3a3a5a; font-size: 0.85rem; margin-top: 0.4rem; }
</style>
""", unsafe_allow_html=True)


# ── Core functions (same logic as rag_chatbot_graph.py) ──────────────────────
@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


def extract_keywords(text: str) -> list[str]:
    stopwords = {
        "this","that","with","from","have","been","will","they","their",
        "there","which","about","would","could","should","these","those",
        "when","what","where","then","than","also","some","more","into",
        "over","after","before","through","between","during","because",
        "while","each","only","just","both","very","such","same","even",
        "most","other","many","much","well","data","used","using","uses",
        "make","made","like","name","named","here","were",
    }
    words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9_\-]{3,}\b', text.lower())
    keywords = [w for w in words if w not in stopwords]
    freq = {}
    for w in keywords:
        freq[w] = freq.get(w, 0) + 1
    sorted_kw = sorted(freq, key=lambda x: freq[x], reverse=True)
    return sorted_kw[:20]


def load_uploaded_files(uploaded_files) -> list[Document]:
    docs = []
    for uf in uploaded_files:
        ext      = os.path.splitext(uf.name)[1].lower()
        filename = uf.name
        try:
            if ext == ".csv":
                df   = pd.read_csv(uf)
                text = df.to_string(index=False)
                docs.append(Document(page_content=text, metadata={"source": filename, "type": "csv"}))

            elif ext in [".xlsx", ".xls"]:
                xl = pd.ExcelFile(uf)
                for sheet in xl.sheet_names:
                    df   = xl.parse(sheet)
                    text = f"Sheet: {sheet}\n" + df.to_string(index=False)
                    docs.append(Document(page_content=text, metadata={"source": filename, "sheet": sheet, "type": "excel"}))

            elif ext == ".pdf":
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uf.read())
                    tmp_path = tmp.name
                from langchain_community.document_loaders import PyPDFLoader
                pages = PyPDFLoader(tmp_path).load()
                docs.extend(pages)
                os.unlink(tmp_path)

            elif ext == ".txt":
                text = uf.read().decode("utf-8", errors="ignore")
                docs.append(Document(page_content=text, metadata={"source": filename, "type": "txt"}))

        except Exception as e:
            st.warning(f"Could not load {filename}: {e}")
    return docs


def build_index(docs: list[Document]):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(docs)

    embeddings = get_embeddings()
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(FAISS_INDEX)

    # Build knowledge graph
    G = nx.Graph()
    for i, chunk in enumerate(chunks):
        keywords = extract_keywords(chunk.page_content)
        G.add_node(i, text=chunk.page_content,
                   source=chunk.metadata.get("source", "unknown"),
                   keywords=keywords)

    nodes = list(G.nodes(data=True))
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            idx_i, data_i = nodes[i]
            idx_j, data_j = nodes[j]
            shared = set(data_i["keywords"]) & set(data_j["keywords"])
            if len(shared) >= 2:
                G.add_edge(idx_i, idx_j, weight=len(shared))

    with open(GRAPH_FILE, "wb") as f:
        pickle.dump(G, f)

    return db, G, chunks


def graph_search(query, G, all_chunks, seed_indices):
    visited        = set(seed_indices)
    collected      = []
    query_keywords = set(extract_keywords(query))

    for seed in seed_indices:
        if seed not in G:
            continue
        frontier = [seed]
        for _ in range(MAX_GRAPH_HOPS):
            next_frontier = []
            for node in frontier:
                for neighbor in G.neighbors(node):
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)
                    node_keywords = set(G.nodes[neighbor].get("keywords", []))
                    if query_keywords & node_keywords:
                        collected.append(neighbor)
                        next_frontier.append(neighbor)
            frontier = next_frontier

    result = []
    for idx in collected[:TOP_K_GRAPH]:
        if idx < len(all_chunks):
            result.append(all_chunks[idx])
    return result


def hybrid_retrieve(query, db, G, all_chunks):
    vector_results = db.similarity_search(query, k=TOP_K_VECTOR)

    seed_indices = []
    for doc in vector_results:
        for node_id, data in G.nodes(data=True):
            if data.get("text", "") == doc.page_content:
                seed_indices.append(node_id)
                break

    graph_results = graph_search(query, G, all_chunks, seed_indices)

    seen, merged = set(), []
    for doc in vector_results + graph_results:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            merged.append(doc)
    return merged, {d.page_content for d in vector_results}


def get_answer(question, db, G, all_chunks):
    retrieved, vector_texts = hybrid_retrieve(question, db, G, all_chunks)
    context = "\n\n---\n\n".join(doc.page_content for doc in retrieved)

    llm = ChatGroq(api_key=GROQ_API_KEY, model_name=GROQ_MODEL,
                   temperature=0.0, max_tokens=1024)

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a helpful and precise data assistant.
Answer the question using ONLY the context provided below.
If the answer is not found in the context, say:
"I don't have enough information in the provided data to answer this."
Never guess. Never make up information. Be concise and factual.

Context:
{context}

Question: {question}

Answer:"""
    )
    chain  = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})

    sources = []
    seen_src = set()
    for doc in retrieved:
        src  = doc.metadata.get("source", "unknown")
        mode = "vector+graph" if doc.page_content in vector_texts else "graph"
        if src not in seen_src:
            seen_src.add(src)
            sources.append({"src": src, "mode": mode})

    return answer, sources, len(retrieved)


# ── Session state init ────────────────────────────────────────────────────────
if "messages"   not in st.session_state: st.session_state.messages   = []
if "db"         not in st.session_state: st.session_state.db         = None
if "G"          not in st.session_state: st.session_state.G          = None
if "chunks"     not in st.session_state: st.session_state.chunks     = []
if "file_names" not in st.session_state: st.session_state.file_names = []
if "indexed"    not in st.session_state: st.session_state.indexed    = False


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Data sources")
    st.markdown('<div class="upload-label">Upload your files</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        label="Drop files here",
        type=["csv", "xlsx", "xls", "pdf", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded:
        st.markdown("**Files ready:**")
        for f in uploaded:
            ext_icon = {"csv": "📊", "pdf": "📄", "xlsx": "📗",
                        "xls": "📗", "txt": "📝"}.get(f.name.split(".")[-1], "📁")
            st.markdown(f"{ext_icon} `{f.name}`")

    st.markdown("")
    build_btn = st.button("Build knowledge base", use_container_width=True)

    if build_btn:
        if not uploaded:
            st.error("Please upload at least one file first.")
        else:
            with st.spinner("Loading files..."):
                docs = load_uploaded_files(uploaded)

            if not docs:
                st.error("No content found in uploaded files.")
            else:
                progress = st.progress(0, text="Building vector index...")
                docs_loaded = len(docs)

                with st.spinner("Creating embeddings + knowledge graph..."):
                    db, G, chunks = build_index(docs)

                progress.progress(100, text="Done!")

                st.session_state.db         = db
                st.session_state.G          = G
                st.session_state.chunks     = chunks
                st.session_state.file_names = [f.name for f in uploaded]
                st.session_state.indexed    = True
                st.session_state.messages   = []

                st.success(f"Ready! {G.number_of_nodes()} chunks indexed.")

    # Stats
    if st.session_state.indexed:
        st.markdown("---")
        st.markdown("**Knowledge graph**")
        G = st.session_state.G
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-num">{G.number_of_nodes()}</div>
                <div class="stat-lbl">Nodes</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-num">{G.number_of_edges()}</div>
                <div class="stat-lbl">Edges</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("")
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.experimental_rerun()


# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown('<div class="app-title">RAG Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">graph retrieval · vector search · groq llm</div>', unsafe_allow_html=True)

# Chat history
if not st.session_state.indexed:
    st.markdown("""
    <div class="welcome-box">
        <div class="welcome-icon">🧠</div>
        <div class="welcome-title">Upload your data to get started</div>
        <div class="welcome-sub">Supports CSV · Excel · PDF · TXT — drag and drop on the left</div>
    </div>
    """, unsafe_allow_html=True)
else:
    # Render chat messages
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="msg-label-user">you</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="msg-user">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="msg-label-bot">assistant</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="msg-bot">{msg["content"]}</div>', unsafe_allow_html=True)
            if msg.get("sources"):
                pills = "".join(
                    f'<span class="source-pill">{s["src"]} · {s["mode"]}</span>'
                    for s in msg["sources"]
                )
                st.markdown(
                    f'<div class="source-tag">Sources: {pills}</div>',
                    unsafe_allow_html=True
                )

    # Input
    st.markdown("")
    col_input, col_btn = st.columns([6, 1])
    with col_input:
        user_q = st.text_input(
            "Ask anything",
            placeholder="Ask a question about your data...",
            label_visibility="collapsed",
            key="chat_input"
        )
    with col_btn:
        send = st.button("Send", use_container_width=True)

    if send and user_q.strip():
        st.session_state.messages.append({"role": "user", "content": user_q})

        with st.spinner("Thinking..."):
            answer, sources, n_chunks = get_answer(
                user_q,
                st.session_state.db,
                st.session_state.G,
                st.session_state.chunks
            )

        st.session_state.messages.append({
            "role":    "assistant",
            "content": answer,
            "sources": sources,
            "chunks":  n_chunks
        })
        st.experimental_rerun()