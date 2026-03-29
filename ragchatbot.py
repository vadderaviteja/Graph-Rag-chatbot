import os
import sys
import glob

import pandas as pd
from dotenv import load_dotenv

# ── Modern LangChain imports (zero yellow warnings) ───────────────────────────
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# ── Config ────────────────────────────────────────────────────────────────────
DATA_FOLDER   = "D:\VSCODE\inn_ai\chatbot\data"
FAISS_INDEX   = "./faiss_index"
EMBED_MODEL   = "all-MiniLM-L6-v2"
GROQ_MODEL    = "llama-3.3-70b-versatile"
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50
TOP_K_CHUNKS  = 5

# ── Load API key ──────────────────────────────────────────────────────────────
load_dotenv()
GROQ_API_KEY= "xxxxxxxxxxxxxxxxxxxxxxxxxx"
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

if not "GROQ_API_KEY":
    print("GROQ_API_KEY not found in .env file.")
    print("Add this line to your .env file:")
    print("GROQ_API_KEY=your_key_here")
    print("Get a free key at: https://console.groq.com")
    sys.exit(1)


# ── Document loader ───────────────────────────────────────────────────────────
def load_documents(folder: str) -> list[Document]:
    files = [f for f in glob.glob(f"{folder}/**/*", recursive=True) if os.path.isfile(f)]
    files += [f for f in glob.glob(f"{folder}/*") if os.path.isfile(f)]
    files = list(set(files))

    if not files:
        print(f"No files found in '{folder}/'.")
        print("Add CSV / Excel / PDF / TXT files there and try again.")
        sys.exit(1)

    docs = []
    for filepath in files:
        ext      = os.path.splitext(filepath)[1].lower()
        filename = os.path.basename(filepath)
        try:
            if ext == ".csv":
                df   = pd.read_csv(filepath)
                text = df.to_string(index=False)
                docs.append(Document(
                    page_content=text,
                    metadata={"source": filename, "type": "csv"}
                ))
                print(f"   CSV   : {filename}  ({len(df)} rows)")

            elif ext in [".xlsx", ".xls"]:
                xl = pd.ExcelFile(filepath)
                for sheet in xl.sheet_names:
                    df   = xl.parse(sheet)
                    text = f"Sheet: {sheet}\n" + df.to_string(index=False)
                    docs.append(Document(
                        page_content=text,
                        metadata={"source": filename, "sheet": sheet, "type": "excel"}
                    ))
                print(f"   Excel : {filename}  ({len(xl.sheet_names)} sheet(s))")

            elif ext == ".pdf":
                from langchain_community.document_loaders import PyPDFLoader
                pages = PyPDFLoader(filepath).load()
                docs.extend(pages)
                print(f"   PDF   : {filename}  ({len(pages)} page(s))")

            elif ext == ".txt":
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                docs.append(Document(
                    page_content=text,
                    metadata={"source": filename, "type": "txt"}
                ))
                print(f"   TXT   : {filename}")

            else:
                print(f"   Skipped (unsupported): {filename}")

        except Exception as e:
            print(f"   Failed to load {filename}: {e}")

    return docs


# ── Vector store ──────────────────────────────────────────────────────────────
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


def build_vector_store(docs: list[Document]) -> FAISS:
    print("\nBuilding vector index...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    print(f"   Chunks created : {len(chunks)}")

    db = FAISS.from_documents(chunks, get_embeddings())
    db.save_local(FAISS_INDEX)
    print(f"   Index saved to : {FAISS_INDEX}/\n")
    return db


def load_vector_store() -> FAISS:
    db = FAISS.load_local(
        FAISS_INDEX,
        get_embeddings(),
        allow_dangerous_deserialization=True
    )
    print(f"Loaded existing index from '{FAISS_INDEX}/'\n")
    return db


# ── RAG chain (modern LCEL, replaces deprecated RetrievalQA) ─────────────────
def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(db: FAISS):
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=GROQ_MODEL,
        temperature=0.0,
        max_tokens=1024,
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a helpful data assistant.
Answer the question using ONLY the context provided below.
If the answer is not in the context, say:
"I don't have enough information in the provided data to answer this."
Never guess or make up information.

Context:
{context}

Question: {question}

Answer:"""
    )

    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K_CHUNKS}
    )

    # Modern LCEL chain - no deprecated RetrievalQA
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever


# ── Chat loop ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("     RAG Chatbot  |  Groq + FAISS + HuggingFace")
    print("=" * 55)

    if not os.path.isdir(DATA_FOLDER):
        os.mkdir(DATA_FOLDER)

    rebuild = "--rebuild" in sys.argv

    if os.path.exists(FAISS_INDEX) and not rebuild:
        db = load_vector_store()
    else:
        reason = "Rebuilding" if rebuild else "First run - building"
        print(f"\n{reason} index from '{DATA_FOLDER}/'...\n")
        docs = load_documents(DATA_FOLDER)
        print(f"\n   Documents loaded : {len(docs)}")
        db = build_vector_store(docs)

    print("Connecting to Groq...")
    chain, retriever = build_rag_chain(db)
    print("Ready! Ask anything about your data.\n")
    print("   quit    -> exit")
    print("   rebuild -> re-index data folder")
    print("-" * 55)

    while True:
        try:
            user_input = input("\n You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nBye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ["quit", "exit", "q"]:
            print("Bye!")
            break

        if user_input.lower() == "rebuild":
            print("\nRebuilding index...\n")
            docs             = load_documents(DATA_FOLDER)
            db               = build_vector_store(docs)
            chain, retriever = build_rag_chain(db)
            print("Done. Ready!")
            continue

        try:
            print("\nThinking...", end="\r")

            answer      = chain.invoke(user_input)
            source_docs = retriever.invoke(user_input)

            print(f"\n Bot: {answer}")

            if source_docs:
                seen = set()
                print("\n   Sources used:")
                for doc in source_docs:
                    src = doc.metadata.get("source", "unknown")
                    if src not in seen:
                        seen.add(src)
                        sheet = doc.metadata.get("sheet", "")
                        extra = f" -> sheet: {sheet}" if sheet else ""
                        print(f"      - {src}{extra}")

        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()
