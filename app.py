# --- Fix for ChromaDB on older SQLite ---
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import streamlit as st
from dotenv import load_dotenv
from pathlib import Path
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
from langdetect import detect

# LLM imports
import requests
from groq import Groq
import google.generativeai as genai

# Load env
load_dotenv()

# --- ENV VARIABLES ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
LLM_BACKEND = os.getenv("LLM_BACKEND", "gemini")  # 'gemini', 'groq', 'hf'
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")
HF_MODEL = os.getenv("HF_MODEL", "HuggingFaceH4/zephyr-7b-beta")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-pro")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-70b-8192")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
TOP_K = int(os.getenv("TOP_K", 4))
DB_DIR = "chroma_db"

st.set_page_config(page_title="🎓 Multilingual College Chatbot", page_icon="🎓")

# --- EMBEDDING MODEL ---
@st.cache_resource
def get_embedder():
    return SentenceTransformer(EMBEDDING_MODEL)

embedder = get_embedder()

# --- PDF TO TEXT ---
def pdf_to_text(path: Path) -> str:
    text = []
    with fitz.open(path) as doc:
        for page in doc:
            content = page.get_text()
            if content:
                text.append(content)
    return "\n".join(text)

# --- CHUNKING ---
def chunk_text(text: str, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks

# --- CHROMA DB INIT ---
client = chromadb.PersistentClient(path=DB_DIR)
collection = client.get_or_create_collection(
    name="college_docs",
    embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(EMBEDDING_MODEL)
)

# --- INDEXING PDFs ---
def index_pdfs(search_path: Path):
    existing_ids = set(collection.get()["ids"])
    for pdf in search_path.glob("*.pdf"):
        text = pdf_to_text(pdf)
        if not text.strip():
            continue
        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        for i, chunk in enumerate(chunks):
            doc_id = f"{pdf.name}_{i}"
            if doc_id in existing_ids:
                continue
            collection.add(
                documents=[chunk],
                metadatas=[{"source": pdf.name}],
                ids=[doc_id]
            )
    # No persist() needed for PersistentClient

# --- Scan for PDFs in root or 'pdfs/' ---
root_pdfs = list(Path(".").glob("*.pdf"))
pdfs_folder = Path("pdfs")
folder_pdfs = list(pdfs_folder.glob("*.pdf")) if pdfs_folder.exists() else []
all_pdfs = root_pdfs + folder_pdfs

if not all_pdfs:
    st.warning("No PDF files found in repo root or 'pdfs/' folder.")
else:
    index_pdfs(Path("."))
    if pdfs_folder.exists():
        index_pdfs(pdfs_folder)

# --- RETRIEVAL ---
def retrieve(query: str, k=TOP_K):
    results = collection.query(query_texts=[query], n_results=k)
    docs = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        docs.append({"text": doc, "source": meta.get("source", "unknown")})
    return docs

# --- PROMPT BUILDER ---
def compose_prompt_bullet(question: str, contexts, lang="en"):
    ctx_texts = "\n\n".join([f"Source: {c['source']}\n{c['text']}" for c in contexts])

    if lang == "ar":
        prompt = (
            "أنت مساعد ذكي لمعلومات الكلية.\n"
            "استخدم فقط المعلومات المقدمة للإجابة.\n"
            "قدم الإجابة في شكل نقاط.\n"
            "اجعل عناوين البريد الإلكتروني قابلة للنقر باستخدام mailto: وأرقام الهاتف باستخدام tel:.\n"
            "إذا لم تكن الإجابة في النص، قل أنك لا تعرف واقترح التواصل مع الكلية.\n\n"
            f"المحتوى:\n{ctx_texts}\n\nالسؤال: {question}\n\n"
            "أجب في نقاط واضحة ودقيقة."
        )
    else:
        prompt = (
            "You are an intelligent college information assistant.\n"
            "Use ONLY the provided context to answer.\n"
            "Provide the answer in BULLET POINTS.\n"
            "Make emails clickable using mailto: and phone numbers clickable using tel:.\n"
            "If the answer is not in context, say you don't know and suggest contacting the college.\n\n"
            f"CONTEXT:\n{ctx_texts}\n\nQUESTION: {question}\n\n"
            "Answer in bullet points, concise, with direct facts."
        )
    return prompt

# --- LLM CALLS ---
def call_hf_chat(model: str, prompt: str):
    if not HF_API_TOKEN:
        raise RuntimeError("HF_API_TOKEN not set.")
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 400, "temperature": 0.2}}
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    out = r.json()
    if isinstance(out, list) and len(out) > 0 and "generated_text" in out[0]:
        return out[0]["generated_text"]
    return str(out)

def call_gemini_chat(model: str, prompt: str):
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set.")
    genai.configure(api_key=GEMINI_API_KEY)
    m = genai.GenerativeModel(model)
    resp = m.generate_content(prompt)
    return resp.text

def call_groq_chat(model: str, prompt: str):
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set.")
    client = Groq(api_key=GROQ_API_KEY)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=500
    )
    return resp.choices[0].message.content

# --- STREAMLIT UI ---
st.title("🎓 Multilingual College Enquiry Chatbot")
st.write("Ask questions in English or Arabic, get answers in the same language.")

question = st.text_input("Your question (English or Arabic):")
if st.button("Ask") and question.strip():
    try:
        lang = detect(question)
    except:
        lang = "en"

    with st.spinner("Retrieving context..."):
        contexts = retrieve(question, k=TOP_K)

    st.subheader("Retrieved Contexts")
    for c in contexts:
        st.markdown(f"**Source:** {c['source']}")
        st.write(c['text'][:500] + ("..." if len(c['text']) > 500 else ""))

    prompt = compose_prompt_bullet(question, contexts, lang="ar" if lang == "ar" else "en")

    with st.spinner(f"Querying {LLM_BACKEND}..."):
        try:
            if LLM_BACKEND == "hf":
                answer = call_hf_chat(HF_MODEL, prompt)
            elif LLM_BACKEND == "gemini":
                answer = call_gemini_chat(GEMINI_MODEL, prompt)
            elif LLM_BACKEND == "groq":
                answer = call_groq_chat(GROQ_MODEL, prompt)
            else:
                answer = "Invalid LLM_BACKEND in .env"
        except Exception as e:
            answer = f"Error: {e}"

    st.subheader("Answer")
    st.markdown(answer)
