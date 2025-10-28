import streamlit as st
import fitz
import numpy as np
import ollama
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk
import time
import pandas as pd

# -------------------------
# Setup & persistent storage
# -------------------------
nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))

SAVE_DIR = "saved_data"
os.makedirs(SAVE_DIR, exist_ok=True)

MEMORY_FILE = os.path.join(SAVE_DIR, "memory.json")
MEMORY_CSV = os.path.join(SAVE_DIR, "memory.csv")

# Init memory CSV if missing
if not os.path.exists(MEMORY_CSV):
    pd.DataFrame(columns=["pdf_name", "page"]).to_csv(MEMORY_CSV, index=False)

# Load persistent Q&A memory (from previous questions)
if os.path.exists(MEMORY_FILE):
    try:
        with open(MEMORY_FILE, "r") as f:
            memory = json.load(f)
    except Exception:
        memory = {}
else:
    memory = {}

memory.setdefault("qa", {})


def save_memory():
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f)


# -------------------------
# PDF ingestion → stored inside memory.csv
# -------------------------
def save_pdf_to_memory(data: bytes, pdf_name: str):
    df = pd.read_csv(MEMORY_CSV)
    doc = fitz.open(stream=data, filetype="pdf")
    pages = [p.get_text() or "" for p in doc]   # extract text

    new_rows = pd.DataFrame({"pdf_name": pdf_name, "page": pages})
    df = pd.concat([df, new_rows], ignore_index=True)
    df.to_csv(MEMORY_CSV, index=False)


@st.cache_resource
def load_vector_memory():
    df = pd.read_csv(MEMORY_CSV)
    pages = df["page"].tolist()

    vectorizer = TfidfVectorizer(stop_words="english", max_features=2000)
    vectors = vectorizer.fit_transform(pages).toarray()
    return pages, vectorizer, vectors


# -------------------------
# Retrieval
# -------------------------
def extract_keywords(text: str):
    words = text.lower().split()
    return [w for w in words if w not in stop_words and len(w) > 3]


def retrieve_context(question):
    pages, vectorizer, vectors = load_vector_memory()

    query_vec = vectorizer.transform([question])
    scores = (vectors @ query_vec.T).flatten()

    top_idx = int(np.argmax(scores))
    if scores[top_idx] < 0.03:  # threshold
        return None

    return pages[top_idx][:1200]  # limit for model prompt


# -------------------------
# Memory Check
# -------------------------
def check_memory(question):
    return memory["qa"].get(question)


# -------------------------
# Local Model
# -------------------------
def generate_response(question, context=None):
    if context:
        prompt = f"""
You are answering from stored PDF memory.
Context:
{context}

Question: {question}
Answer accurately & concisely:
"""
    else:
        prompt = f"Question: {question}\nAnswer concisely:"

    resp = ollama.chat(
        model="llama3-local",
        messages=[{"role": "user", "content": prompt}],
        options={"num_predict": 160, "temperature": 0.1},
    )
    return resp["message"]["content"].strip()


# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="Corporate PDF Intelligence", layout="wide")
st.title("🔍 Corporate PDF Intelligence — Unified PDF Memory Mode")

with st.sidebar:
    st.subheader("Upload PDF")
    uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])
    st.markdown("---")

question = st.text_input("💬 Ask a question:")

if st.button("🔎 Get Answer"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        saved_ans = check_memory(question)
        if saved_ans:
            st.success("⚡ Answer from memory.json (learned)")
            st.write(saved_ans)
            st.caption("⏱️ 0 sec")
        else:
            start = time.time()

            if uploaded_pdf:  # add new knowledge
                save_pdf_to_memory(uploaded_pdf.read(), uploaded_pdf.name)
                load_vector_memory.clear()  # refresh cache

            keywords = extract_keywords(question)
            context = retrieve_context(" ".join(keywords) if keywords else question)

            answer = generate_response(question, context)
            end = time.time()

            st.write(answer)
            st.caption(f"⏱️ {round(end - start, 2)} sec")

            memory["qa"][question] = answer
            save_memory()

else:
    st.info("Upload PDFs & ask questions ✅")
