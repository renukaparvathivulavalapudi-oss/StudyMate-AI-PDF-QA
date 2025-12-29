import streamlit as st
import fitz  # PyMuPDF
import faiss
import os
import tempfile
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="StudyMate AI – PDF Q&A", layout="wide")
st.title("📘 StudyMate – AI Powered PDF Q&A")
st.write("Upload a PDF and ask any question related to it. StudyMate finds the answer intelligently 😊")

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2")
    return embed_model, qa_model

embed_model, qa_model = load_models()


# ---------------- FUNCTIONS ----------------
def extract_text_chunks(pdf_path, chunk_size=700):
    doc = fitz.open(pdf_path)
    chunks = []

    for page in doc:
        text = page.get_text()
        text = text.replace("\n", " ").strip()

        if len(text) == 0:
            continue
        
        # Smart Chunking
        words = text.split()
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i+chunk_size])
            chunks.append(chunk)

    return chunks


def create_faiss_index(chunks):
    embeddings = embed_model.encode(chunks)
    embedding_dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(embedding_dim)
    index.add(np.array(embeddings, dtype="float32"))

    return index, embeddings


def search_chunks(question, chunks, index, k=3):
    q_embed = embed_model.encode([question]).astype("float32")
    distances, indices = index.search(q_embed, k)

    selected_chunks = [chunks[i] for i in indices[0]]
    similarity_scores = distances[0]

    return selected_chunks, similarity_scores


# ---------------- MAIN APP ----------------
uploaded_pdf = st.file_uploader("📤 Upload your PDF", type="pdf")

if uploaded_pdf is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_pdf.read())
        pdf_path = tmp.name

    with st.spinner("⏳ Reading PDF and preparing AI brain..."):
        chunks = extract_text_chunks(pdf_path)
        index, embeddings = create_faiss_index(chunks)

    st.success("✅ PDF processed successfully! Ask your question now.")
    question = st.text_input("❓ Ask a question based on the uploaded PDF")

    if question:
        with st.spinner("🤖 Thinking..."):
            top_chunks, scores = search_chunks(question, chunks, index)

            # Hallucination Protection
            if scores[0] > 2.0:
                st.warning("⚠ The answer may not exist clearly in the PDF. Showing best attempt.")
            
            context = " ".join(top_chunks)

            result = qa_model(question=question, context=context)

        st.subheader("📌 Answer:")
        st.write(result["answer"])

        with st.expander("📄 Source Context"):
            st.write(context)
