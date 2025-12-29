📘 StudyMate – AI Powered PDF Q&A Assistant

StudyMate is an AI-based learning assistant that allows users to upload a PDF and ask questions related to it. The system understands the content using AI, finds the most relevant part from the PDF, and provides accurate answers to help students study smarter.

🚀 Features:

Upload any academic PDF

Ask questions in natural language

AI extracts and understands PDF content

Uses vector search to find the right answer

Shows answer + source context

Simple Streamlit interface

Works locally, no API key needed

🧠 How It Works

1. Extracts text from the uploaded PDF

2. Breaks text into meaningful chunks

3. Converts text into embeddings

4. Stores them using FAISS vector index

5. Converts your question into embedding

6. Finds the closest matching text

7. Uses QA model to generate the best answer


🛠 Tech Stack

Python

Streamlit

Sentence Transformers

HuggingFace Transformers

FAISS

PyMuPDF

▶ How to Run

1. Install dependencies

pip install streamlit pymupdf faiss-cpu sentence-transformers transformers

2. Run the app

streamlit run app.py

3. Open in browser

If not auto-opened:

http://localhost:8501

🎯 Purpose

Helps students study from notes / textbooks

Quickly understands long PDFs

Useful for exam preparation and learning

🌟 Future Enhancements

Chat mode (Chat with PDF)

Multi-PDF support

Highlight exact answer sentence

Online deployment

👩‍💻 Developer

Renuka Parvathi

