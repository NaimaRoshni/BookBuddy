import chromadb
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter

pdf_path = "booklist.pdf"

# Load PDF and extract text
def load_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

# Chunk the extracted text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

def chunk_text(text):
    return text_splitter.split_text(text)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create ChromaDB client
chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_or_create_collection(name="book_recommendations")

# Process PDF
def process_pdf(pdf_path):
    text = load_pdf(pdf_path)
    chunks = chunk_text(text)
    embeddings = model.encode(chunks).tolist()
    
    # Store in ChromaDB
    for i, chunk in enumerate(chunks):
        collection.add(
            ids=[str(i)],
            embeddings=[embeddings[i]],
            metadatas=[{"chunk": chunk}]
        )
    
    print("Book embeddings stored in ChromaDB.")


