import os
import faiss
import pickle
import numpy as np
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableMap
from langchain_core.messages import HumanMessage, AIMessage
from sentence_transformers import SentenceTransformer

# Paths and constants
FAISS_PATH = "faiss_index"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Load FAISS index
index = faiss.read_index(os.path.join(FAISS_PATH, "book_index.faiss"))

# Load metadata for book descriptions
with open(os.path.join(FAISS_PATH, "metadata.pkl"), "rb") as f:
    metadata = pickle.load(f)

# Load SentenceTransformer model for embedding queries
model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# Define the LLM prompt template
template = """You are a book recommendation assistant.
Use the retrieved book context to provide the best recommendations.

Retrieved Context:
{context}

User Question:
{question}
"""
prompt = ChatPromptTemplate.from_template(template)

# Convert Streamlit chat history into LangChain message format
def format_chat_history(chat_history):
    formatted_history = []
    for entry in chat_history:
        role, content = entry.get("role"), entry.get("content")
        if role == "user":
            formatted_history.append(HumanMessage(content=content))
        elif role == "assistant":
            formatted_history.append(AIMessage(content=content))
    return formatted_history

# Retrieve the most relevant books from FAISS index
def retrieve_books(question, k=5):
    query_embedding = model.encode([question]).astype(np.float32)
    distances, indices = index.search(query_embedding, k)
    return [metadata[i] for i in indices[0] if i < len(metadata)]

# Generate recommendation response using selected Ollama model
def generate_ans(question, chat_history, model_name="llama3.2:latest"):
    try:
        llm = OllamaLLM(model=model_name)
    except Exception as e:
        return f"⚠️ Error loading model '{model_name}': {str(e)}"

    question = str(question).strip()

    # Retrieve book context
    retrieved_docs = retrieve_books(question)
    retrieved_context = "\n\n".join(retrieved_docs) if retrieved_docs else "No relevant books found."

    # Construct LangChain pipeline
    chain = (
        RunnableMap({
            "context": RunnablePassthrough(),
            "question": RunnablePassthrough()
        })
        | prompt
        | llm
        | StrOutputParser()
    )

    # Run the chain and return result
    return chain.invoke({
        "context": retrieved_context,
        "question": question
    })

# Wrapper for Streamlit frontend
def query_book_recommendation(user_input, llm_model):
    return generate_ans(user_input, [], llm_model)
