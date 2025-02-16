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

# Load FAISS index and metadata
FAISS_PATH = "faiss_index"
index = faiss.read_index(os.path.join(FAISS_PATH, "book_index.faiss"))

with open(os.path.join(FAISS_PATH, "metadata.pkl"), "rb") as f:
    metadata = pickle.load(f)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Define LLM prompt template
template = """You are a book recommendation assistant.
Use the retrieved book context to provide the best recommendations.

Retrieved Context:
{context}

User Question:
{question}
"""
prompt = ChatPromptTemplate.from_template(template)

# Function to format chat history
def format_chat_history(chat_history):
    formatted_history = []
    for entry in chat_history:
        role, content = entry.get("role"), entry.get("content")
        if role == "user":
            formatted_history.append(HumanMessage(content=content))
        elif role == "assistant":
            formatted_history.append(AIMessage(content=content))
    return formatted_history

# Function to retrieve relevant books using FAISS
def retrieve_books(question, k=5):
    query_embedding = model.encode([question]).astype(np.float32)
    distances, indices = index.search(query_embedding, k)
    
    retrieved_docs = [metadata[i] for i in indices[0] if i < len(metadata)]
    return retrieved_docs

# Generate book recommendations using Llama3
def generate_ans(question, chat_history, model_name="llama3.2:latest"):  # 🔄 Change default to Llama3
    llm = OllamaLLM(model=model_name)

    # Ensure question is a string
    question = str(question).strip()

    # Retrieve relevant documents from FAISS
    retrieved_docs = retrieve_books(question)

    # Convert retrieved docs to a single string
    retrieved_context = "\n\n".join(retrieved_docs) if retrieved_docs else "No relevant books found."

    chain = (
        RunnableMap({
            "context": RunnablePassthrough(),
            "question": RunnablePassthrough()
        })
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke({"context": retrieved_context, "question": question, "chat_history": format_chat_history(chat_history)})

# Function for Streamlit UI
def query_book_recommendation(user_input, llm_model):
    return generate_ans(user_input, [], llm_model)
