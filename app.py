import streamlit as st
from main import query_book_recommendation

#  Streamlit UI setup
st.set_page_config(page_title="Book Recommendation Chatbot", page_icon="📚", layout="wide")
st.title("📚 AI-Powered Book Recommendation Chatbot")

#  Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar for model selection
with st.sidebar:
    st.subheader("Settings")
    llm_model = st.selectbox("Select LLM Model", ["llama3.2:latest"], index=0)  # 🔄 Only show Llama3
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.success("Chat history cleared!")

# Display chat history
for message in st.session_state.chat_history:
    role, content = message["role"], message["content"]
    avatar = "🤖" if role == "assistant" else "🧑"
    st.chat_message(role, avatar=avatar).markdown(content)

# User input field
user_input = st.text_input("Ask for book recommendations:", "", placeholder="Enter your query...")

# Process user query
if st.button("Submit"):
    if user_input.strip():
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.chat_message("user", avatar="🧑").markdown(user_input)

        with st.spinner("Fetching recommendations..."):
            response = query_book_recommendation(user_input, llm_model)

            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.chat_message("assistant", avatar="🤖").markdown(response)
    else:
        st.warning("Please enter a query.")
