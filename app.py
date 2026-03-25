import os
import time

import streamlit as st
from dotenv import load_dotenv

from src.config import CREDENTIALS, PROJECT_ID, DEFAULT_MODEL_ID, DEFAULT_PARAMETERS
from src.document_processor import process_document, create_vectorstore
from src.llm_service import get_llm, get_ollama_llm
from src.rag_chain import create_conversational_chain

# Load environment variables
load_dotenv()

st.set_page_config(page_title="RAG AI Chatbot", page_icon="🤖", layout="wide")

# Sidebar for configuration and file uploads
with st.sidebar:
    st.title("🤖 RAG Configuration")
    st.markdown("---")

    # Select Provider
    provider = st.radio("Select LLM Provider", ["IBM Watsonx", "Local (Ollama)"])

    if provider == "Local (Ollama)":
        ollama_model = st.selectbox(
            "Select Local Model",
            ["qwen2.5:1.5b-instruct", "phi3:mini", "llama3.2", "phi3:latest"],
            index=0,
        )
        st.info("qwen2.5 1.5B hızlı ve hafif; phi3:mini de küçük, hız için uygun.")

    api_key = None
    if provider == "IBM Watsonx":
        # Check for API Key
        api_key = os.getenv("IBM_CLOUD_API_KEY")
        if not api_key:
            api_key = st.text_input("Enter IBM Cloud API Key", type="password")
            if api_key:
                os.environ["IBM_CLOUD_API_KEY"] = api_key
                st.success("API Key updated local session!")
            else:
                st.warning("Please provide an API Key.")

    st.markdown("### 📄 Document Upload")
    uploaded_file = st.file_uploader(
        "Upload a document (PDF, TXT, XLSX)", type=["pdf", "txt", "xlsx"]
    )

    if st.button("Process Document") and uploaded_file:
        if provider == "IBM Watsonx" and not api_key:
            st.error("Please provide an API Key for IBM Watsonx.")
        else:
            with st.spinner("Processing document..."):
                # Save uploaded file temporarily
                temp_path = os.path.join("tmp", uploaded_file.name)
                os.makedirs("tmp", exist_ok=True)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Process and index
                texts = process_document(temp_path)
                embed_backend = os.getenv("EMBED_BACKEND", "ollama")
                embed_model = os.getenv("EMBED_MODEL", "nomic-embed-text")
                persist_dir = os.path.join("tmp", "chroma", uploaded_file.name)
                vectorstore = create_vectorstore(
                    texts,
                    embedding_model=embed_model,
                    prefer=embed_backend,
                    persist_directory=persist_dir,
                )

                # Initialize LLM and Chain Based on Provider
                if provider == "IBM Watsonx":
                    llm = get_llm(
                        DEFAULT_MODEL_ID,
                        DEFAULT_PARAMETERS,
                        CREDENTIALS,
                        PROJECT_ID,
                    )
                else:
                    llm = get_ollama_llm(ollama_model)

                st.session_state.qa_chain = create_conversational_chain(
                    llm, vectorstore
                )
                st.success(f"Document processed with {provider}!")

# Main Chat Interface
st.title("💬 Chat with your Document")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your document..."):
    if "qa_chain" not in st.session_state:
        st.error("Please upload and process a document first.")
    else:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    start = time.perf_counter()
                    result = st.session_state.qa_chain.invoke({"question": prompt})
                    elapsed = time.perf_counter() - start
                    answer = result.get("answer", "No answer found.")
                    st.markdown(answer)
                    st.caption(f"Answered in {elapsed:.1f}s")
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )
                except Exception as e:
                    st.error(f"Error: {e}")

st.markdown("---")
st.caption("Powered by IBM Watsonx AI & LangChain")
