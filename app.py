import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import CrossEncoder

import rag_chat
from rag_chat import (
    load_collection, 
    retrieve_chunks, 
    format_context, 
    generate_answer, 
    GEN_MODEL, 
    RERANK_MODEL
)
from memory import ConversationMemory

# 1. Page Configuration
st.set_page_config(page_title="Financial RAG Assistant", page_icon="📊", layout="centered")
st.title("📊 Financial Announcements RAG Assistant")
st.markdown("Ask questions about company financial results, dividends, and contracts.")


# Sidebar: Sample Questions
with st.sidebar:
    st.header("💡 Sample Questions")
    
    st.subheader("Single Turn")
    st.markdown("""
    - Which companies reported financial results?
    - Tell me about SAL's 2023 performance.
    """)
    
    st.subheader("🧠 Test Memory (Multi-turn)")
    st.markdown("""
    Try asking these in order:
    1. What dividends were announced by ZAIN KSA?
    2. When was **it** announced? 
    3. Did **they** sign any other contracts?
    """)
    
    st.divider()
    st.caption("This assistant uses Sentence-BERT for retrieval, Cross-Encoder for re-ranking, and Flan-T5 for generation.")

# 2. Resource Initialization
@st.cache_resource(show_spinner="Loading Database and Models (This may take a minute)...")
import os
import sys
import subprocess

# 2. Resource Initialization
@st.cache_resource(show_spinner="Loading Database and Models (This may take a minute)...")
def init_system():
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
    if not os.path.exists(db_path):
        st.info("First time setup: Building Vector Database natively on the cloud... This will take 1-2 minutes.")
        try:
            subprocess.run([sys.executable, "build_vector_db.py"], check=True)
            st.success("Database built successfully!")
        except Exception as e:
            st.error(f"Error building database: {e}")

    collection = load_collection()
    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL)
    reranker = CrossEncoder(RERANK_MODEL)
    return collection, tokenizer, model, reranker
collection, tokenizer, model, reranker = init_system()

# 3. Session State Setup
if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationMemory(max_turns=5)

# 4. Chat Interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("View Source Citations"):
                for source in message["sources"]:
                    st.markdown(f"- {source}")

if prompt := st.chat_input("E.g., What dividends were announced by ZAIN KSA?"):
    
    # Input validation: Reject inputs that lack alphanumeric characters
    if not any(char.isalnum() for char in prompt):
        st.warning("Please enter a valid question containing text, rather than just punctuation.")
    else:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.memory.add_user(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching and generating answer..."):
                
                search_query = prompt
                user_msgs = [msg for role, msg in st.session_state.memory.buffer if role == "User"]
                
                if len(user_msgs) >= 2:
                    last_topic = user_msgs[-2]
                    search_query = f"{last_topic} {prompt}"

                chunks = retrieve_chunks(collection, search_query, reranker)

                if not chunks:
                    response = "This question does not appear related to financial announcements."
                    citations = []
                    st.session_state.memory.buffer.pop() 
                    st.markdown(response)
                else:
                    context_text, citations = format_context(chunks)
                    
                    rag_chat.memory = st.session_state.memory
                    
                    response = generate_answer(tokenizer, model, prompt, context_text)
                    st.session_state.memory.add_assistant(response)
                    
                    st.markdown(response)
                    
                    with st.expander("View Source Citations"):
                        for cite in citations:
                            st.markdown(f"- {cite}")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "sources": citations if chunks else []
                })
