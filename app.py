"""Main Streamlit application for RAG Chatbot."""

import os
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

from src.document_loader import DocumentLoader
from src.vector_store import VectorStore
from src.rag_chain import RAGChain


# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📄 RAG Chatbot")
st.write("Upload a PDF or text document and ask questions about it!")

# Initialize session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False

# Sidebar for file upload
with st.sidebar:
    st.header("📤 Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF or text file",
        type=["pdf", "txt", "md"]
    )
    
    if uploaded_file is not None:
        # Save uploaded file
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        
        file_path = upload_dir / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Process document
        with st.spinner("Processing document..."):
            try:
                # Load document
                loader = DocumentLoader()
                documents = loader.load_document(str(file_path))
                
                # Create vector store
                vector_store = VectorStore()
                vector_store.add_documents(documents)
                
                # Create RAG chain
                retriever = vector_store.get_retriever(k=4)
                rag_chain = RAGChain(retriever)
                
                # Store in session state
                st.session_state.vector_store = vector_store
                st.session_state.rag_chain = rag_chain
                st.session_state.documents_loaded = True
                
                st.success(f"✅ Loaded {len(documents)} document chunks!")
                st.info(f"📄 File: {uploaded_file.name}")
                
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
                st.session_state.documents_loaded = False

# Main chat interface
if st.session_state.documents_loaded:
    st.header("💬 Ask Questions")
    
    # Question input
    question = st.text_input(
        "Enter your question:",
        placeholder="What is this document about?",
        key="question_input"
    )
    
    if question:
        with st.spinner("Searching and generating answer..."):
            try:
                result = st.session_state.rag_chain.query(question)
                
                # Display answer
                st.subheader("Answer")
                st.write(result["answer"])
                
                # Display source documents
                if result["source_documents"]:
                    with st.expander("📚 Source Documents"):
                        for i, doc in enumerate(result["source_documents"], 1):
                            st.write(f"**Source {i}:**")
                            st.write(doc.page_content[:500] + "...")
                            if doc.metadata:
                                st.caption(str(doc.metadata))
                
            except Exception as e:
                st.error(f"❌ Error generating answer: {str(e)}")
else:
    st.info("👈 Please upload a document in the sidebar to get started!")

# Footer
st.divider()
st.caption("Powered by OpenAI GPT and LangChain | RAG Chatbot v1.0")
