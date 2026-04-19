"""Module for managing vector embeddings and similarity search."""

import os
from typing import List
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document


class VectorStore:
    """Manage vector embeddings and retrieve similar documents."""

    def __init__(self, embeddings=None):
        """
        Initialize the vector store.
        
        Args:
            embeddings: Embeddings model to use (defaults to HuggingFace)
        """
        if embeddings is None:
            # Use HuggingFace embeddings (free, no API key required)
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        else:
            self.embeddings = embeddings
        
        self.vector_store = None

    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of documents to add
        """
        if not documents:
            raise ValueError("No documents provided")
        
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(
                documents, 
                self.embeddings
            )
        else:
            self.vector_store.add_documents(documents)

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of similar documents
        """
        if self.vector_store is None:
            raise ValueError("No documents in vector store")
        
        return self.vector_store.similarity_search(query, k=k)

    def get_retriever(self, k: int = 4):
        """
        Get a retriever for use with LangChain chains.
        
        Args:
            k: Number of documents to retrieve
            
        Returns:
            Retriever object
        """
        if self.vector_store is None:
            raise ValueError("No documents in vector store")
        
        return self.vector_store.as_retriever(search_kwargs={"k": k})

    def clear(self) -> None:
        """Clear the vector store."""
        self.vector_store = None
