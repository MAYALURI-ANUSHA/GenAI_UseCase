"""Module for creating RAG chain with LLM."""

import os
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


class RAGChain:
    """Create and manage RAG chain for question answering."""

    def __init__(self, retriever, llm_model: str = "llama-3.1-8b-instant", temperature: float = 0.7):
        """
        Initialize the RAG chain with Groq LLM.
        
        Args:
            retriever: Vector store retriever
            llm_model: Groq model to use (default: llama-3.1-8b-instant)
            temperature: Model temperature (0-1)
        """
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
        
        self.llm = ChatGroq(
            model=llm_model,
            temperature=temperature,
            groq_api_key=api_key
        )
        
        self.retriever = retriever
        self.qa_chain = self._create_chain()

    def _create_chain(self):
        """Create the RAG QA chain."""
        prompt_template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        
        return chain

    def query(self, question: str) -> dict:
        """
        Query the RAG chain.
        
        Args:
            question: User's question
            
        Returns:
            Dictionary with answer and source documents
        """
        result = self.qa_chain({"query": question})
        return {
            "answer": result["result"],
            "source_documents": result.get("source_documents", [])
        }
