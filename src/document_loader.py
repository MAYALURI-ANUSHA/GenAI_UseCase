"""Module for loading and processing documents (PDF and text files)."""

import os
from typing import List
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


class DocumentLoader:
    """Load and process PDF and text documents."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the document loader.
        
        Args:
            chunk_size: Size of each text chunk
            chunk_overlap: Overlap between chunks
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def load_pdf(self, file_path: str) -> List[Document]:
        """
        Load and process a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of document chunks
        """
        try:
            from pypdf import PdfReader
            
            documents = []
            reader = PdfReader(file_path)
            
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text.strip():
                    doc = Document(
                        page_content=text,
                        metadata={
                            "source": file_path,
                            "page": page_num + 1
                        }
                    )
                    documents.append(doc)
            
            return self.text_splitter.split_documents(documents)
        
        except Exception as e:
            raise ValueError(f"Error loading PDF: {str(e)}")

    def load_text(self, file_path: str) -> List[Document]:
        """
        Load and process a text file.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            List of document chunks
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            doc = Document(
                page_content=text,
                metadata={"source": file_path}
            )
            
            return self.text_splitter.split_documents([doc])
        
        except Exception as e:
            raise ValueError(f"Error loading text file: {str(e)}")

    def load_document(self, file_path: str) -> List[Document]:
        """
        Load a document based on file extension.
        
        Args:
            file_path: Path to the document
            
        Returns:
            List of document chunks
        """
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension == '.pdf':
            return self.load_pdf(file_path)
        elif file_extension in ['.txt', '.md']:
            return self.load_text(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
