# RAG Chatbot

A simple Retrieval-Augmented Generation (RAG) chatbot that allows users to upload PDF or text documents and ask questions about them.

## Features

- 📤 Upload PDF and text files
- 🔍 Semantic search using vector embeddings
- 🤖 Question answering powered by Groq API (Mixtral 8x7B)
- 📚 Source document references
- 💬 Interactive chat interface
- 🚀 Fast inference with Groq

## Project Structure

```
GenAI/
├── src/
│   ├── document_loader.py    # PDF and text file loading
│   ├── vector_store.py       # Vector embeddings and similarity search
│   └── rag_chain.py          # RAG chain creation and querying
├── uploads/                  # Directory for uploaded files
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── .env.example             # Environment variable template
└── README.md                # This file
```

## Prerequisites

- Python 3.8 or higher
- Groq API key (Get one at https://console.groq.com)

## Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd GenAI
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env and add your Groq API key
   # GROQ_API_KEY=your_key_here
   ```

## Usage

1. **Start the application:**
   ```bash
   streamlit run app.py
   ```

2. **Open your browser:**
   The app will open at `http://localhost:8501`

3. **Upload a document:**
   - Click the upload button in the sidebar
   - Select a PDF or text file

4. **Ask questions:**
   - Enter your question in the text box
   - Get instant answers with source references

## How It Works

### RAG Pipeline

1. **Document Loading**: PDF and text files are loaded and split into chunks
2. **Embeddings**: Text chunks are converted to vector embeddings using OpenAI
3. **Vector Store**: Embeddings are stored in a FAISS index for fast retrieval
4. **Retrieval**: User questions are converted to embeddings and matched against stored documents
5. **Generation**: Retrieved context is passed to GPT-3.5-turbo to generate answers

## Configuration

### Document Chunking
Adjust chunk size and overlap in `src/document_loader.py`:
```python
DocumentLoader(chunk_size=1000, chunk_overlap=200)
```

### LLM Settings
Modify temperature and model in `src/rag_chain.py`:
```python
RAGChain(retriever, llm_model="llama-3.1-70b-versatile", temperature=0.7)
# Available Groq models (check https://console.groq.com/docs/models):
# - llama-3.1-70b-versatile (recommended, powerful)
# - llama-3.1-8b-instant (faster, lighter)
# - mixtral-8x7b-32768 (if available)
```

### Retrieval
Change number of retrieved documents in `app.py`:
```python
retriever = vector_store.get_retriever(k=4)  # Change k value
```

## Dependencies
- **streamlit**: Web UI framework
- **langchain**: LLM orchestration
- **langchain-groq**: Groq API integration
- **pypdf**: PDF parsing
- **faiss-cpu**: Vector similarity search
- **sentence-transformers**: Embedding model (HuggingFace)
- **python-dotenv**: Environment variable management

## Groq API Models

The following Groq models are available:

1. **llama-3.1-70b-versatile** (Recommended)
   - Powerful reasoning
   - Best for RAG use cases
   - Excellent quality
   
2. **llama2-70b-4096**
   - More powerful reasoning
   - Larger context window

## Troubleshooting

### "GROQ_API_KEY not set"
Make sure your `.env` file exists and contains a valid Groq API key.

### "No documents in vector store"
Ensure a document is uploaded before asking questions.

### Import errors
Reinstall dependencies:
```bash
pip install --upgrade -r requirements.txt
```

## Future Enhancements

- Support for more file formats (DOCX, PPT)
- Multi-document queries
- Document summarization
- Custom LLM model selection
- Conversation history
- Document metadata filtering
- Caching for improved performance

## License

MIT License

## Support

For issues or questions, please refer to the project documentation or LangChain documentation at https://python.langchain.com/
