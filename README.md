# RAG Chatbot

AI chatbot that answers questions from your documents using Retrieval-Augmented Generation. Upload a PDF, TXT, or Markdown file and ask questions -- the chatbot answers using only the content from your documents.

**Built with:** Python, Flask, LangChain, FAISS, HuggingFace Embeddings, Groq (Llama 3.1)

## How It Works

1. **Upload** a document -- it gets split into chunks and embedded into vectors using HuggingFace's `all-MiniLM-L6-v2` model
2. **Store** the vectors in a local FAISS index for fast similarity search
3. **Ask** a question -- the system retrieves the most relevant chunks and sends them to Llama 3.1 (via Groq) to generate a grounded answer

## Quick Start

```bash
git clone https://github.com/ahmaddbaig/rag-chatbot.git
cd rag-chatbot
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your Groq API key (free at https://console.groq.com)
python app.py
```

Open http://127.0.0.1:8000

Note: The first upload will be slower as it downloads the embedding model (~80MB).

## Architecture

```
Document --> PyPDF/TextLoader --> RecursiveCharacterTextSplitter --> HuggingFace Embeddings --> FAISS
                                                                                                |
Question --> HuggingFace Embeddings --> FAISS similarity search --> Top 4 chunks --> Groq LLM --> Answer
```

## Switching to OpenAI

In `rag.py`, change two lines:

```python
# Before
from langchain_groq import ChatGroq
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# After
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
```

Then set `OPENAI_API_KEY` in your `.env` file.

## Technologies

- **LangChain** -- orchestrates the RAG pipeline (document loading, chunking, retrieval, QA chain)
- **FAISS** -- Meta's vector similarity search library, runs locally with no external server
- **HuggingFace Embeddings** -- `all-MiniLM-L6-v2` model runs on CPU, no API key needed
- **Groq** -- free-tier LLM inference for Llama 3.1, fast response times
- **Flask** -- lightweight Python web framework for the API and UI
