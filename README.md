# RAG Pipeline for LLMs

This repository contains a complete implementation of a Retrieval-Augmented Generation (RAG) pipeline for Large Language Models. RAG enhances LLM responses by retrieving relevant information from your documents before generating answers.

## Features

- 📚 Document processing from multiple sources (text files, PDF, webpages)
- 🔍 Semantic search via vector embeddings
- 💾 Persistent vector database for efficient retrieval
- 🧠 Context-aware LLM responses
- 📊 Basic evaluation toolkit

## Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API key

### Installation

1. Clone this repository:

   ```
   git clone https://github.com/yourusername/Rag-pipeline-llms.git
   cd Rag-pipeline-llms
   ```

2. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Set up your OpenAI API key:
   - Rename `.env.example` to `.env`
   - Add your OpenAI API key in the `.env` file:
     ```
     OPENAI_API_KEY=your_openai_api_key
     ```

### Usage

1. **Add your documents**:

   - Place text files in the `data/documents` directory
   - For PDFs or web pages, use the provided loaders

2. **Run the pipeline**:

   ```
   python rag.py
   ```

3. **First-time setup**:

   - Uncomment the `rag.ingest_documents()` line in `rag.py` for the first run to index your documents
   - Re-comment this line for subsequent runs to avoid re-indexing

4. **Ask questions**:
   - Type your questions when prompted
   - Type "exit" to quit

## Components

### Core RAG Pipeline (`rag.py`)

The main component that ties everything together:

- Document ingestion and chunking
- Vector database management
- Retrieval system setup
- LLM integration

### Document Loaders

- **Text Files**: Built into the main pipeline
- **PDF Files** (`pdf_loader.py`): Load and process PDF documents
- **Web Pages** (`web_loader.py`): Fetch and process web content

### Evaluation Tool (`evaluation.py`)

Basic evaluation tool to measure RAG pipeline performance against ground truth.

## Advanced Usage

### Using PDF Loader
