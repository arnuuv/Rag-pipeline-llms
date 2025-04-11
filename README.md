# RAG Pipeline for LLMs

This repository contains a complete implementation of a Retrieval-Augmented Generation (RAG) pipeline for Large Language Models. RAG enhances LLM responses by retrieving relevant information from your documents before generating answers.

## Features

- 📚 Document processing from multiple sources (text files, PDF, webpages)
- 🔍 Semantic search via vector embeddings
- 💾 Persistent vector database for efficient retrieval
- 🧠 Context-aware LLM responses
- 📊 Basic evaluation toolkit
- 📝 Query history and logging system

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
   - Type "history" to see your 5 most recent queries
   - Type "search: keyword" to find relevant past queries

## Components

### Core RAG Pipeline (`rag.py`)

The main component that ties everything together:

- Document ingestion and chunking
- Vector database management
- Retrieval system setup
- LLM integration
- Query history and logging

### Document Loaders

- **Text Files**: Built into the main pipeline
- **PDF Files** (`pdf_loader.py`): Load and process PDF documents
- **Web Pages** (`web_loader.py`): Fetch and process web content

### Evaluation Tool (`evaluation.py`)

Basic evaluation tool to measure RAG pipeline performance against ground truth.

## Advanced Usage

### Using PDF Loader

```python
from pdf_loader import load_pdf

pdf_docs = load_pdf("path/to/your/document.pdf")
# Process pdf_docs with your RAG pipeline
```

### Using Web Loader

```python
from web_loader import load_webpage

web_docs = load_webpage("https://example.com/page")
# Process web_docs with your RAG pipeline
```

### Using Query History

The RAG pipeline automatically logs all queries and their responses to a file called `query_history.json`. You can:

1. View recent queries with the `history` command
2. Search through past queries with `search: keyword`
3. Access the history programmatically:

```python
from rag import RAGPipeline

rag = RAGPipeline()
# Get all history
all_history = rag.get_query_history()

# Get last 10 queries
recent = rag.get_query_history(10)

# Search for specific queries
results = rag.search_query_history("specific topic")
```

### Evaluating Performance

```python
from evaluation import evaluate_rag
from rag import RAGPipeline

rag = RAGPipeline()
questions = ["What is RAG?", "How does vector search work?"]
ground_truth = ["Retrieval-Augmented Generation", "similarity search"]

accuracy = evaluate_rag(rag, questions, ground_truth)
print(f"Accuracy: {accuracy}")
```

## Customization

You can customize various aspects of the RAG pipeline:

- **Chunk Size**: Modify `chunk_size` and `chunk_overlap` in `ingest_documents()`
- **Retrieval Parameters**: Adjust the `k` value in `setup_retriever()`
- **LLM Parameters**: Change the `temperature` in the `RAGPipeline` initialization
- **Prompt Template**: Customize the prompt in `setup_qa_chain()`
- **Query Log Path**: Change the file path in the `RAGPipeline` initialization

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
