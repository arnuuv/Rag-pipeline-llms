import os
from typing import List, Dict, Any, Optional
import json
import chromadb
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

class RAGPipeline:
    def __init__(self, documents_dir: str = "data/documents", query_log_path: str = "query_history.json"):
        self.documents_dir = documents_dir
        self.embeddings = OpenAIEmbeddings()
        self.vector_db = None
        self.llm = OpenAI(temperature=0)
        self.query_log_path = query_log_path
        self.query_history = self._load_query_history()
        
    def _load_query_history(self) -> List[Dict[str, Any]]:
        """Load query history from file if it exists."""
        if os.path.exists(self.query_log_path):
            try:
                with open(self.query_log_path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"Error loading query history file. Starting with empty history.")
                return []
        return []
        
    def _save_query_history(self):
        """Save query history to file."""
        with open(self.query_log_path, 'w') as f:
            json.dump(self.query_history, f, indent=2)
            
    def log_query(self, query: str, answer: str, sources: List[str], metadata: Optional[Dict[str, Any]] = None):
        """Log a query and its results to the query history."""
        timestamp = datetime.now().isoformat()
        
        # Create log entry
        log_entry = {
            "timestamp": timestamp,
            "query": query,
            "answer": answer,
            "sources": sources,
            "metadata": metadata or {}
        }
        
        # Add to history
        self.query_history.append(log_entry)
        
        # Save updated history
        self._save_query_history()
        
        return log_entry
    
    def get_query_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get the most recent query history entries."""
        if limit:
            return self.query_history[-limit:]
        return self.query_history
    
    def search_query_history(self, search_term: str) -> List[Dict[str, Any]]:
        """Search the query history for a term in queries or answers."""
        search_term = search_term.lower()
        results = []
        
        for entry in self.query_history:
            if (search_term in entry["query"].lower() or 
                search_term in entry["answer"].lower()):
                results.append(entry)
                
        return results
        
    def ingest_documents(self):
        """Load documents and create vector store."""
        # Load documents
        loader = DirectoryLoader(self.documents_dir, glob="**/*.txt", loader_cls=TextLoader)
        documents = loader.load()
        
        if not documents:
            print(f"No documents found in {self.documents_dir}")
            return
        
        print(f"Loaded {len(documents)} documents")
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Split into {len(chunks)} chunks")
        
        # Create vector store
        self.vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )
        print("Vector database created and persisted")
        
    def load_vector_db(self):
        """Load an existing vector database."""
        if os.path.exists("./chroma_db"):
            self.vector_db = Chroma(
                persist_directory="./chroma_db",
                embedding_function=self.embeddings
            )
            print("Loaded existing vector database")
            return True
        return False
        
    def setup_retriever(self, k: int = 4):
        """Set up the retriever with the specified number of results."""
        if not self.vector_db:
            if not self.load_vector_db():
                print("No vector database found. Please ingest documents first.")
                return None
                
        return self.vector_db.as_retriever(search_kwargs={"k": k})
        
    def setup_qa_chain(self):
        """Set up the QA chain with custom prompt."""
        retriever = self.setup_retriever()
        if not retriever:
            return None
            
        template = """
        You are an assistant that answers questions based on the provided context.
        
        Context:
        {context}
        
        Question:
        {question}
        
        Answer the question based on the context provided. If you cannot answer 
        the question from the context, say "I don't have enough information to answer this question."
        """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
        
        return chain
        
    def answer_question(self, question: str, log_query: bool = True) -> Dict[str, Any]:
        """Process a question through the RAG pipeline."""
        chain = self.setup_qa_chain()
        if not chain:
            return {"answer": "System not ready. Please ingest documents first."}
            
        result = chain({"query": question})
        
        response = {
            "answer": result["result"],
            "source_documents": [doc.page_content for doc in result["source_documents"]]
        }
        
        # Log the query if requested
        if log_query:
            self.log_query(
                query=question,
                answer=response["answer"],
                sources=response["source_documents"],
                metadata={"timestamp": datetime.now().isoformat()}
            )
        
        return response

# Example usage
if __name__ == "__main__":
    rag = RAGPipeline()
    
    # Ingest documents - uncomment this line for first run or when adding new documents
    # rag.ingest_documents()
    
    # Answer questions
    while True:
        question = input("\nEnter your question (or 'exit' to quit, 'history' to see past queries): ")
        
        if question.lower() == 'exit':
            break
        elif question.lower() == 'history':
            history = rag.get_query_history(5)  # Get the last 5 queries
            if history:
                print("\n===== Recent Queries =====")
                for i, entry in enumerate(history):
                    print(f"\n[{i+1}] Q: {entry['query']}")
                    print(f"    A: {entry['answer'][:100]}...")
                    print(f"    Time: {entry['timestamp']}")
            else:
                print("\nNo query history found.")
        elif question.lower().startswith('search:'):
            search_term = question[7:].strip()
            results = rag.search_query_history(search_term)
            if results:
                print(f"\nFound {len(results)} matches for '{search_term}':")
                for i, entry in enumerate(results):
                    print(f"\n[{i+1}] Q: {entry['query']}")
                    print(f"    A: {entry['answer'][:100]}...")
            else:
                print(f"\nNo matches found for '{search_term}'")
        else:
            result = rag.answer_question(question)
            print(f"\nAnswer: {result['answer']}")
            
            print("\nSources:")
            for i, doc in enumerate(result["source_documents"]):
                print(f"\nSource {i+1}:\n{doc[:200]}...") 