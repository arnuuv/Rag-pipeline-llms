import os
from typing import List, Dict, Any
import chromadb
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class RAGPipeline:
    def __init__(self, documents_dir: str = "data/documents"):
        self.documents_dir = documents_dir
        self.embeddings = OpenAIEmbeddings()
        self.vector_db = None
        self.llm = OpenAI(temperature=0)
        
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
        
    def answer_question(self, question: str) -> Dict[str, Any]:
        """Process a question through the RAG pipeline."""
        chain = self.setup_qa_chain()
        if not chain:
            return {"answer": "System not ready. Please ingest documents first."}
            
        result = chain({"query": question})
        return {
            "answer": result["result"],
            "source_documents": [doc.page_content for doc in result["source_documents"]]
        }

# Example usage
if __name__ == "__main__":
    rag = RAGPipeline()
    
    # Ingest documents - uncomment this line for first run or when adding new documents
    # rag.ingest_documents()
    
    # Answer questions
    while True:
        question = input("\nEnter your question (or 'exit' to quit): ")
        if question.lower() == 'exit':
            break
            
        result = rag.answer_question(question)
        print(f"\nAnswer: {result['answer']}")
        
        print("\nSources:")
        for i, doc in enumerate(result["source_documents"]):
            print(f"\nSource {i+1}:\n{doc[:200]}...") 