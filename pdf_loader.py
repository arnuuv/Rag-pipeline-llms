from langchain.document_loaders import PyPDFLoader

def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load() 
    
def evaluate_rag(rag_pipeline, test_questions, ground_truth):
    """Basic evaluation of RAG pipeline."""
    scores = []
    for i, question in enumerate(test_questions):
        answer = rag_pipeline.answer_question(question)["answer"]
        # Here you could implement more sophisticated evaluation
        # Like semantic similarity or asking an LLM to grade the answer
        contains_truth = ground_truth[i].lower() in answer.lower()
        scores.append(1 if contains_truth else 0)
    
    return sum(scores) / len(scores) if scores else 0 