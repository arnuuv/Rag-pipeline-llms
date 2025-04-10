from langchain.document_loaders import WebBaseLoader

def load_webpage(url):
    loader = WebBaseLoader(url)
    return loader.load() 