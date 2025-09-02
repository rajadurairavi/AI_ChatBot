# pip install sentence-transformers langchain-community faiss-cpu
 
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
 
# 1) Tiny knowledge base
docs = [
    "Raja is learning GenAI.",
    "FastAPI is a Python framework for building APIs.",
    "Clouds are made of tiny water droplets called water vapor.",
    "RAG retrieves relevant text from a vector database before the LLM answers."
]
 
# 2) Local embeddings (downloads a small model on first run)
emb = HuggingFaceEmbeddings(model_name=r"C:\models\all-MiniLM-L6-v2")
 
# 3) Build vector store (FAISS) and retriever
vs = FAISS.from_texts(docs, embedding=emb)
retriever = vs.as_retriever(search_kwargs={"k": 2})
 
# 4) Helper to ask and see retrieved context
def ask(q: str):
    hits = retriever.invoke(q)
    print(f"\nQ: {q}")
    print("Retrieved context:")
    for d in hits:
        print("-", d.page_content)
 
# 5) Try a few
ask("What is Raja learning?")
ask("What is FastAPI?")
ask("Explain RAG in one line.")
ask("How many legs does a spider have?")  # not in our data