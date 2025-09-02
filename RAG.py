# Day 7: RAG = Retrieval + LLM Generation
# ----------------------------------------
# pip install langchain langchain-community sentence-transformers faiss-cpu langchain-groq
 
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
 
# 1) Knowledge base
docs = [
    "Raja is learning GenAI.",
    "FastAPI is a Python framework for building APIs.",
    "Clouds are made of tiny water droplets called water vapor.",
    "RAG retrieves relevant text from a vector database before the LLM answers.",
    "Python is widely used for test automation and AI."
]
 
# 2) Embeddings + FAISS
embeddings = HuggingFaceEmbeddings(
    model_name=r"C:\models\all-MiniLM-L6-v2"   # local path model
)
vectorstore = FAISS.from_texts(docs, embedding=embeddings)
 
# 3) Connect LLM (Groq)
llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),  # never hardcode keys
        model_name="llama-3.1-8b-instant",             # Groq-hosted model
    )
 
 
# 4) RetrievalQA chain = Retriever + LLM
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
    return_source_documents=True
)
 
# 5) Ask questions
def ask(q: str):
    result = qa_chain.invoke({"query": q})
    print(f"\nQ: {q}")
    print("Answer:", result["result"])
    print("Retrieved context:")
    for d in result["source_documents"]:
        print("-", d.page_content)
 
# 6) Try it
ask("What is Raja learning?")
ask("What is FastAPI?")
ask("Where does Raja live?")  # Not in docs → should say "I don't know"