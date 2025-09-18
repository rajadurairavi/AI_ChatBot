# RAG_PDF.py
# Retrieval Augmented Generation (RAG) with PDF
# ------------------------------------------------
# pip install langchain langchain-community sentence-transformers faiss-cpu langchain-groq pypdf
 
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
 
# 1) Load PDF file
pdf = PyPDFLoader("C:\Knowledge_Base\TamilCinema.pdf")
docs = pdf.load()
 
# 2) Split into chunks (for better retrieval)
splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=500,     # each chunk max ~500 characters
    chunk_overlap=50    # overlap of 50 characters between chunks
)
chunks = splitter.split_documents(docs)

 
# 3) Convert to embeddings + store in FAISS (vector DB)
embeddings = HuggingFaceEmbeddings(
    model_name=r"C:\models\all-MiniLM-L6-v2"   # local embedding model
)
vectorstore = FAISS.from_documents(chunks, embedding=embeddings)

 
# 4) Connect Groq LLM
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant"
)
 
# 5) Build RetrievalQA chain (Retriever + LLM)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
    return_source_documents=True
)
 
# 6) Helper function to ask questions
def ask(q: str):
    result = qa_chain.invoke({"query": q})   # MUST use "query", not "input"
    print(f"\nQ: {q}")
    print("Answer:", result["result"])
    print("Retrieved context:")
    for d in result["source_documents"]:
        print("-", d.page_content[:200])  # show only first 200 chars
 
# 7) Try it
ask("What is Tamil cinema?")
ask("who is Rajinikanth ?")
ask("who is Kamal Hassan ?")
ask("who is Vijay ?")

 