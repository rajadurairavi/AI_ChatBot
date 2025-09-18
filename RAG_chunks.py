# Day 8: RAG + Chunking
# ------------------------------------------------------
# Why? Large text docs can’t be embedded as one big block.
# Instead, we split them into smaller chunks (like paragraphs).
# This helps the retriever find precise answers.
 
import os
from langchain_community.embeddings import HuggingFaceEmbeddings   # for embeddings
from langchain_community.vectorstores import FAISS                 # vector database
from langchain.text_splitter import CharacterTextSplitter          # chunk splitter
from langchain.chains import RetrievalQA                           # RAG chain
from langchain_groq import ChatGroq                                # LLM (Groq)
 
# 1) Knowledge base (big text instead of tiny sentences)
long_text = """
Raja is learning GenAI. He also works with FastAPI to build APIs.
Clouds are made of tiny water droplets called water vapor.
RAG retrieves relevant text from a vector database before the LLM answers.
Python is widely used for test automation and AI.
"""
 
# 2) Split the text into chunks
# - separator="\n" → split at newlines
# - chunk_size=50 → each chunk max 50 characters
# - chunk_overlap=10 → repeat 10 chars between chunks (keeps context)
splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=50,
    chunk_overlap=10,
)
chunks = splitter.split_text(long_text)
print("Chunks:", chunks)   # just to see how text got split
 
# 3) Convert chunks into embeddings + store in FAISS
embeddings = HuggingFaceEmbeddings(
    model_name=r"C:\models\all-MiniLM-L6-v2"   # local path to model
)
vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
 
# 4) Connect LLM (Groq’s llama3-8b-8192)
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant"
)
 
# 5) Create RAG chain (Retriever + LLM)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),  # top 2 chunks
    return_source_documents=True
)
 
# 6) Function to ask questions
def ask(q: str):
    result = qa_chain.invoke({"query": q})   # pass query to chain
    print(f"\nQ: {q}")
    print("Answer:", result["result"])       # LLM’s answer
    print("Retrieved context:")
    for d in result["source_documents"]:     # chunks retrieved
        print("-", d.page_content)
 
# 7) Try it out
ask("What is Raja learning?")
ask("What is FastAPI?")
ask("What are clouds made of?")
ask("Where does Raja live?")  # not in docs → should say "I don’t know"