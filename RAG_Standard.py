# Final RAG pipeline (Steps 1–10 complete)
# ------------------------------------------------------- 
import os
from langchain_community.document_loaders import PyPDFLoader   # to load PDFs
from langchain.text_splitter import CharacterTextSplitter      # to split into chunks
from langchain_community.embeddings import HuggingFaceEmbeddings # embeddings model
from langchain_community.vectorstores import FAISS             # vector DB
from langchain_groq import ChatGroq                            # Groq LLM
from langchain.chains import RetrievalQA                       # RAG chain
from langchain.prompts import PromptTemplate                   # custom prompt
 
# -------------------------------------------------------
# 1) Load multiple PDFs from folder
# -------------------------------------------------------
pdf1 = PyPDFLoader(r"C:\Knowledge_Base\TamilCinema.pdf")
pdf2 = PyPDFLoader(r"C:\Knowledge_Base\Test_Specialist_Rajadurai.pdf")
doc1 = pdf1.load()
doc2 = pdf2.load()
docs = doc1 + doc2  # combine lists
 
# -------------------------------------------------------
# 2) Split into chunks
# -------------------------------------------------------
splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=500,   # chunk size = 500 characters
    chunk_overlap=50  # overlap between chunks
)
chunks = splitter.split_documents(docs)
 
# -------------------------------------------------------
# 3) Embeddings + FAISS (with persistence)
# -------------------------------------------------------
embeddings = HuggingFaceEmbeddings(model_name=r"C:\models\all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embedding=embeddings)
 
# -------------------------------------------------------
# 4) Connect LLM (Groq)
# -------------------------------------------------------
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),   # API key from env variable
    model_name="llama-3.1-8b-instant"         # stable Groq LLM
)
 
# -------------------------------------------------------
# 5) Concise Prompt (with "I don't know" handling)
# -------------------------------------------------------
concise_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful assistant.
Answer the question using ONLY the information in the provided context.
Do not use prior knowledge.
If the answer is not in the context, say "I don't know".
 
Context:
{context}
 
Question:
{question}
 
Answer (concise and accurate):
"""
)
 
# -------------------------------------------------------
# 6) RetrievalQA chain
# -------------------------------------------------------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,   # Groq LLM
    retriever=vectorstore.as_retriever(
        search_type="mmr",  # Maximal Marginal Relevance (Step 5)
        search_kwargs={"k": 2}  # retrieve top 3 chunks
    ),
    chain_type_kwargs={"prompt": concise_prompt},  # custom prompt
    return_source_documents=True   # enable source tracking
)
 
# -------------------------------------------------------
# 7) Helper function with source tracking
# -------------------------------------------------------
def ask(q: str):
    result = qa_chain.invoke({"query": q})
    print(f"\nQ: {q}")
    print("Answer:", result["result"])
    
 
# -------------------------------------------------------
# 8) Test queries (manual test)
# -------------------------------------------------------
ask("Who are some actors in Tamil cinema?")
ask("When did Tamil cinema start?")
ask("What is the history of Kollywood?")
ask("Where does Raja live?")  # not in docs → should say "I don't know"
 
# -------------------------------------------------------