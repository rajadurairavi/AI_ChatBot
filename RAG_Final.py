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
pdf_folder = r"C:\Knowledge_Base"   # path to your PDF folder
docs = []
for file in os.listdir(pdf_folder):
    if file.endswith(".pdf"):  # load only PDFs
        loader = PyPDFLoader(os.path.join(pdf_folder, file))
        docs.extend(loader.load())  # add all pages
        print(f"✅ Loaded {file}")
 
# -------------------------------------------------------
# 2) Split into chunks
# -------------------------------------------------------
splitter = CharacterTextSplitter(
    chunk_size=500,   # chunk size = 500 characters
    chunk_overlap=50  # overlap between chunks
)
chunks = splitter.split_documents(docs)
print(f"📄 Created {len(chunks)} chunks")
 
# -------------------------------------------------------
# 3) Embeddings + FAISS (with persistence)
# -------------------------------------------------------
faiss_path = r"C:\Knowledge_Base\faiss_index"   # where FAISS index will be saved
embeddings = HuggingFaceEmbeddings(model_name=r"C:\models\all-MiniLM-L6-v2")
 
if os.path.exists(faiss_path):
    print("✅ Loading existing FAISS index...")
    vectorstore = FAISS.load_local(
        faiss_path,
        embeddings,
        allow_dangerous_deserialization=True  # required by LangChain for FAISS
    )
else:
    print("⚡ No FAISS found, creating new one...")
    vectorstore = FAISS.from_documents(chunks, embedding=embeddings)
    vectorstore.save_local(faiss_path)
    print("✅ FAISS index saved!")
 
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
        search_kwargs={"k": 3}  # retrieve top 3 chunks
    ),
    chain_type_kwargs={"prompt": concise_prompt},  # custom prompt
    return_source_documents=True   # enable source tracking
)
 
# -------------------------------------------------------
# 7) Helper function with source tracking
# -------------------------------------------------------
def ask(q: str):
    """Ask a question to the RAG pipeline and print answer + sources"""
    result = qa_chain.invoke({"query": q})
    print(f"\nQ: {q}")
    print("Answer:", result["result"])
    print("Retrieved context:")
    for d in result["source_documents"]:
        page_num = d.metadata.get("page", "N/A") + 1 if "page" in d.metadata else "N/A"
        source = os.path.basename(d.metadata.get("source", "Unknown"))
        snippet = d.page_content[:200].replace("\n", " ")
        print(f"- {source} (Page {page_num}): {snippet}...")
 
# -------------------------------------------------------
# 8) Test queries (manual test)
# -------------------------------------------------------
ask("Who are some actors in Tamil cinema?")
ask("When did Tamil cinema start?")
ask("What is the history of Kollywood?")
ask("Where does Raja live?")  # not in docs → should say "I don't know"
 
# -------------------------------------------------------
# 9) Evaluation (Step 10)
# -------------------------------------------------------
# Small test set: question → expected answer
test_questions = {
    "Who is Raja?": "Raja is learning GenAI.",
    "What is FastAPI?": "FastAPI is a Python framework for building APIs.",
    "Where does Raja live?": "I don't know"  # not in docs
}
 
def evaluate():
    """Evaluate RAG answers against expected outputs"""
    print("\n📊 Running Evaluation...\n")
    for q, expected in test_questions.items():
        result = qa_chain.invoke({"query": q})
        answer = result["result"]
        print("Q:", q)
        print("Expected:", expected)
        print("Got:", answer)
        print("Match:", "✅" if expected.lower() in answer.lower() else "❌", "\n")
 
# Run evaluation
evaluate()