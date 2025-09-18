import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# 1) Load multiple PDFs
# =========================
pdf1 = PyPDFLoader(r"C:\Knowledge_Base\TamilCinema.pdf")
pdf2 = PyPDFLoader(r"C:\Knowledge_Base\Test_Specialist_Rajadurai.pdf")
 
doc1 = pdf1.load()
doc2 = pdf2.load()
docs = doc1 + doc2   # combine lists

# 2) Split into chunks
# =========================
splitter = CharacterTextSplitter(
    chunk_size=500,     # max 500 characters
    chunk_overlap=50    # overlap between chunks
)
chunks = splitter.split_documents(docs)

# 3) Embeddings + FAISS (Vector DB)
# =========================

embeddings = HuggingFaceEmbeddings(model_name=r"C:\models\all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embedding=embeddings)

# 4) Connect LLM (Groq)
# =========================
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant"   # stable Groq LLM
)

# 5) Concise Prompt (with "I don’t know" handling)
# =========================
concise_prompt = PromptTemplate(
    input_variables=["context", "input"],
    template="""You are a helpful assistant.
Answer the question using ONLY the information in the provided context.
Do not use prior knowledge. If the answer is not in the context, say "I don't know".
 
Context:
{context}
 
Question:
{input}
 
Answer (concise and accurate):"""
)

# 6) RetrievalQA chain
# =========================
llm_prompt_chain = create_stuff_documents_chain(llm, concise_prompt)
 
qa_chain = create_retrieval_chain(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
    combine_docs_chain=llm_prompt_chain
)

# 7) Helper function with source tracking
# =========================
def ask(question: str):
    result = qa_chain.invoke({"input": question})  # ✅ FIXED: use "question"
    print(f"\nQ: {question}")
    print("Answer:", result["answer"])
    print("Retrieved context:")
    for doc in result["context"]:
        print("-", doc.page_content[:200])  # only print first 200 chars
 
# =========================
# 8) Test queries
# =========================
ask("Who are some actors in Tamil cinema?")
ask("When did Tamil cinema start?")
ask("What is the history of Kollywood?")
ask("Where does Rajai live?")   # not in docs → should say "I don't know"
 
 

