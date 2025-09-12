import os
from langchain_community.document_loaders import PyPDFLoader   # to load PDFs
from langchain.text_splitter import CharacterTextSplitter      # to split into chunks
from langchain_huggingface import HuggingFaceEmbeddings  # embeddings model
from langchain_community.vectorstores import FAISS             # vector DB
from langchain_groq import ChatGroq                            # Groq LLM
 
# =========================
# 1) Load multiple PDFs
# =========================
pdf1 = PyPDFLoader(r"C:\Knowledge_Base\TamilCinema.pdf")
pdf2 = PyPDFLoader(r"C:\Knowledge_Base\Test_Specialist_Rajadurai.pdf")
 
doc1 = pdf1.load()
doc2 = pdf2.load()
docs = doc1 + doc2   # combine lists
 
# =========================
# 2) Split into chunks
# =========================
splitter = CharacterTextSplitter(
    chunk_size=500,     # max 500 characters
    chunk_overlap=50    # overlap between chunks
)
chunks = splitter.split_documents(docs)
 
# =========================
# 3) Embeddings + FAISS (Vector DB)
# =========================

embeddings = HuggingFaceEmbeddings(model_name=r"C:\models\all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embedding=embeddings)

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),   # API key from env variable
    model_name="llama-3.1-8b-instant"         # stable Groq LLM
)

def retrieve(question: str):
    retriever = vectorstore.as_retriever(search_kwargs={"k":3})
    context = retriever.invoke(question)
    return context

def augment(question, context):
    return f""" You are a helpful assistant.
    Answer the questions ONLY from the info given by context.
    Do not have prior knowledge.
    if you do not find answer in context, say "I don't know" 
    
    Input : {question}
    context : {context}
    Please answer consise and factual.
    """
def generate(prompt):
    response =  llm.invoke(prompt)
    return response



if __name__=="__main__":

    question = "what is tamil cinema ? "
    context = retrieve(question)
    prompts = augment(question, context)
    result = generate(prompts)

    print(result.content)


