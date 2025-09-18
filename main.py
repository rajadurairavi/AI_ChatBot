import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

pdf = PyPDFLoader(r"C:\Knowledge_Base\TamilCinema.pdf")
docs = pdf.load()

spilitter = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=50)
chunks = spilitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name=r"C:\models\all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embedding = embeddings)

retriever = vectorstore.as_retriever(search_kwargs = {"k": 2})

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name = "llama-3.1-8b-instant")

prompt = ChatPromptTemplate.from_template("""
You are a helpful assistane. Answer the question using ONLY the information provided in the context.
Do not use any prior knowledge. 
If you don't find the answer in the context, simply state that "I don't know".

"Context":{context}
"Input":{question}

answer(be consise and factual)
                                          
""")

rag_chain = {"context":retriever, "question": RunnablePassthrough()}|prompt|llm|StrOutputParser()

def ask(question: str):
    response = rag_chain.invoke(question)
    print(f"\n Q: {question}")
    print(response)

ask("what is tamil cinema ?")
