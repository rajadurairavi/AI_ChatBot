import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

pdf = PyPDFLoader(r"C:\models\TamilCinema.pdf")
text = TextLoader(r"C:\models\Temp.txt")

docs = pdf.load() + text.load()

spilitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=100
)

embeddings = HuggingFaceEmbeddings(model_name=r"C:\models\all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k":3})

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant"
)

prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant.
Answer the question using ONLY the information in the provided context.
Do not use prior knowledge.
If the answer is not in the context, say "I don't know".
Context: {context}
Question: {question}
Answer (concise and accurate):
""")

chain = {"context":retriever, "question": RunnablePassthrough()}|prompt|llm|StrOutputParser()

def ask(question:str):
    response = chain.invoke(question)
    print("\nQ:", question)
    print("\nA:", response)

ask("Who is Raja?")
ask("List some Tamil movies.")
