import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# Document loading
pdf = PyPDFLoader(r"C:\Knowledge_Base\TamilCinema.pdf")
docs = pdf.load()

# Text splitting
splitter = CharacterTextSplitter(separator="\n", chunk_size = 500, chunk_overlap = 50)
chunks = splitter.split_documents(docs)

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name = r"C:\models\all-MiniLM-L6-v2")

# vectorstore
vectorstore = FAISS.from_documents(chunks, embedding = embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k":2})

#llm
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant"
)

#prompt template
prompt = ChatPromptTemplate.from_template(""" You are a helpful assistant.
Answer the questions ONLY from the info given by context.
Do not have prior knowledge.
if you do not find answer in context, say "I don't know"
Question : {question}
context : {context}
Please answer consise and factual.
""")

rag_chain = {"context": retriever, "question": RunnablePassthrough()}| prompt | llm | StrOutputParser()

def ask(question: str):
    response = rag_chain.invoke(question)
    print(f"\nQ:{question}")
    print(response)

ask("what is tamil cinema ?")