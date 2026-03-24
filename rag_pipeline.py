import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

PDF_PATH = "constitution_of_india.pdf"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_and_index_pdf():
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    prompt = ChatPromptTemplate.from_template("""
You are a legal assistant specializing in the Indian Constitution.
Use the context below (articles/sections from the Constitution) to answer the user's question.

Structure your response in exactly this format:

🔍 **Understanding Your Problem:**
[Briefly explain the legal issue the user is facing in simple terms]

⚖️ **Relevant Articles & Sections:**
[List all relevant Article numbers, Part names, and their descriptions from the Constitution]

✅ **Best Solution:**
[Give a clear, practical, step-by-step solution based on the Constitutional provisions. Explain what rights the person has and what actions they can take.]

📌 **Important Note:**
[Any important caveats, limitations, or advice to consult a lawyer if needed]

If the answer is not in the context, say "I could not find a relevant provision in the Indian Constitution for this query."

Context:
{context}

User Question: {question}

Answer:""")

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.3,
        max_tokens=1024,
        api_key=os.getenv("GROQ_API_KEY"),
    )

    rag_chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["question"]))
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever


def get_answer(chain_and_retriever, query: str) -> dict:
    chain, retriever = chain_and_retriever

    answer = chain.invoke({"question": query})

    source_docs = retriever.invoke(query)
    source_info = []
    for doc in source_docs:
        page = doc.metadata.get("page", "?")
        snippet = doc.page_content[:200].strip()
        source_info.append({"page": page + 1, "snippet": snippet})

    return {"answer": answer, "sources": source_info}