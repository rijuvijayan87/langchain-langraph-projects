import os
from encodings import search_function
from importlib import metadata

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import max_retries

load_dotenv()

# create a embedder using embedding model
model = OpenAIEmbeddings(model="text-embedding-3-small")
docs = [
    {
        "question": "What services does FinCorp Solutions offer?",
        "answer": "FinCorp Solutions offers a range of financial services including personal banking, business loans, investment advisory, retirement planning, and digital payment solutions tailored for individuals and enterprises.",
    },
    {
        "question": "How do I open a bank account with FinCorp?",
        "answer": "You can open an account online through our website or visit any of our 200+ branches. You will need a government-issued ID, proof of address, and an initial deposit of $100.",
    },
    {
        "question": "What are FinCorp's customer support hours?",
        "answer": "Our customer support team is available Monday through Friday, 8 AM to 8 PM EST, and Saturday from 9 AM to 5 PM EST. You can reach us via phone, email, or live chat on our website.",
    },
    {
        "question": "How does FinCorp protect my personal and financial data?",
        "answer": "We use 256-bit AES encryption, two-factor authentication, and comply with SOC 2 Type II and PCI-DSS standards to ensure your data is always secure.",
    },
    {
        "question": "What is the interest rate on FinCorp personal loans?",
        "answer": "Personal loan interest rates range from 5.9% to 18.5% APR depending on your credit score, loan amount, and repayment term. You can get a pre-approval quote with no impact to your credit score.",
    },
    {
        "question": "Can I invest through FinCorp?",
        "answer": "Yes. FinCorp offers self-directed brokerage accounts, managed portfolios, and access to ETFs, mutual funds, and fixed-income instruments. A minimum investment of $500 is required to get started.",
    },
    {
        "question": "How do I reset my online banking password?",
        "answer": "Click the 'Forgot Password' link on the login page, enter your registered email address, and follow the instructions sent to your inbox. For security reasons, the reset link expires in 15 minutes.",
    },
    {
        "question": "what is your Leadership team",
        "answer": "Fincorp is headed by Sundar Pichai, with Raghuram rajan as the CFO",
    },
]


def chroma_db():

    documents = [
        Document(
            page_content=f"Q: {doc["question"]} \n A: {doc["answer"]}",
            metadata={"question": doc["question"]},
        )
        for doc in docs
    ]

    # create data chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    doc_chunks = text_splitter.split_documents(documents)

    CHROMA_PATH = "./chroma_db"

    # store it in the vector db (e.g. chromadb)
    if not os.path.exists(CHROMA_PATH):
        print("chromadb not found. creating...")
        vector_db = Chroma.from_documents(
            documents=doc_chunks, embedding=model, persist_directory=CHROMA_PATH
        )
        print(
            f"chromadb created in {CHROMA_PATH} with {vector_db._collection.count()} documents."
        )
    else:
        vector_db = Chroma(embedding_function=model, persist_directory=CHROMA_PATH)
        print(
            f"documents are read from {CHROMA_PATH} with {vector_db._collection.count()} documents."
        )

    # print answers
    results = vector_db.similarity_search_with_score("how secure is your bank?", k=1)
    for doc, score in results:
        print(f"Score: {score:.4f}\n{doc.page_content}\n")

    print("=== now as retriever ===")
    results_as_retriever = vector_db.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    )

    # retriever_result = results_as_retriever.invoke("how secure is your bank?")
    # for doc in retriever_result:
    #     print(doc.page_content)

    prompt = ChatPromptTemplate.from_template("""
        You are an useful assistant. Answer using the provided context. if you do not know the answer
        just say so. 

        context: {context}
        question: {question}
    """)

    llm = init_chat_model(model="gpt-4o-mini", max_retries=2)

    chain = (
        {"context": results_as_retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    question = input("ask your question: ")
    result = chain.invoke(question)

    print(result)


if __name__ == "__main__":
    chroma_db()
