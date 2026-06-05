import os
import shutil
from datetime import datetime
from typing import TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

load_dotenv()


MODEL = "text-embedding-3-small"
DB_PATH = "./research_db"


class DocumentMetadata(TypedDict, total=False):
    source: str
    indexed_at: str


class ResearchResponse(BaseModel):
    """Structured response from the research assistant"""

    answer: str = Field(description="the answer to the question")
    confidence: str = Field(description="high, medium, or low based on source")
    sources: list[str] = Field(description="list of source documents used")
    key_quotes: list[str] = Field(
        description="relevant quotes from sources", default=[]
    )
    follow_up_questions: list[str] = Field(description="suggested follow up questions")


class AIResearchAssistant:
    """AI Research assistant with document ingestion and retrieval"""

    def __init__(
        self,
        persist_directory: str = DB_PATH,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> None:
        self.persist_directory = persist_directory

        # 1. embedding turn text into vectors
        self.embeddings = OpenAIEmbeddings(model=MODEL)

        self.llm = init_chat_model(model="gpt-4o-mini", temperature=0)

        # 2. Splitter - break big docs into chunks
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        # 3. Vector store - stores and searches embeddings
        # OpenAI embeddings are unit-normalized, so cosine distance is the correct metric.
        # Chroma's default L2 distance produces relevance scores well below 0.5 for these
        # embeddings, causing all documents to be filtered out by the threshold.
        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings,
            collection_name="research_docs",
            collection_metadata={"hnsw:space": "cosine"},
        )

        print(f"Research Assistant initialized")
        print(f"    Vector store: {persist_directory}")
        print(f"    Documents indexed: {self.vectorstore._collection.count()}")

    def add_documents(
        self, documents: list[Document], source_name: str | None = None
    ) -> int:
        """Add documents to the research database"""

        # Tag with source name
        if source_name:
            for doc in documents:
                doc.metadata["source"] = source_name

        # Split into chunks
        chunks = self.splitter.split_documents(documents=documents)

        # Timestamp each chunk
        for chunk in chunks:
            chunk.metadata["indexed_at"] = datetime.now().isoformat()

        # Store in vector DB
        self.vectorstore.add_documents(chunks)

        print(f"Added {len(chunks)} chunks from {len(documents)} documents")
        return len(chunks)

    def add_text(
        self, text: str, source: str, metadata: DocumentMetadata | None = None
    ) -> int:
        """Add a single text string as a document"""

        doc = Document(
            page_content=text, metadata={"source": source, **(metadata or {})}
        )
        return self.add_documents([doc])

    def add_texts(self, texts: list[str], source: str) -> int:
        """Add multiple text strings from the same source"""

        docs = [Document(page_content=t, metadata={source: source}) for t in texts]

        return self.add_documents(docs)

    def get_document_count(self) -> int:
        """Get total number of indexed chunks"""
        return self.vectorstore._collection.count()

    def list_sources(self) -> list[str]:
        """List all unique sources in the database"""
        results = self.vectorstore._collection.get()
        sources = set()
        for metadata in results.get("metadatas") or []:
            if metadata and "source" in metadata:
                sources.add(metadata["source"])
        return sorted(list(sources))

    def _build_retriever(self, score_threshold: float = 0.5):
        return self.vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 4, "score_threshold": score_threshold},
        )

    def _format_docs_for_context(self, docs: list[Document]) -> str:
        """Format retrieved documents into a string for the prompt"""

        if not docs:
            return "No relevant documents found"

        formatted = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "unknown")
            formatted.append(f"[Source {i+1}: {source}]\n{doc.page_content}")
        return "\n\n--\n\n".join(formatted)

    def ask(self, question: str) -> str:
        """Ask a question against research documents"""

        # Step 1: Retrieve relevant chunks
        retriever = self._build_retriever()
        docs = retriever.invoke(question)

        # Step 2: Format into context string
        context = self._format_docs_for_context(docs)

        # Step 3: Build the prompt
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are an AI research assistant. Answer questions based ONLY on the the 
                    provided context documents. 

                    Rules:
                    1. Only use information from the context below
                    2. If the context doesn't have the answer, say so
                    3. cite which source you used (e.g. "According to Source 1...")
                    4. Rate your confidence: high, medium or low
                    """,
                ),
                (
                    "human",
                    """
                    Context documents:
                    {context}
                    Question: {question}
                    Provide a clear answer with source citations
                    """,
                ),
            ]
        )

        # Step 4: Build and run the chain
        chain = prompt | self.llm | StrOutputParser()

        response = chain.invoke({"context": context, "question": question})

        return response


if __name__ == "__main__":

    shutil.rmtree(DB_PATH, ignore_errors=True)

    assistant = AIResearchAssistant()

    # doc = Document(
    #     page_content="The capital of France is Paris. It is known for th Eiffel tower",
    #     metadata={"source": "general_knowledge.txt"},
    # )
    # assistant.add_documents([doc], source_name="general_knowledge.txt")

    assistant.add_text(
        """
        Attention mechanisms in neural networks

        The attention mechanism was introduced in "Attention is all you need" by Vaswani et al.
        (2017). It allows models to focus on relevant parts of the input when generating output.

        Key concepts:
        - Query, Key, value (QKV) triplets
        - Scaled dot-product attention
        - Multi-head attention for parallel processing

        The transformer architecture has become the foundation for modern NLP models 
        including BERT, GPT and T5.
        """,
        source="attention_mechanisms.pdf",
    )

    assistant.add_text(
        """
        Retrieval Augmented Generation (RAG)

        RAG combines retrieval systems with generative models. First introduced by Lewis et al.
        Lewis et al. (2020), RAG addresses the limitation of LLM being limited to 
        to their training data.

        Components of a RAG system:
        1. Document store with vector embeddings
        2. Retriever to find relevant documents
        3. Generator (LLM) to produce responses

        Benefits include reduced hallucination, up-to-date information, and source
        attribution,
        """,
        source="rag_survey.pdf",
    )

    assistant.add_text(
        """
        Langchain and Langraph framework overview

        Langchain is an open-source framework for building LLM applications.
        Key features include modular components, integration with 50+ LLM providers,
        and built-in RAG utilities.

        Langraph extends Langchain for stateful applications with graph-based state
        management, support for cycles and loops, and human-in-the-loop workflows.    
        """,
        source="langchain_docs.md",
    )

    # prove it worked

    print(f"\n Files on disk: {os.listdir(DB_PATH)}")

    print(f"\nTotal chunks indexed: {assistant.get_document_count()}")
    print(f"Sources: {assistant.list_sources()}")

    # Question 1: Direct answer
    print("\n" + "=" * 60)
    print("QUESTION 1: Direct factual question")
    print("\n" + "=" * 60)

    q1 = "What is RAG and what are it's main components"
    print(f"\nUser: {q1}")
    print(f"\nAssistant: {assistant.ask(question=q1)}")

    # Question 2: Cross-document
    print("\n" + "=" * 60)
    print("QUESTION 2: Requires info from multiple source")
    print("\n" + "=" * 60)

    q2 = "How does the attention mechanism relate to Langchain?"
    print(f"\nUser: {q2}")
    print(f"\nAssistant: {assistant.ask(question=q2)}")

    # Question 3: THE FAILURE - follow-up question
    print("\n" + "=" * 60)
    print("QUESTION 3: Follow-up (this will fail!)")
    print("\n" + "=" * 60)

    q3 = "Can you expand on the second component you just mentioned"
    print(f"\nUser: {q3}")
    print(f"\nAssistant: {assistant.ask(q3)}")

    print("\n" + "=" * 60)
    print("PROBLEM: It has not idea what 'you just mentioned' means!")
    print("Each question is independent -- there is not memory")
    print("\n" + "=" * 60)

    # Cleanup
    shutil.rmtree(DB_PATH, ignore_errors=True)
