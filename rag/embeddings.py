from pydoc import doc

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


def similiarity_search():
    docs = [
        "Python is a programming language.",
        "Java is a programming language.",
        "Python is great for data science.",
        "Deep learning is a subset of machine learning.",
        "C++ is a programming language.",
        "Cats are cute animals.",
    ]

    query = "What programming languages exists?"

    # embedding documents and query
    doc_embeddings = embeddings.embed_documents(docs)
    query_embedding = embeddings.embed_query(query)

    # calculate cosine similarity between query and documents
    def cosine_similarity(vec1, vec2):
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude_vec1 = sum(a**2 for a in vec1) ** 0.5
        magnitude_vec2 = sum(b**2 for b in vec2) ** 0.5
        if magnitude_vec1 == 0 or magnitude_vec2 == 0:
            return 0.0
        return dot_product / (magnitude_vec1 * magnitude_vec2)

    similarities = [
        cosine_similarity(query_embedding, doc_embedding)
        for doc_embedding in doc_embeddings
    ]

    ranked_docs = sorted(zip(docs, similarities), key=lambda x: x[1], reverse=True)

    print(f"Query: {query}\n")
    print("Ranked by similarity:")

    for doc, sim in ranked_docs:
        print(f"Document: {doc} | Similarity: {sim:.4f}")


if __name__ == "__main__":
    similiarity_search()
