import faiss
import numpy as np
import os
from sentence_transformers import SentenceTransformer


# Embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# Load knowledge
with open("rag/knowledge.txt", "r", encoding="utf-8") as file:
    knowledge = file.read()


# Split knowledge into disease sections
documents = [
    doc.strip()
    for doc in knowledge.split("\n\n")
    if doc.strip()
]


# Create embeddings
embeddings = embedding_model.encode(
    documents,
    convert_to_numpy=True
)

embeddings = embeddings.astype("float32")


# FAISS index
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)


def retrieve_information(query, top_k=3):

    # Convert query into embedding
    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True
    ).astype("float32")

    # Search
    distances, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for i in indices[0]:

        if i < len(documents):
            results.append(documents[i])

    return results


if __name__ == "__main__":

    query = input(
        "Enter fish symptoms: "
    )

    results = retrieve_information(query)

    print("\n========== RAG RESULTS ==========")

    for i, result in enumerate(results, 1):

        print(f"\nResult {i}")
        print(result)