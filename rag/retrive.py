import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(
    path="chroma_db"
)

collection = client.get_collection(
    "mental_health"
)

def retrieve_similar_cases(text):

    embedding = model.encode(text).tolist()

    results = collection.query(
        query_embeddings=[embedding],
        n_results=3
    )

    return results


if __name__ == "__main__":

    query = input("Enter text: ")

    results = retrieve_similar_cases(query)

    print("\nRetrieved Cases:\n")

    for doc, meta in zip(
        results["documents"][0],
        results["metadatas"][0]
    ):
        print(f"Label: {meta['label']}")
        print(f"Text: {doc}")
        print("-" * 50)