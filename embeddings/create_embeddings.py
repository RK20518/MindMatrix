import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

print("Loading dataset...")

df = pd.read_csv(
    "dataset/processed_mental_health.csv"
)

df = df.sample(
    n=5000,
    random_state=42
)

print(f"Total Records: {len(df)}")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

client = chromadb.PersistentClient(
    path="chroma_db"
)

try:
    client.delete_collection("mental_health")
except:
    pass

collection = client.get_or_create_collection(
    "mental_health"
)

texts = df["text"].tolist()

print("Generating embeddings...")

embeddings = model.encode(
    texts,
    show_progress_bar=True
)

ids = [str(i) for i in range(len(df))]

metadatas = [
    {"label": label}
    for label in df["label"]
]

collection.add(
    ids=ids,
    embeddings=embeddings.tolist(),
    documents=texts,
    metadatas=metadatas
)

print("Vector Database Created Successfully")
# import pandas as pd
# import chromadb
# from sentence_transformers import SentenceTransformer

# df = pd.read_csv("dataset/mental_health.csv")

# model = SentenceTransformer("all-MiniLM-L6-v2")

# client = chromadb.PersistentClient(
#     path="chroma_db"
# )

# collection = client.get_or_create_collection(
#     "mental_health"
# )

# for idx, row in df.iterrows():

#     embedding = model.encode(
#         row["text"]
#     ).tolist()

#     collection.add(
#         ids=[str(idx)],
#         embeddings=[embedding],
#         documents=[row["text"]],
#         metadatas=[
#             {"label": row["label"]}
#         ]
#     )

# print("Embeddings stored successfully")