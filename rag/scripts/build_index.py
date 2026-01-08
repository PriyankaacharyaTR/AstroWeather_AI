import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer

# Load texts
texts = []
sources = []

for fname, source in [
    ("data/raw_text/surya_siddhanta.txt", "Surya Siddhanta"),
    ("data/raw_text/brihat_samhita.txt", "Brihat Samhita"),
]:
    with open(fname, encoding="utf-8") as f:
        text = f.read().replace("\n", " ")
        words = text.split()

        # chunking
        for i in range(0, len(words), 350):
            chunk = " ".join(words[i:i+350])
            texts.append(chunk)
            sources.append(source)

# Embed
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(texts, show_progress_bar=True)

# FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

os.makedirs("vectorstore", exist_ok=True)
faiss.write_index(index, "vectorstore/index.faiss")

with open("vectorstore/meta.pkl", "wb") as f:
    pickle.dump(list(zip(texts, sources)), f)

print("✅ Vector index built successfully")
