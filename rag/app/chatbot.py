import os
import pickle
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from groq import Groq

load_dotenv()

# ----------------------------
# Groq client
# ----------------------------
client = Groq(api_key="" + os.getenv("GROQ_API_KEY"))
MODEL_NAME = "llama-3.1-8b-instant"

# ----------------------------
# Load FAISS index + metadata
# ----------------------------
# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
vectorstore_dir = os.path.join(script_dir, '..', 'vectorstore')

index = faiss.read_index(
    os.path.join(vectorstore_dir, 'index.faiss')
)

with open(
    os.path.join(vectorstore_dir, 'meta.pkl'), "rb"
) as f:
    data = pickle.load(f)

texts, sources = zip(*data)

# ----------------------------
# Embedding model
# ----------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ----------------------------
# System prompt
# ----------------------------
SYSTEM_PROMPT = """
You are a research assistant specializing in ancient Indian astronomy and meteorology.

Answer ONLY using the provided manuscript excerpts.
Explicitly mention whether information comes from Surya Siddhanta or Brihat Samhita.
If the answer is not present in the text, clearly state that it is not found.
Do NOT introduce modern scientific explanations or interpretations.
"""

# ----------------------------
# Retrieval
# ----------------------------
def retrieve(query, k=5):
    q_emb = embedder.encode([query])
    _, idxs = index.search(q_emb, k)
    return [(texts[i], sources[i]) for i in idxs[0]]

# ----------------------------
# Ask function
# ----------------------------
def ask(query):
    passages = retrieve(query)

    context = "\n\n".join(
        f"[{src}]\n{text}" for text, src in passages
    )

    user_prompt = f"""
Context:
{context}

Question:
{query}

Answer:
"""

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
        temperature=0.2,
        max_tokens=500,
    )

    answer = completion.choices[0].message.content

    sources = list(set(src for _, src in passages))

    return {"answer": answer, "sources": sources}

# ----------------------------
# CLI loop
# ----------------------------
if __name__ == "__main__":
    while True:
        q = input("\nAsk a question (or type exit): ")
        if q.lower() == "exit":
            break
        ask(q)
