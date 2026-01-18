from fastapi import FastAPI
from pydantic import BaseModel
from chatbot import qa

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask_question(query: Query):
    result = qa(query.question)
    return {
        "answer": result["result"],
        "sources": list(set(
            doc.metadata["source"]
            for doc in result["source_documents"]
        ))
    }
