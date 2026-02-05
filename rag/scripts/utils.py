def clean_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = " ".join(text.split())
    return text


def chunk_text(text, chunk_size=400, overlap=80):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks


def make_chunks(text, source):
    cleaned = clean_text(text)
    return [
        {
            "text": chunk,
            "source": source
        }
        for chunk in chunk_text(cleaned)
    ]
