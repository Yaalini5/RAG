"""
Extract text from a PDF (or plain .txt file) and split it into overlapping chunks.
"""
from pypdf import PdfReader


def extract_text(file_path: str) -> str:
    """Read a PDF or .txt file and return its raw text."""
    if file_path.lower().endswith(".pdf"):
        reader = PdfReader(file_path)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    """
    Split text into overlapping word-based chunks.
    chunk_size / overlap are in words, not tokens - simple and good enough
    for a first version. Be ready to explain this choice: word-count chunking
    is easy to reason about, though a production system would usually chunk
    on tokens or sentence boundaries instead.
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end >= len(words):
            break
        start = end - overlap  # step forward, keeping some overlap between chunks
    return chunks


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "sample.txt"
    text = extract_text(path)
    chunks = chunk_text(text)
    print(f"Extracted {len(text)} characters -> {len(chunks)} chunks")
    for i, c in enumerate(chunks[:3]):
        print(f"\n--- chunk {i} ({len(c.split())} words) ---\n{c[:200]}...")
