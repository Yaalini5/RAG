"""
Command-line entry point: ingest a document, build the index, then answer
questions from the terminal. Run this first to make sure the core pipeline
works before touching the Flask layer.

Usage:
    python rag.py path/to/document.pdf
"""
import sys

from ingest import extract_text, chunk_text
from vector_store import VectorStore
from generate import generate_answer


def build_index(file_path: str) -> VectorStore:
    print(f"Reading {file_path}...")
    text = extract_text(file_path)
    chunks = chunk_text(text)
    print(f"Split into {len(chunks)} chunks. Embedding locally "
          f"(first run downloads the model, ~90MB)...")
    store = VectorStore()
    store.build(chunks)
    print("Index built.")
    return store


def ask(store: VectorStore, question: str, top_k: int = 3) -> str:
    retrieved = store.search(question, top_k=top_k)
    return generate_answer(question, retrieved)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python rag.py <path_to_pdf_or_txt>")
        sys.exit(1)

    store = build_index(sys.argv[1])

    print("\nIndex ready. Ask questions about the document (type 'exit' to quit).\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue
        answer = ask(store, question)
        print(f"\nAnswer: {answer}\n")
