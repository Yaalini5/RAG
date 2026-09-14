"""
Flask API wrapping the RAG pipeline: upload a document, then ask questions
about it. Keeps the index in memory for the process lifetime - fine for a
demo, not for production (that's a deliberate scope cut, be ready to say so).
"""
import os
from flask import Flask, request, jsonify, render_template

from ingest import extract_text, chunk_text
from vector_store import VectorStore
from generate import generate_answer

app = Flask(__name__)
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# In-memory store - one document at a time, kept simple on purpose.
state = {"store": None, "filename": None}


@app.get("/")
def index():
    return render_template("index.html", filename=state["filename"])


@app.get("/health")
def health():
    return jsonify({"status": "ok", "filename": state["filename"]})


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    text = extract_text(file_path)
    chunks = chunk_text(text)
    if not chunks:
        return jsonify({"error": "Couldn't extract any text from that file"}), 400

    store = VectorStore()
    store.build(chunks)
    state["store"] = store
    state["filename"] = file.filename

    return jsonify({"message": f"Indexed {file.filename} into {len(chunks)} chunks."})


@app.route("/ask", methods=["POST"])
def ask():
    if state["store"] is None:
        return jsonify({"error": "Upload a document first"}), 400

    question = (request.json or {}).get("question", "").strip()
    if not question:
        return jsonify({"error": "Question is empty"}), 400

    retrieved = state["store"].search(question, top_k=3)
    answer = generate_answer(question, retrieved)

    return jsonify({
        "answer": answer,
        "sources": [
            {"text": chunk[:200], "score": round(score, 3)}
            for chunk, score in retrieved
        ],
    })


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
