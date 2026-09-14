"""
Embed text chunks and store/query them in a local FAISS index.
Embeddings run fully locally via sentence-transformers - no API key, no cost,
no rate limits. First run downloads the model (~90MB) and caches it.
"""
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, good enough for a demo


class VectorStore:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []  # keeps chunk text aligned with FAISS row order

    def build(self, chunks: list):
        """Embed all chunks and build a fresh FAISS index."""
        self.chunks = chunks
        embeddings = self.model.encode(
            chunks, convert_to_numpy=True, normalize_embeddings=True
        )
        dim = embeddings.shape[1]
        # Inner product on normalized vectors == cosine similarity.
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings.astype(np.float32))

    def search(self, query: str, top_k: int = 3):
        """Return the top_k most similar chunks to the query, with similarity scores."""
        if self.index is None:
            raise RuntimeError("Vector store is empty - call build() first.")
        query_vec = self.model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )
        scores, indices = self.index.search(query_vec.astype(np.float32), top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))
        return results
