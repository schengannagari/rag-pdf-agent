from typing import List

import numpy as np
from pypdf import PdfReader
from google import genai

EMBED_MODEL = "gemini-embedding-001"

class PDFKnowledgeBase:
    def __init__(self, client: genai.Client, pdf_path: str,
                 chunk_size: int = 200, chunk_overlap: int = 40):
        self.client = client
        self.chunks = self._load_and_chunk(pdf_path, chunk_size, chunk_overlap)
        self.embeddings = self._embed(self.chunks)

    def _load_and_chunk(self, pdf_path, chunk_size, chunk_overlap):
        reader = PdfReader(pdf_path)
        full_text = "\n".join(page.extract_text() or "" for page in reader.pages)

        chunks = []
        start = 0
        while start < len(full_text):
            end = start + chunk_size
            chunks.append(full_text[start:end])
            start = end - chunk_overlap  # overlap keeps context across chunk edges
        return [c.strip() for c in chunks if c.strip()]

    def _embed(self, texts: List[str]) -> np.ndarray:
        """Embeds any list of strings with Gemini's embedding model.
        Used both for the chunks at load time and for a single query at
        search time, so there's one code path calling the embedding API,
        not two."""
        result = self.client.models.embed_content(model=EMBED_MODEL, contents=texts)
        return np.array([e.values for e in result.embeddings])

    def embed_query(self, query: str) -> np.ndarray:
        """Embeds a single query string in the same vector space as the
        chunks (same model, same call path as _embed)."""
        return self._embed([query])[0]

    def similarities(self, query_vec: np.ndarray) -> np.ndarray:
        """Cosine similarity of query_vec against every chunk embedding,
        one score per chunk."""
        return (self.embeddings @ query_vec) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_vec) + 1e-8
        )

    def rank(self, sims: np.ndarray, top_k: int) -> np.ndarray:
        """Indices of the top_k highest-scoring chunks, best first."""
        return np.argsort(sims)[::-1][:top_k]

    def search(self, query: str, top_k: int = 3) -> str:
        """Full retrieval in one call -- composes the three steps above.
        This is what agent.py and normal callers use."""
        query_vec = self.embed_query(query)
        sims = self.similarities(query_vec)
        top_idx = self.rank(sims, top_k)
        results = [self.chunks[i] for i in top_idx]
        return "\n\n---\n\n".join(results)