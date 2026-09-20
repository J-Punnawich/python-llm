import os
import json
import math
from typing import Optional
import re
from collections import Counter
from http_helper import call_openai

class CrossLingualRAG:
    def __init__(self, collection_name: str = "documents", embedding_model: str = "BAAI/bge-m3"):
        self.collection_name = collection_name
        self.doc_counter = 0

        self.vectors = []

    def add_documents(self, texts: list[str], metadata_list: Optional[list[dict]] = None):
        """Add documents to in-memory store with chunking"""
        chunks = []
        chunk_metadata = []
        chunk_ids = []

        for doc_idx, text in enumerate(texts):
            doc_chunks = self.split_text(text, chunk_size=1000, chunk_overlap=200)

            for chunk_idx, chunk in enumerate(doc_chunks):
                chunks.append(chunk)
                chunk_id = f"chunk_{self.doc_counter}_{len(chunks)-1}"
                chunk_ids.append(chunk_id)

                meta = {
                    "doc_id": str(doc_idx),
                    "chunk_id": str(chunk_idx),
                    "doc_name": metadata_list[doc_idx].get("name", f"doc_{doc_idx}") if metadata_list else f"doc_{doc_idx}",
                }
                if metadata_list and metadata_list[doc_idx]:
                    meta.update(metadata_list[doc_idx])
                chunk_metadata.append(meta)

        for chunk_id, chunk, meta in zip(chunk_ids, chunks, chunk_metadata):
            self.vectors.append({
                "chunk_id": chunk_id,
                "content": chunk,
                "embedding": self._embed(chunk),
                "metadata": meta,
            })

        self.doc_counter += len(texts)
        print(f"Added {len(chunks)} chunks from {len(texts)} documents")

    def query(self, query_text: str, n_results: int = 3) -> list[dict]:
        """Query in-memory vectors using cosine similarity search"""
        if not self.vectors:
            return []

        query_embedding = self._embed(query_text)

        similarities = []
        for vector in self.vectors:
            stored_embedding = vector['embedding']
            similarity = self._cosine_similarity(query_embedding, stored_embedding)
            similarities.append({
                "content": vector['content'],
                "metadata": vector['metadata'],
                "distance": 1 - similarity
            })

        similarities.sort(key=lambda x: x['distance'])
        return similarities[:n_results]

    def rag_query(self, query_text: str, system_prompt: str = "You are a helpful assistant") -> str:
        """Query vector store and generate response with OpenAI"""
        retrieved = self.query(query_text, n_results=3)

        context = "\n\n".join([f"[{r['metadata']['doc_name']}]\n{r['content']}" for r in retrieved])

        return call_openai(query_text, context, system_prompt)

    @staticmethod
    def _cosine_similarity(
        vec1: dict[str, float],
        vec2: dict[str, float],
    ) -> float:
        dot_product = sum(vec1.get(key, 0) * value for key, value in vec2.items())
        norm1 = math.sqrt(sum(value * value for value in vec1.values()))
        norm2 = math.sqrt(sum(value * value for value in vec2.values()))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    @staticmethod
    def _embed(text: str) -> dict[str, float]:
        tokens = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
        counts = Counter(tokens)
        total = sum(counts.values()) or 1
        return {token: count / total for token, count in counts.items()}

    def split_text(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list[str]:
        chunks = []
        start = 0
        text_length = len(text)
        step = chunk_size - chunk_overlap

        if step <= 0:
            raise ValueError("chunk_size must be greater than chunk_overlap")

        while start < text_length:
            end = min(start + chunk_size, text_length)
            chunk = text[start:end].strip()

            if chunk:
                chunks.append(chunk)

            if end == text_length:
                break

            start += step

        return chunks
