import os
import json
import math
from typing import Optional
from sentence_transformers import SentenceTransformer  # type: ignore[reportMissingImports]
from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore[reportMissingImports]
from openai import OpenAI

class CrossLingualRAG:
    def __init__(self, collection_name: str = "documents", embedding_model: str = "BAAI/bge-m3"):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.collection_name = collection_name
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        self.doc_counter = 0

        self.vectors = []

    def add_documents(self, texts: list[str], metadata_list: Optional[list[dict]] = None):
        """Add documents to in-memory store with chunking"""
        chunks = []
        chunk_metadata = []
        chunk_ids = []

        for doc_idx, text in enumerate(texts):
            doc_chunks = self.splitter.split_text(text)

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

        embeddings = self.embedding_model.encode(chunks, convert_to_numpy=True)

        for i, (chunk_id, chunk, embedding, meta) in enumerate(zip(chunk_ids, chunks, embeddings, chunk_metadata)):
            self.vectors.append({
                "chunk_id": chunk_id,
                "content": chunk,
                "embedding": embedding.tolist() if hasattr(embedding, "tolist") else list(embedding),
                "metadata": meta
            })

        self.doc_counter += len(texts)
        print(f"Added {len(chunks)} chunks from {len(texts)} documents")

    def query(self, query_text: str, n_results: int = 3) -> list[dict]:
        """Query in-memory vectors using cosine similarity search"""
        if not self.vectors:
            return []

        query_embedding = self.embedding_model.encode(query_text, convert_to_numpy=True)
        if hasattr(query_embedding, "tolist"):
            query_embedding = query_embedding.tolist()

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

    @staticmethod
    def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        norm1 = math.sqrt(sum(value * value for value in vec1))
        norm2 = math.sqrt(sum(value * value for value in vec2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return sum(a * b for a, b in zip(vec1, vec2)) / (norm1 * norm2)

    def rag_query(self, query_text: str, system_prompt: str = "You are a helpful assistant") -> str:
        """Query vector store and generate response with OpenAI"""
        retrieved = self.query(query_text, n_results=3)

        context = "\n\n".join([f"[{r['metadata']['doc_name']}]\n{r['content']}" for r in retrieved])

        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query_text}"}
            ]
        )
        return response.choices[0].message.content

    def rag_query(self, query_text: str, system_prompt: str = "You are a helpful assistant") -> str:
        """Query vector store and generate response with OpenAI"""
        retrieved = self.query(query_text, n_results=3)

        context = "\n\n".join([f"[{r['metadata']['doc_name']}]\n{r['content']}" for r in retrieved])

        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query_text}"}
            ]
        )
        return response.choices[0].message.content
