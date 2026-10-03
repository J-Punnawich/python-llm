import os
import hashlib
from datetime import datetime, timezone
from typing import Optional
import truststore
# Use the Windows OS certificate store (trusts the corporate proxy's root CA)
# for all SSL connections instead of the certifi bundle.
truststore.inject_into_ssl()
from astrapy import DataAPIClient
from astrapy.info import CollectionDefinition
from http_helper import call_openai, get_embedding, EMBEDDING_MODEL, EMBEDDING_DIM

class CrossLingualRAG:
    def __init__(self, collection_name: str = "documents", embedding_model: str = EMBEDDING_MODEL):
        self.embedding_model = embedding_model

        client = DataAPIClient(os.environ["ASTRA_DB_APPLICATION_TOKEN"])
        database = client.get_database(os.environ["ASTRA_DB_API_ENDPOINT"])

        chunks_definition = (
            CollectionDefinition.builder()
            .with_vector_dimension(EMBEDDING_DIM)
            .with_vector_metric("cosine")
            .build()
        )
        self.chunks = database.create_collection(collection_name, definition=chunks_definition)
        # Tracks which documents were ingested, independent of per-chunk vector rows
        self.registry = database.create_collection(f"{collection_name}_registry")

    def add_documents(self, texts: list[str], metadata_list: Optional[list[dict]] = None):
        chunk_records = []
        total_new_chunks = 0

        for doc_idx, text in enumerate(texts):
            doc_name = metadata_list[doc_idx].get("name", f"doc_{doc_idx}") if metadata_list else f"doc_{doc_idx}"
            source_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            doc_id = source_hash[:16]

            if self.registry.find_one({"_id": doc_id}):
                print(f"Skipping '{doc_name}' (already ingested, hash unchanged)")
                continue

            doc_chunks = self.split_text(text, chunk_size=1000, chunk_overlap=200)

            meta = dict(metadata_list[doc_idx]) if metadata_list and metadata_list[doc_idx] else {}
            meta["doc_name"] = doc_name

            for chunk_idx, chunk in enumerate(doc_chunks):
                chunk_records.append({
                    "_id": f"{doc_id}_{chunk_idx}",
                    "content": chunk,
                    "$vector": get_embedding(chunk),
                    "doc_id": doc_id,
                    "chunk_index": chunk_idx,
                    "metadata": meta,
                })

            self.registry.insert_one({
                "_id": doc_id,
                "doc_name": doc_name,
                "source_hash": source_hash,
                "chunk_count": len(doc_chunks),
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "embedding_model": self.embedding_model,
                "embedding_dim": EMBEDDING_DIM,
                "ingested_at": datetime.now(timezone.utc).isoformat(),
                "metadata": meta,
            })
            total_new_chunks += len(doc_chunks)

        if chunk_records:
            self.chunks.insert_many(chunk_records)

        print(f"Added {total_new_chunks} chunks from {len(texts)} documents")

    def list_documents(self) -> list[dict]:
        """Return the ingestion registry, to track back what was added and how"""
        return list(self.registry.find({}))

    def delete_document(self, doc_id: str):
        self.chunks.delete_many({"doc_id": doc_id})
        self.registry.delete_one({"_id": doc_id})

    def query(self, query_text: str, n_results: int = 3) -> list[dict]:
        """Query Astra DB vector collection using ANN similarity search"""
        query_embedding = get_embedding(query_text)

        results = self.chunks.find(
            {},
            sort={"$vector": query_embedding},
            limit=n_results,
            include_similarity=True,
        )

        return [
            {
                "content": r["content"],
                "metadata": r["metadata"],
                "distance": 1 - r.get("$similarity", 0),
            }
            for r in results
        ]

    def rag_query(self, query_text: str, system_prompt: str = "You are a helpful assistant") -> str:
        """Query vector store and generate response with OpenAI"""
        retrieved = self.query(query_text, n_results=3)

        context = "\n\n".join([f"[{r['metadata']['doc_name']}]\n{r['content']}" for r in retrieved])

        return call_openai(query_text, context, system_prompt)

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
