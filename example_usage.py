import os
from dotenv import load_dotenv
from rag_pipeline import CrossLingualRAG

load_dotenv()

rag = CrossLingualRAG(collection_name="multilingual_docs")

thai_docs = [
    """
    ปัญญาเทียม (AI) คือการจำลองความฉลาดของมนุษย์ด้วยเครื่องจักร
    AI ช่วยให้คอมพิวเตอร์สามารถเรียนรู้จากประสบการณ์และปรับปรุงตนเองได้
    งานประยุกต์ AI ได้แก่ การทำความเข้าใจภาษาธรรมชาติ การมองเห็น และการตัดสินใจ
    """
]

english_docs = [
    """
    Retrieval-Augmented Generation (RAG) combines retrieval and generation models.
    It retrieves relevant documents first, then generates responses based on the context.
    RAG improves accuracy and reduces hallucination in LLM responses.
    """,
    """
    Vector databases store high-dimensional embeddings efficiently.
    They enable semantic search by finding similar items in vector space.
    Common vector databases include Pinecone, Weaviate, and Chroma.
    """
]

metadata = [
    {"name": "Thai_AI_Overview", "language": "thai", "topic": "AI"},
    {"name": "RAG_Explanation", "language": "english", "topic": "RAG"},
    {"name": "Vector_DB_Guide", "language": "english", "topic": "Database"}
]

rag.add_documents(thai_docs + english_docs, metadata)

print("\n--- Query in Thai ---")
thai_query = "ปัญญาเทียมคืออะไร?"
print(f"Query: {thai_query}")
retrieved = rag.query(thai_query, n_results=2)
for r in retrieved:
    print(f"\nDoc: {r['metadata']['doc_name']}")
    print(f"Content: {r['content'][:100]}...")

print("\n--- RAG Response in Thai ---")
response = rag.rag_query(thai_query)
print(response)

print("\n--- Query in English ---")
eng_query = "What is RAG?"
print(f"Query: {eng_query}")
response = rag.rag_query(eng_query)
print(response)

print("\n--- Cross-lingual Query (TH + EN) ---")
mixed_query = "Vector database และการค้นหา semantic search"
print(f"Query: {mixed_query}")
response = rag.rag_query(mixed_query)
print(response)
