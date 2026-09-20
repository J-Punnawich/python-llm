# Cross-Lingual RAG Pipeline (Thai + English)

A Python RAG system with BGE-M3 embeddings, Chroma vector DB, and OpenAI LLM integration.

## Setup

```bash
pip install -r requirements.txt
# Add your OPENAI_API_KEY to .env
```

## Features

- **BGE-M3 Embeddings**: Cross-lingual Thai + English support
- **Chroma Vector DB**: Lightweight, in-memory, stores vectors + metadata
- **Document Chunking**: Recursive splitting with overlap
- **RAG Query**: Retrieves context + generates with OpenAI
- **Metadata Support**: Track doc_name, doc_id, chunk_id, language, topic

## Usage

```python
from rag_pipeline import CrossLingualRAG

rag = CrossLingualRAG()

# Add documents
rag.add_documents(
    texts=["Document 1", "Document 2"],
    metadata_list=[
        {"name": "doc1", "language": "thai"},
        {"name": "doc2", "language": "english"}
    ]
)

# Query and retrieve
results = rag.query("What is AI?", n_results=3)

# Query with LLM response
response = rag.rag_query("What is AI?")
```

## Architecture

```
Documents
    ↓
Chunking (RecursiveCharacterTextSplitter)
    ↓
BGE-M3 Embeddings (cross-lingual)
    ↓
Chroma Vector DB (with metadata)
    ↓
Query → Retrieve Top-K → OpenAI LLM → Response
```

## Next Steps: Function Calling

To add function calling:

1. Define tools in `rag_pipeline.py`:
```python
def add_function_calling(self, tools: list[dict]):
    """Add tool definitions for function calling"""
    self.tools = tools

def rag_query_with_functions(self, query_text: str, tools: list[dict]) -> dict:
    """Query with function calling capability"""
    retrieved = self.query(query_text)
    context = "\n".join([r['content'] for r in retrieved])
    
    response = self.openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"{context}\n\n{query_text}"}],
        tools=tools,
        tool_choice="auto"
    )
    return response
```

2. Example tools:
```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_database",
            "description": "Search the vector database",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                }
            }
        }
    }
]
```

## File Structure

```
.
├── rag_pipeline.py       # Main RAG class
├── example_usage.py      # Usage examples
├── requirements.txt      # Dependencies
├── .env          # Environment template
└── README.md             # This file
```
