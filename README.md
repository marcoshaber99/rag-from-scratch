# rag-from-scratch

A small RAG system over a personal writing corpus, built from primitives instead of a framework.

## Why

To understand how RAG works under the hood. The goal was to build the mechanics directly rather than hide them behind LangChain or a vector database.

## How it works

```text
corpus/ → chunks → embeddings → SQLite → retrieval → LLM → answer
```

Chunk documents, embed each chunk, store in SQLite, retrieve top matches by cosine similarity, generate an answer grounded in the retrieved chunks.

## Stack

Python, OpenAI API (`text-embedding-3-small`, `gpt-4o-mini`), SQLite, NumPy, tiktoken. No LangChain, no vector database.

## Setup

```bash
uv sync
```

Add an `.env` file with `OPENAI_API_KEY`, put markdown files in `corpus/`, then:

```bash
uv run python main.py
```

## Usage

```python
from src.generate import generate_answer
print(generate_answer("How do git worktrees work?"))
```

## Design choices

- **Chunking:** 500 tokens, 100 overlap. Sliding window, no structural splitting.
- **Embeddings:** `text-embedding-3-small`, 1536 dims, stored as `float32` BLOBs.
- **Retrieval:** cosine similarity, top K=3.
- **Generation:** prompt instructs the model to use only the retrieved context.

## Example

A query about git worktrees retrieves the right document and produces a grounded answer.

A query about Tokyo weather returns:

> The provided context does not contain any information about the current weather forecast in Tokyo.

That second behavior is the prompt doing real work.

## Limitations

This is a learning project, not production. A real system would add reranking, hybrid search, query rewriting, evals, citations, and a production vector database (pgvector, Pinecone, Qdrant).
