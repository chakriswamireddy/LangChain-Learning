# LangChain Learning

A personal learning repository for exploring LangChain v1 concepts with local Ollama models. This is a study notebook, not a production project — each script demonstrates one concept in isolation.

## Prerequisites

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/) running locally with `llama3.2`, `gemma3:1b`, and `nomic-embed-text` pulled

## Setup

```bash
uv sync
```

## Examples

| Concept | File |
|---|---|
| Simple AI Agent | `basic_agent.py` |
| Standalone Model Inference | `simple_chat.py` |
| Conversations | `simple_chat.py` |
| Streaming Responses | `simple_chat.py` |
| Advanced Agent (Context, Memory, Structured Output) | `multi_tools.py` |
| RAG (Embeddings, Vector Stores, Retrieval) | `rag/ingest.py`, `rag/retriever.py`, `rag/chat.py` |
| Dynamic System Prompts | `multi-modal.py` |
| Dynamic Model Choice | `multi-modal.py` |
| Custom Agent Middleware | `custom-agent-middleware.py` |

### RAG

```bash
uv run rag/ingest.py    # load PDF from data/, chunk, embed into chroma_db/
uv run rag/chat.py      # chat over the ingested documents
```

## Notes

- `chroma_db/` is a generated vector store and is git-ignored; re-run `rag/ingest.py` to rebuild it.
- `multi-modal.py` currently holds the dynamic prompt/model middleware experiments.
