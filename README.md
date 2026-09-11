# Agentic RAG

A FastAPI service for building and querying document knowledge bases with agentic retrieval-augmented generation. Documents are uploaded into named knowledge bases, parsed and chunked with Docling, indexed with FAISS and BM25, and queried through a LangGraph-based answer pipeline.

## Features

- Multi-file document uploads grouped by knowledge base
- Docling-based ingestion with image captioning support
- Hybrid retrieval using dense FAISS search and sparse BM25 search
- Answer generation and verification through an agentic RAG pipeline
- JSON and server-sent events (SSE) query endpoints
- Retrieval and answer caching
- OpenAPI documentation at `/docs`

## Requirements

- Python 3.11 or 3.12
- [`uv`](https://docs.astral.sh/uv/)
- A Groq API key
- CUDA is recommended for the configured PyTorch and ONNX dependencies; CPU-only environments may require dependency adjustments

## Setup

1. Clone the repository and enter its directory.

2. Install the locked dependencies:

   ```powershell
   uv sync
   ```

3. Configure the required secret. The application reads settings from environment variables, `.env`, and the ignored `config.json` file. Environment variables take precedence.

   Create a `.env` file in the repository root:

   ```dotenv
   GROQ_API_KEY=your-groq-api-key
   INGESTION_STRATEGY=docling_caption
   HYBRID_SEARCH_ENABLED=true
   HYBRID_DENSE_WEIGHT=0.5
   HYBRID_SPARSE_WEIGHT=0.5
   ```

   Never commit API keys. If a key has been exposed, revoke it and create a replacement.

## Run the API

Start the development server with:

```powershell
uv run uvicorn app.main:app --host 0.0.0.0 --port 6001 --reload
```

The API is available at `http://localhost:6001`. Interactive API documentation is available at `http://localhost:6001/docs`.

## Typical workflow

### 1. Upload documents

```powershell
curl.exe -X POST http://localhost:6001/api/v1/knowledge/upload `
  -F "knowledge_base=product-docs" `
  -F "files=@path\to\document.pdf"
```

Replace `path\to\document.pdf` with the path to a local document. Repeat the command or include multiple `-F "files=@..."` fields to upload more files.

### 2. Build or update the index

```powershell
curl.exe -X POST http://localhost:6001/api/v1/knowledge/reindex `
  -H "Content-Type: application/json" `
  -d '{"knowledge_base":"product-docs"}'
```

Reindexing processes new and modified files for the selected knowledge base and updates the FAISS and BM25 indexes.

### 3. Ask a question

```powershell
curl.exe -X POST http://localhost:6001/api/v1/query/ `
  -H "Content-Type: application/json" `
  -d '{"knowledge_base":"product-docs","question":"What is the cancellation policy?","chat_id":"demo-chat"}'
```

The response includes the generated answer, document citations, and whether the answer came from cache.

### Streaming responses

Use `/api/v1/query/stream` with the same JSON body to receive server-sent events. The endpoint emits `answer`, `citations`, and `error` events.

## API endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `POST` | `/api/v1/knowledge/upload` | Upload one or more files to a knowledge base |
| `GET` | `/api/v1/knowledge/files?knowledge_name=...` | List files in a knowledge base |
| `GET` | `/api/v1/knowledge/knowledge-bases` | List knowledge bases |
| `POST` | `/api/v1/knowledge/reindex` | Process new or modified files and update indexes |
| `DELETE` | `/api/v1/knowledge/{knowledge_base}` | Delete a knowledge base and related cached data |
| `POST` | `/api/v1/query/` | Run a query and return a JSON response |
| `POST` | `/api/v1/query/stream` | Run a query and return SSE events |

## Project layout

```text
app/
|-- api/          FastAPI route handlers
|-- core/         Application settings
|-- ingestion/    Loading, chunking, embedding, and indexing
|-- prompts/      Agent and RAG prompts
|-- retrieval/    Retrieval, reranking, caching, and answer generation
`-- schemas/      Request and response models

data/
`-- cache/        Runtime cache and generated index data
```

Runtime uploads, indexes, caches, `.env`, and `config.json` are ignored by Git. Keep production secrets in your deployment environment or a secret manager.

## Development

Run the test suite when tests are available:

```powershell
uv run pytest
```

The application can also be started directly with:

```powershell
uv run python -m app.main
```
