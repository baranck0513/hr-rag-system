# HR RAG System

A Retrieval-Augmented Generation (RAG) system for HR operations, built with FastAPI, Qdrant, and OpenAI.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green)
![Tests](https://img.shields.io/badge/Tests-207%20passing-brightgreen)

## Overview

This system allows HR departments to:
- **Upload documents** (PDF, TXT, MD) with automatic PII masking
- **Query documents** using natural language
- **Control access** with role-based permissions

### Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI Application                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│   │   Document   │    │   Retrieval  │    │     RBAC     │  │
│   │   Ingestion  │    │   Pipeline   │    │   Security   │  │
│   └──────────────┘    └──────────────┘    └──────────────┘  │
│          │                   │                    │          │
│          ▼                   ▼                    ▼          │
│   ┌──────────────────────────────────────────────────────┐  │
│   │                    Service Layer                      │  │
│   │  Parser │ PII Masker │ Chunker │ Embedder │ Retriever │  │
│   └──────────────────────────────────────────────────────┘  │
│                              │                               │
└──────────────────────────────┼───────────────────────────────┘
                               │
                ┌──────────────┴──────────────┐
                │                             │
         ┌──────▼──────┐              ┌───────▼──────┐
         │   Qdrant    │              │   MongoDB    │
         │  (Vectors)  │              │  (Metadata)  │
         └─────────────┘              └──────────────┘
```

## Features

| Feature | Description |
|---------|-------------|
| **Document Processing** | Parse PDF, TXT, MD files and extract text |
| **PII Masking** | Automatically mask UK-specific PII (NI numbers, emails, phone numbers, etc.) |
| **Smart Chunking** | Recursive chunking that preserves document structure |
| **Semantic Search** | Find relevant documents using natural language queries |
| **Role-Based Access Control** | Restrict document access based on user roles |
| **Evaluation Metrics** | Measure RAG quality with Recall@k, Precision@k, MRR |

## Quick Start

### Prerequisites

- Python 3.12+
- Docker & Docker Compose (for containerised deployment)

### Option 1: Run with Docker
```bash
# Clone the repository
git clone https://github.com/baranck0513/hr-rag-system.git
cd hr-rag-system

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Access the API
open http://localhost:8000/docs
```

### Option 2: Run Locally
```bash
# Clone the repository
git clone https://github.com/baranck0513/hr-rag-system.git
cd hr-rag-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the API
uvicorn app.main:app --reload

# Access the API
open http://localhost:8000/docs
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/documents/upload` | Upload and process a document |
| `GET` | `/api/v1/documents` | List all documents |
| `GET` | `/api/v1/documents/{id}` | Get document details |
| `DELETE` | `/api/v1/documents/{id}` | Delete a document |
| `POST` | `/api/v1/query` | Query the RAG system |
| `GET` | `/api/v1/health` | Health check |

### Example: Upload a Document
```bash
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -F "file=@leave_policy.pdf" \
  -F "department=HR" \
  -F "access_roles=all_staff"
```

### Example: Query
```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "How much annual leave do I get?", "top_k": 5}'
```

## Project Structure
```
hr-rag-system/
├── app/
│   ├── main.py                 # FastAPI application entry point
│   ├── api/
│   │   └── routes/
│   │       ├── documents.py    # Document endpoints
│   │       └── query.py        # Query endpoints
│   ├── models/
│   │   └── schemas.py          # Pydantic models
│   └── services/
│       ├── document_parser.py  # PDF/TXT parsing
│       ├── pii_masker.py       # PII detection and masking
│       ├── chunker.py          # Text chunking
│       ├── embedder.py         # Text to vector conversion
│       ├── vector_store.py     # Qdrant operations
│       ├── retriever.py        # Search orchestration
│       ├── ingestion.py        # Ingestion pipeline
│       ├── rbac.py             # Access control
│       └── evaluation.py       # Quality metrics
├── tests/                      # Test suite (207 tests)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Testing
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html

# Run specific test file
pytest tests/test_retriever.py -v
```

## Configuration

Environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `QDRANT_HOST` | Qdrant server host | `localhost` |
| `QDRANT_PORT` | Qdrant server port | `6333` |
| `OPENAI_API_KEY` | OpenAI API key for embeddings | - |
| `LOG_LEVEL` | Logging level | `INFO` |

## PII Types Detected

The system automatically detects and masks UK-specific PII:

- National Insurance numbers
- Email addresses
- UK phone numbers (mobile and landline)
- UK postcodes
- Bank sort codes and account numbers
- Dates of birth
- UK passport numbers

## Evaluation Metrics

The system includes built-in evaluation:

- **Recall@k**: Percentage of relevant documents found in top-k results
- **Precision@k**: Percentage of top-k results that are relevant
- **MRR**: Mean Reciprocal Rank - how high the first relevant result ranks
