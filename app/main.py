"""
HR RAG System - Main Application

A production-grade Retrieval-Augmented Generation system
for HR operations, optimized for UK-sponsored graduate-level roles.
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import time

from app.api.routes import documents, query
from app.models.schemas import HealthResponse, ErrorResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Application metadata
APP_TITLE = "HR RAG System"
APP_DESCRIPTION = """
A Retrieval-Augmented Generation system for HR operations.

## Features

* **Document Upload** - Upload PDF, TXT, or MD files
* **PII Masking** - Automatically masks UK-specific PII (NI numbers, emails, etc.)
* **Semantic Search** - Find relevant information using natural language queries
* **Department Filtering** - Filter results by department

## Usage

1. Upload documents via `POST /api/v1/documents/upload`
2. Query the system via `POST /api/v1/query`
"""
APP_VERSION = "0.1.0"

# Create FastAPI application
app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
    docs_url="/docs",           # Swagger UI
    redoc_url="/redoc",         # ReDoc
    openapi_url="/openapi.json" # OpenAPI schema
)

# Configure CORS (Cross-Origin Resource Sharing)
# This allows the API to be called from web browsers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Middleware ---

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time to response headers."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(round(process_time * 1000, 2))
    return response


# --- Exception Handlers ---

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle ValueError exceptions."""
    return JSONResponse(
        status_code=400,
        content={"error": "ValidationError", "detail": str(exc)}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.error(f"Unexpected error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "detail": "An unexpected error occurred"
        }
    )


# --- Routes ---

# Include routers with API version prefix
app.include_router(documents.router, prefix="/api/v1")
app.include_router(query.router, prefix="/api/v1")


# --- Health Check ---

@app.get(
    "/api/v1/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health check",
    description="Check if the API is running and get basic statistics."
)
async def health_check():
    """
    Health check endpoint.
    
    Returns the API status and basic statistics.
    """
    from app.api.routes.documents import get_document_store
    from app.api.routes.query import get_retriever
    
    doc_store = get_document_store()
    retriever = get_retriever()
    stats = retriever.get_stats()
    
    return HealthResponse(
        status="healthy",
        version=APP_VERSION,
        total_documents=len(doc_store),
        total_vectors=stats.get("total_vectors", 0)
    )


# --- Root Endpoint ---

@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint.
    
    Provides basic information and links to documentation.
    """
    return {
        "name": APP_TITLE,
        "version": APP_VERSION,
        "documentation": {
            "swagger": "/docs",
            "redoc": "/redoc",
            "openapi": "/openapi.json"
        },
        "endpoints": {
            "health": "/api/v1/health",
            "upload": "/api/v1/documents/upload",
            "documents": "/api/v1/documents",
            "query": "/api/v1/query"
        }
    }


# --- Startup/Shutdown Events ---

@app.on_event("startup")
async def startup_event():
    """Run on application startup."""
    logger.info(f"Starting {APP_TITLE} v{APP_VERSION}")
    logger.info("API documentation available at /docs")


@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown."""
    logger.info("Shutting down application")


# --- Run with uvicorn ---

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True  # Auto-reload on code changes (development only)
    )