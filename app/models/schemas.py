"""
API Schemas

Pydantic models that define the shape of API requests and responses.
These provide automatic validation, serialisation, and documentation.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


# --- Document Schemas ---

class DocumentUploadResponse(BaseModel):
    """Response after successfully uploading a document."""
    
    document_id: str = Field(
        description="Unique identifier for the uploaded document"
    )
    filename: str = Field(
        description="Original filename"
    )
    chunk_count: int = Field(
        description="Number of chunks created from the document"
    )
    pii_masked: dict[str, int] = Field(
        default_factory=dict,
        description="Count of PII types that were masked"
    )
    processing_time_ms: float = Field(
        description="Time taken to process the document in milliseconds"
    )
    message: str = Field(
        default="Document uploaded and processed successfully"
    )


class DocumentMetadataResponse(BaseModel):
    """Metadata about a stored document."""
    
    document_id: str
    filename: str
    file_type: str
    file_size_bytes: int
    uploaded_at: datetime
    uploaded_by: Optional[str] = None
    department: Optional[str] = None
    access_roles: list[str] = Field(default_factory=list)
    chunk_count: int = 0


class DocumentListResponse(BaseModel):
    """Response containing a list of documents."""
    
    documents: list[DocumentMetadataResponse]
    total: int


class DocumentDeleteResponse(BaseModel):
    """Response after deleting a document."""
    
    document_id: str
    message: str = "Document deleted successfully"


# --- Query Schemas ---

class QueryRequest(BaseModel):
    """Request to query the RAG system."""
    
    question: str = Field(
        min_length=1,
        max_length=1000,
        description="The question to ask"
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of results to return"
    )
    department: Optional[str] = Field(
        default=None,
        description="Filter results by department"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "question": "How much annual leave do I get?",
                    "top_k": 5,
                    "department": "HR"
                }
            ]
        }
    }


class RetrievedChunk(BaseModel):
    """A single retrieved chunk from the search."""
    
    text: str = Field(description="The chunk text content")
    score: float = Field(description="Similarity score (0-1)")
    document_id: Optional[str] = Field(
        default=None,
        description="ID of the source document"
    )
    filename: Optional[str] = Field(
        default=None,
        description="Name of the source file"
    )


class QueryResponse(BaseModel):
    """Response to a query request."""
    
    question: str = Field(description="The original question")
    results: list[RetrievedChunk] = Field(
        description="Retrieved chunks relevant to the question"
    )
    total_results: int = Field(
        description="Number of results returned"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "question": "How much annual leave do I get?",
                    "results": [
                        {
                            "text": "All employees are entitled to 25 days of annual leave per year.",
                            "score": 0.92,
                            "document_id": "abc123",
                            "filename": "leave_policy.pdf"
                        }
                    ],
                    "total_results": 1
                }
            ]
        }
    }


# --- Health Check Schemas ---

class HealthResponse(BaseModel):
    """Response for health check endpoint."""
    
    status: str = Field(
        default="healthy",
        description="Current health status"
    )
    version: str = Field(
        description="API version"
    )
    total_documents: int = Field(
        default=0,
        description="Number of documents in the system"
    )
    total_vectors: int = Field(
        default=0,
        description="Number of vectors in the store"
    )


# --- Error Schemas ---

class ErrorResponse(BaseModel):
    """Standard error response."""
    
    error: str = Field(description="Error type")
    detail: str = Field(description="Detailed error message")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "error": "ValidationError",
                    "detail": "Question cannot be empty"
                }
            ]
        }
    }