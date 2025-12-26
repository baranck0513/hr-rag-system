"""
Document Routes

API endpoints for document management:
- Upload documents
- List documents
- Get document details
- Delete documents
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from typing import Optional
import logging

from app.models.schemas import (
    DocumentUploadResponse,
    DocumentMetadataResponse,
    DocumentListResponse,
    DocumentDeleteResponse,
    ErrorResponse,
)
from app.services.ingestion import IngestionService
from app.services.document_parser import ParserFactory

logger = logging.getLogger(__name__)

# Create router with prefix and tags for OpenAPI docs
router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        404: {"model": ErrorResponse, "description": "Not Found"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    }
)

# In-memory storage for document metadata (will be MongoDB in production)
# This is a simple dict for now - documents are stored by document_id
_document_store: dict[str, DocumentMetadataResponse] = {}

# Ingestion service instance
_ingestion_service = IngestionService()


def get_document_store() -> dict[str, DocumentMetadataResponse]:
    """Get the document store. Allows for dependency injection in tests."""
    return _document_store


def get_ingestion_service() -> IngestionService:
    """Get the ingestion service. Allows for dependency injection in tests."""
    return _ingestion_service


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a document",
    description="""
    Upload a document for processing and indexing.
    
    The document will be:
    1. Parsed to extract text
    2. Scanned for PII which will be masked
    3. Split into chunks for retrieval
    
    Supported file types: PDF, TXT, MD
    """
)
async def upload_document(
    file: UploadFile = File(..., description="The document file to upload"),
    department: Optional[str] = Form(
        default=None,
        description="Department this document belongs to"
    ),
    access_roles: Optional[str] = Form(
        default=None,
        description="Comma-separated list of roles that can access this document"
    ),
    uploaded_by: Optional[str] = Form(
        default=None,
        description="User ID of the uploader"
    )
):
    """
    Upload and process a document.
    
    The document will be parsed, PII will be masked, and it will be
    chunked for later retrieval.
    """
    # Validate file type
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is required"
        )
    
    if not ParserFactory.is_supported(file.filename):
        supported = ", ".join(sorted(ParserFactory.SUPPORTED_EXTENSIONS))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type. Supported types: {supported}"
        )
    
    try:
        # Read file content
        content = await file.read()
    except Exception as e:
        logger.error(f"Error reading file: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error reading uploaded file"
        )
    
    # Check for empty file (outside try block so HTTPException propagates)
    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File is empty"
        )
    
    try:
        
        # Parse access roles from comma-separated string
        roles_list = []
        if access_roles:
            roles_list = [r.strip() for r in access_roles.split(",") if r.strip()]
        
        # Process the document
        logger.info(f"Processing uploaded file: {file.filename}")
        
        result = _ingestion_service.ingest(
            content=content,
            filename=file.filename,
            uploaded_by=uploaded_by,
            department=department,
            access_roles=roles_list
        )
        
        # Store document metadata
        doc_metadata = DocumentMetadataResponse(
            document_id=result.metadata.document_id,
            filename=result.metadata.filename,
            file_type=result.metadata.file_type,
            file_size_bytes=result.metadata.file_size_bytes,
            uploaded_at=result.metadata.uploaded_at,
            uploaded_by=result.metadata.uploaded_by,
            department=result.metadata.department,
            access_roles=result.metadata.access_roles,
            chunk_count=result.metadata.chunk_count
        )
        _document_store[result.metadata.document_id] = doc_metadata
        
        logger.info(
            f"Document processed successfully: {result.metadata.document_id}"
        )
        
        return DocumentUploadResponse(
            document_id=result.metadata.document_id,
            filename=result.metadata.filename,
            chunk_count=result.metadata.chunk_count,
            pii_masked=result.metadata.pii_detected,
            processing_time_ms=result.metadata.processing_time_ms
        )
        
    except ValueError as e:
        logger.error(f"Validation error during upload: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing the document"
        )


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List all documents",
    description="Get a list of all uploaded documents with their metadata."
)
async def list_documents(
    department: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
):
    """
    List all documents, optionally filtered by department.
    """
    documents = list(_document_store.values())
    
    # Filter by department if specified
    if department:
        documents = [d for d in documents if d.department == department]
    
    # Apply pagination
    total = len(documents)
    documents = documents[offset:offset + limit]
    
    return DocumentListResponse(
        documents=documents,
        total=total
    )


@router.get(
    "/{document_id}",
    response_model=DocumentMetadataResponse,
    summary="Get document details",
    description="Get metadata for a specific document by ID."
)
async def get_document(document_id: str):
    """
    Get details for a specific document.
    """
    if document_id not in _document_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document not found: {document_id}"
        )
    
    return _document_store[document_id]


@router.delete(
    "/{document_id}",
    response_model=DocumentDeleteResponse,
    summary="Delete a document",
    description="Delete a document and all its associated chunks."
)
async def delete_document(document_id: str):
    """
    Delete a document and its chunks from the system.
    """
    if document_id not in _document_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document not found: {document_id}"
        )
    
    # Remove from document store
    del _document_store[document_id]
    
    # Note: In production, we would also delete from Qdrant here
    # using retriever.delete_document(document_id)
    
    logger.info(f"Document deleted: {document_id}")
    
    return DocumentDeleteResponse(document_id=document_id)