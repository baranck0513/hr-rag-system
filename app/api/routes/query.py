"""
Query Routes

API endpoints for querying the RAG system:
- Ask questions and retrieve relevant chunks
"""

from fastapi import APIRouter, HTTPException, status
import logging

from app.models.schemas import (
    QueryRequest,
    QueryResponse,
    RetrievedChunk,
    ErrorResponse,
)
from app.services.retriever import RetrieverBuilder

logger = logging.getLogger(__name__)

# Create router with prefix and tags for OpenAPI docs
router = APIRouter(
    prefix="/query",
    tags=["Query"],
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    }
)

# Retriever instance (using mock for now - will use real Qdrant in production)
_retriever = (
    RetrieverBuilder()
    .with_mock_embedder()
    .with_mock_vector_store()
    .with_top_k(5)
    .build()
)

# Initialise the collection
_retriever.create_collection()


def get_retriever():
    """Get the retriever instance. Allows for dependency injection in tests."""
    return _retriever


@router.post(
    "",
    response_model=QueryResponse,
    summary="Query the system",
    description="""
    Ask a question and retrieve relevant document chunks.
    
    The system will:
    1. Convert your question to an embedding
    2. Search for semantically similar chunks
    3. Return the most relevant results
    
    You can optionally filter by department.
    """
)
async def query(request: QueryRequest):
    """
    Query the RAG system with a question.
    
    Returns the most relevant document chunks based on semantic similarity.
    """
    try:
        logger.info(f"Processing query: {request.question[:50]}...")
        
        # Build filters if department specified
        filters = None
        if request.department:
            filters = {"department": request.department}
        
        # Perform retrieval
        result = _retriever.retrieve(
            query=request.question,
            top_k=request.top_k,
            filters=filters
        )
        
        # Convert to response format
        chunks = [
            RetrievedChunk(
                text=r.text,
                score=r.score,
                document_id=r.metadata.get("document_id"),
                filename=r.metadata.get("filename")
            )
            for r in result.results
        ]
        
        logger.info(f"Query returned {len(chunks)} results")
        
        return QueryResponse(
            question=request.question,
            results=chunks,
            total_results=len(chunks)
        )
        
    except ValueError as e:
        logger.error(f"Validation error during query: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing your query"
        )