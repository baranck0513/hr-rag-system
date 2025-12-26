"""
Vector Store Service

Handles storage and retrieval of vectors in Qdrant.
Provides operations for:
- Creating collections
- Storing document chunks with their embeddings
- Searching for similar vectors
- Filtering by metadata
"""

from dataclasses import dataclass, field
from typing import Optional, Any
import logging
import uuid

logger = logging.getLogger(__name__)


@dataclass
class VectorDocument:
    """
    A document chunk with its vector and metadata.
    
    This is what we store in Qdrant.
    
    Attributes:
        id: Unique identifier for this vector
        vector: The embedding vector
        text: Original text content
        metadata: Additional data (document_id, filename, etc.)
    """
    id: str
    vector: list[float]
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """
    A single search result from Qdrant.
    
    Attributes:
        id: The vector ID
        text: The chunk text
        score: Similarity score (higher = more similar)
        metadata: Associated metadata
    """
    id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QdrantConfig:
    """
    Configuration for Qdrant connection.
    
    Attributes:
        host: Qdrant server host
        port: Qdrant server port
        collection_name: Name of the collection to use
        vector_size: Dimensionality of vectors
        distance: Distance metric ("cosine", "euclid", "dot")
    """
    host: str = "localhost"
    port: int = 6333
    collection_name: str = "hr_documents"
    vector_size: int = 1536
    distance: str = "cosine"


class QdrantVectorStore:
    """
    Vector store implementation using Qdrant.
    
    Handles all interactions with the Qdrant database for
    storing and retrieving document embeddings.
    
    Usage:
        config = QdrantConfig(collection_name="hr_docs")
        store = QdrantVectorStore(config)
        
        # Store vectors
        store.upsert([vector_doc1, vector_doc2])
        
        # Search
        results = store.search(query_vector, top_k=5)
    """
    
    def __init__(self, config: Optional[QdrantConfig] = None):
        """
        Initialise the Qdrant vector store.
        
        Args:
            config: Qdrant configuration. Uses defaults if not provided.
        """
        self.config = config or QdrantConfig()
        self._client = None
    
    @property
    def client(self):
        """Lazy initialisation of Qdrant client."""
        if self._client is None:
            try:
                from qdrant_client import QdrantClient
                self._client = QdrantClient(
                    host=self.config.host,
                    port=self.config.port
                )
                logger.info(
                    f"Connected to Qdrant at "
                    f"{self.config.host}:{self.config.port}"
                )
            except ImportError:
                raise ImportError(
                    "qdrant-client package is required. Install with: "
                    "pip install qdrant-client"
                )
        return self._client
    
    def create_collection(self, recreate: bool = False) -> None:
        """
        Create the vector collection in Qdrant.
        
        Args:
            recreate: If True, delete existing collection first
        """
        from qdrant_client.models import Distance, VectorParams
        
        collection_name = self.config.collection_name
        
        # Check if collection exists
        collections = self.client.get_collections().collections
        exists = any(c.name == collection_name for c in collections)
        
        if exists and recreate:
            logger.info(f"Deleting existing collection: {collection_name}")
            self.client.delete_collection(collection_name)
            exists = False
        
        if not exists:
            # Map distance string to Qdrant Distance enum
            distance_map = {
                "cosine": Distance.COSINE,
                "euclid": Distance.EUCLID,
                "dot": Distance.DOT
            }
            distance = distance_map.get(
                self.config.distance.lower(), 
                Distance.COSINE
            )
            
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self.config.vector_size,
                    distance=distance
                )
            )
            logger.info(
                f"Created collection: {collection_name} "
                f"(size={self.config.vector_size}, distance={self.config.distance})"
            )
        else:
            logger.info(f"Collection already exists: {collection_name}")
    
    def upsert(self, documents: list[VectorDocument]) -> int:
        """
        Insert or update vectors in the collection.
        
        Args:
            documents: List of VectorDocument objects to store
            
        Returns:
            Number of documents upserted
        """
        if not documents:
            return 0
        
        from qdrant_client.models import PointStruct
        
        points = [
            PointStruct(
                id=doc.id,
                vector=doc.vector,
                payload={
                    "text": doc.text,
                    **doc.metadata
                }
            )
            for doc in documents
        ]
        
        self.client.upsert(
            collection_name=self.config.collection_name,
            points=points
        )
        
        logger.info(f"Upserted {len(documents)} vectors")
        return len(documents)
    
    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filters: Optional[dict] = None,
        score_threshold: Optional[float] = None
    ) -> list[SearchResult]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: The query embedding vector
            top_k: Number of results to return
            filters: Metadata filters (e.g., {"department": "HR"})
            score_threshold: Minimum similarity score
            
        Returns:
            List of SearchResult objects, ordered by similarity
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        # Build filter if provided
        qdrant_filter = None
        if filters:
            conditions = [
                FieldCondition(
                    key=key,
                    match=MatchValue(value=value)
                )
                for key, value in filters.items()
            ]
            qdrant_filter = Filter(must=conditions)
        
        results = self.client.search(
            collection_name=self.config.collection_name,
            query_vector=query_vector,
            limit=top_k,
            query_filter=qdrant_filter,
            score_threshold=score_threshold
        )
        
        search_results = [
            SearchResult(
                id=str(result.id),
                text=result.payload.get("text", ""),
                score=result.score,
                metadata={
                    k: v for k, v in result.payload.items() 
                    if k != "text"
                }
            )
            for result in results
        ]
        
        logger.debug(f"Search returned {len(search_results)} results")
        return search_results
    
    def delete(self, ids: list[str]) -> int:
        """
        Delete vectors by ID.
        
        Args:
            ids: List of vector IDs to delete
            
        Returns:
            Number of vectors deleted
        """
        if not ids:
            return 0
        
        from qdrant_client.models import PointIdsList
        
        self.client.delete(
            collection_name=self.config.collection_name,
            points_selector=PointIdsList(points=ids)
        )
        
        logger.info(f"Deleted {len(ids)} vectors")
        return len(ids)
    
    def delete_by_filter(self, filters: dict) -> None:
        """
        Delete vectors matching a filter.
        
        Useful for deleting all chunks from a specific document.
        
        Args:
            filters: Metadata filters (e.g., {"document_id": "abc123"})
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        conditions = [
            FieldCondition(
                key=key,
                match=MatchValue(value=value)
            )
            for key, value in filters.items()
        ]
        
        self.client.delete(
            collection_name=self.config.collection_name,
            points_selector=Filter(must=conditions)
        )
        
        logger.info(f"Deleted vectors matching filter: {filters}")
    
    def count(self) -> int:
        """
        Get the number of vectors in the collection.
        
        Returns:
            Number of vectors
        """
        info = self.client.get_collection(self.config.collection_name)
        return info.points_count


class MockVectorStore:
    """
    Mock vector store for testing without Qdrant.
    
    Stores vectors in memory and provides basic search functionality.
    """
    
    def __init__(self, config: Optional[QdrantConfig] = None):
        self.config = config or QdrantConfig()
        self._documents: dict[str, VectorDocument] = {}
    
    def create_collection(self, recreate: bool = False) -> None:
        """Create/reset the in-memory store."""
        if recreate:
            self._documents = {}
        logger.info("Mock collection ready")
    
    def upsert(self, documents: list[VectorDocument]) -> int:
        """Store documents in memory."""
        for doc in documents:
            self._documents[doc.id] = doc
        return len(documents)
    
    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filters: Optional[dict] = None,
        score_threshold: Optional[float] = None
    ) -> list[SearchResult]:
        """
        Search using cosine similarity.
        
        Simple implementation for testing.
        """
        import math
        
        def cosine_similarity(v1: list[float], v2: list[float]) -> float:
            """Calculate cosine similarity between two vectors."""
            dot_product = sum(a * b for a, b in zip(v1, v2))
            norm1 = math.sqrt(sum(a * a for a in v1))
            norm2 = math.sqrt(sum(b * b for b in v2))
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return dot_product / (norm1 * norm2)
        
        # Calculate similarities
        scored_docs = []
        for doc in self._documents.values():
            # Apply filters if provided
            if filters:
                if not all(
                    doc.metadata.get(k) == v 
                    for k, v in filters.items()
                ):
                    continue
            
            score = cosine_similarity(query_vector, doc.vector)
            
            # Apply score threshold
            if score_threshold and score < score_threshold:
                continue
            
            scored_docs.append((doc, score))
        
        # Sort by score (descending) and take top_k
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        top_docs = scored_docs[:top_k]
        
        return [
            SearchResult(
                id=doc.id,
                text=doc.text,
                score=score,
                metadata=doc.metadata
            )
            for doc, score in top_docs
        ]
    
    def delete(self, ids: list[str]) -> int:
        """Delete documents by ID."""
        count = 0
        for id in ids:
            if id in self._documents:
                del self._documents[id]
                count += 1
        return count
    
    def delete_by_filter(self, filters: dict) -> None:
        """Delete documents matching filter."""
        to_delete = [
            id for id, doc in self._documents.items()
            if all(doc.metadata.get(k) == v for k, v in filters.items())
        ]
        for id in to_delete:
            del self._documents[id]
    
    def count(self) -> int:
        """Return number of stored documents."""
        return len(self._documents)


class VectorStoreFactory:
    """Factory for creating vector store instances."""
    
    @staticmethod
    def create(
        provider: str = "qdrant",
        config: Optional[QdrantConfig] = None
    ):
        """
        Create a vector store instance.
        
        Args:
            provider: "qdrant" or "mock"
            config: Configuration for the store
            
        Returns:
            A vector store instance
        """
        if provider == "qdrant":
            return QdrantVectorStore(config)
        elif provider == "mock":
            return MockVectorStore(config)
        else:
            raise ValueError(f"Unknown vector store provider: {provider}")