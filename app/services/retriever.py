"""
Retriever Service

Orchestrates the retrieval pipeline:
1. Convert query to embedding
2. Search vector store for similar chunks
3. (Optional) Re-rank results
4. Return top-k relevant chunks

This is the main interface for finding relevant documents
to answer user questions.
"""

from dataclasses import dataclass, field
from typing import Optional, Protocol
import logging
import uuid

from .embedder import EmbedderProtocol, EmbedderFactory, MockEmbedder
from .vector_store import (
    VectorStoreFactory,
    MockVectorStore,
    VectorDocument,
    SearchResult,
    QdrantConfig,
)
from .chunker import Chunk

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """
    Result of a retrieval operation.
    
    Attributes:
        query: The original query text
        results: List of matching chunks with scores
        total_results: Number of results found
        filters_applied: Any filters that were used
    """
    query: str
    results: list[SearchResult]
    total_results: int
    filters_applied: dict = field(default_factory=dict)
    
    @property
    def texts(self) -> list[str]:
        """Get just the text content of results."""
        return [r.text for r in self.results]
    
    @property
    def top_result(self) -> Optional[SearchResult]:
        """Get the highest-scoring result."""
        return self.results[0] if self.results else None


@dataclass
class RetrieverConfig:
    """
    Configuration for the retriever.
    
    Attributes:
        top_k: Number of results to return
        score_threshold: Minimum similarity score
        use_reranking: Whether to re-rank results
        embedder_provider: Which embedder to use
        vector_store_provider: Which vector store to use
    """
    top_k: int = 5
    score_threshold: Optional[float] = None
    use_reranking: bool = False
    embedder_provider: str = "openai"
    vector_store_provider: str = "qdrant"


class Retriever:
    """
    Main retrieval service for finding relevant document chunks.
    
    Combines embedding and vector search to find chunks that
    are semantically similar to a user's query.
    
    Usage:
        retriever = Retriever()
        
        # First, index some documents
        retriever.index_chunks(chunks)
        
        # Then, search
        results = retriever.retrieve("What is the leave policy?")
        
        for result in results.results:
            print(f"Score: {result.score:.3f}")
            print(f"Text: {result.text[:100]}...")
    """
    
    def __init__(
        self,
        config: Optional[RetrieverConfig] = None,
        embedder: Optional[EmbedderProtocol] = None,
        vector_store = None,
        qdrant_config: Optional[QdrantConfig] = None
    ):
        """
        Initialise the retriever.
        
        Args:
            config: Retriever configuration
            embedder: Custom embedder instance (optional)
            vector_store: Custom vector store instance (optional)
            qdrant_config: Qdrant configuration (optional)
        """
        self.config = config or RetrieverConfig()
        
        # Initialise embedder
        if embedder:
            self.embedder = embedder
        else:
            self.embedder = EmbedderFactory.create(
                self.config.embedder_provider
            )
        
        # Initialise vector store
        if vector_store:
            self.vector_store = vector_store
        else:
            self.vector_store = VectorStoreFactory.create(
                self.config.vector_store_provider,
                qdrant_config
            )
        
        logger.info(
            f"Retriever initialised with {self.config.embedder_provider} "
            f"embedder and {self.config.vector_store_provider} vector store"
        )
    
    def create_collection(self, recreate: bool = False) -> None:
        """
        Create the vector collection.
        
        Must be called before indexing documents.
        
        Args:
            recreate: If True, delete existing collection first
        """
        self.vector_store.create_collection(recreate=recreate)
    
    def index_chunks(
        self,
        chunks: list[Chunk],
        document_id: Optional[str] = None
    ) -> int:
        """
        Index document chunks for retrieval.
        
        Embeds each chunk and stores it in the vector database.
        
        Args:
            chunks: List of Chunk objects to index
            document_id: Optional document ID to associate with chunks
            
        Returns:
            Number of chunks indexed
        """
        if not chunks:
            return 0
        
        logger.info(f"Indexing {len(chunks)} chunks")
        
        # Get embeddings for all chunks
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedder.embed_batch(texts)
        
        # Create vector documents
        vector_docs = []
        for chunk, embedding in zip(chunks, embeddings):
            # Generate unique ID for this vector
            vector_id = str(uuid.uuid4())
            
            # Build metadata
            metadata = {
                **chunk.metadata,
                "chunk_index": chunk.index,
            }
            if document_id:
                metadata["document_id"] = document_id
            
            vector_docs.append(VectorDocument(
                id=vector_id,
                vector=embedding.vector,
                text=chunk.text,
                metadata=metadata
            ))
        
        # Store in vector database
        count = self.vector_store.upsert(vector_docs)
        logger.info(f"Indexed {count} chunks successfully")
        
        return count
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[dict] = None
    ) -> RetrievalResult:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: The user's question
            top_k: Number of results (overrides config if provided)
            filters: Metadata filters (e.g., {"department": "HR"})
            
        Returns:
            RetrievalResult containing matching chunks
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        logger.info(f"Retrieving for query: {query[:50]}...")
        
        # Embed the query
        query_embedding = self.embedder.embed(query)
        
        # Search vector store
        results = self.vector_store.search(
            query_vector=query_embedding.vector,
            top_k=top_k or self.config.top_k,
            filters=filters,
            score_threshold=self.config.score_threshold
        )
        
        # Optionally re-rank results
        if self.config.use_reranking and results:
            results = self._rerank(query, results)
        
        logger.info(f"Retrieved {len(results)} results")
        
        return RetrievalResult(
            query=query,
            results=results,
            total_results=len(results),
            filters_applied=filters or {}
        )
    
    def _rerank(
        self,
        query: str,
        results: list[SearchResult]
    ) -> list[SearchResult]:
        """
        Re-rank results for better relevance.
        
        This is a placeholder for more sophisticated re-ranking.
        Could use a cross-encoder model for better accuracy.
        
        Args:
            query: The original query
            results: Initial search results
            
        Returns:
            Re-ranked results
        """
        # Simple re-ranking: boost exact phrase matches
        query_lower = query.lower()
        
        for result in results:
            text_lower = result.text.lower()
            
            # Boost if query words appear in text
            query_words = query_lower.split()
            word_matches = sum(
                1 for word in query_words 
                if word in text_lower
            )
            
            # Adjust score (simple boost)
            boost = 1.0 + (word_matches * 0.1)
            result.score *= boost
        
        # Re-sort by adjusted score
        results.sort(key=lambda r: r.score, reverse=True)
        
        return results
    
    def delete_document(self, document_id: str) -> None:
        """
        Delete all chunks for a document.
        
        Args:
            document_id: The document ID to delete
        """
        self.vector_store.delete_by_filter({"document_id": document_id})
        logger.info(f"Deleted chunks for document: {document_id}")
    
    def get_stats(self) -> dict:
        """
        Get statistics about the retriever.
        
        Returns:
            Dict with stats like total vectors, etc.
        """
        return {
            "total_vectors": self.vector_store.count(),
            "embedder": self.config.embedder_provider,
            "vector_store": self.config.vector_store_provider,
            "top_k": self.config.top_k,
        }


class RetrieverBuilder:
    """
    Builder for creating customised Retriever instances.
    
    Usage:
        retriever = (
            RetrieverBuilder()
            .with_mock_embedder()
            .with_mock_vector_store()
            .with_top_k(10)
            .build()
        )
    """
    
    def __init__(self):
        self._config = RetrieverConfig()
        self._embedder = None
        self._vector_store = None
        self._qdrant_config = None
    
    def with_top_k(self, k: int) -> "RetrieverBuilder":
        """Set number of results to return."""
        self._config.top_k = k
        return self
    
    def with_score_threshold(self, threshold: float) -> "RetrieverBuilder":
        """Set minimum similarity score."""
        self._config.score_threshold = threshold
        return self
    
    def with_reranking(self, enabled: bool = True) -> "RetrieverBuilder":
        """Enable or disable re-ranking."""
        self._config.use_reranking = enabled
        return self
    
    def with_openai_embedder(self, api_key: Optional[str] = None) -> "RetrieverBuilder":
        """Use OpenAI embedder."""
        self._config.embedder_provider = "openai"
        self._embedder = EmbedderFactory.create("openai", api_key=api_key)
        return self
    
    def with_mock_embedder(self) -> "RetrieverBuilder":
        """Use mock embedder (for testing)."""
        self._config.embedder_provider = "mock"
        self._embedder = MockEmbedder()
        return self
    
    def with_qdrant(self, config: Optional[QdrantConfig] = None) -> "RetrieverBuilder":
        """Use Qdrant vector store."""
        self._config.vector_store_provider = "qdrant"
        self._qdrant_config = config
        return self
    
    def with_mock_vector_store(self) -> "RetrieverBuilder":
        """Use mock vector store (for testing)."""
        self._config.vector_store_provider = "mock"
        self._vector_store = MockVectorStore()
        return self
    
    def build(self) -> Retriever:
        """Build and return the configured Retriever."""
        return Retriever(
            config=self._config,
            embedder=self._embedder,
            vector_store=self._vector_store,
            qdrant_config=self._qdrant_config
        )