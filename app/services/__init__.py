"""
Services package - contains core business logic.
"""

from .document_parser import (
    DocumentParser,
    TextParser,
    PDFParser,
    ParserFactory,
)
from .pii_masker import PIIMasker, PIIPattern
from .chunker import (
    Chunk,
    ChunkerConfig,
    RecursiveChunker,
    SentenceChunker,
    FixedSizeChunker,
    ChunkerFactory,
)
from .ingestion import (
    IngestionService,
    IngestionServiceBuilder,
    DocumentMetadata,
    ProcessedDocument,
)
from .embedder import (
    EmbeddingResult,
    OpenAIEmbedder,
    MockEmbedder,
    EmbedderFactory,
)
from .vector_store import (
    VectorDocument,
    SearchResult,
    QdrantConfig,
    QdrantVectorStore,
    MockVectorStore,
    VectorStoreFactory,
)
from .retriever import (
    Retriever,
    RetrieverBuilder,
    RetrieverConfig,
    RetrievalResult,
)

from .rbac import (
    User,
    RBACService,
    RBACMiddleware,
    ALL_STAFF_ROLE,
)

from .evaluation import (
    EvaluationQuery,
    EvaluationResult,
    AggregateResults,
    EvaluationService,
    RetrievalEvaluator,
)

__all__ = [
    # Document Parser
    "DocumentParser",
    "TextParser",
    "PDFParser",
    "ParserFactory",
    # PII Masker
    "PIIMasker",
    "PIIPattern",
    # Chunker
    "Chunk",
    "ChunkerConfig",
    "RecursiveChunker",
    "SentenceChunker",
    "FixedSizeChunker",
    "ChunkerFactory",
    # Ingestion
    "IngestionService",
    "IngestionServiceBuilder",
    "DocumentMetadata",
    "ProcessedDocument",
    # Embedder
    "EmbeddingResult",
    "OpenAIEmbedder",
    "MockEmbedder",
    "EmbedderFactory",
    # Vector Store
    "VectorDocument",
    "SearchResult",
    "QdrantConfig",
    "QdrantVectorStore",
    "MockVectorStore",
    "VectorStoreFactory",
    # Retriever
    "Retriever",
    "RetrieverBuilder",
    "RetrieverConfig",
    "RetrievalResult",
    # RBAC
    "User",
    "RBACService",
    "RBACMiddleware",
    "ALL_STAFF_ROLE",
    # Evaluation
    "EvaluationQuery",
    "EvaluationResult",
    "AggregateResults",
    "EvaluationService",
    "RetrievalEvaluator",
]