"""
Ingestion Service

Orchestrates the complete document ingestion pipeline:
1. Parse document (extract text from PDF/TXT/MD)
2. Mask PII (remove sensitive information)
3. Chunk text (split into optimal pieces for embedding)

This service acts as the main entry point for document processing,
coordinating the individual services and managing the flow of data.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import logging
import hashlib

from .document_parser import ParserFactory, DocumentParser
from .pii_masker import PIIMasker
from .chunker import (
    Chunk,
    ChunkerConfig,
    ChunkerFactory,
    RecursiveChunker,
)

logger = logging.getLogger(__name__)


@dataclass
class DocumentMetadata:
    """
    Metadata about an ingested document.
    
    This information is stored in MongoDB alongside the document
    for tracking, filtering, and access control purposes.
    """
    document_id: str
    filename: str
    file_type: str
    file_size_bytes: int
    uploaded_at: datetime
    uploaded_by: Optional[str] = None
    department: Optional[str] = None
    access_roles: list[str] = field(default_factory=list)
    chunk_count: int = 0
    pii_detected: dict[str, int] = field(default_factory=dict)
    processing_time_ms: float = 0.0


@dataclass
class ProcessedDocument:
    """
    Result of document ingestion.
    
    Contains the chunks ready for embedding, plus metadata
    for storage and tracking.
    """
    metadata: DocumentMetadata
    chunks: list[Chunk]
    
    @property
    def total_characters(self) -> int:
        """Total characters across all chunks."""
        return sum(len(chunk.text) for chunk in self.chunks)


class IngestionService:
    """
    Main service for document ingestion.
    
    Coordinates parsing, PII masking, and chunking into a single
    pipeline. Designed for use with HR documents in a UK context.
    
    Usage:
        service = IngestionService()
        result = service.ingest(
            content=file_bytes,
            filename="leave_policy.pdf",
            uploaded_by="hr_admin",
            department="HR",
            access_roles=["all_staff"]
        )
        
        # result.chunks are ready for embedding
        # result.metadata should be stored in MongoDB
    
    Example:
        >>> service = IngestionService()
        >>> result = service.ingest(b"Employee NI: AB123456C", "test.txt")
        >>> result.chunks[0].text
        'Employee NI: [NI_NUMBER]'
    """
    
    def __init__(
        self,
        chunker_config: Optional[ChunkerConfig] = None,
        chunking_strategy: str = "recursive"
    ):
        """
        Initialise the ingestion service.
        
        Args:
            chunker_config: Configuration for text chunking.
                           Uses sensible defaults if not provided.
            chunking_strategy: Which chunking strategy to use.
                              Options: "recursive", "sentence", "fixed"
        """
        self.pii_masker = PIIMasker()
        self.chunker_config = chunker_config or ChunkerConfig(
            chunk_size=1000,
            chunk_overlap=200,
            min_chunk_size=100
        )
        self.chunker = ChunkerFactory.get_chunker(
            chunking_strategy, 
            self.chunker_config
        )
        
        logger.info(
            f"IngestionService initialised with {chunking_strategy} chunking"
        )
    
    def ingest(
        self,
        content: bytes,
        filename: str,
        uploaded_by: Optional[str] = None,
        department: Optional[str] = None,
        access_roles: Optional[list[str]] = None
    ) -> ProcessedDocument:
        """
        Process a document through the complete ingestion pipeline.
        
        Args:
            content: Raw bytes of the document file
            filename: Original filename (used to determine parser)
            uploaded_by: User ID of the uploader (for audit trail)
            department: Department the document belongs to
            access_roles: List of roles that can access this document
            
        Returns:
            ProcessedDocument containing chunks and metadata
            
        Raises:
            ValueError: If file type is not supported
        """
        import time
        start_time = time.time()
        
        logger.info(f"Starting ingestion of {filename}")
        
        # Generate document ID from content hash
        document_id = self._generate_document_id(content, filename)
        
        # Step 1: Parse document
        logger.debug(f"Step 1: Parsing {filename}")
        parser = ParserFactory.get_parser(filename)
        raw_text = parser.parse(content, filename)
        logger.info(f"Parsed {len(raw_text)} characters from {filename}")
        
        # Step 2: Mask PII
        logger.debug("Step 2: Masking PII")
        masked_text, pii_stats = self.pii_masker.mask_with_stats(raw_text)
        if pii_stats:
            logger.info(f"Masked PII: {pii_stats}")
        
        # Step 3: Chunk text
        logger.debug("Step 3: Chunking text")
        chunks = self.chunker.chunk(masked_text)
        logger.info(f"Created {len(chunks)} chunks")
        
        # Add document reference to chunk metadata
        for chunk in chunks:
            chunk.metadata["document_id"] = document_id
            chunk.metadata["filename"] = filename
        
        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Build metadata
        metadata = DocumentMetadata(
            document_id=document_id,
            filename=filename,
            file_type=filename.split(".")[-1].lower(),
            file_size_bytes=len(content),
            uploaded_at=datetime.utcnow(),
            uploaded_by=uploaded_by,
            department=department,
            access_roles=access_roles or [],
            chunk_count=len(chunks),
            pii_detected=pii_stats,
            processing_time_ms=processing_time_ms
        )
        
        logger.info(
            f"Completed ingestion of {filename} in {processing_time_ms:.2f}ms"
        )
        
        return ProcessedDocument(metadata=metadata, chunks=chunks)
    
    def _generate_document_id(self, content: bytes, filename: str) -> str:
        """
        Generate a unique document ID based on content hash.
        
        Using content hash means the same document uploaded twice
        will get the same ID (useful for deduplication).
        
        Args:
            content: Document bytes
            filename: Filename (included in hash for uniqueness)
            
        Returns:
            Hex string document ID
        """
        hasher = hashlib.sha256()
        hasher.update(content)
        hasher.update(filename.encode("utf-8"))
        return hasher.hexdigest()[:16]  # First 16 chars is enough


class IngestionServiceBuilder:
    """
    Builder pattern for creating customised IngestionService instances.
    
    Useful when you need fine-grained control over configuration.
    
    Usage:
        service = (
            IngestionServiceBuilder()
            .with_chunk_size(500)
            .with_chunk_overlap(100)
            .with_strategy("sentence")
            .build()
        )
    """
    
    def __init__(self):
        self._chunk_size = 1000
        self._chunk_overlap = 200
        self._min_chunk_size = 100
        self._strategy = "recursive"
    
    def with_chunk_size(self, size: int) -> "IngestionServiceBuilder":
        """Set the target chunk size in characters."""
        self._chunk_size = size
        return self
    
    def with_chunk_overlap(self, overlap: int) -> "IngestionServiceBuilder":
        """Set the overlap between chunks in characters."""
        self._chunk_overlap = overlap
        return self
    
    def with_min_chunk_size(self, size: int) -> "IngestionServiceBuilder":
        """Set the minimum chunk size in characters."""
        self._min_chunk_size = size
        return self
    
    def with_strategy(self, strategy: str) -> "IngestionServiceBuilder":
        """Set the chunking strategy (recursive, sentence, fixed)."""
        self._strategy = strategy
        return self
    
    def build(self) -> IngestionService:
        """Build and return the configured IngestionService."""
        config = ChunkerConfig(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            min_chunk_size=self._min_chunk_size
        )
        return IngestionService(
            chunker_config=config,
            chunking_strategy=self._strategy
        )