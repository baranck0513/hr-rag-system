"""
Tests for the ingestion service.
"""

import pytest
from datetime import datetime
from app.services.ingestion import (
    IngestionService,
    IngestionServiceBuilder,
    DocumentMetadata,
    ProcessedDocument,
)


class TestDocumentMetadata:
    """Tests for the DocumentMetadata dataclass."""
    
    def test_metadata_creation(self):
        """Should create metadata with required fields."""
        metadata = DocumentMetadata(
            document_id="abc123",
            filename="test.pdf",
            file_type="pdf",
            file_size_bytes=1024,
            uploaded_at=datetime.utcnow()
        )
        
        assert metadata.document_id == "abc123"
        assert metadata.filename == "test.pdf"
        assert metadata.file_type == "pdf"
        assert metadata.file_size_bytes == 1024
    
    def test_metadata_optional_fields(self):
        """Should handle optional fields with defaults."""
        metadata = DocumentMetadata(
            document_id="abc123",
            filename="test.pdf",
            file_type="pdf",
            file_size_bytes=1024,
            uploaded_at=datetime.utcnow()
        )
        
        assert metadata.uploaded_by is None
        assert metadata.department is None
        assert metadata.access_roles == []
        assert metadata.pii_detected == {}


class TestProcessedDocument:
    """Tests for the ProcessedDocument dataclass."""
    
    def test_total_characters(self):
        """Should calculate total characters across chunks."""
        from app.services.chunker import Chunk
        
        metadata = DocumentMetadata(
            document_id="abc123",
            filename="test.pdf",
            file_type="pdf",
            file_size_bytes=1024,
            uploaded_at=datetime.utcnow()
        )
        
        chunks = [
            Chunk(text="Hello", index=0),  # 5 chars
            Chunk(text="World", index=1),  # 5 chars
        ]
        
        doc = ProcessedDocument(metadata=metadata, chunks=chunks)
        
        assert doc.total_characters == 10


class TestIngestionService:
    """Tests for the IngestionService class."""
    
    @pytest.fixture
    def service(self):
        """Create an ingestion service with default settings."""
        return IngestionService()
    
    @pytest.fixture
    def small_chunk_service(self):
        """Create an ingestion service with small chunk size for testing."""
        return IngestionServiceBuilder() \
            .with_chunk_size(100) \
            .with_chunk_overlap(20) \
            .build()
    
    # --- Basic Functionality Tests ---
    
    def test_ingest_simple_text(self, service):
        """Should ingest a simple text file."""
        content = b"This is a simple test document."
        filename = "test.txt"
        
        result = service.ingest(content, filename)
        
        assert result.metadata.filename == filename
        assert result.metadata.file_type == "txt"
        assert len(result.chunks) >= 1
        assert "simple test document" in result.chunks[0].text
    
    def test_ingest_returns_processed_document(self, service):
        """Should return a ProcessedDocument instance."""
        content = b"Test content"
        
        result = service.ingest(content, "test.txt")
        
        assert isinstance(result, ProcessedDocument)
        assert isinstance(result.metadata, DocumentMetadata)
    
    def test_ingest_sets_document_id(self, service):
        """Should generate a document ID."""
        content = b"Test content"
        
        result = service.ingest(content, "test.txt")
        
        assert result.metadata.document_id is not None
        assert len(result.metadata.document_id) == 16
    
    def test_same_content_same_id(self, service):
        """Same content should generate same document ID."""
        content = b"Identical content"
        
        result1 = service.ingest(content, "test.txt")
        result2 = service.ingest(content, "test.txt")
        
        assert result1.metadata.document_id == result2.metadata.document_id
    
    def test_different_content_different_id(self, service):
        """Different content should generate different document IDs."""
        result1 = service.ingest(b"Content A", "test.txt")
        result2 = service.ingest(b"Content B", "test.txt")
        
        assert result1.metadata.document_id != result2.metadata.document_id
    
    # --- PII Masking Tests ---
    
    def test_ingest_masks_pii(self, service):
        """Should mask PII in the document."""
        content = b"Employee NI: AB123456C, Email: john@test.com"
        
        result = service.ingest(content, "employee.txt")
        
        # Check PII was masked
        chunk_text = result.chunks[0].text
        assert "AB123456C" not in chunk_text
        assert "[NI_NUMBER]" in chunk_text
        assert "john@test.com" not in chunk_text
        assert "[EMAIL]" in chunk_text
    
    def test_ingest_tracks_pii_stats(self, service):
        """Should track what PII was detected."""
        content = b"NI: AB123456C, AB654321D. Email: a@b.com"
        
        result = service.ingest(content, "test.txt")
        
        assert "NI_NUMBER" in result.metadata.pii_detected
        assert result.metadata.pii_detected["NI_NUMBER"] == 2
        assert "EMAIL" in result.metadata.pii_detected
        assert result.metadata.pii_detected["EMAIL"] == 1
    
    # --- Chunking Tests ---
    
    def test_ingest_creates_chunks(self, small_chunk_service):
        """Should create multiple chunks for large documents."""
        # Create content larger than chunk size
        content = ("This is a test sentence. " * 50).encode("utf-8")
        
        result = small_chunk_service.ingest(content, "large.txt")
        
        assert len(result.chunks) > 1
    
    def test_chunks_have_document_metadata(self, service):
        """Chunks should include document reference in metadata."""
        content = b"Test content for chunking"
        
        result = service.ingest(content, "test.txt")
        
        for chunk in result.chunks:
            assert "document_id" in chunk.metadata
            assert "filename" in chunk.metadata
            assert chunk.metadata["filename"] == "test.txt"
    
    # --- Metadata Tests ---
    
    def test_ingest_with_optional_metadata(self, service):
        """Should store optional metadata."""
        content = b"Test content"
        
        result = service.ingest(
            content,
            "policy.txt",
            uploaded_by="user123",
            department="HR",
            access_roles=["hr_team", "managers"]
        )
        
        assert result.metadata.uploaded_by == "user123"
        assert result.metadata.department == "HR"
        assert "hr_team" in result.metadata.access_roles
        assert "managers" in result.metadata.access_roles
    
    def test_ingest_records_file_size(self, service):
        """Should record the file size in bytes."""
        content = b"x" * 500
        
        result = service.ingest(content, "test.txt")
        
        assert result.metadata.file_size_bytes == 500
    
    def test_ingest_records_chunk_count(self, service):
        """Should record the number of chunks created."""
        content = b"Test content"
        
        result = service.ingest(content, "test.txt")
        
        assert result.metadata.chunk_count == len(result.chunks)
    
    def test_ingest_records_processing_time(self, service):
        """Should record processing time."""
        content = b"Test content"
        
        result = service.ingest(content, "test.txt")
        
        assert result.metadata.processing_time_ms > 0
    
    def test_ingest_records_timestamp(self, service):
        """Should record upload timestamp."""
        content = b"Test content"
        before = datetime.utcnow()
        
        result = service.ingest(content, "test.txt")
        
        after = datetime.utcnow()
        assert before <= result.metadata.uploaded_at <= after
    
    # --- Error Handling Tests ---
    
    def test_ingest_unsupported_file_type(self, service):
        """Should raise error for unsupported file types."""
        content = b"Test content"
        
        with pytest.raises(ValueError) as exc_info:
            service.ingest(content, "document.docx")
        
        assert "Unsupported file type" in str(exc_info.value)


class TestIngestionServiceBuilder:
    """Tests for the IngestionServiceBuilder class."""
    
    def test_build_with_defaults(self):
        """Should create service with default settings."""
        service = IngestionServiceBuilder().build()
        
        assert isinstance(service, IngestionService)
    
    def test_build_with_custom_chunk_size(self):
        """Should apply custom chunk size."""
        service = IngestionServiceBuilder() \
            .with_chunk_size(500) \
            .build()
        
        assert service.chunker_config.chunk_size == 500
    
    def test_build_with_custom_overlap(self):
        """Should apply custom overlap."""
        service = IngestionServiceBuilder() \
            .with_chunk_overlap(50) \
            .build()
        
        assert service.chunker_config.chunk_overlap == 50
    
    def test_build_with_custom_strategy(self):
        """Should apply custom chunking strategy."""
        from app.services.chunker import SentenceChunker
        
        service = IngestionServiceBuilder() \
            .with_strategy("sentence") \
            .build()
        
        assert isinstance(service.chunker, SentenceChunker)
    
    def test_builder_chaining(self):
        """Should support method chaining."""
        service = IngestionServiceBuilder() \
            .with_chunk_size(500) \
            .with_chunk_overlap(100) \
            .with_min_chunk_size(50) \
            .with_strategy("recursive") \
            .build()
        
        assert service.chunker_config.chunk_size == 500
        assert service.chunker_config.chunk_overlap == 100
        assert service.chunker_config.min_chunk_size == 50


class TestIngestionPipeline:
    """Integration tests for the complete ingestion pipeline."""
    
    @pytest.fixture
    def service(self):
        return IngestionService()
    
    def test_full_pipeline_with_hr_document(self, service):
        """Should process a realistic HR document."""
        content = b"""
ANNUAL LEAVE POLICY

Employee: John Smith
NI Number: AB123456C
Email: john.smith@company.com

1. Entitlement
All employees receive 25 days annual leave.

2. Booking Process
Submit requests via HR portal.
Contact HR at hr@company.com for queries.
"""
        
        result = service.ingest(
            content,
            "leave_policy.txt",
            department="HR",
            access_roles=["all_staff"]
        )
        
        # Check PII was masked
        full_text = " ".join(c.text for c in result.chunks)
        assert "AB123456C" not in full_text
        assert "john.smith@company.com" not in full_text
        assert "[NI_NUMBER]" in full_text
        assert "[EMAIL]" in full_text
        
        # Check metadata
        assert result.metadata.department == "HR"
        assert "all_staff" in result.metadata.access_roles
        assert result.metadata.pii_detected.get("NI_NUMBER", 0) >= 1
        assert result.metadata.pii_detected.get("EMAIL", 0) >= 1
        
        # Check chunks were created
        assert len(result.chunks) >= 1
        assert result.metadata.chunk_count == len(result.chunks)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])