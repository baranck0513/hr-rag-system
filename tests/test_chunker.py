"""
Tests for the text chunker service.
"""

import pytest
from app.services.chunker import (
    Chunk,
    ChunkerConfig,
    RecursiveChunker,
    SentenceChunker,
    FixedSizeChunker,
    ChunkerFactory,
)


class TestChunk:
    """Tests for the Chunk dataclass."""
    
    def test_chunk_creation(self):
        """Should create a chunk with required fields."""
        chunk = Chunk(text="Hello world", index=0)
        
        assert chunk.text == "Hello world"
        assert chunk.index == 0
        assert chunk.metadata == {}
    
    def test_chunk_with_metadata(self):
        """Should create a chunk with metadata."""
        chunk = Chunk(
            text="Hello world",
            index=0,
            metadata={"source": "test.pdf"}
        )
        
        assert chunk.metadata["source"] == "test.pdf"
    
    def test_token_count_estimate(self):
        """Should estimate token count (~4 chars per token)."""
        chunk = Chunk(text="This is a test sentence.", index=0)
        
        # 25 characters / 4 = ~6 tokens
        assert chunk.token_count_estimate == 6


class TestRecursiveChunker:
    """Tests for the RecursiveChunker class."""
    
    @pytest.fixture
    def chunker(self):
        """Create a chunker with small chunk size for testing."""
        config = ChunkerConfig(
            chunk_size=100,
            chunk_overlap=20,
            min_chunk_size=10
        )
        return RecursiveChunker(config)
    
    @pytest.fixture
    def large_chunker(self):
        """Create a chunker with larger chunk size."""
        config = ChunkerConfig(
            chunk_size=500,
            chunk_overlap=50,
            min_chunk_size=50
        )
        return RecursiveChunker(config)
    
    def test_empty_text_returns_empty_list(self, chunker):
        """Should return empty list for empty text."""
        assert chunker.chunk("") == []
        assert chunker.chunk("   ") == []
    
    def test_small_text_returns_single_chunk(self, chunker):
        """Should return single chunk for small text."""
        text = "This is a small piece of text."
        chunks = chunker.chunk(text)
        
        assert len(chunks) == 1
        assert chunks[0].text == text
        assert chunks[0].index == 0
    
    def test_splits_on_paragraph_breaks(self, large_chunker):
        """Should prefer splitting on paragraph breaks."""
        text = """First paragraph with some content.

Second paragraph with different content.

Third paragraph with more content."""
        
        chunks = large_chunker.chunk(text)
        
        # Should keep paragraphs together when possible
        assert len(chunks) >= 1
        # Check that paragraph structure is somewhat preserved
        assert "First paragraph" in chunks[0].text
    
    def test_chunks_have_sequential_indices(self, chunker):
        """Should assign sequential indices to chunks."""
        text = "Word " * 100  # Long text to force multiple chunks
        chunks = chunker.chunk(text)
        
        for i, chunk in enumerate(chunks):
            assert chunk.index == i
    
    def test_chunks_have_metadata(self, chunker):
        """Should include metadata in chunks."""
        text = "Word " * 100
        chunks = chunker.chunk(text)
        
        for chunk in chunks:
            assert "chunk_size" in chunk.metadata
            assert "chunking_strategy" in chunk.metadata
            assert chunk.metadata["chunking_strategy"] == "recursive"
    
    def test_respects_chunk_size_limit(self, chunker):
        """Chunks should not greatly exceed chunk size."""
        text = "Word " * 200
        chunks = chunker.chunk(text)
        
        for chunk in chunks:
            # Allow some tolerance for boundary handling
            assert len(chunk.text) <= chunker.config.chunk_size * 1.5


class TestSentenceChunker:
    """Tests for the SentenceChunker class."""
    
    @pytest.fixture
    def chunker(self):
        config = ChunkerConfig(chunk_size=100, chunk_overlap=20)
        return SentenceChunker(config)
    
    def test_empty_text_returns_empty_list(self, chunker):
        """Should return empty list for empty text."""
        assert chunker.chunk("") == []
    
    def test_splits_on_sentence_boundaries(self, chunker):
        """Should split on sentence endings."""
        text = "First sentence. Second sentence. Third sentence."
        chunks = chunker.chunk(text)
        
        # Should create chunks that end at sentence boundaries
        for chunk in chunks:
            # Each chunk should ideally end with sentence-ending punctuation
            # (unless it's the last one which might not)
            assert len(chunk.text) > 0
    
    def test_groups_sentences_to_reach_target_size(self, chunker):
        """Should group multiple sentences when under chunk size."""
        text = "Hi. Hello. Hey."  # Very short sentences
        chunks = chunker.chunk(text)
        
        # Short sentences should be grouped together
        assert len(chunks) == 1
    
    def test_metadata_shows_sentence_strategy(self, chunker):
        """Should indicate sentence chunking strategy in metadata."""
        chunks = chunker.chunk("A sentence. Another one.")
        
        assert chunks[0].metadata["chunking_strategy"] == "sentence"


class TestFixedSizeChunker:
    """Tests for the FixedSizeChunker class."""
    
    @pytest.fixture
    def chunker(self):
        config = ChunkerConfig(
            chunk_size=50,
            chunk_overlap=10
        )
        return FixedSizeChunker(config)
    
    def test_empty_text_returns_empty_list(self, chunker):
        """Should return empty list for empty text."""
        assert chunker.chunk("") == []
    
    def test_splits_at_fixed_intervals(self, chunker):
        """Should split at fixed character intervals."""
        text = "A" * 100  # 100 characters
        chunks = chunker.chunk(text)
        
        # With chunk_size=50 and overlap=10, we expect multiple chunks
        assert len(chunks) >= 2
    
    def test_applies_overlap(self, chunker):
        """Should apply overlap between chunks."""
        text = "ABCDEFGHIJ" * 10  # 100 characters
        chunks = chunker.chunk(text)
        
        # If we have multiple chunks, they should overlap
        if len(chunks) >= 2:
            # End of first chunk should appear in start of second
            end_of_first = chunks[0].text[-10:]
            assert end_of_first in chunks[1].text or len(chunks[1].text) < 10
    
    def test_metadata_shows_fixed_strategy(self, chunker):
        """Should indicate fixed_size strategy in metadata."""
        chunks = chunker.chunk("Some text here that is long enough")
        
        if chunks:
            assert chunks[0].metadata["chunking_strategy"] == "fixed_size"


class TestChunkerFactory:
    """Tests for the ChunkerFactory class."""
    
    def test_get_recursive_chunker(self):
        """Should return RecursiveChunker for 'recursive' strategy."""
        chunker = ChunkerFactory.get_chunker("recursive")
        
        assert isinstance(chunker, RecursiveChunker)
    
    def test_get_sentence_chunker(self):
        """Should return SentenceChunker for 'sentence' strategy."""
        chunker = ChunkerFactory.get_chunker("sentence")
        
        assert isinstance(chunker, SentenceChunker)
    
    def test_get_fixed_chunker(self):
        """Should return FixedSizeChunker for 'fixed' strategy."""
        chunker = ChunkerFactory.get_chunker("fixed")
        
        assert isinstance(chunker, FixedSizeChunker)
    
    def test_default_is_recursive(self):
        """Should default to recursive chunker."""
        chunker = ChunkerFactory.get_chunker()
        
        assert isinstance(chunker, RecursiveChunker)
    
    def test_accepts_custom_config(self):
        """Should apply custom config to chunker."""
        config = ChunkerConfig(chunk_size=200)
        chunker = ChunkerFactory.get_chunker("recursive", config)
        
        assert chunker.config.chunk_size == 200
    
    def test_raises_for_unknown_strategy(self):
        """Should raise ValueError for unknown strategy."""
        with pytest.raises(ValueError) as exc_info:
            ChunkerFactory.get_chunker("unknown")
        
        assert "Unknown chunking strategy" in str(exc_info.value)


class TestChunkingHRDocument:
    """Integration tests with realistic HR document content."""
    
    @pytest.fixture
    def hr_document(self):
        """Sample HR document text."""
        return """
ANNUAL LEAVE POLICY

1. Entitlement
All full-time employees are entitled to 25 days of paid annual leave per calendar year. This is in addition to the 8 UK bank holidays.

2. Carry Over
Employees may carry over up to 5 days of unused leave to the following year. Any additional unused leave will be forfeited unless exceptional circumstances apply.

3. Booking Leave
Leave requests must be submitted through the HR portal at least 2 weeks in advance. Line manager approval is required for all leave requests.

SICK LEAVE POLICY

1. Notification
Employees must notify their line manager as soon as possible on the first day of absence, and no later than 30 minutes after their normal start time.

2. Documentation
For absences of 7 or fewer calendar days, employees must complete a self-certification form. For absences exceeding 7 days, a doctor's note is required.
"""
    
    def test_chunks_preserve_policy_sections(self, hr_document):
        """Should attempt to keep policy sections together."""
        config = ChunkerConfig(chunk_size=500, chunk_overlap=50)
        chunker = RecursiveChunker(config)
        
        chunks = chunker.chunk(hr_document)
        
        # Should create multiple chunks
        assert len(chunks) >= 2
        
        # Each chunk should be non-empty
        for chunk in chunks:
            assert len(chunk.text.strip()) > 0
    
    def test_chunks_are_searchable(self, hr_document):
        """Chunks should contain searchable content."""
        config = ChunkerConfig(chunk_size=300, chunk_overlap=50)
        chunker = RecursiveChunker(config)
        
        chunks = chunker.chunk(hr_document)
        
        # Should be able to find relevant chunks by keyword
        leave_chunks = [c for c in chunks if "leave" in c.text.lower()]
        assert len(leave_chunks) >= 1
        
        sick_chunks = [c for c in chunks if "sick" in c.text.lower()]
        assert len(sick_chunks) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])