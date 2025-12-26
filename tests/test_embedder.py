"""
Tests for the embedder service.
"""

import pytest
from app.services.embedder import (
    EmbeddingResult,
    MockEmbedder,
    EmbedderFactory,
)


class TestEmbeddingResult:
    """Tests for the EmbeddingResult dataclass."""
    
    def test_embedding_result_creation(self):
        """Should create an embedding result."""
        result = EmbeddingResult(
            text="Hello world",
            vector=[0.1, 0.2, 0.3],
            model="test-model",
            dimensions=3
        )
        
        assert result.text == "Hello world"
        assert result.vector == [0.1, 0.2, 0.3]
        assert result.model == "test-model"
        assert result.dimensions == 3
    
    def test_text_hash_is_consistent(self):
        """Same text should produce same hash."""
        result1 = EmbeddingResult(
            text="Hello",
            vector=[0.1],
            model="test",
            dimensions=1
        )
        result2 = EmbeddingResult(
            text="Hello",
            vector=[0.2],  # Different vector
            model="test",
            dimensions=1
        )
        
        assert result1.text_hash == result2.text_hash
    
    def test_text_hash_differs_for_different_text(self):
        """Different text should produce different hash."""
        result1 = EmbeddingResult(
            text="Hello",
            vector=[0.1],
            model="test",
            dimensions=1
        )
        result2 = EmbeddingResult(
            text="World",
            vector=[0.1],
            model="test",
            dimensions=1
        )
        
        assert result1.text_hash != result2.text_hash


class TestMockEmbedder:
    """Tests for the MockEmbedder class."""
    
    @pytest.fixture
    def embedder(self):
        return MockEmbedder(dimensions=128)
    
    def test_embed_returns_result(self, embedder):
        """Should return an EmbeddingResult."""
        result = embedder.embed("test text")
        
        assert isinstance(result, EmbeddingResult)
        assert result.text == "test text"
        assert len(result.vector) == 128
        assert result.dimensions == 128
    
    def test_embed_is_deterministic(self, embedder):
        """Same text should produce same embedding."""
        result1 = embedder.embed("Hello world")
        result2 = embedder.embed("Hello world")
        
        assert result1.vector == result2.vector
    
    def test_different_text_different_embedding(self, embedder):
        """Different text should produce different embeddings."""
        result1 = embedder.embed("Hello")
        result2 = embedder.embed("World")
        
        assert result1.vector != result2.vector
    
    def test_embed_empty_text_raises_error(self, embedder):
        """Should raise error for empty text."""
        with pytest.raises(ValueError) as exc_info:
            embedder.embed("")
        
        assert "empty" in str(exc_info.value).lower()
    
    def test_embed_whitespace_only_raises_error(self, embedder):
        """Should raise error for whitespace-only text."""
        with pytest.raises(ValueError):
            embedder.embed("   ")
    
    def test_embed_batch_returns_list(self, embedder):
        """Should embed multiple texts."""
        texts = ["Hello", "World", "Test"]
        results = embedder.embed_batch(texts)
        
        assert len(results) == 3
        assert all(isinstance(r, EmbeddingResult) for r in results)
    
    def test_embed_batch_skips_empty_texts(self, embedder):
        """Should skip empty texts in batch."""
        texts = ["Hello", "", "World"]
        results = embedder.embed_batch(texts)
        
        assert len(results) == 2
    
    def test_vector_values_in_valid_range(self, embedder):
        """Vector values should be between -1 and 1."""
        result = embedder.embed("test text")
        
        for value in result.vector:
            assert -1 <= value <= 1


class TestEmbedderFactory:
    """Tests for the EmbedderFactory class."""
    
    def test_create_mock_embedder(self):
        """Should create a MockEmbedder."""
        embedder = EmbedderFactory.create("mock")
        
        assert isinstance(embedder, MockEmbedder)
    
    def test_create_mock_with_custom_dimensions(self):
        """Should pass dimensions to MockEmbedder."""
        embedder = EmbedderFactory.create("mock", dimensions=256)
        result = embedder.embed("test")
        
        assert len(result.vector) == 256
    
    def test_create_unknown_provider_raises_error(self):
        """Should raise error for unknown provider."""
        with pytest.raises(ValueError) as exc_info:
            EmbedderFactory.create("unknown")
        
        assert "Unknown embedder provider" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])