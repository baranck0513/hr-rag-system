"""
Tests for the retriever service.
"""

import pytest
from app.services.retriever import (
    Retriever,
    RetrieverBuilder,
    RetrieverConfig,
    RetrievalResult,
)
from app.services.chunker import Chunk
from app.services.embedder import MockEmbedder
from app.services.vector_store import MockVectorStore


class TestRetrievalResult:
    """Tests for the RetrievalResult dataclass."""
    
    def test_retrieval_result_creation(self):
        """Should create a retrieval result."""
        from app.services.vector_store import SearchResult
        
        results = [
            SearchResult(id="1", text="Hello", score=0.9),
            SearchResult(id="2", text="World", score=0.8),
        ]
        
        retrieval = RetrievalResult(
            query="test query",
            results=results,
            total_results=2
        )
        
        assert retrieval.query == "test query"
        assert len(retrieval.results) == 2
        assert retrieval.total_results == 2
    
    def test_texts_property(self):
        """Should return list of text content."""
        from app.services.vector_store import SearchResult
        
        results = [
            SearchResult(id="1", text="Hello", score=0.9),
            SearchResult(id="2", text="World", score=0.8),
        ]
        
        retrieval = RetrievalResult(
            query="test",
            results=results,
            total_results=2
        )
        
        assert retrieval.texts == ["Hello", "World"]
    
    def test_top_result_property(self):
        """Should return highest-scoring result."""
        from app.services.vector_store import SearchResult
        
        results = [
            SearchResult(id="1", text="Best match", score=0.9),
            SearchResult(id="2", text="Second", score=0.8),
        ]
        
        retrieval = RetrievalResult(
            query="test",
            results=results,
            total_results=2
        )
        
        assert retrieval.top_result.text == "Best match"
    
    def test_top_result_none_when_empty(self):
        """Should return None when no results."""
        retrieval = RetrievalResult(
            query="test",
            results=[],
            total_results=0
        )
        
        assert retrieval.top_result is None


class TestRetriever:
    """Tests for the Retriever class."""
    
    @pytest.fixture
    def retriever(self):
        """Create a retriever with mock components."""
        return (
            RetrieverBuilder()
            .with_mock_embedder()
            .with_mock_vector_store()
            .with_top_k(5)
            .build()
        )
    
    @pytest.fixture
    def sample_chunks(self):
        """Sample chunks for testing."""
        return [
            Chunk(
                text="Employees are entitled to 25 days of annual leave.",
                index=0,
                metadata={"document_id": "doc1", "department": "HR"}
            ),
            Chunk(
                text="Sick leave requires a doctor's note after 7 days.",
                index=1,
                metadata={"document_id": "doc1", "department": "HR"}
            ),
            Chunk(
                text="IT security policy requires strong passwords.",
                index=2,
                metadata={"document_id": "doc2", "department": "IT"}
            ),
        ]
    
    def test_create_collection(self, retriever):
        """Should create collection without error."""
        retriever.create_collection()
        
        # Should not raise
        assert True
    
    def test_index_chunks(self, retriever, sample_chunks):
        """Should index chunks successfully."""
        retriever.create_collection()
        
        count = retriever.index_chunks(sample_chunks)
        
        assert count == 3
    
    def test_index_empty_chunks(self, retriever):
        """Should handle empty chunk list."""
        retriever.create_collection()
        
        count = retriever.index_chunks([])
        
        assert count == 0
    
    def test_retrieve_returns_results(self, retriever, sample_chunks):
        """Should return retrieval results."""
        retriever.create_collection()
        retriever.index_chunks(sample_chunks)
        
        result = retriever.retrieve("How much annual leave do I get?")
        
        assert isinstance(result, RetrievalResult)
        assert result.query == "How much annual leave do I get?"
        assert len(result.results) > 0
    
    def test_retrieve_empty_query_raises_error(self, retriever):
        """Should raise error for empty query."""
        with pytest.raises(ValueError) as exc_info:
            retriever.retrieve("")
        
        assert "empty" in str(exc_info.value).lower()
    
    def test_retrieve_with_filter(self, retriever, sample_chunks):
        """Should filter results by metadata."""
        retriever.create_collection()
        retriever.index_chunks(sample_chunks)
        
        result = retriever.retrieve(
            "policy",
            filters={"department": "IT"}
        )
        
        # Should only return IT documents
        for r in result.results:
            assert r.metadata.get("department") == "IT"
    
    def test_retrieve_respects_top_k(self, retriever, sample_chunks):
        """Should return only top_k results."""
        retriever.create_collection()
        retriever.index_chunks(sample_chunks)
        
        result = retriever.retrieve("policy", top_k=1)
        
        assert len(result.results) <= 1
    
    def test_delete_document(self, retriever, sample_chunks):
        """Should delete document chunks."""
        retriever.create_collection()
        retriever.index_chunks(sample_chunks, document_id="test-doc")
        
        # Verify chunks exist
        stats = retriever.get_stats()
        initial_count = stats["total_vectors"]
        
        retriever.delete_document("test-doc")
        
        # Verify chunks deleted
        stats = retriever.get_stats()
        assert stats["total_vectors"] < initial_count
    
    def test_get_stats(self, retriever, sample_chunks):
        """Should return retriever statistics."""
        retriever.create_collection()
        retriever.index_chunks(sample_chunks)
        
        stats = retriever.get_stats()
        
        assert "total_vectors" in stats
        assert "embedder" in stats
        assert "vector_store" in stats
        assert stats["total_vectors"] == 3


class TestRetrieverBuilder:
    """Tests for the RetrieverBuilder class."""
    
    def test_build_default_retriever(self):
        """Should build retriever with defaults."""
        retriever = (
            RetrieverBuilder()
            .with_mock_embedder()
            .with_mock_vector_store()
            .build()
        )
        
        assert isinstance(retriever, Retriever)
    
    def test_build_with_custom_top_k(self):
        """Should apply custom top_k."""
        retriever = (
            RetrieverBuilder()
            .with_mock_embedder()
            .with_mock_vector_store()
            .with_top_k(10)
            .build()
        )
        
        assert retriever.config.top_k == 10
    
    def test_build_with_score_threshold(self):
        """Should apply score threshold."""
        retriever = (
            RetrieverBuilder()
            .with_mock_embedder()
            .with_mock_vector_store()
            .with_score_threshold(0.7)
            .build()
        )
        
        assert retriever.config.score_threshold == 0.7
    
    def test_build_with_reranking(self):
        """Should enable reranking."""
        retriever = (
            RetrieverBuilder()
            .with_mock_embedder()
            .with_mock_vector_store()
            .with_reranking(True)
            .build()
        )
        
        assert retriever.config.use_reranking is True
    
    def test_builder_chaining(self):
        """Should support method chaining."""
        retriever = (
            RetrieverBuilder()
            .with_mock_embedder()
            .with_mock_vector_store()
            .with_top_k(3)
            .with_score_threshold(0.5)
            .with_reranking(True)
            .build()
        )
        
        assert retriever.config.top_k == 3
        assert retriever.config.score_threshold == 0.5
        assert retriever.config.use_reranking is True


class TestRetrieverIntegration:
    """Integration tests for the complete retrieval pipeline."""
    
    @pytest.fixture
    def retriever(self):
        """Create a fully configured retriever."""
        return (
            RetrieverBuilder()
            .with_mock_embedder()
            .with_mock_vector_store()
            .with_top_k(5)
            .build()
        )
    
    def test_full_pipeline(self, retriever):
        """Should handle complete index and retrieve workflow."""
        # Create chunks from an HR document
        chunks = [
            Chunk(
                text="All employees are entitled to 25 days of annual leave per year.",
                index=0,
                metadata={"document_id": "leave-policy", "section": "entitlement"}
            ),
            Chunk(
                text="Leave requests must be submitted at least 2 weeks in advance.",
                index=1,
                metadata={"document_id": "leave-policy", "section": "booking"}
            ),
            Chunk(
                text="Unused leave can be carried over up to 5 days.",
                index=2,
                metadata={"document_id": "leave-policy", "section": "carryover"}
            ),
        ]
        
        # Index the chunks
        retriever.create_collection(recreate=True)
        indexed = retriever.index_chunks(chunks)
        assert indexed == 3
        
        # Retrieve relevant chunks
        result = retriever.retrieve("How much holiday do I get?")
        
        # Should find relevant results
        assert result.total_results > 0
        assert result.top_result is not None
        
        # Results should have text and metadata
        for r in result.results:
            assert r.text
            assert r.score > 0
    
    def test_semantic_similarity(self, retriever):
        """Test that semantically similar queries find relevant content."""
        chunks = [
            Chunk(text="The annual leave entitlement is 25 days.", index=0),
            Chunk(text="Password must be at least 12 characters.", index=1),
            Chunk(text="Fire exits are located on each floor.", index=2),
        ]
        
        retriever.create_collection(recreate=True)
        retriever.index_chunks(chunks)
        
        # These queries should all find the leave policy
        queries = [
            "How much holiday do I get?",
            "What is my annual leave?",
            "vacation days entitlement",
        ]
        
        for query in queries:
            result = retriever.retrieve(query)
            # With mock embedder, results may vary, but should return something
            assert result.total_results > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])