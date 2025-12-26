"""
Tests for the vector store service.
"""

import pytest
from app.services.vector_store import (
    VectorDocument,
    SearchResult,
    QdrantConfig,
    MockVectorStore,
    VectorStoreFactory,
)


class TestVectorDocument:
    """Tests for the VectorDocument dataclass."""
    
    def test_vector_document_creation(self):
        """Should create a vector document."""
        doc = VectorDocument(
            id="doc-1",
            vector=[0.1, 0.2, 0.3],
            text="Hello world"
        )
        
        assert doc.id == "doc-1"
        assert doc.vector == [0.1, 0.2, 0.3]
        assert doc.text == "Hello world"
        assert doc.metadata == {}
    
    def test_vector_document_with_metadata(self):
        """Should create document with metadata."""
        doc = VectorDocument(
            id="doc-1",
            vector=[0.1],
            text="Hello",
            metadata={"department": "HR"}
        )
        
        assert doc.metadata["department"] == "HR"


class TestSearchResult:
    """Tests for the SearchResult dataclass."""
    
    def test_search_result_creation(self):
        """Should create a search result."""
        result = SearchResult(
            id="doc-1",
            text="Hello world",
            score=0.95
        )
        
        assert result.id == "doc-1"
        assert result.text == "Hello world"
        assert result.score == 0.95


class TestMockVectorStore:
    """Tests for the MockVectorStore class."""
    
    @pytest.fixture
    def store(self):
        """Create a fresh mock store for each test."""
        store = MockVectorStore()
        store.create_collection()
        return store
    
    @pytest.fixture
    def sample_docs(self):
        """Sample documents for testing."""
        return [
            VectorDocument(
                id="doc-1",
                vector=[1.0, 0.0, 0.0],  # Points in x direction
                text="Annual leave policy",
                metadata={"department": "HR"}
            ),
            VectorDocument(
                id="doc-2",
                vector=[0.0, 1.0, 0.0],  # Points in y direction
                text="Sick leave policy",
                metadata={"department": "HR"}
            ),
            VectorDocument(
                id="doc-3",
                vector=[0.0, 0.0, 1.0],  # Points in z direction
                text="IT security policy",
                metadata={"department": "IT"}
            ),
        ]
    
    def test_upsert_documents(self, store, sample_docs):
        """Should store documents."""
        count = store.upsert(sample_docs)
        
        assert count == 3
        assert store.count() == 3
    
    def test_upsert_empty_list(self, store):
        """Should handle empty list."""
        count = store.upsert([])
        
        assert count == 0
    
    def test_search_returns_results(self, store, sample_docs):
        """Should return search results."""
        store.upsert(sample_docs)
        
        # Search with vector similar to doc-1
        results = store.search(
            query_vector=[0.9, 0.1, 0.0],
            top_k=3
        )
        
        assert len(results) == 3
        assert all(isinstance(r, SearchResult) for r in results)
    
    def test_search_orders_by_similarity(self, store, sample_docs):
        """Should order results by similarity score."""
        store.upsert(sample_docs)
        
        # Search with vector pointing mostly in x direction
        results = store.search(
            query_vector=[0.9, 0.1, 0.0],
            top_k=3
        )
        
        # doc-1 should be most similar (both point in x direction)
        assert results[0].text == "Annual leave policy"
        assert results[0].score > results[1].score
    
    def test_search_respects_top_k(self, store, sample_docs):
        """Should return only top_k results."""
        store.upsert(sample_docs)
        
        results = store.search(
            query_vector=[1.0, 0.0, 0.0],
            top_k=1
        )
        
        assert len(results) == 1
    
    def test_search_with_filter(self, store, sample_docs):
        """Should filter results by metadata."""
        store.upsert(sample_docs)
        
        # Search only HR documents
        results = store.search(
            query_vector=[1.0, 0.0, 0.0],
            top_k=10,
            filters={"department": "HR"}
        )
        
        assert len(results) == 2
        assert all(r.metadata["department"] == "HR" for r in results)
    
    def test_search_with_score_threshold(self, store, sample_docs):
        """Should filter results below score threshold."""
        store.upsert(sample_docs)
        
        # Search with high threshold
        results = store.search(
            query_vector=[1.0, 0.0, 0.0],  # Exactly matches doc-1
            top_k=10,
            score_threshold=0.9
        )
        
        # Only doc-1 should match with high score
        assert len(results) <= 1
    
    def test_search_empty_store(self, store):
        """Should return empty list when store is empty."""
        results = store.search(
            query_vector=[1.0, 0.0, 0.0],
            top_k=5
        )
        
        assert results == []
    
    def test_delete_by_id(self, store, sample_docs):
        """Should delete documents by ID."""
        store.upsert(sample_docs)
        
        count = store.delete(["doc-1"])
        
        assert count == 1
        assert store.count() == 2
    
    def test_delete_nonexistent_id(self, store, sample_docs):
        """Should handle deletion of nonexistent ID."""
        store.upsert(sample_docs)
        
        count = store.delete(["nonexistent"])
        
        assert count == 0
        assert store.count() == 3
    
    def test_delete_by_filter(self, store, sample_docs):
        """Should delete documents matching filter."""
        store.upsert(sample_docs)
        
        store.delete_by_filter({"department": "HR"})
        
        assert store.count() == 1
        
        # Only IT document should remain
        results = store.search([1.0, 0.0, 0.0], top_k=10)
        assert all(r.metadata["department"] == "IT" for r in results)
    
    def test_recreate_collection(self, store, sample_docs):
        """Should clear data when recreating collection."""
        store.upsert(sample_docs)
        assert store.count() == 3
        
        store.create_collection(recreate=True)
        
        assert store.count() == 0


class TestVectorStoreFactory:
    """Tests for the VectorStoreFactory class."""
    
    def test_create_mock_store(self):
        """Should create a MockVectorStore."""
        store = VectorStoreFactory.create("mock")
        
        assert isinstance(store, MockVectorStore)
    
    def test_create_with_custom_config(self):
        """Should pass config to store."""
        config = QdrantConfig(collection_name="test_collection")
        store = VectorStoreFactory.create("mock", config)
        
        assert store.config.collection_name == "test_collection"
    
    def test_create_unknown_provider_raises_error(self):
        """Should raise error for unknown provider."""
        with pytest.raises(ValueError) as exc_info:
            VectorStoreFactory.create("unknown")
        
        assert "Unknown vector store provider" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])