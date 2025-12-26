"""
Tests for the FastAPI application.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.api.routes.documents import _document_store


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def clear_document_store():
    """Clear the document store before each test."""
    _document_store.clear()
    yield
    _document_store.clear()


class TestRootEndpoint:
    """Tests for the root endpoint."""
    
    def test_root_returns_info(self, client):
        """Should return application information."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "documentation" in data
        assert "endpoints" in data


class TestHealthEndpoint:
    """Tests for the health check endpoint."""
    
    def test_health_check(self, client):
        """Should return healthy status."""
        response = client.get("/api/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "total_documents" in data
        assert "total_vectors" in data


class TestDocumentUpload:
    """Tests for document upload endpoint."""
    
    def test_upload_text_file(self, client):
        """Should upload and process a text file."""
        content = b"This is a test document about annual leave policy."
        
        response = client.post(
            "/api/v1/documents/upload",
            files={"file": ("test.txt", content, "text/plain")}
        )
        
        assert response.status_code == 201
        data = response.json()
        assert "document_id" in data
        assert data["filename"] == "test.txt"
        assert data["chunk_count"] >= 1
        assert "processing_time_ms" in data
    
    def test_upload_with_metadata(self, client):
        """Should store optional metadata."""
        content = b"HR Policy document content"
        
        response = client.post(
            "/api/v1/documents/upload",
            files={"file": ("policy.txt", content, "text/plain")},
            data={
                "department": "HR",
                "access_roles": "hr_team,managers",
                "uploaded_by": "user123"
            }
        )
        
        assert response.status_code == 201
        
        # Verify metadata was stored
        doc_id = response.json()["document_id"]
        get_response = client.get(f"/api/v1/documents/{doc_id}")
        assert get_response.status_code == 200
        doc_data = get_response.json()
        assert doc_data["department"] == "HR"
        assert "hr_team" in doc_data["access_roles"]
        assert "managers" in doc_data["access_roles"]
    
    def test_upload_masks_pii(self, client):
        """Should mask PII in uploaded documents."""
        content = b"Employee NI: AB123456C, Email: john@test.com"
        
        response = client.post(
            "/api/v1/documents/upload",
            files={"file": ("employee.txt", content, "text/plain")}
        )
        
        assert response.status_code == 201
        data = response.json()
        
        # Check that PII was detected and masked
        assert "NI_NUMBER" in data["pii_masked"] or "EMAIL" in data["pii_masked"]
    
    def test_upload_unsupported_type(self, client):
        """Should reject unsupported file types."""
        content = b"Some content"
        
        response = client.post(
            "/api/v1/documents/upload",
            files={"file": ("document.docx", content, "application/octet-stream")}
        )
        
        assert response.status_code == 400
        assert "Unsupported file type" in response.json()["detail"]
    
    def test_upload_empty_file(self, client):
        """Should reject empty files."""
        response = client.post(
            "/api/v1/documents/upload",
            files={"file": ("empty.txt", b"", "text/plain")}
        )
        
        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()


class TestDocumentList:
    """Tests for document listing endpoint."""
    
    def test_list_empty(self, client):
        """Should return empty list when no documents."""
        response = client.get("/api/v1/documents")
        
        assert response.status_code == 200
        data = response.json()
        assert data["documents"] == []
        assert data["total"] == 0
    
    def test_list_after_upload(self, client):
        """Should list uploaded documents."""
        # Upload a document
        client.post(
            "/api/v1/documents/upload",
            files={"file": ("test.txt", b"Test content", "text/plain")}
        )
        
        # List documents
        response = client.get("/api/v1/documents")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert len(data["documents"]) == 1
        assert data["documents"][0]["filename"] == "test.txt"
    
    def test_list_filter_by_department(self, client):
        """Should filter documents by department."""
        # Upload documents to different departments
        client.post(
            "/api/v1/documents/upload",
            files={"file": ("hr_doc.txt", b"HR content", "text/plain")},
            data={"department": "HR"}
        )
        client.post(
            "/api/v1/documents/upload",
            files={"file": ("it_doc.txt", b"IT content", "text/plain")},
            data={"department": "IT"}
        )
        
        # Filter by HR
        response = client.get("/api/v1/documents?department=HR")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert data["documents"][0]["department"] == "HR"


class TestDocumentGet:
    """Tests for getting document details."""
    
    def test_get_document(self, client):
        """Should return document details."""
        # Upload a document
        upload_response = client.post(
            "/api/v1/documents/upload",
            files={"file": ("test.txt", b"Test content", "text/plain")}
        )
        doc_id = upload_response.json()["document_id"]
        
        # Get document
        response = client.get(f"/api/v1/documents/{doc_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["document_id"] == doc_id
        assert data["filename"] == "test.txt"
    
    def test_get_document_not_found(self, client):
        """Should return 404 for non-existent document."""
        response = client.get("/api/v1/documents/nonexistent")
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


class TestDocumentDelete:
    """Tests for document deletion."""
    
    def test_delete_document(self, client):
        """Should delete a document."""
        # Upload a document
        upload_response = client.post(
            "/api/v1/documents/upload",
            files={"file": ("test.txt", b"Test content", "text/plain")}
        )
        doc_id = upload_response.json()["document_id"]
        
        # Delete document
        response = client.delete(f"/api/v1/documents/{doc_id}")
        
        assert response.status_code == 200
        assert response.json()["document_id"] == doc_id
        
        # Verify it's gone
        get_response = client.get(f"/api/v1/documents/{doc_id}")
        assert get_response.status_code == 404
    
    def test_delete_document_not_found(self, client):
        """Should return 404 for non-existent document."""
        response = client.delete("/api/v1/documents/nonexistent")
        
        assert response.status_code == 404


class TestQueryEndpoint:
    """Tests for the query endpoint."""
    
    def test_query_basic(self, client):
        """Should accept a basic query."""
        response = client.post(
            "/api/v1/query",
            json={"question": "What is the leave policy?"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["question"] == "What is the leave policy?"
        assert "results" in data
        assert "total_results" in data
    
    def test_query_with_options(self, client):
        """Should accept query with options."""
        response = client.post(
            "/api/v1/query",
            json={
                "question": "What is the leave policy?",
                "top_k": 3,
                "department": "HR"
            }
        )
        
        assert response.status_code == 200
    
    def test_query_empty_question(self, client):
        """Should reject empty question."""
        response = client.post(
            "/api/v1/query",
            json={"question": ""}
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_query_invalid_top_k(self, client):
        """Should reject invalid top_k values."""
        response = client.post(
            "/api/v1/query",
            json={"question": "Test?", "top_k": 100}
        )
        
        assert response.status_code == 422  # Validation error


class TestAPIDocumentation:
    """Tests for API documentation endpoints."""
    
    def test_swagger_docs_available(self, client):
        """Swagger UI should be accessible."""
        response = client.get("/docs")
        
        assert response.status_code == 200
    
    def test_openapi_schema_available(self, client):
        """OpenAPI schema should be accessible."""
        response = client.get("/openapi.json")
        
        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "paths" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])