"""
Tests for the RBAC service.
"""

import pytest
from app.services.rbac import (
    User,
    RBACService,
    RBACMiddleware,
    ALL_STAFF_ROLE,
)


class TestUser:
    """Tests for the User dataclass."""
    
    def test_user_creation(self):
        """Should create a user with required fields."""
        user = User(
            user_id="123",
            username="john.smith",
            roles=["employee", "hr"]
        )
        
        assert user.user_id == "123"
        assert user.username == "john.smith"
        assert user.roles == ["employee", "hr"]
    
    def test_user_default_roles(self):
        """Should default to empty roles list."""
        user = User(user_id="123", username="john")
        
        assert user.roles == []
    
    def test_has_role_true(self):
        """Should return True when user has role."""
        user = User(user_id="123", username="john", roles=["hr", "manager"])
        
        assert user.has_role("hr") is True
        assert user.has_role("manager") is True
    
    def test_has_role_false(self):
        """Should return False when user doesn't have role."""
        user = User(user_id="123", username="john", roles=["employee"])
        
        assert user.has_role("hr") is False
    
    def test_has_any_role_true(self):
        """Should return True when user has any of the roles."""
        user = User(user_id="123", username="john", roles=["employee", "hr"])
        
        assert user.has_any_role(["hr", "admin"]) is True
    
    def test_has_any_role_false(self):
        """Should return False when user has none of the roles."""
        user = User(user_id="123", username="john", roles=["employee"])
        
        assert user.has_any_role(["hr", "admin"]) is False
    
    def test_has_any_role_empty_list(self):
        """Should return False for empty role list."""
        user = User(user_id="123", username="john", roles=["employee"])
        
        assert user.has_any_role([]) is False


class TestRBACService:
    """Tests for the RBACService class."""
    
    @pytest.fixture
    def rbac(self):
        """Create an RBAC service instance."""
        return RBACService()
    
    @pytest.fixture
    def employee_user(self):
        """Regular employee user."""
        return User(
            user_id="1",
            username="alice",
            roles=["employee"],
            department="Engineering"
        )
    
    @pytest.fixture
    def hr_user(self):
        """HR team member."""
        return User(
            user_id="2",
            username="bob",
            roles=["employee", "hr"],
            department="HR"
        )
    
    @pytest.fixture
    def manager_user(self):
        """Manager user."""
        return User(
            user_id="3",
            username="carol",
            roles=["employee", "manager"],
            department="Sales"
        )
    
    # --- can_access tests ---
    
    def test_can_access_public_document(self, rbac, employee_user):
        """Should allow access to documents with empty access_roles."""
        # Empty access_roles means public
        document_roles = []
        
        assert rbac.can_access(employee_user, document_roles) is True
    
    def test_can_access_all_staff_document(self, rbac, employee_user):
        """Should allow access to documents with 'all_staff' role."""
        document_roles = ["all_staff"]
        
        assert rbac.can_access(employee_user, document_roles) is True
    
    def test_can_access_matching_role(self, rbac, hr_user):
        """Should allow access when user has matching role."""
        document_roles = ["hr", "admin"]
        
        assert rbac.can_access(hr_user, document_roles) is True
    
    def test_cannot_access_no_matching_role(self, rbac, employee_user):
        """Should deny access when user has no matching role."""
        document_roles = ["hr", "admin"]
        
        assert rbac.can_access(employee_user, document_roles) is False
    
    def test_can_access_one_of_many_roles(self, rbac, manager_user):
        """Should allow access if user has at least one matching role."""
        document_roles = ["hr", "manager", "executive"]
        
        assert rbac.can_access(manager_user, document_roles) is True
    
    def test_cannot_access_restricted_document(self, rbac, employee_user):
        """Should deny access to restricted documents."""
        document_roles = ["executive"]
        
        assert rbac.can_access(employee_user, document_roles) is False
    
    # --- filter_results tests ---
    
    def test_filter_results_allows_public(self, rbac, employee_user):
        """Should include public documents in filtered results."""
        results = [
            MockResult(access_roles=[]),
            MockResult(access_roles=["hr"]),
        ]
        
        filtered = rbac.filter_results(employee_user, results)
        
        assert len(filtered) == 1
    
    def test_filter_results_allows_all_staff(self, rbac, employee_user):
        """Should include all_staff documents in filtered results."""
        results = [
            MockResult(access_roles=["all_staff"]),
            MockResult(access_roles=["hr"]),
        ]
        
        filtered = rbac.filter_results(employee_user, results)
        
        assert len(filtered) == 1
    
    def test_filter_results_matching_roles(self, rbac, hr_user):
        """Should include documents where user has matching role."""
        results = [
            MockResult(access_roles=["hr"]),
            MockResult(access_roles=["executive"]),
            MockResult(access_roles=["hr", "manager"]),
        ]
        
        filtered = rbac.filter_results(hr_user, results)
        
        # HR user should see first and third document
        assert len(filtered) == 2
    
    def test_filter_results_empty_input(self, rbac, employee_user):
        """Should return empty list for empty input."""
        filtered = rbac.filter_results(employee_user, [])
        
        assert filtered == []
    
    def test_filter_results_all_denied(self, rbac, employee_user):
        """Should return empty list when all results are denied."""
        results = [
            MockResult(access_roles=["hr"]),
            MockResult(access_roles=["executive"]),
        ]
        
        filtered = rbac.filter_results(employee_user, results)
        
        assert filtered == []
    
    def test_filter_results_all_allowed(self, rbac, hr_user):
        """Should return all results when all are accessible."""
        results = [
            MockResult(access_roles=["hr"]),
            MockResult(access_roles=["all_staff"]),
            MockResult(access_roles=[]),
        ]
        
        filtered = rbac.filter_results(hr_user, results)
        
        assert len(filtered) == 3
    
    def test_filter_results_dict_format(self, rbac, hr_user):
        """Should handle results in dict format."""
        results = [
            {"metadata": {"access_roles": ["hr"]}},
            {"metadata": {"access_roles": ["executive"]}},
        ]
        
        filtered = rbac.filter_results(hr_user, results)
        
        assert len(filtered) == 1


class TestRBACServiceCustomRoles:
    """Tests for RBAC with custom public roles."""
    
    def test_custom_public_roles(self):
        """Should use custom public roles."""
        rbac = RBACService(public_roles=["everyone", "public"])
        user = User(user_id="1", username="alice", roles=["employee"])
        
        # "everyone" should work as public role
        assert rbac.can_access(user, ["everyone"]) is True
        
        # Default "all_staff" should NOT work
        assert rbac.can_access(user, ["all_staff"]) is False


class TestRBACMiddleware:
    """Tests for the RBACMiddleware class."""
    
    def test_middleware_filters_results(self):
        """Should filter retriever results based on user roles."""
        # Create mock retriever
        mock_retriever = MockRetriever(results=[
            MockResult(access_roles=["hr"]),
            MockResult(access_roles=["all_staff"]),
            MockResult(access_roles=["executive"]),
        ])
        
        middleware = RBACMiddleware(mock_retriever)
        user = User(user_id="1", username="alice", roles=["employee"])
        
        result = middleware.retrieve("test query", user, top_k=5)
        
        # Employee should only see "all_staff" document
        assert len(result.results) == 1
    
    def test_middleware_respects_top_k(self):
        """Should return at most top_k results after filtering."""
        mock_retriever = MockRetriever(results=[
            MockResult(access_roles=["all_staff"]),
            MockResult(access_roles=["all_staff"]),
            MockResult(access_roles=["all_staff"]),
            MockResult(access_roles=["all_staff"]),
        ])
        
        middleware = RBACMiddleware(mock_retriever)
        user = User(user_id="1", username="alice", roles=["employee"])
        
        result = middleware.retrieve("test query", user, top_k=2)
        
        assert len(result.results) == 2


class TestRBACRealWorldScenarios:
    """Integration tests with realistic scenarios."""
    
    @pytest.fixture
    def rbac(self):
        return RBACService()
    
    def test_scenario_leave_policy(self, rbac):
        """Leave policy should be accessible to all staff."""
        user = User(user_id="1", username="new_hire", roles=["employee"])
        document_roles = ["all_staff"]
        
        assert rbac.can_access(user, document_roles) is True
    
    def test_scenario_salary_bands(self, rbac):
        """Salary bands should only be accessible to HR and managers."""
        hr_user = User(user_id="1", username="hr_person", roles=["employee", "hr"])
        manager = User(user_id="2", username="manager", roles=["employee", "manager"])
        employee = User(user_id="3", username="employee", roles=["employee"])
        
        document_roles = ["hr", "manager"]
        
        assert rbac.can_access(hr_user, document_roles) is True
        assert rbac.can_access(manager, document_roles) is True
        assert rbac.can_access(employee, document_roles) is False
    
    def test_scenario_disciplinary_records(self, rbac):
        """Disciplinary records should only be accessible to HR."""
        hr_user = User(user_id="1", username="hr_person", roles=["employee", "hr"])
        manager = User(user_id="2", username="manager", roles=["employee", "manager"])
        
        document_roles = ["hr"]
        
        assert rbac.can_access(hr_user, document_roles) is True
        assert rbac.can_access(manager, document_roles) is False
    
    def test_scenario_public_faq(self, rbac):
        """Public FAQ should be accessible to everyone."""
        user = User(user_id="1", username="anyone", roles=[])
        document_roles = []  # Empty means public
        
        assert rbac.can_access(user, document_roles) is True


# --- Mock classes for testing ---

class MockResult:
    """Mock search result for testing."""
    
    def __init__(self, access_roles: list[str]):
        self.metadata = {"access_roles": access_roles}
        self.text = "Mock result text"
        self.score = 0.9


class MockRetrievalResult:
    """Mock retrieval result object."""
    
    def __init__(self, results: list):
        self.results = results
        self.total_results = len(results)


class MockRetriever:
    """Mock retriever for testing middleware."""
    
    def __init__(self, results: list):
        self._results = results
    
    def retrieve(self, query: str, top_k: int = 5, **kwargs):
        return MockRetrievalResult(self._results[:top_k * 3])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])