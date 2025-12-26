"""
RBAC Service

Role-Based Access Control for document access.
Ensures users only see documents they're authorised to access.

Key concepts:
- Users have roles (e.g., ["employee", "hr", "manager"])
- Documents have access_roles (e.g., ["hr", "managers"])
- A user can access a document if their roles overlap with document's access_roles

Special cases:
- Empty access_roles [] means public (everyone can access)
- "all_staff" role means all authenticated users can access
"""

from dataclasses import dataclass, field
from typing import Optional
import logging

logger = logging.getLogger(__name__)


# Special role that grants access to all authenticated users
ALL_STAFF_ROLE = "all_staff"


@dataclass
class User:
    """
    Represents an authenticated user.
    
    Attributes:
        user_id: Unique identifier for the user
        username: Display name
        roles: List of roles assigned to this user
        department: User's department (optional)
    """
    user_id: str
    username: str
    roles: list[str] = field(default_factory=list)
    department: Optional[str] = None
    
    def has_role(self, role: str) -> bool:
        """Check if user has a specific role."""
        return role in self.roles
    
    def has_any_role(self, roles: list[str]) -> bool:
        """Check if user has any of the specified roles."""
        return bool(set(self.roles) & set(roles))


class RBACService:
    """
    Service for managing role-based access control.
    
    Usage:
        rbac = RBACService()
        user = User(user_id="123", username="john", roles=["employee", "hr"])
        
        # Check single document access
        if rbac.can_access(user, document_roles=["hr", "managers"]):
            # User can see this document
            
        # Filter a list of results
        filtered = rbac.filter_results(user, results)
    """
    
    def __init__(self, public_roles: Optional[list[str]] = None):
        """
        Initialise the RBAC service.
        
        Args:
            public_roles: Roles that grant access to everyone.
                         Defaults to ["all_staff"].
        """
        self.public_roles = public_roles or [ALL_STAFF_ROLE]
    
    def can_access(
        self,
        user: User,
        document_roles: list[str]
    ) -> bool:
        """
        Check if a user can access a document.
        
        Access is granted if:
        1. Document has no access_roles (public)
        2. Document has a public role (e.g., "all_staff")
        3. User has at least one role that matches document's access_roles
        
        Args:
            user: The user requesting access
            document_roles: The roles required to access the document
            
        Returns:
            True if user can access, False otherwise
        """
        # Case 1: Empty access_roles means public document
        if not document_roles:
            logger.debug(f"Document is public (no access_roles)")
            return True
        
        # Case 2: Document has a public role like "all_staff"
        for public_role in self.public_roles:
            if public_role in document_roles:
                logger.debug(f"Document has public role: {public_role}")
                return True
        
        # Case 3: Check if user has any matching role
        user_roles_set = set(user.roles)
        document_roles_set = set(document_roles)
        matching_roles = user_roles_set & document_roles_set
        
        if matching_roles:
            logger.debug(f"User has matching roles: {matching_roles}")
            return True
        
        logger.debug(
            f"Access denied. User roles: {user.roles}, "
            f"Document requires: {document_roles}"
        )
        return False
    
    def filter_results(
        self,
        user: User,
        results: list,
        roles_field: str = "access_roles"
    ) -> list:
        """
        Filter search results to only include accessible documents.
        
        Args:
            user: The user requesting access
            results: List of search results (must have metadata with access_roles)
            roles_field: Name of the field containing access roles in metadata
            
        Returns:
            Filtered list containing only accessible results
        """
        filtered = []
        
        for result in results:
            # Get access_roles from result metadata
            if hasattr(result, 'metadata'):
                document_roles = result.metadata.get(roles_field, [])
            elif isinstance(result, dict):
                document_roles = result.get('metadata', {}).get(roles_field, [])
            else:
                # If we can't determine roles, deny access (secure default)
                logger.warning(f"Cannot determine access_roles for result")
                continue
            
            if self.can_access(user, document_roles):
                filtered.append(result)
        
        logger.info(
            f"RBAC filtered {len(results)} results to {len(filtered)} "
            f"for user {user.username}"
        )
        
        return filtered
    
    def get_accessible_departments(self, user: User) -> list[str]:
        """
        Get list of departments a user can access.
        
        This is a simple implementation where users can access
        their own department plus any departments matching their roles.
        
        Args:
            user: The user to check
            
        Returns:
            List of accessible department names
        """
        departments = set()
        
        # User can always access their own department
        if user.department:
            departments.add(user.department)
        
        # Map roles to departments (simplified)
        role_department_map = {
            "hr": "HR",
            "it": "IT",
            "finance": "Finance",
            "engineering": "Engineering",
            "admin": None,  # Admin can access all (handled separately)
        }
        
        for role in user.roles:
            if role == "admin":
                # Admin can access all departments
                return ["*"]
            
            dept = role_department_map.get(role)
            if dept:
                departments.add(dept)
        
        return list(departments)


class RBACMiddleware:
    """
    Middleware for applying RBAC checks to retrieval results.
    
    Can be used to wrap the retriever and automatically filter results.
    
    Usage:
        retriever = Retriever(...)
        rbac_retriever = RBACMiddleware(retriever, rbac_service)
        
        # Results are automatically filtered
        results = rbac_retriever.retrieve(query, user)
    """
    
    def __init__(self, retriever, rbac_service: Optional[RBACService] = None):
        """
        Initialise the RBAC middleware.
        
        Args:
            retriever: The underlying retriever to wrap
            rbac_service: RBAC service instance. Creates default if not provided.
        """
        self.retriever = retriever
        self.rbac = rbac_service or RBACService()
    
    def retrieve(
        self,
        query: str,
        user: User,
        top_k: int = 5,
        **kwargs
    ):
        """
        Retrieve results with RBAC filtering.
        
        Args:
            query: The search query
            user: The user making the request
            top_k: Number of results to return (after filtering)
            **kwargs: Additional arguments passed to retriever
            
        Returns:
            Filtered retrieval results
        """
        # Request more results than needed since some may be filtered out
        fetch_multiplier = 3
        raw_results = self.retriever.retrieve(
            query=query,
            top_k=top_k * fetch_multiplier,
            **kwargs
        )
        
        # Filter results based on user's roles
        filtered_results = self.rbac.filter_results(user, raw_results.results)
        
        # Trim to requested top_k
        filtered_results = filtered_results[:top_k]
        
        # Update the results object
        raw_results.results = filtered_results
        raw_results.total_results = len(filtered_results)
        
        return raw_results