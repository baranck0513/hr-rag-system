"""
Embedder Service

Converts text into vector embeddings using various embedding models.
Supports OpenAI embeddings and can be extended for other providers.

The embedder is used in two places:
1. Ingestion: Convert document chunks to vectors for storage
2. Retrieval: Convert user questions to vectors for search
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Protocol
import logging
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """
    Result of an embedding operation.
    
    Attributes:
        text: The original text that was embedded
        vector: The embedding vector (list of floats)
        model: The model used to create the embedding
        dimensions: Number of dimensions in the vector
    """
    text: str
    vector: list[float]
    model: str
    dimensions: int
    
    @property
    def text_hash(self) -> str:
        """Hash of the text for caching purposes."""
        return hashlib.md5(self.text.encode()).hexdigest()


class EmbedderProtocol(Protocol):
    """Protocol defining the interface for embedders."""
    
    def embed(self, text: str) -> EmbeddingResult:
        """Embed a single text."""
        ...
    
    def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        """Embed multiple texts."""
        ...


class OpenAIEmbedder:
    """
    Embedder using OpenAI's embedding models.
    
    Uses the text-embedding-3-small model by default, which offers
    a good balance of quality and cost for most use cases.
    
    Usage:
        embedder = OpenAIEmbedder(api_key="sk-...")
        result = embedder.embed("What is the leave policy?")
        print(result.vector)  # [0.123, -0.456, ...]
    
    Note:
        Requires the OPENAI_API_KEY environment variable or
        explicit api_key parameter.
    """
    
    # Available OpenAI embedding models
    MODELS = {
        "text-embedding-3-small": 1536,   # Recommended for most use cases
        "text-embedding-3-large": 3072,   # Higher quality, higher cost
        "text-embedding-ada-002": 1536,   # Legacy model
    }
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small"
    ):
        """
        Initialise the OpenAI embedder.
        
        Args:
            api_key: OpenAI API key. If not provided, will look for
                    OPENAI_API_KEY environment variable.
            model: Which embedding model to use.
        """
        self.model = model
        self.dimensions = self.MODELS.get(model, 1536)
        self._api_key = api_key
        self._client = None
        
        if model not in self.MODELS:
            logger.warning(
                f"Unknown model {model}, using default dimensions of 1536"
            )
    
    @property
    def client(self):
        """Lazy initialisation of OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self._api_key)
            except ImportError:
                raise ImportError(
                    "openai package is required. Install with: "
                    "pip install openai"
                )
        return self._client
    
    def embed(self, text: str) -> EmbeddingResult:
        """
        Embed a single text string.
        
        Args:
            text: The text to embed
            
        Returns:
            EmbeddingResult containing the vector
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")
        
        # Clean the text
        text = text.replace("\n", " ").strip()
        
        response = self.client.embeddings.create(
            input=text,
            model=self.model
        )
        
        vector = response.data[0].embedding
        
        return EmbeddingResult(
            text=text,
            vector=vector,
            model=self.model,
            dimensions=len(vector)
        )
    
    def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        """
        Embed multiple texts in a single API call.
        
        More efficient than calling embed() multiple times.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of EmbeddingResult objects
        """
        if not texts:
            return []
        
        # Clean texts
        cleaned_texts = [t.replace("\n", " ").strip() for t in texts]
        
        # Filter out empty texts
        valid_texts = [(i, t) for i, t in enumerate(cleaned_texts) if t]
        
        if not valid_texts:
            raise ValueError("All texts are empty")
        
        indices, texts_to_embed = zip(*valid_texts)
        
        response = self.client.embeddings.create(
            input=list(texts_to_embed),
            model=self.model
        )
        
        results = []
        for i, embedding_data in enumerate(response.data):
            results.append(EmbeddingResult(
                text=texts_to_embed[i],
                vector=embedding_data.embedding,
                model=self.model,
                dimensions=len(embedding_data.embedding)
            ))
        
        return results


class MockEmbedder:
    """
    Mock embedder for testing without API calls.
    
    Generates deterministic fake embeddings based on text content.
    Useful for unit tests and development.
    
    Usage:
        embedder = MockEmbedder()
        result = embedder.embed("test text")
    """
    
    def __init__(self, dimensions: int = 1536):
        """
        Initialise mock embedder.
        
        Args:
            dimensions: Number of dimensions for fake vectors
        """
        self.dimensions = dimensions
        self.model = "mock-embedder"
    
    def embed(self, text: str) -> EmbeddingResult:
        """Generate a deterministic fake embedding."""
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")
        
        # Generate deterministic vector based on text hash
        vector = self._generate_vector(text)
        
        return EmbeddingResult(
            text=text,
            vector=vector,
            model=self.model,
            dimensions=self.dimensions
        )
    
    def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        """Generate fake embeddings for multiple texts."""
        return [self.embed(text) for text in texts if text.strip()]
    
    def _generate_vector(self, text: str) -> list[float]:
        """
        Generate a deterministic vector from text.
        
        Same text always produces same vector (for testing).
        """
        import hashlib
        
        # Create a seed from the text
        hash_bytes = hashlib.sha256(text.encode()).digest()
        
        # Generate vector values from hash
        vector = []
        for i in range(self.dimensions):
            # Use different parts of hash for different dimensions
            byte_idx = i % len(hash_bytes)
            # Normalise to range [-1, 1]
            value = (hash_bytes[byte_idx] / 255.0) * 2 - 1
            vector.append(value)
        
        return vector


class EmbedderFactory:
    """
    Factory for creating embedder instances.
    
    Usage:
        # For production
        embedder = EmbedderFactory.create("openai", api_key="sk-...")
        
        # For testing
        embedder = EmbedderFactory.create("mock")
    """
    
    @staticmethod
    def create(
        provider: str = "openai",
        **kwargs
    ) -> EmbedderProtocol:
        """
        Create an embedder instance.
        
        Args:
            provider: Which provider to use ("openai" or "mock")
            **kwargs: Additional arguments passed to the embedder
            
        Returns:
            An embedder instance
        """
        if provider == "openai":
            return OpenAIEmbedder(**kwargs)
        elif provider == "mock":
            return MockEmbedder(**kwargs)
        else:
            raise ValueError(f"Unknown embedder provider: {provider}")