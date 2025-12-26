"""
Text Chunker Service

Splits documents into optimal chunks for embedding and retrieval.
Uses a recursive approach: semantic splitting first, then fixed-size
splitting for chunks that exceed the token limit.

This module provides multiple chunking strategies that can be selected
based on document type and use case.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Protocol
import re
import logging

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """
    Represents a single chunk of text with metadata.
    
    Attributes:
        text: The actual text content of the chunk
        index: Position of this chunk in the original document (0-based)
        metadata: Additional information about this chunk
    """
    text: str
    index: int
    metadata: dict = field(default_factory=dict)
    
    @property
    def token_count_estimate(self) -> int:
        """
        Estimate token count (rough approximation).
        
        OpenAI uses ~4 characters per token on average for English text.
        This is an estimate — for precise counting, use tiktoken library.
        """
        return len(self.text) // 4


class ChunkingStrategy(Protocol):
    """Protocol defining the interface for chunking strategies."""
    
    def chunk(self, text: str) -> list[Chunk]:
        """Split text into chunks."""
        ...


@dataclass
class ChunkerConfig:
    """
    Configuration for the text chunker.
    
    Attributes:
        chunk_size: Target maximum size of each chunk (in characters)
        chunk_overlap: Number of characters to overlap between chunks
        min_chunk_size: Minimum chunk size (smaller chunks are merged)
        separators: List of separators to try, in order of preference
    """
    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size: int = 100
    separators: list[str] = field(default_factory=lambda: [
        "\n\n\n",      # Multiple blank lines (section breaks)
        "\n\n",         # Paragraph breaks
        "\n",           # Line breaks
        ". ",           # Sentence endings
        "? ",           # Question endings
        "! ",           # Exclamation endings
        "; ",           # Semicolon breaks
        ", ",           # Comma breaks
        " ",            # Word breaks
        ""              # Character-level (last resort)
    ])


class RecursiveChunker:
    """
    Recursively splits text using a hierarchy of separators.
    
    This chunker tries to split on semantic boundaries (paragraphs,
    sentences) first, falling back to smaller units only when necessary.
    
    This is the recommended chunker for HR documents as it preserves
    the natural structure of policies and procedures.
    
    Usage:
        config = ChunkerConfig(chunk_size=1000, chunk_overlap=200)
        chunker = RecursiveChunker(config)
        chunks = chunker.chunk(document_text)
    """
    
    def __init__(self, config: ChunkerConfig | None = None):
        """
        Initialise the chunker with configuration.
        
        Args:
            config: Chunking configuration. Uses defaults if not provided.
        """
        self.config = config or ChunkerConfig()
    
    def chunk(self, text: str) -> list[Chunk]:
        """
        Split text into chunks using recursive splitting.
        
        Args:
            text: The text to split
            
        Returns:
            List of Chunk objects
        """
        if not text or not text.strip():
            return []
        
        # Perform the recursive splitting
        raw_chunks = self._split_recursive(text, self.config.separators)
        
        # Merge small chunks and apply overlap
        merged_chunks = self._merge_small_chunks(raw_chunks)
        
        # Create Chunk objects with metadata
        chunks = [
            Chunk(
                text=chunk_text.strip(),
                index=i,
                metadata={
                    "chunk_size": len(chunk_text),
                    "chunking_strategy": "recursive"
                }
            )
            for i, chunk_text in enumerate(merged_chunks)
            if chunk_text.strip()  # Skip empty chunks
        ]
        
        logger.info(f"Created {len(chunks)} chunks from text of length {len(text)}")
        
        return chunks
    
    def _split_recursive(
        self, 
        text: str, 
        separators: list[str]
    ) -> list[str]:
        """
        Recursively split text using the separator hierarchy.
        
        Args:
            text: Text to split
            separators: Remaining separators to try
            
        Returns:
            List of text chunks
        """
        if not separators:
            # No more separators — return text as-is (will be split by chars)
            return [text]
        
        separator = separators[0]
        remaining_separators = separators[1:]
        
        # If text is small enough, return it
        if len(text) <= self.config.chunk_size:
            return [text]
        
        # Split by current separator
        if separator:
            splits = text.split(separator)
        else:
            # Empty separator means split by character
            splits = list(text)
        
        # Process each split
        chunks = []
        current_chunk = ""
        
        for split in splits:
            # Add separator back (except for empty separator)
            piece = split + separator if separator else split
            
            # If adding this piece exceeds chunk size
            if len(current_chunk) + len(piece) > self.config.chunk_size:
                # Save current chunk if it exists
                if current_chunk:
                    # If current chunk is still too big, split it further
                    if len(current_chunk) > self.config.chunk_size:
                        chunks.extend(
                            self._split_recursive(current_chunk, remaining_separators)
                        )
                    else:
                        chunks.append(current_chunk)
                
                # Start new chunk
                current_chunk = piece
            else:
                current_chunk += piece
        
        # Don't forget the last chunk
        if current_chunk:
            if len(current_chunk) > self.config.chunk_size:
                chunks.extend(
                    self._split_recursive(current_chunk, remaining_separators)
                )
            else:
                chunks.append(current_chunk)
        
        return chunks
    
    def _merge_small_chunks(self, chunks: list[str]) -> list[str]:
        """
        Merge chunks that are too small and apply overlap.
        
        Args:
            chunks: List of text chunks
            
        Returns:
            List of merged chunks with overlap applied
        """
        if not chunks:
            return []
        
        merged = []
        current = chunks[0]
        
        for next_chunk in chunks[1:]:
            # If merging wouldn't exceed limit, merge
            if len(current) + len(next_chunk) <= self.config.chunk_size:
                current += next_chunk
            else:
                # Save current and start new
                if len(current) >= self.config.min_chunk_size:
                    merged.append(current)
                
                # Apply overlap from end of current to start of next
                overlap_text = current[-self.config.chunk_overlap:] if len(current) > self.config.chunk_overlap else current
                current = overlap_text + next_chunk
        
        # Don't forget the last chunk
        if current and len(current) >= self.config.min_chunk_size:
            merged.append(current)
        elif current and merged:
            # If last chunk is too small, append to previous
            merged[-1] += current
        elif current:
            # If it's the only chunk, keep it
            merged.append(current)
        
        return merged


class SentenceChunker:
    """
    Splits text by sentences, grouping them to reach target chunk size.
    
    This is a simpler alternative to RecursiveChunker that works well
    for documents with clear sentence structure.
    """
    
    def __init__(self, config: ChunkerConfig | None = None):
        self.config = config or ChunkerConfig()
        # Simple sentence-ending pattern
        self.sentence_pattern = re.compile(r'(?<=[.!?])\s+')
    
    def chunk(self, text: str) -> list[Chunk]:
        """Split text into sentence-based chunks."""
        if not text or not text.strip():
            return []
        
        sentences = self.sentence_pattern.split(text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.config.chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return [
            Chunk(
                text=chunk_text,
                index=i,
                metadata={
                    "chunk_size": len(chunk_text),
                    "chunking_strategy": "sentence"
                }
            )
            for i, chunk_text in enumerate(chunks)
        ]


class FixedSizeChunker:
    """
    Splits text into fixed-size chunks with overlap.
    
    This is the simplest chunking strategy. Use it when document
    structure is unknown or irrelevant.
    """
    
    def __init__(self, config: ChunkerConfig | None = None):
        self.config = config or ChunkerConfig()
    
    def chunk(self, text: str) -> list[Chunk]:
        """Split text into fixed-size chunks."""
        if not text or not text.strip():
            return []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.config.chunk_size
            chunk_text = text[start:end]
            
            if chunk_text.strip():
                chunks.append(Chunk(
                    text=chunk_text.strip(),
                    index=len(chunks),
                    metadata={
                        "chunk_size": len(chunk_text),
                        "chunking_strategy": "fixed_size"
                    }
                ))
            
            # Move start, accounting for overlap
            start = end - self.config.chunk_overlap
        
        return chunks


class ChunkerFactory:
    """
    Factory for creating chunker instances.
    
    Usage:
        chunker = ChunkerFactory.get_chunker("recursive")
        chunks = chunker.chunk(text)
    """
    
    CHUNKERS = {
        "recursive": RecursiveChunker,
        "sentence": SentenceChunker,
        "fixed": FixedSizeChunker,
    }
    
    @classmethod
    def get_chunker(
        cls, 
        strategy: str = "recursive",
        config: ChunkerConfig | None = None
    ) -> ChunkingStrategy:
        """
        Get a chunker instance by strategy name.
        
        Args:
            strategy: Name of the chunking strategy
            config: Optional configuration
            
        Returns:
            A chunker instance
            
        Raises:
            ValueError: If strategy is not recognised
        """
        chunker_class = cls.CHUNKERS.get(strategy)
        
        if not chunker_class:
            available = ", ".join(cls.CHUNKERS.keys())
            raise ValueError(
                f"Unknown chunking strategy: {strategy}. "
                f"Available: {available}"
            )
        
        return chunker_class(config)