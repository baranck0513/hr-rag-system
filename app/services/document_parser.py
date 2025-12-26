"""
Document Parser Service

Handles extraction of text content from various document formats.
Supports PDF, TXT, and MD files with extensible architecture.
"""

from pathlib import Path
from typing import Protocol
import io
import logging

# Configure logging for production use
logger = logging.getLogger(__name__)


class DocumentParser(Protocol):
    """
    Protocol defining the interface for document parsers.
    
    Any parser must implement the parse() method.
    This enables type checking and ensures consistency across parsers.
    """
    
    def parse(self, content: bytes, filename: str) -> str:
        """
        Extract text from document bytes.
        
        Args:
            content: Raw bytes of the document file
            filename: Original filename (used for logging/debugging)
            
        Returns:
            Extracted text as a string
        """
        ...


class TextParser:
    """
    Handles plain text files (.txt and .md).
    
    These files are already text, so we simply decode the bytes.
    Supports UTF-8 encoding with fallback to latin-1.
    """
    
    def parse(self, content: bytes, filename: str) -> str:
        """Decode text file bytes to string."""
        try:
            return content.decode("utf-8")
        except UnicodeDecodeError:
            # Fallback for files with different encoding
            logger.warning(f"UTF-8 decode failed for {filename}, trying latin-1")
            return content.decode("latin-1")


class PDFParser:
    """
    Handles PDF files using pdfplumber.
    
    pdfplumber is chosen over PyPDF2 for better table extraction
    and handling of complex layouts.
    """
    
    def parse(self, content: bytes, filename: str) -> str:
        """Extract text from PDF bytes."""
        import pdfplumber
        
        text_parts = []
        
        try:
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    page_text = page.extract_text()
                    
                    if page_text:
                        text_parts.append(page_text)
                    else:
                        logger.warning(
                            f"No text extracted from page {page_num} of {filename}"
                        )
            
            if not text_parts:
                logger.error(f"No text extracted from {filename}")
                raise ValueError(f"Could not extract text from PDF: {filename}")
                
        except Exception as e:
            logger.error(f"Failed to parse PDF {filename}: {str(e)}")
            raise
        
        # Join pages with double newlines to preserve page boundaries
        return "\n\n".join(text_parts)


class ParserFactory:
    """
    Factory class that returns the appropriate parser for a file type.
    
    This pattern centralises the logic for selecting parsers,
    making it easy to add new file types without changing calling code.
    
    Usage:
        parser = ParserFactory.get_parser("document.pdf")
        text = parser.parse(file_content, "document.pdf")
    """
    
    # Maps file extensions to parser classes
    PARSERS: dict[str, type[DocumentParser]] = {
        ".txt": TextParser,
        ".md": TextParser,
        ".pdf": PDFParser,
    }
    
    # File extensions we support (for validation)
    SUPPORTED_EXTENSIONS: set[str] = {".txt", ".md", ".pdf"}
    
    @classmethod
    def get_parser(cls, filename: str) -> DocumentParser:
        """
        Get the appropriate parser for a given filename.
        
        Args:
            filename: Name of the file to parse
            
        Returns:
            An instance of the appropriate parser
            
        Raises:
            ValueError: If the file type is not supported
        """
        extension = Path(filename).suffix.lower()
        
        parser_class = cls.PARSERS.get(extension)
        if not parser_class:
            supported = ", ".join(sorted(cls.SUPPORTED_EXTENSIONS))
            raise ValueError(
                f"Unsupported file type: {extension}. "
                f"Supported types: {supported}"
            )
        
        return parser_class()
    
    @classmethod
    def is_supported(cls, filename: str) -> bool:
        """
        Check if a file type is supported.
        
        Args:
            filename: Name of the file to check
            
        Returns:
            True if supported, False otherwise
        """
        extension = Path(filename).suffix.lower()
        return extension in cls.SUPPORTED_EXTENSIONS