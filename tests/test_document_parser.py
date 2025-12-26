"""
Tests for the document parser service.
"""

import pytest
from app.services.document_parser import (
    TextParser,
    ParserFactory,
)


class TestTextParser:
    """Tests for TextParser class."""
    
    def test_parse_utf8_text(self):
        """Should correctly decode UTF-8 encoded text."""
        content = "Hello, this is a test document.".encode("utf-8")
        parser = TextParser()
        
        result = parser.parse(content, "test.txt")
        
        assert result == "Hello, this is a test document."
    
    def test_parse_text_with_unicode(self):
        """Should handle Unicode characters like £ symbol."""
        content = "Salary: £50,000 per annum".encode("utf-8")
        parser = TextParser()
        
        result = parser.parse(content, "salary.txt")
        
        assert "£50,000" in result


class TestParserFactory:
    """Tests for ParserFactory class."""
    
    def test_get_parser_for_txt(self):
        """Should return TextParser for .txt files."""
        parser = ParserFactory.get_parser("document.txt")
        
        assert isinstance(parser, TextParser)
    
    def test_get_parser_for_md(self):
        """Should return TextParser for .md files."""
        parser = ParserFactory.get_parser("readme.md")
        
        assert isinstance(parser, TextParser)
    
    def test_get_parser_unsupported_type(self):
        """Should raise ValueError for unsupported file types."""
        with pytest.raises(ValueError) as exc_info:
            ParserFactory.get_parser("document.docx")
        
        assert "Unsupported file type" in str(exc_info.value)
    
    def test_is_supported_returns_true_for_valid_types(self):
        """Should return True for supported file types."""
        assert ParserFactory.is_supported("test.txt") is True
        assert ParserFactory.is_supported("test.pdf") is True
        assert ParserFactory.is_supported("test.md") is True
    
    def test_is_supported_returns_false_for_invalid_types(self):
        """Should return False for unsupported file types."""
        assert ParserFactory.is_supported("test.docx") is False
        assert ParserFactory.is_supported("test.xlsx") is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])