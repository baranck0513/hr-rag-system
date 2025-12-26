"""
Tests for the PII masker service.
"""

import pytest
from app.services.pii_masker import PIIMasker


class TestPIIMasker:
    """Tests for PIIMasker class."""
    
    @pytest.fixture
    def masker(self):
        """Create a PIIMasker instance for tests."""
        return PIIMasker()
    
    # --- National Insurance Number Tests ---
    
    def test_mask_ni_number_standard_format(self, masker):
        """Should mask standard NI number format."""
        text = "Employee NI: AB123456C"
        result = masker.mask(text)
        
        assert result == "Employee NI: [NI_NUMBER]"
        assert "AB123456C" not in result
    
    def test_mask_ni_number_lowercase(self, masker):
        """Should mask lowercase NI numbers."""
        text = "NI number is ab123456c"
        result = masker.mask(text)
        
        assert result == "NI number is [NI_NUMBER]"
    
    def test_mask_ni_number_in_sentence(self, masker):
        """Should mask NI number within longer text."""
        text = "Please update record for AB123456C in the system."
        result = masker.mask(text)
        
        assert "[NI_NUMBER]" in result
        assert "AB123456C" not in result
    
    # --- Email Tests ---
    
    def test_mask_email_simple(self, masker):
        """Should mask simple email addresses."""
        text = "Contact: john.smith@company.com"
        result = masker.mask(text)
        
        assert result == "Contact: [EMAIL]"
    
    def test_mask_email_with_numbers(self, masker):
        """Should mask emails containing numbers."""
        text = "Email hr.team2024@example.co.uk for queries"
        result = masker.mask(text)
        
        assert "[EMAIL]" in result
        assert "@" not in result
    
    def test_mask_multiple_emails(self, masker):
        """Should mask multiple emails in same text."""
        text = "From: alice@test.com To: bob@test.com"
        result = masker.mask(text)
        
        assert result == "From: [EMAIL] To: [EMAIL]"
    
    # --- Phone Number Tests ---
    
    def test_mask_uk_mobile_with_spaces(self, masker):
        """Should mask UK mobile numbers with spaces."""
        text = "Call me on 07700 900123"
        result = masker.mask(text)
        
        assert "[PHONE]" in result
        assert "07700" not in result
    
    def test_mask_uk_mobile_no_spaces(self, masker):
        """Should mask UK mobile numbers without spaces."""
        text = "Mobile: 07700900123"
        result = masker.mask(text)
        
        assert "[PHONE]" in result
    
    def test_mask_uk_mobile_international_format(self, masker):
        """Should mask UK mobile in international format."""
        text = "Phone: +44 7700 900123"
        result = masker.mask(text)
        
        assert "[PHONE]" in result
        assert "+44" not in result
    
    # --- Sort Code Tests ---
    
    def test_mask_sort_code_with_hyphens(self, masker):
        """Should mask sort codes with hyphens."""
        text = "Sort code: 12-34-56"
        result = masker.mask(text)
        
        assert result == "Sort code: [SORT_CODE]"
    
    def test_mask_sort_code_with_spaces(self, masker):
        """Should not mask sort codes with spaces (to avoid false positives)."""
        text = "Sort code: 12 34 56"
        result = masker.mask(text)
        
        # With stricter pattern, spaces are not matched to avoid false positives
        # Only hyphenated format is matched: 12-34-56
        assert result == "Sort code: 12 34 56"
    
    # --- Bank Account Tests ---
    
    def test_mask_bank_account(self, masker):
        """Should mask 8-digit bank account numbers."""
        text = "Account number: 12345678"
        result = masker.mask(text)
        
        assert result == "Account number: [BANK_ACCOUNT]"
    
    # --- Date of Birth Tests ---
    
    def test_mask_dob_slash_format(self, masker):
        """Should mask DOB in DD/MM/YYYY format."""
        text = "Date of birth: 15/03/1990"
        result = masker.mask(text)
        
        assert result == "Date of birth: [DOB]"
    
    def test_mask_dob_hyphen_format(self, masker):
        """Should mask DOB in DD-MM-YYYY format."""
        text = "DOB: 01-12-1985"
        result = masker.mask(text)
        
        assert "[DOB]" in result
    
    # --- Postcode Tests ---
    
    def test_mask_postcode_standard(self, masker):
        """Should mask standard UK postcodes."""
        text = "Address: 10 Downing Street, SW1A 2AA"
        result = masker.mask(text)
        
        assert "[POSTCODE]" in result
        assert "SW1A 2AA" not in result
    
    def test_mask_postcode_short_format(self, masker):
        """Should mask short format postcodes."""
        text = "Located in M1 1AE area"
        result = masker.mask(text)
        
        assert "[POSTCODE]" in result
    
    # --- Combined Tests ---
    
    def test_mask_multiple_pii_types(self, masker):
        """Should mask multiple PII types in same text."""
        text = (
            "Employee John Smith (NI: AB123456C) can be reached at "
            "john.smith@company.com or 07700 900123."
        )
        result = masker.mask(text)
        
        assert "[NI_NUMBER]" in result
        assert "[EMAIL]" in result
        assert "[PHONE]" in result
        assert "AB123456C" not in result
        assert "john.smith@company.com" not in result
        assert "07700" not in result
    
    def test_mask_empty_string(self, masker):
        """Should handle empty strings."""
        result = masker.mask("")
        assert result == ""
    
    def test_mask_no_pii(self, masker):
        """Should return unchanged text when no PII present."""
        text = "This is a normal sentence with no PII."
        result = masker.mask(text)
        
        assert result == text
    
    # --- Stats Tests ---
    
    def test_mask_with_stats_returns_counts(self, masker):
        """Should return correct counts of masked items."""
        text = "Contact john@test.com or jane@test.com about NI AB123456C"
        
        masked_text, stats = masker.mask_with_stats(text)
        
        assert stats["EMAIL"] == 2
        assert stats["NI_NUMBER"] == 1
        assert "[EMAIL]" in masked_text
        assert "[NI_NUMBER]" in masked_text
    
    def test_mask_with_stats_empty_when_no_pii(self, masker):
        """Should return empty stats when no PII found."""
        text = "No personal information here."
        
        masked_text, stats = masker.mask_with_stats(text)
        
        assert stats == {}
        assert masked_text == text


class TestPIIMaskerDetection:
    """Tests for PII detection without masking."""
    
    @pytest.fixture
    def masker(self):
        return PIIMasker()
    
    def test_detect_only_returns_actual_values(self, masker):
        """Should return actual PII values found."""
        text = "NI number is AB123456C and email is test@example.com"
        
        detections = masker.detect_only(text)
        
        assert "AB123456C" in detections.get("NI_NUMBER", [])
        assert "test@example.com" in detections.get("EMAIL", [])
    
    def test_detect_only_empty_when_no_pii(self, masker):
        """Should return empty dict when no PII found."""
        text = "No personal data here"
        
        detections = masker.detect_only(text)
        
        assert detections == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])