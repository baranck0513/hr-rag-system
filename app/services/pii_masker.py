"""
PII Masker Service

Detects and masks Personally Identifiable Information (PII) in text.
Focused on UK-specific formats commonly found in HR documents.

This module uses regex patterns for predictable formats (NI numbers, 
phone numbers, etc.) and can be extended with ML-based detection 
for names and addresses using Microsoft Presidio.
"""

import re
from dataclasses import dataclass
from typing import Callable
import logging

logger = logging.getLogger(__name__)


@dataclass
class PIIPattern:
    """
    Represents a PII detection pattern.
    
    Attributes:
        name: Human-readable name for this PII type
        pattern: Compiled regex pattern
        replacement: What to replace matches with
    """
    name: str
    pattern: re.Pattern
    replacement: str


class PIIMasker:
    """
    Masks PII in text using regex patterns.
    
    This class is designed for UK HR documents and handles:
    - National Insurance numbers
    - UK phone numbers
    - Email addresses
    - Sort codes
    - Bank account numbers
    - Dates of birth (UK format)
    
    Usage:
        masker = PIIMasker()
        masked_text = masker.mask(original_text)
        
    Example:
        >>> masker = PIIMasker()
        >>> masker.mask("Contact John at john@example.com")
        "Contact John at [EMAIL]"
    """
    
    def __init__(self):
        """Initialise the masker with UK-specific PII patterns."""
        self.patterns = self._build_patterns()
    
    def _build_patterns(self) -> list[PIIPattern]:
        """
        Build the list of PII patterns to detect.
        
        Returns:
            List of PIIPattern objects, ordered by specificity
            (more specific patterns first to avoid partial matches)
            
        IMPORTANT: Order matters! More specific patterns must come before
        less specific ones to prevent partial matching.
        """
        return [
            # --- MOST SPECIFIC PATTERNS FIRST ---
            
            # National Insurance Number: AB123456C
            # Format: 2 letters, 6 digits, 1 letter
            # Note: First letter cannot be D, F, I, Q, U, V
            # Second letter cannot be D, F, I, O, Q, U, V
            PIIPattern(
                name="NI_NUMBER",
                pattern=re.compile(
                    r"\b[A-CEGHJ-PR-TW-Z][A-CEGHJ-NPR-TW-Z]\d{6}[A-D]\b",
                    re.IGNORECASE
                ),
                replacement="[NI_NUMBER]"
            ),
            
            # Date of Birth - UK formats (MUST come before sort code!)
            # DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY
            PIIPattern(
                name="DOB",
                pattern=re.compile(
                    r"\b(?:0[1-9]|[12]\d|3[01])[-/.](?:0[1-9]|1[0-2])[-/.](?:19|20)\d{2}\b"
                ),
                replacement="[DOB]"
            ),
            
            # UK Phone Numbers - multiple formats (MUST come before sort code!)
            # Mobile: 07xxx xxxxxx or 07xxx-xxx-xxx or +44 7xxx xxxxxx
            # Landline: 01xxx or 02xxx
            PIIPattern(
                name="PHONE_UK",
                pattern=re.compile(
                    r"(?:\+44\s?)?(?:0|\(0\))?\s?7\d{3}[\s.-]?\d{3}[\s.-]?\d{3}"  # Mobile
                    r"|(?:\+44\s?)?(?:0|\(0\))?\s?[1-2]\d{2,3}[\s.-]?\d{3}[\s.-]?\d{3,4}"  # Landline
                ),
                replacement="[PHONE]"
            ),
            
            # Email addresses
            PIIPattern(
                name="EMAIL",
                pattern=re.compile(
                    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
                ),
                replacement="[EMAIL]"
            ),
            
            # Postcode (UK): e.g., SW1A 1AA, M1 1AE, B33 8TH
            PIIPattern(
                name="POSTCODE",
                pattern=re.compile(
                    r"\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b",
                    re.IGNORECASE
                ),
                replacement="[POSTCODE]"
            ),
            
            # --- LESS SPECIFIC PATTERNS LAST ---
            
            # Sort Code: 12-34-56 (with hyphens only, to avoid false matches)
            # Requiring hyphens makes this more specific
            PIIPattern(
                name="SORT_CODE",
                pattern=re.compile(
                    r"\b\d{2}-\d{2}-\d{2}\b"
                ),
                replacement="[SORT_CODE]"
            ),
            
            # UK Bank Account Number: 8 digits
            PIIPattern(
                name="BANK_ACCOUNT",
                pattern=re.compile(
                    r"\b\d{8}\b"
                ),
                replacement="[BANK_ACCOUNT]"
            ),
            
            # Passport Number (UK): 9 digits
            PIIPattern(
                name="PASSPORT",
                pattern=re.compile(
                    r"\b\d{9}\b"
                ),
                replacement="[PASSPORT]"
            ),
        ]
    
    def mask(self, text: str) -> str:
        """
        Mask all detected PII in the given text.
        
        Args:
            text: The original text potentially containing PII
            
        Returns:
            Text with all detected PII replaced with placeholders
        """
        if not text:
            return text
        
        masked_text = text
        
        for pii_pattern in self.patterns:
            matches = pii_pattern.pattern.findall(masked_text)
            if matches:
                logger.debug(
                    f"Found {len(matches)} {pii_pattern.name} matches"
                )
            masked_text = pii_pattern.pattern.sub(
                pii_pattern.replacement, 
                masked_text
            )
        
        return masked_text
    
    def mask_with_stats(self, text: str) -> tuple[str, dict[str, int]]:
        """
        Mask PII and return statistics about what was masked.
        
        Useful for logging and auditing purposes.
        
        Args:
            text: The original text potentially containing PII
            
        Returns:
            Tuple of (masked_text, stats_dict)
            stats_dict maps PII type names to count of matches
        """
        if not text:
            return text, {}
        
        masked_text = text
        stats: dict[str, int] = {}
        
        for pii_pattern in self.patterns:
            matches = pii_pattern.pattern.findall(masked_text)
            if matches:
                stats[pii_pattern.name] = len(matches)
                masked_text = pii_pattern.pattern.sub(
                    pii_pattern.replacement, 
                    masked_text
                )
        
        return masked_text, stats
    
    def detect_only(self, text: str) -> dict[str, list[str]]:
        """
        Detect PII without masking (for analysis purposes).
        
        WARNING: The returned dict contains actual PII values.
        Use with caution and ensure proper data handling.
        
        Args:
            text: The text to analyse
            
        Returns:
            Dict mapping PII type names to list of detected values
        """
        if not text:
            return {}
        
        detections: dict[str, list[str]] = {}
        
        for pii_pattern in self.patterns:
            matches = pii_pattern.pattern.findall(text)
            if matches:
                detections[pii_pattern.name] = matches
        
        return detections