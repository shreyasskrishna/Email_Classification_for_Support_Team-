#!/usr/bin/env python3
"""
Tests for the utility functions of the Email Classification System.
"""

import pytest
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import detect_pii, mask_pii, unmask_pii, preprocess_text

def test_detect_pii():
    """Test PII detection"""
    # Test with email and name
    text = "Hello, my name is John Doe and my email is john.doe@example.com"
    entities = detect_pii(text)
    
    # Check if entities are detected
    assert len(entities) >= 2
    
    # Check if name and email are detected
    entity_types = [e["classification"] for e in entities]
    assert "full_name" in entity_types
    assert "email" in entity_types
    
    # Test with phone number
    text = "Please call me at (123) 456-7890"
    entities = detect_pii(text)
    
    # Check if phone number is detected
    assert len(entities) >= 1
    assert "phone_number" in [e["classification"] for e in entities]
    
    # Test with credit card number
    text = "My credit card number is 1234 5678 9012 3456"
    entities = detect_pii(text)
    
    # Check if credit card number is detected
    assert len(entities) >= 1
    assert "credit_debit_no" in [e["classification"] for e in entities]

def test_mask_pii():
    """Test PII masking"""
    text = "Hello, my name is John Doe and my email is john.doe@example.com"
    entities = detect_pii(text)
    masked_text = mask_pii(text, entities)
    
    # Check if entities are masked
    assert "John Doe" not in masked_text
    assert "john.doe@example.com" not in masked_text
    
    # Check if masks are present
    assert "[full_name]" in masked_text
    assert "[email]" in masked_text

def test_unmask_pii():
    """Test PII unmasking"""
    text = "Hello, my name is John Doe and my email is john.doe@example.com"
    entities = detect_pii(text)
    masked_text = mask_pii(text, entities)
    unmasked_text = unmask_pii(masked_text, entities)
    
    # Check if original text is restored
    assert unmasked_text == text

def test_preprocess_text():
    """Test text preprocessing"""
    text = "Hello, this is a test! With some punctuation."
    processed_text = preprocess_text(text)
    
    # Check if text is preprocessed correctly
    assert processed_text == "hello this is a test with some punctuation"
    
    # Check if special characters are removed
    assert "," not in processed_text
    assert "!" not in processed_text
    assert "." not in processed_text
    
    # Check if text is lowercase
    assert processed_text == processed_text.lower()
