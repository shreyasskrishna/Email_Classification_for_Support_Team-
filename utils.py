#!/usr/bin/env python3
"""
Utility functions for the Email Classification System.
This module provides functions for PII masking, text preprocessing, and other utilities.
"""

import re
import spacy
import pandas as pd
import logging
import os
from typing import Dict, List, Tuple, Any, Optional, Union
import sys
from datetime import datetime

# Configure logging
def setup_logging(log_file: Optional[str] = None, log_level: str = "INFO") -> None:
    """
    Set up logging configuration
    
    Args:
        log_file: Path to log file (if None, logs to console only)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Create logs directory if it doesn't exist
    if log_file and not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file))
    
    # Set up logging format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Configure logging
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format=log_format,
        handlers=handlers
    )

# Load spaCy model for NER
try:
    nlp = spacy.load("en_core_web_md")
except:
    # If model not found, download it
    import spacy.cli
    spacy.cli.download("en_core_web_md")
    nlp = spacy.load("en_core_web_md")

# Define regex patterns for PII detection
PII_PATTERNS = {
    "full_name": r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',
    "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    "phone_number": r'\b(?:\+\d{1,3}[- ]?)?$$?\d{3}$$?[- ]?\d{3}[- ]?\d{4}\b',
    "dob": r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
    "aadhar_num": r'\b\d{4}[ -]?\d{4}[ -]?\d{4}\b',
    "credit_debit_no": r'\b(?:\d{4}[ -]?){4}\b',
    "cvv_no": r'\bCVV:? \d{3,4}\b',
    "expiry_no": r'\b(?:0[1-9]|1[0-2])[/-]\d{2,4}\b'
}

def detect_pii(text: str) -> List[Dict[str, Any]]:
    """
    Detect PII in text using regex patterns and NER
    
    Args:
        text: Input text to scan for PII
        
    Returns:
        List of dictionaries containing detected PII entities
    """
    entities = []
    
    # Use regex patterns to detect PII
    for entity_type, pattern in PII_PATTERNS.items():
        for match in re.finditer(pattern, text):
            entities.append({
                "position": [match.start(), match.end()],
                "classification": entity_type,
                "entity": match.group()
            })
    
    # Use spaCy NER to detect names and other entities
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON" and not any(e["position"][0] <= ent.start_char and e["position"][1] >= ent.end_char for e in entities):
            entities.append({
                "position": [ent.start_char, ent.end_char],
                "classification": "full_name",
                "entity": ent.text
            })
    
    # Sort entities by position
    entities.sort(key=lambda x: x["position"][0])
    
    return entities

def mask_pii(text: str, entities: List[Dict[str, Any]]) -> str:
    """
    Mask PII in text based on detected entities
    
    Args:
        text: Original text
        entities: List of detected PII entities
        
    Returns:
        Text with PII masked
    """
    # Sort entities in reverse order to avoid position shifts
    sorted_entities = sorted(entities, key=lambda x: x["position"][0], reverse=True)
    
    # Create a mutable list of characters
    chars = list(text)
    
    # Replace each entity with its mask
    for entity in sorted_entities:
        start, end = entity["position"]
        entity_type = entity["classification"]
        mask = f"[{entity_type}]"
        
        # Replace the entity with the mask
        chars[start:end] = list(mask)
    
    return ''.join(chars)

def unmask_pii(masked_text: str, entities: List[Dict[str, Any]]) -> str:
    """
    Restore original text from masked text
    
    Args:
        masked_text: Text with PII masked
        entities: List of detected PII entities
        
    Returns:
        Original text with PII restored
    """
    # Create a copy of the masked text
    unmasked_text = masked_text
    
    # Replace each mask with the original entity
    for entity in entities:
        start, end = entity["position"]
        entity_value = entity["entity"]
        entity_type = entity["classification"]
        mask = f"[{entity_type}]"
        
        # Replace the mask with the original entity
        unmasked_text = unmasked_text.replace(mask, entity_value, 1)
    
    return unmasked_text

def preprocess_text(text: str) -> str:
    """
    Preprocess text for classification
    
    Args:
        text: Input text
        
    Returns:
        Preprocessed text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and extra whitespace
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load and preprocess the email dataset
    
    Args:
        file_path: Path to the dataset file
        
    Returns:
        Preprocessed DataFrame
    """
    # Determine file type from extension
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.json'):
        df = pd.read_json(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    
    # Basic preprocessing
    if 'email' in df.columns:
        df['processed_text'] = df['email'].apply(preprocess_text)
    
    return df

def extract_features(text: str) -> Dict[str, int]:
    """
    Extract features from text for classification
    
    Args:
        text: Input text
        
    Returns:
        Dictionary of features
    """
    # Simple feature extraction based on keyword presence
    features = {
        'has_issue': 1 if 'issue' in text.lower() else 0,
        'has_problem': 1 if 'problem' in text.lower() else 0,
        'has_request': 1 if 'request' in text.lower() else 0,
        'has_urgent': 1 if 'urgent' in text.lower() else 0,
        'has_support': 1 if 'support' in text.lower() else 0,
        'has_data': 1 if 'data' in text.lower() else 0,
        'has_security': 1 if 'security' in text.lower() else 0,
        'has_update': 1 if 'update' in text.lower() else 0,
    }
    
    return features

def save_results(results: Dict[str, Any], output_path: str) -> None:
    """
    Save classification results to a file
    
    Args:
        results: Dictionary of classification results
        output_path: Path to save the results
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save results as JSON
    if output_path.endswith('.json'):
        import json
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
    # Save results as CSV
    elif output_path.endswith('.csv'):
        pd.DataFrame([results]).to_csv(output_path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {output_path}")
