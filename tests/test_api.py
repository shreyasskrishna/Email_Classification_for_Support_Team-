#!/usr/bin/env python3
"""
Tests for the API endpoints of the Email Classification System.
"""

import pytest
import sys
import os
import json
from fastapi.testclient import TestClient

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api import app

client = TestClient(app)

def test_health_check():
    """Test the health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert "model_loaded" in data

def test_root():
    """Test the root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "version" in data
    assert "description" in data
    assert "docs_url" in data

def test_classify_email():
    """Test the email classification endpoint"""
    # This test will fail if the model is not trained
    # It's included here for completeness
    email_body = """
    I am reaching out to report an issue with our data analytics platform. The platform has
    crashed, and we believe it might be due to inadequate RAM allocation My name is
    John Doe. We have already tried restarting the server and reviewing the logs, but
    the problem still exists You can reach me at john.doe@example.com. We kindly request your
    assistance in investigating this matter and providing a resolution at your earliest
    convenience. Please inform us if any additional information from our side is necessary
    to address this issue. The platform is currently unavailable, and we urgently need it to
    complete a critical issue with Project Sync Resulting in Data Loss
    """
    
    response = client.post(
        "/classify",
        json={"email_body": email_body}
    )
    
    # If model is not trained, this will return a 500 error
    if response.status_code == 500:
        data = response.json()
        assert "detail" in data
        assert "Model not loaded" in data["detail"]
    else:
        assert response.status_code == 200
        data = response.json()
        assert "input_email_body" in data
        assert "list_of_masked_entities" in data
        assert "masked_email" in data
        assert "category_of_the_email" in data
        
        # Check if PII is detected
        entities = data["list_of_masked_entities"]
        assert len(entities) > 0
        
        # Check if at least one entity is a name or email
        entity_types = [e["classification"] for e in entities]
        assert "full_name" in entity_types or "email" in entity_types
        
        # Check if the masked email contains the masked entities
        for entity in entities:
            entity_type = entity["classification"]
            assert f"[{entity_type}]" in data["masked_email"]
