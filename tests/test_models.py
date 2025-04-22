#!/usr/bin/env python3
"""
Tests for the model functions of the Email Classification System.
"""

import pytest
import sys
import os
import pandas as pd
import numpy as np

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import EmailClassifier

@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    data = {
        'email': [
            "I am having an issue with the platform. It keeps crashing.",
            "Please help me with my account setup request.",
            "I need technical support for my application.",
            "The system is down and we need urgent assistance.",
            "I would like to request a feature enhancement."
        ],
        'type': [
            "Problem",
            "Request",
            "Request",
            "Incident",
            "Change"
        ]
    }
    return pd.DataFrame(data)

def test_email_classifier_init():
    """Test EmailClassifier initialization"""
    classifier = EmailClassifier()
    assert classifier.pipeline is not None
    assert hasattr(classifier.pipeline, 'steps')
    assert len(classifier.pipeline.steps) == 2
    assert classifier.pipeline.steps[0][0] == 'tfidf'
    assert classifier.pipeline.steps[1][0] == 'classifier'

def test_email_classifier_train(sample_data):
    """Test EmailClassifier training"""
    classifier = EmailClassifier()
    X_train = sample_data['email'].values
    y_train = sample_data['type'].values
    
    classifier.train(X_train, y_train)
    
    # Check if classes are set
    assert classifier.classes is not None
    assert len(classifier.classes) == len(set(y_train))
    
    # Check if pipeline is fitted
    assert hasattr(classifier.pipeline, 'classes_')
    assert len(classifier.pipeline.classes_) == len(set(y_train))

def test_email_classifier_predict(sample_data):
    """Test EmailClassifier prediction"""
    classifier = EmailClassifier()
    X_train = sample_data['email'].values
    y_train = sample_data['type'].values
    
    classifier.train(X_train, y_train)
    
    # Test prediction
    prediction = classifier.predict("I have a problem with my account")
    assert prediction in classifier.classes
    
    # Test prediction with new text
    prediction = classifier.predict("The system crashed and is not responding")
    assert prediction in classifier.classes

def test_email_classifier_evaluate(sample_data):
    """Test EmailClassifier evaluation"""
    classifier = EmailClassifier()
    X = sample_data['email'].values
    y = sample_data['type'].values
    
    # Use the same data for training and testing (not recommended in practice)
    classifier.train(X, y)
    metrics = classifier.evaluate(X, y)
    
    # Check if metrics are returned
    assert 'classification_report' in metrics
    assert 'f1_score' in metrics
    assert 'confusion_matrix' in metrics
    assert 'classes' in metrics
    
    # Check if f1_score is a float
    assert isinstance(metrics['f1_score'], float)
    
    # Check if confusion_matrix has correct shape
    assert metrics['confusion_matrix'].shape == (len(classifier.classes), len(classifier.classes))
