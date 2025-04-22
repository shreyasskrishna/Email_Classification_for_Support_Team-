#!/usr/bin/env python3
"""Configuration settings for the Email Classification System"""

import os
from datetime import datetime

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "dataset")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# Create directories if they don't exist
for directory in [MODEL_DIR, DATA_DIR, LOG_DIR]:
    os.makedirs(directory, exist_ok=True)

# Model settings
MODEL_PATH = os.path.join(MODEL_DIR, "email_classifier.joblib")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")

# API settings
API_HOST = "0.0.0.0"
API_PORT = 8000
API_WORKERS = 4

# Dataset settings
DATASET_PATH = os.path.join(DATA_DIR, "emails.csv")
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Logging settings
LOG_LEVEL = "INFO"
LOG_FILE = os.path.join(LOG_DIR, f"app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# PII masking settings
PII_ENTITY_TYPES = [
    "full_name",
    "email",
    "phone_number",
    "dob",
    "aadhar_num",
    "credit_debit_no",
    "cvv_no",
    "expiry_no"
]

# Classification settings
CATEGORIES = [
    "Incident",
    "Request",
    "Problem",
    "Change"
]
