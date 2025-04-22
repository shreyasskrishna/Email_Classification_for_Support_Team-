#!/usr/bin/env python3
"""
API implementation for the Email Classification System.
This module defines the FastAPI endpoints for email classification and PII masking.
"""

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import joblib
import os
import logging
import time
from datetime import datetime

from utils import detect_pii, mask_pii, setup_logging
from models import EmailClassifier
from config import MODEL_PATH

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Email Classification API",
    description="API for classifying emails and masking PII",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model
classifier = EmailClassifier()
try:
    classifier.load_model()
    logger.info("Model loaded successfully")
except FileNotFoundError:
    logger.warning(f"Model file {MODEL_PATH} not found. API will return errors until model is trained.")

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"Request: {request.method} {request.url.path} - Status: {response.status_code} - Time: {process_time:.4f}s")
    return response

# Input and output models
class EmailInput(BaseModel):
    """Email input model"""
    email_body: str = Field(..., description="The body of the email to classify")

class MaskedEntity(BaseModel):
    """Masked entity model"""
    position: List[int] = Field(..., description="Start and end positions of the entity in the text")
    classification: str = Field(..., description="Type of the entity (e.g., full_name, email)")
    entity: str = Field(..., description="Original value of the entity")

class EmailOutput(BaseModel):
    """Email output model"""
    input_email_body: str = Field(..., description="Original email body")
    list_of_masked_entities: List[MaskedEntity] = Field(..., description="List of detected PII entities")
    masked_email: str = Field(..., description="Email with PII masked")
    category_of_the_email: str = Field(..., description="Predicted category of the email")

# API endpoints
@app.post("/classify", response_model=EmailOutput, tags=["Classification"])
async def classify_email(email_input: EmailInput) -> Dict[str, Any]:
    """
    Classify an email and mask PII
    
    This endpoint takes an email body, detects and masks PII, and classifies the email into a category.
    
    Args:
        email_input: Email body
        
    Returns:
        Classification result and masked email
    """
    # Check if model is loaded
    if not hasattr(classifier, 'pipeline') or classifier.pipeline is None:
        logger.error("Model not loaded. Please train the model first.")
        raise HTTPException(status_code=500, detail="Model not loaded. Please train the model first.")
    
    try:
        # Get email body
        email_body = email_input.email_body
        logger.debug(f"Received email with length: {len(email_body)}")
        
        # Detect PII
        entities = detect_pii(email_body)
        logger.debug(f"Detected {len(entities)} PII entities")
        
        # Mask PII
        masked_email = mask_pii(email_body, entities)
        
        # Classify email
        category = classifier.predict(masked_email)
        logger.info(f"Email classified as: {category}")
        
        # Prepare response
        response = {
            "input_email_body": email_body,
            "list_of_masked_entities": entities,
            "masked_email": masked_email,
            "category_of_the_email": category
        }
        
        return response
    
    except Exception as e:
        logger.error(f"Error processing email: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing email: {str(e)}")

@app.get("/health", tags=["Monitoring"])
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint
    
    Returns the status of the API and whether the model is loaded
    """
    model_loaded = hasattr(classifier, 'pipeline') and classifier.pipeline is not None
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model_loaded
    }

@app.get("/", tags=["Info"])
async def root() -> Dict[str, str]:
    """
    Root endpoint
    
    Returns basic information about the API
    """
    return {
        "name": "Email Classification API",
        "version": "1.0.0",
        "description": "API for classifying emails and masking PII",
        "docs_url": "/docs"
    }
