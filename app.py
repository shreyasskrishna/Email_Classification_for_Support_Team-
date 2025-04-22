import uvicorn
import argparse
import os
import pandas as pd
import logging
from typing import Dict, Any, Optional

from models import train_model, EmailClassifier
from utils import load_dataset, setup_logging
from config import (
    DATASET_PATH, 
    API_HOST, 
    API_PORT, 
    API_WORKERS, 
    MODEL_PATH,
    LOG_LEVEL
)

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)

def main() -> None:
    """Main function to run the application"""
    parser = argparse.ArgumentParser(description="Email Classification System")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--data", type=str, default=DATASET_PATH, help="Path to the dataset")
    parser.add_argument("--serve", action="store_true", help="Start the API server")
    parser.add_argument("--port", type=int, default=API_PORT, help="Port for the API server")
    parser.add_argument("--host", type=str, default=API_HOST, help="Host for the API server")
    parser.add_argument("--workers", type=int, default=API_WORKERS, help="Number of workers for the API server")
    parser.add_argument("--log-level", type=str, default=LOG_LEVEL, 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level")
    
    args = parser.parse_args()
    
    # Set log level from command line if provided
    if args.log_level:
        logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    if args.train:
        logger.info(f"Training model using dataset: {args.data}")
        if not os.path.exists(args.data):
            logger.error(f"Dataset file {args.data} not found")
            return
        
        try:
            # Load dataset
            df = load_dataset(args.data)
            logger.info(f"Loaded dataset with {len(df)} records")
            
            # Train model
            classifier, metrics = train_model(df)
            
            logger.info("\nTraining completed!")
            logger.info(f"F1 Score: {metrics['f1_score']:.4f}")
            logger.info("Classification Report:")
            for cls, values in metrics['classification_report'].items():
                if isinstance(values, dict):
                    logger.info(f"  {cls}:")
                    for metric, value in values.items():
                        if isinstance(value, float):
                            logger.info(f"    {metric}: {value:.4f}")
                        else:
                            logger.info(f"    {metric}: {value}")
            
            logger.info("\nConfusion Matrix saved to confusion_matrix.png")
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise
    
    if args.serve:
        logger.info(f"Starting API server on {args.host}:{args.port} with {args.workers} workers")
        try:
            uvicorn.run(
                "api:app", 
                host=args.host, 
                port=args.port, 
                workers=args.workers,
                log_level=args.log_level.lower()
            )
        except Exception as e:
            logger.error(f"Error starting API server: {str(e)}")
            raise
    
    if not args.train and not args.serve:
        logger.warning("No action specified. Use --train to train the model or --serve to start the API server.")
        parser.print_help()

if __name__ == "__main__":
    main()
