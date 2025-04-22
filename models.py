
import pandas as pd
import numpy as np
import joblib
import os
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Any, List, Optional, Union

from config import MODEL_PATH, VECTORIZER_PATH, TEST_SIZE, RANDOM_STATE
from utils import preprocess_text

# Set up logging
logger = logging.getLogger(__name__)

class EmailClassifier:
    """
    Email classification model using Logistic Regression
    """
    
    def __init__(self, model_path: str = MODEL_PATH):
        """
        Initialize the EmailClassifier
        
        Args:
            model_path: Path to save/load the model
        """
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('classifier', LogisticRegression(max_iter=1000, C=1.0, solver='liblinear'))
        ])
        self.model_path = model_path
        self.classes = None
        
    def train(self, X_train: List[str], y_train: List[str], 
              perform_grid_search: bool = False) -> None:
        """
        Train the email classification model
        
        Args:
            X_train: List of email texts
            y_train: List of email categories
            perform_grid_search: Whether to perform grid search for hyperparameter tuning
        """
        logger.info("Starting model training")
        self.classes = sorted(list(set(y_train)))
        logger.info(f"Found {len(self.classes)} classes: {self.classes}")
        
        if perform_grid_search:
            logger.info("Performing grid search for hyperparameter tuning")
            param_grid = {
                'tfidf__max_features': [3000, 5000, 10000],
                'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
                'classifier__C': [0.1, 1.0, 10.0],
                'classifier__solver': ['liblinear', 'saga']
            }
            
            grid_search = GridSearchCV(
                self.pipeline, 
                param_grid, 
                cv=5, 
                scoring='f1_weighted',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            logger.info(f"Best parameters: {grid_search.best_params_}")
            logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
            
            self.pipeline = grid_search.best_estimator_
        else:
            logger.info("Training with default parameters")
            self.pipeline.fit(X_train, y_train)
        
        logger.info("Model training completed")
        
    def predict(self, text: str) -> str:
        """
        Predict the category of an email
        
        Args:
            text: Email text
            
        Returns:
            Predicted category
        """
        # Preprocess the text
        processed_text = preprocess_text(text)
        
        # Make prediction
        return self.pipeline.predict([processed_text])[0]
    
    def predict_proba(self, text: str) -> Dict[str, float]:
        """
        Get prediction probabilities for each category
        
        Args:
            text: Email text
            
        Returns:
            Dictionary mapping categories to probabilities
        """
        # Preprocess the text
        processed_text = preprocess_text(text)
        
        # Get probabilities
        proba = self.pipeline.predict_proba([processed_text])[0]
        
        # Map probabilities to classes
        return {cls: float(prob) for cls, prob in zip(self.pipeline.classes_, proba)}
    
    def save_model(self) -> None:
        """Save the trained model to disk"""
        logger.info(f"Saving model to {self.model_path}")
        joblib.dump(self.pipeline, self.model_path)
        logger.info("Model saved successfully")
        
    def load_model(self) -> None:
        """Load the trained model from disk"""
        logger.info(f"Loading model from {self.model_path}")
        if os.path.exists(self.model_path):
            self.pipeline = joblib.load(self.model_path)
            self.classes = self.pipeline.classes_
            logger.info(f"Model loaded successfully with classes: {self.classes}")
        else:
            logger.error(f"Model file {self.model_path} not found")
            raise FileNotFoundError(f"Model file {self.model_path} not found")
    
    def evaluate(self, X_test: List[str], y_test: List[str]) -> Dict[str, Any]:
       
        logger.info("Evaluating model performance")
        
        # Preprocess test data
        X_test_processed = [preprocess_text(text) for text in X_test]
        
        # Make predictions
        y_pred = self.pipeline.predict(X_test_processed)
        
        # Calculate metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Generate confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=self.classes)
        
        logger.info(f"Evaluation complete. F1 score: {f1:.4f}")
        
        return {
            'classification_report': report,
            'f1_score': f1,
            'confusion_matrix': cm,
            'classes': self.classes
        }
    
    def plot_confusion_matrix(self, cm: np.ndarray, classes: List[str], 
                             output_path: str = 'confusion_matrix.png') -> None:
        """
        Plot confusion matrix
        
        Args:
            cm: Confusion matrix
            classes: Class labels
            output_path: Path to save the plot
        """
        logger.info(f"Plotting confusion matrix to {output_path}")
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        logger.info("Confusion matrix plot saved")

def train_model(data: Union[str, pd.DataFrame], 
                test_size: float = TEST_SIZE, 
                random_state: int = RANDOM_STATE,
                perform_grid_search: bool = False) -> Tuple[EmailClassifier, Dict[str, Any]]:
    """
    Train and evaluate the email classification model
    
    Args:
        data: Path to the dataset or DataFrame
        test_size: Proportion of the dataset to include in the test split
        random_state: Random seed for reproducibility
        perform_grid_search: Whether to perform grid search for hyperparameter tuning
        
    Returns:
        Trained model and evaluation metrics
    """
    logger.info("Starting model training process")
    
    # Load dataset if path is provided
    if isinstance(data, str):
        logger.info(f"Loading dataset from {data}")
        if data.endswith('.csv'):
            df = pd.read_csv(data)
        elif data.endswith('.json'):
            df = pd.read_json(data)
        else:
            raise ValueError(f"Unsupported file format: {data}")
    else:
        df = data
    
    logger.info(f"Dataset loaded with {len(df)} records")
    
    # Check if required columns exist
    if 'email' not in df.columns or 'type' not in df.columns:
        logger.error("Dataset must contain 'email' and 'type' columns")
        raise ValueError("Dataset must contain 'email' and 'type' columns")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['email'].values, 
        df['type'].values, 
        test_size=test_size, 
        random_state=random_state,
        stratify=df['type'] if len(df['type'].unique()) > 1 else None
    )
    
    logger.info(f"Data split into {len(X_train)} training and {len(X_test)} testing samples")
    
    # Initialize and train model
    classifier = EmailClassifier()
    classifier.train(X_train, y_train, perform_grid_search=perform_grid_search)
    
    # Evaluate model
    metrics = classifier.evaluate(X_test, y_test)
    
    # Plot confusion matrix
    classifier.plot_confusion_matrix(metrics['confusion_matrix'], metrics['classes'])
    
    # Save model
    classifier.save_model()
    
    logger.info("Model training and evaluation complete")
    
    return classifier, metrics
