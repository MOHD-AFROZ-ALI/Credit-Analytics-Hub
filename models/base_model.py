"""
Base Model Module for Credit Analytics Hub

This module provides the foundational classes and utilities for building
and managing machine learning models in the credit analytics system.
"""

import os
import pickle
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor(BaseEstimator, TransformerMixin):
    """
    Data preprocessing pipeline for credit risk modeling.

    This class handles missing values, categorical encoding, feature scaling,
    and other preprocessing steps required for credit risk models.
    """

    def __init__(self, 
                 handle_missing: str = 'median',
                 encode_categorical: str = 'label',
                 scale_features: bool = True,
                 remove_outliers: bool = False,
                 outlier_threshold: float = 3.0):
        """
        Initialize the data preprocessor.

        Args:
            handle_missing: Strategy for handling missing values ('median', 'mean', 'mode', 'drop')
            encode_categorical: Method for encoding categorical variables ('label', 'onehot')
            scale_features: Whether to scale numerical features
            remove_outliers: Whether to remove outliers
            outlier_threshold: Z-score threshold for outlier detection
        """
        self.handle_missing = handle_missing
        self.encode_categorical = encode_categorical
        self.scale_features = scale_features
        self.remove_outliers = remove_outliers
        self.outlier_threshold = outlier_threshold

        # Initialize transformers
        self.scaler = StandardScaler() if scale_features else None
        self.label_encoders = {}
        self.feature_names = None
        self.numerical_features = None
        self.categorical_features = None
        self.missing_value_fills = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'DataPreprocessor':
        """
        Fit the preprocessor to the training data.

        Args:
            X: Input features
            y: Target variable (optional)

        Returns:
            self: Fitted preprocessor
        """
        X = X.copy()
        self.feature_names = X.columns.tolist()

        # Identify numerical and categorical features
        self.numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

        # Handle missing values
        self._fit_missing_value_strategy(X)

        # Fit categorical encoders
        if self.encode_categorical == 'label':
            for col in self.categorical_features:
                self.label_encoders[col] = LabelEncoder()
                # Handle missing values before fitting encoder
                non_null_values = X[col].dropna()
                if len(non_null_values) > 0:
                    self.label_encoders[col].fit(non_null_values)

        # Fit scaler on numerical features
        if self.scaler is not None and self.numerical_features:
            # Apply missing value handling first
            X_processed = self._handle_missing_values(X)
            numerical_data = X_processed[self.numerical_features]
            self.scaler.fit(numerical_data)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input data using fitted preprocessor.

        Args:
            X: Input features to transform

        Returns:
            Transformed features
        """
        X = X.copy()

        # Handle missing values
        X = self._handle_missing_values(X)

        # Remove outliers if specified
        if self.remove_outliers:
            X = self._remove_outliers(X)

        # Encode categorical variables
        X = self._encode_categorical_variables(X)

        # Scale numerical features
        if self.scaler is not None and self.numerical_features:
            X[self.numerical_features] = self.scaler.transform(X[self.numerical_features])

        return X

    def _fit_missing_value_strategy(self, X: pd.DataFrame) -> None:
        """Fit the missing value handling strategy."""
        for col in X.columns:
            if X[col].isnull().any():
                if col in self.numerical_features:
                    if self.handle_missing == 'median':
                        self.missing_value_fills[col] = X[col].median()
                    elif self.handle_missing == 'mean':
                        self.missing_value_fills[col] = X[col].mean()
                    else:
                        self.missing_value_fills[col] = X[col].median()  # default
                elif col in self.categorical_features:
                    if self.handle_missing == 'mode':
                        mode_value = X[col].mode()
                        self.missing_value_fills[col] = mode_value[0] if len(mode_value) > 0 else 'Unknown'
                    else:
                        self.missing_value_fills[col] = 'Unknown'

    def _handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        X = X.copy()

        if self.handle_missing == 'drop':
            return X.dropna()

        for col, fill_value in self.missing_value_fills.items():
            if col in X.columns:
                X[col] = X[col].fillna(fill_value)

        return X

    def _remove_outliers(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers using Z-score method."""
        X = X.copy()

        for col in self.numerical_features:
            if col in X.columns:
                z_scores = np.abs((X[col] - X[col].mean()) / X[col].std())
                X = X[z_scores <= self.outlier_threshold]

        return X

    def _encode_categorical_variables(self, X: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables."""
        X = X.copy()

        if self.encode_categorical == 'label':
            for col in self.categorical_features:
                if col in X.columns and col in self.label_encoders:
                    # Handle unseen categories
                    encoder = self.label_encoders[col]
                    X[col] = X[col].map(lambda x: encoder.transform([x])[0] 
                                      if x in encoder.classes_ else -1)

        return X

class ModelValidator:
    """
    Model validation utilities for credit risk models.

    Provides comprehensive validation metrics and cross-validation
    functionality for evaluating model performance.
    """

    def __init__(self, cv_folds: int = 5, random_state: int = 42):
        """
        Initialize the model validator.

        Args:
            cv_folds: Number of cross-validation folds
            random_state: Random state for reproducibility
        """
        self.cv_folds = cv_folds
        self.random_state = random_state

    def evaluate_model(self, 
                      model: Any, 
                      X: pd.DataFrame, 
                      y: pd.Series,
                      test_size: float = 0.2) -> Dict[str, float]:
        """
        Evaluate model performance using train-test split.

        Args:
            model: Trained model to evaluate
            X: Input features
            y: Target variable
            test_size: Proportion of data for testing

        Returns:
            Dictionary containing evaluation metrics
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted')
        }

        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)

        return metrics

    def cross_validate_model(self, 
                           model: Any, 
                           X: pd.DataFrame, 
                           y: pd.Series,
                           scoring: str = 'accuracy') -> Dict[str, float]:
        """
        Perform cross-validation on the model.

        Args:
            model: Model to validate
            X: Input features
            y: Target variable
            scoring: Scoring metric for cross-validation

        Returns:
            Dictionary containing cross-validation results
        """
        cv_scores = cross_val_score(
            model, X, y, cv=self.cv_folds, scoring=scoring, n_jobs=-1
        )

        return {
            f'cv_{scoring}_mean': cv_scores.mean(),
            f'cv_{scoring}_std': cv_scores.std(),
            f'cv_{scoring}_scores': cv_scores.tolist()
        }

class BaseModel(ABC):
    """
    Abstract base class for all credit risk models.

    This class defines the interface that all models must implement
    and provides common functionality for model management.
    """

    def __init__(self, 
                 model_name: str,
                 model_version: str = "1.0",
                 random_state: int = 42):
        """
        Initialize the base model.

        Args:
            model_name: Name of the model
            model_version: Version of the model
            random_state: Random state for reproducibility
        """
        self.model_name = model_name
        self.model_version = model_version
        self.random_state = random_state
        self.model = None
        self.preprocessor = None
        self.is_trained = False
        self.training_metrics = {}
        self.feature_importance = None
        self.training_timestamp = None

        # Initialize components
        self.validator = ModelValidator(random_state=random_state)

    @abstractmethod
    def _create_model(self) -> Any:
        """
        Create the underlying machine learning model.

        Returns:
            Initialized model instance
        """
        pass

    @abstractmethod
    def _get_model_params(self) -> Dict[str, Any]:
        """
        Get model-specific parameters.

        Returns:
            Dictionary of model parameters
        """
        pass

    def fit(self, 
            X: pd.DataFrame, 
            y: pd.Series,
            validation_split: float = 0.2,
            preprocess_data: bool = True) -> 'BaseModel':
        """
        Train the model on the provided data.

        Args:
            X: Input features
            y: Target variable
            validation_split: Proportion of data for validation
            preprocess_data: Whether to preprocess the data

        Returns:
            self: Trained model instance
        """
        logger.info(f"Training {self.model_name} model...")

        # Initialize model if not already done
        if self.model is None:
            self.model = self._create_model()

        # Preprocess data if required
        if preprocess_data:
            self.preprocessor = DataPreprocessor()
            self.preprocessor.fit(X, y)
            X_processed = self.preprocessor.transform(X)
        else:
            X_processed = X.copy()

        # Train the model
        self.model.fit(X_processed, y)
        self.is_trained = True
        self.training_timestamp = datetime.now()

        # Evaluate model performance
        self.training_metrics = self.validator.evaluate_model(
            self.model, X_processed, y, test_size=validation_split
        )

        # Calculate feature importance if available
        self._calculate_feature_importance(X_processed)

        logger.info(f"Model training completed. Accuracy: {self.training_metrics.get('accuracy', 'N/A'):.4f}")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.

        Args:
            X: Input features

        Returns:
            Predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        # Preprocess data if preprocessor is available
        if self.preprocessor is not None:
            X_processed = self.preprocessor.transform(X)
        else:
            X_processed = X.copy()

        return self.model.predict(X_processed)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Input features

        Returns:
            Class probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        if not hasattr(self.model, 'predict_proba'):
            raise ValueError("Model does not support probability predictions")

        # Preprocess data if preprocessor is available
        if self.preprocessor is not None:
            X_processed = self.preprocessor.transform(X)
        else:
            X_processed = X.copy()

        return self.model.predict_proba(X_processed)

    def _calculate_feature_importance(self, X: pd.DataFrame) -> None:
        """Calculate feature importance if the model supports it."""
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        elif hasattr(self.model, 'coef_'):
            # For linear models, use absolute coefficients
            coef = self.model.coef_[0] if len(self.model.coef_.shape) > 1 else self.model.coef_
            self.feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': np.abs(coef)
            }).sort_values('importance', ascending=False)

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance rankings.

        Returns:
            DataFrame with feature importance scores
        """
        return self.feature_importance

    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk.

        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")

        model_data = {
            'model': self.model,
            'preprocessor': self.preprocessor,
            'model_name': self.model_name,
            'model_version': self.model_version,
            'training_metrics': self.training_metrics,
            'feature_importance': self.feature_importance,
            'training_timestamp': self.training_timestamp,
            'model_params': self._get_model_params()
        }

        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath: str) -> 'BaseModel':
        """
        Load a trained model from disk.

        Args:
            filepath: Path to the saved model

        Returns:
            Loaded model instance
        """
        model_data = joblib.load(filepath)

        # Create new instance
        instance = cls(
            model_name=model_data['model_name'],
            model_version=model_data['model_version']
        )

        # Restore model state
        instance.model = model_data['model']
        instance.preprocessor = model_data['preprocessor']
        instance.training_metrics = model_data['training_metrics']
        instance.feature_importance = model_data['feature_importance']
        instance.training_timestamp = model_data['training_timestamp']
        instance.is_trained = True

        logger.info(f"Model loaded from {filepath}")
        return instance

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information.

        Returns:
            Dictionary containing model metadata and metrics
        """
        return {
            'model_name': self.model_name,
            'model_version': self.model_version,
            'is_trained': self.is_trained,
            'training_timestamp': self.training_timestamp,
            'training_metrics': self.training_metrics,
            'model_params': self._get_model_params() if self.is_trained else None,
            'feature_count': len(self.feature_importance) if self.feature_importance is not None else None
        }

    def validate_input(self, X: pd.DataFrame) -> bool:
        """
        Validate input data format and features.

        Args:
            X: Input features to validate

        Returns:
            True if input is valid, False otherwise
        """
        if not isinstance(X, pd.DataFrame):
            logger.error("Input must be a pandas DataFrame")
            return False

        if X.empty:
            logger.error("Input DataFrame is empty")
            return False

        # Check for required features if preprocessor is available
        if self.preprocessor is not None and self.preprocessor.feature_names is not None:
            missing_features = set(self.preprocessor.feature_names) - set(X.columns)
            if missing_features:
                logger.error(f"Missing required features: {missing_features}")
                return False

        return True

    def cross_validate(self, 
                      X: pd.DataFrame, 
                      y: pd.Series,
                      cv_folds: int = 5,
                      scoring: str = 'accuracy') -> Dict[str, float]:
        """
        Perform cross-validation on the model.

        Args:
            X: Input features
            y: Target variable
            cv_folds: Number of cross-validation folds
            scoring: Scoring metric

        Returns:
            Cross-validation results
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before cross-validation")

        # Preprocess data if preprocessor is available
        if self.preprocessor is not None:
            X_processed = self.preprocessor.transform(X)
        else:
            X_processed = X.copy()

        validator = ModelValidator(cv_folds=cv_folds, random_state=self.random_state)
        return validator.cross_validate_model(self.model, X_processed, y, scoring)

    def __str__(self) -> str:
        """String representation of the model."""
        status = "Trained" if self.is_trained else "Untrained"
        return f"{self.model_name} v{self.model_version} ({status})"

    def __repr__(self) -> str:
        """Detailed string representation of the model."""
        return f"BaseModel(name='{self.model_name}', version='{self.model_version}', trained={self.is_trained})"


class ModelMetrics:
    """
    Comprehensive metrics calculation and reporting for credit risk models.

    Provides detailed performance metrics, risk-specific calculations,
    and reporting functionality.
    """

    def __init__(self):
        """Initialize the metrics calculator."""
        self.metrics_history = []

    def calculate_classification_metrics(self, 
                                       y_true: np.ndarray, 
                                       y_pred: np.ndarray,
                                       y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate comprehensive classification metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)

        Returns:
            Dictionary of classification metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_score_macro': f1_score(y_true, y_pred, average='macro', zero_division=0)
        }

        if y_pred_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            except ValueError:
                metrics['roc_auc'] = 0.0

        return metrics

    def calculate_credit_risk_metrics(self, 
                                    y_true: np.ndarray, 
                                    y_pred: np.ndarray,
                                    y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate credit risk specific metrics.

        Args:
            y_true: True default indicators (1 = default, 0 = no default)
            y_pred: Predicted default indicators
            y_pred_proba: Predicted default probabilities

        Returns:
            Dictionary of credit risk metrics
        """
        # Basic classification metrics
        metrics = self.calculate_classification_metrics(y_true, y_pred, y_pred_proba)

        # Credit-specific metrics
        true_positives = np.sum((y_true == 1) & (y_pred == 1))
        false_positives = np.sum((y_true == 0) & (y_pred == 1))
        true_negatives = np.sum((y_true == 0) & (y_pred == 0))
        false_negatives = np.sum((y_true == 1) & (y_pred == 0))

        # Default detection rate (sensitivity)
        metrics['default_detection_rate'] = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

        # False positive rate (Type I error)
        metrics['false_positive_rate'] = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0

        # Specificity (true negative rate)
        metrics['specificity'] = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0

        # Default rate in predictions
        metrics['predicted_default_rate'] = np.mean(y_pred)
        metrics['actual_default_rate'] = np.mean(y_true)

        return metrics

    def generate_metrics_report(self, 
                              metrics: Dict[str, float],
                              model_name: str = "Model") -> str:
        """
        Generate a formatted metrics report.

        Args:
            metrics: Dictionary of calculated metrics
            model_name: Name of the model

        Returns:
            Formatted metrics report string
        """
        report = f"\n{'='*50}\n"
        report += f"{model_name} Performance Report\n"
        report += f"{'='*50}\n\n"

        # Classification metrics
        report += "Classification Metrics:\n"
        report += f"  Accuracy:           {metrics.get('accuracy', 0):.4f}\n"
        report += f"  Precision (Weighted): {metrics.get('precision', 0):.4f}\n"
        report += f"  Recall (Weighted):    {metrics.get('recall', 0):.4f}\n"
        report += f"  F1-Score (Weighted):  {metrics.get('f1_score', 0):.4f}\n"

        if 'roc_auc' in metrics:
            report += f"  ROC AUC:            {metrics['roc_auc']:.4f}\n"

        # Credit risk specific metrics
        if 'default_detection_rate' in metrics:
            report += "\nCredit Risk Metrics:\n"
            report += f"  Default Detection Rate: {metrics['default_detection_rate']:.4f}\n"
            report += f"  False Positive Rate:    {metrics['false_positive_rate']:.4f}\n"
            report += f"  Specificity:           {metrics['specificity']:.4f}\n"
            report += f"  Predicted Default Rate: {metrics['predicted_default_rate']:.4f}\n"
            report += f"  Actual Default Rate:    {metrics['actual_default_rate']:.4f}\n"

        report += f"\n{'='*50}\n"

        return report

    def save_metrics(self, 
                    metrics: Dict[str, float], 
                    filepath: str,
                    model_name: str = "Model") -> None:
        """
        Save metrics to a file.

        Args:
            metrics: Dictionary of metrics to save
            filepath: Path to save the metrics
            model_name: Name of the model
        """
        metrics_data = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        }

        with open(filepath, 'w') as f:
            import json
            json.dump(metrics_data, f, indent=2)

        logger.info(f"Metrics saved to {filepath}")


class ModelComparison:
    """
    Utility class for comparing multiple models.

    Provides functionality to compare model performance,
    generate comparison reports, and select best models.
    """

    def __init__(self):
        """Initialize the model comparison utility."""
        self.models = {}
        self.comparison_results = {}

    def add_model(self, 
                  model_name: str, 
                  model: BaseModel,
                  X_test: pd.DataFrame,
                  y_test: pd.Series) -> None:
        """
        Add a model to the comparison.

        Args:
            model_name: Name identifier for the model
            model: Trained model instance
            X_test: Test features
            y_test: Test targets
        """
        if not model.is_trained:
            raise ValueError(f"Model {model_name} must be trained before comparison")

        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model.model, 'predict_proba') else None

        # Calculate metrics
        metrics_calc = ModelMetrics()
        metrics = metrics_calc.calculate_credit_risk_metrics(y_test, y_pred, y_pred_proba)

        self.models[model_name] = {
            'model': model,
            'metrics': metrics,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }

    def compare_models(self, 
                      primary_metric: str = 'f1_score') -> pd.DataFrame:
        """
        Compare all added models.

        Args:
            primary_metric: Primary metric for ranking models

        Returns:
            DataFrame with model comparison results
        """
        if not self.models:
            raise ValueError("No models added for comparison")

        comparison_data = []

        for model_name, model_data in self.models.items():
            metrics = model_data['metrics']
            comparison_data.append({
                'Model': model_name,
                'Accuracy': metrics.get('accuracy', 0),
                'Precision': metrics.get('precision', 0),
                'Recall': metrics.get('recall', 0),
                'F1-Score': metrics.get('f1_score', 0),
                'ROC AUC': metrics.get('roc_auc', 0),
                'Default Detection Rate': metrics.get('default_detection_rate', 0),
                'False Positive Rate': metrics.get('false_positive_rate', 0)
            })

        comparison_df = pd.DataFrame(comparison_data)

        # Sort by primary metric (descending for most metrics, ascending for FPR)
        ascending = primary_metric.lower() in ['false_positive_rate', 'false positive rate']
        comparison_df = comparison_df.sort_values(
            by=primary_metric.replace('_', ' ').title() if '_' in primary_metric else primary_metric,
            ascending=ascending
        )

        self.comparison_results = comparison_df
        return comparison_df

    def get_best_model(self, 
                      metric: str = 'f1_score') -> Tuple[str, BaseModel]:
        """
        Get the best performing model based on specified metric.

        Args:
            metric: Metric to use for selection

        Returns:
            Tuple of (model_name, model_instance)
        """
        if not self.models:
            raise ValueError("No models available for selection")

        best_score = -np.inf if metric != 'false_positive_rate' else np.inf
        best_model_name = None

        for model_name, model_data in self.models.items():
            score = model_data['metrics'].get(metric, 0)

            if metric == 'false_positive_rate':
                if score < best_score:
                    best_score = score
                    best_model_name = model_name
            else:
                if score > best_score:
                    best_score = score
                    best_model_name = model_name

        if best_model_name is None:
            raise ValueError(f"Could not find best model using metric: {metric}")

        return best_model_name, self.models[best_model_name]['model']

    def generate_comparison_report(self) -> str:
        """
        Generate a detailed comparison report.

        Returns:
            Formatted comparison report string
        """
        if self.comparison_results is None or self.comparison_results.empty:
            return "No comparison results available. Run compare_models() first."

        report = f"\n{'='*80}\n"
        report += "Model Comparison Report\n"
        report += f"{'='*80}\n\n"

        report += self.comparison_results.to_string(index=False, float_format='%.4f')
        report += f"\n\n{'='*80}\n"

        return report


# Export all classes for easy importing
__all__ = [
    'DataPreprocessor',
    'ModelValidator', 
    'BaseModel',
    'ModelMetrics',
    'ModelComparison'
]
