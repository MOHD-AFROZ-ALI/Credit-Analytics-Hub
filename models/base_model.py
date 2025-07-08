"""
Base Model Module for Credit Analytics Hub

This module provides base classes and interfaces for credit models,
including abstract base classes, model interfaces, and common functionality.
"""

import pandas as pd
import numpy as np
import pickle
import joblib
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from pathlib import Path
import logging
from datetime import datetime
import json
import warnings
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelInterface(ABC):
    """
    Abstract interface for all credit models
    """

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'ModelInterface':
        """
        Train the model

        Args:
            X: Feature matrix
            y: Target vector
            **kwargs: Additional training parameters

        Returns:
            Self: Trained model instance
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions

        Args:
            X: Feature matrix

        Returns:
            np.ndarray: Predictions
        """
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities

        Args:
            X: Feature matrix

        Returns:
            np.ndarray: Class probabilities
        """
        pass

    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores

        Returns:
            Dict[str, float]: Feature importance mapping
        """
        pass

    @abstractmethod
    def save_model(self, filepath: str) -> None:
        """
        Save the model to disk

        Args:
            filepath: Path to save the model
        """
        pass

    @abstractmethod
    def load_model(self, filepath: str) -> 'ModelInterface':
        """
        Load the model from disk

        Args:
            filepath: Path to load the model from

        Returns:
            ModelInterface: Loaded model instance
        """
        pass


class BaseModel(ModelInterface):
    """
    Base class for all credit models with common functionality
    """

    def __init__(self, 
                 model_name: str = "BaseModel",
                 random_state: int = 42,
                 verbose: bool = True):
        """
        Initialize BaseModel

        Args:
            model_name: Name of the model
            random_state: Random state for reproducibility
            verbose: Whether to print verbose output
        """
        self.model_name = model_name
        self.random_state = random_state
        self.verbose = verbose

        # Model state
        self.is_fitted = False
        self.feature_names = None
        self.n_features = None
        self.classes_ = None

        # Training history
        self.training_history = []
        self.model_metadata = {
            'model_name': model_name,
            'created_at': datetime.now().isoformat(),
            'version': '1.0.0',
            'random_state': random_state
        }

        # Performance tracking
        self.training_scores = {}
        self.validation_scores = {}

        # Model-specific attributes (to be set by subclasses)
        self.model = None
        self.preprocessor = None
        self.feature_selector = None

    def _validate_input(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Validate input data

        Args:
            X: Feature matrix
            y: Target vector (optional)

        Returns:
            Tuple[pd.DataFrame, Optional[pd.Series]]: Validated inputs
        """
        # Validate X
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        if X.empty:
            raise ValueError("X cannot be empty")

        # Check for infinite values
        if np.isinf(X.select_dtypes(include=[np.number])).any().any():
            logger.warning("Infinite values detected in X")

        # Validate y if provided
        if y is not None:
            if not isinstance(y, pd.Series):
                raise TypeError("y must be a pandas Series")

            if len(X) != len(y):
                raise ValueError("X and y must have the same number of samples")

            # Check for missing values in target
            if y.isnull().any():
                raise ValueError("Target variable y cannot contain missing values")

        return X, y

    def _check_is_fitted(self) -> None:
        """Check if the model is fitted"""
        if not self.is_fitted:
            raise ValueError(f"{self.model_name} is not fitted yet. Call 'fit' first.")

    def _log_training_step(self, step: str, details: Dict[str, Any]) -> None:
        """Log training step"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'step': step,
            'details': details
        }
        self.training_history.append(log_entry)

        if self.verbose:
            logger.info(f"{self.model_name} - {step}: {details}")

    def fit(self, 
            X: pd.DataFrame, 
            y: pd.Series,
            validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
            **kwargs) -> 'BaseModel':
        """
        Train the model (to be implemented by subclasses)

        Args:
            X: Feature matrix
            y: Target vector
            validation_data: Optional validation data tuple (X_val, y_val)
            **kwargs: Additional training parameters

        Returns:
            Self: Trained model instance
        """
        # Validate inputs
        X, y = self._validate_input(X, y)

        # Store feature information
        self.feature_names = list(X.columns)
        self.n_features = X.shape[1]
        self.classes_ = np.unique(y)

        # Log training start
        self._log_training_step("training_started", {
            'n_samples': len(X),
            'n_features': self.n_features,
            'target_distribution': y.value_counts().to_dict()
        })

        # This method should be overridden by subclasses
        raise NotImplementedError("Subclasses must implement the fit method")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions (to be implemented by subclasses)

        Args:
            X: Feature matrix

        Returns:
            np.ndarray: Predictions
        """
        self._check_is_fitted()
        X, _ = self._validate_input(X)

        # Check feature consistency
        if list(X.columns) != self.feature_names:
            logger.warning("Feature names don't match training data")

        # This method should be overridden by subclasses
        raise NotImplementedError("Subclasses must implement the predict method")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities (to be implemented by subclasses)

        Args:
            X: Feature matrix

        Returns:
            np.ndarray: Class probabilities
        """
        self._check_is_fitted()
        X, _ = self._validate_input(X)

        # This method should be overridden by subclasses
        raise NotImplementedError("Subclasses must implement the predict_proba method")

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores (to be implemented by subclasses)

        Returns:
            Dict[str, float]: Feature importance mapping
        """
        self._check_is_fitted()

        # This method should be overridden by subclasses
        raise NotImplementedError("Subclasses must implement the get_feature_importance method")

    def evaluate(self, 
                X: pd.DataFrame, 
                y: pd.Series,
                metrics: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Evaluate the model on given data

        Args:
            X: Feature matrix
            y: True labels
            metrics: List of metrics to calculate

        Returns:
            Dict[str, float]: Evaluation metrics
        """
        self._check_is_fitted()

        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']

        # Make predictions
        y_pred = self.predict(X)
        y_prob = self.predict_proba(X)[:, 1] if len(self.classes_) == 2 else None

        # Calculate metrics
        results = {}

        if 'accuracy' in metrics:
            results['accuracy'] = accuracy_score(y, y_pred)

        if 'precision' in metrics:
            results['precision'] = precision_score(y, y_pred, average='binary' if len(self.classes_) == 2 else 'weighted')

        if 'recall' in metrics:
            results['recall'] = recall_score(y, y_pred, average='binary' if len(self.classes_) == 2 else 'weighted')

        if 'f1_score' in metrics:
            results['f1_score'] = f1_score(y, y_pred, average='binary' if len(self.classes_) == 2 else 'weighted')

        if 'roc_auc' in metrics and y_prob is not None:
            results['roc_auc'] = roc_auc_score(y, y_prob)

        return results

    def cross_validate(self, 
                      X: pd.DataFrame, 
                      y: pd.Series,
                      cv: int = 5,
                      scoring: str = 'roc_auc') -> Dict[str, Any]:
        """
        Perform cross-validation

        Args:
            X: Feature matrix
            y: Target vector
            cv: Number of cross-validation folds
            scoring: Scoring metric

        Returns:
            Dict[str, Any]: Cross-validation results
        """
        if self.model is None:
            raise ValueError("Model not initialized. Cannot perform cross-validation.")

        # Perform cross-validation
        cv_scores = cross_val_score(self.model, X, y, cv=cv, scoring=scoring)

        results = {
            'cv_scores': cv_scores.tolist(),
            'mean_score': float(np.mean(cv_scores)),
            'std_score': float(np.std(cv_scores)),
            'min_score': float(np.min(cv_scores)),
            'max_score': float(np.max(cv_scores)),
            'scoring_metric': scoring,
            'cv_folds': cv
        }

        if self.verbose:
            logger.info(f"Cross-validation {scoring}: {results['mean_score']:.4f} (+/- {results['std_score']*2:.4f})")

        return results

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information

        Returns:
            Dict[str, Any]: Model information
        """
        info = {
            'model_metadata': self.model_metadata,
            'is_fitted': self.is_fitted,
            'feature_names': self.feature_names,
            'n_features': self.n_features,
            'classes': self.classes_.tolist() if self.classes_ is not None else None,
            'training_history': self.training_history,
            'training_scores': self.training_scores,
            'validation_scores': self.validation_scores
        }

        # Add model-specific parameters if available
        if hasattr(self.model, 'get_params'):
            info['model_parameters'] = self.model.get_params()

        return info

    def save_model(self, filepath: str, save_format: str = 'joblib') -> None:
        """
        Save the model to disk

        Args:
            filepath: Path to save the model
            save_format: Format to save ('joblib', 'pickle')
        """
        self._check_is_fitted()

        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # Prepare model data
        model_data = {
            'model': self.model,
            'model_name': self.model_name,
            'feature_names': self.feature_names,
            'n_features': self.n_features,
            'classes_': self.classes_,
            'model_metadata': self.model_metadata,
            'training_history': self.training_history,
            'preprocessor': self.preprocessor,
            'feature_selector': self.feature_selector,
            'is_fitted': self.is_fitted
        }

        # Save based on format
        if save_format == 'joblib':
            joblib.dump(model_data, filepath)
        elif save_format == 'pickle':
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
        else:
            raise ValueError(f"Unsupported save format: {save_format}")

        if self.verbose:
            logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str, load_format: str = 'auto') -> 'BaseModel':
        """
        Load the model from disk

        Args:
            filepath: Path to load the model from
            load_format: Format to load ('joblib', 'pickle', 'auto')

        Returns:
            BaseModel: Loaded model instance
        """
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        # Auto-detect format if needed
        if load_format == 'auto':
            if filepath.endswith('.pkl') or filepath.endswith('.pickle'):
                load_format = 'pickle'
            else:
                load_format = 'joblib'

        # Load based on format
        try:
            if load_format == 'joblib':
                model_data = joblib.load(filepath)
            elif load_format == 'pickle':
                with open(filepath, 'rb') as f:
                    model_data = pickle.load(f)
            else:
                raise ValueError(f"Unsupported load format: {load_format}")

            # Restore model state
            self.model = model_data['model']
            self.model_name = model_data['model_name']
            self.feature_names = model_data['feature_names']
            self.n_features = model_data['n_features']
            self.classes_ = model_data['classes_']
            self.model_metadata = model_data['model_metadata']
            self.training_history = model_data['training_history']
            self.preprocessor = model_data.get('preprocessor')
            self.feature_selector = model_data.get('feature_selector')
            self.is_fitted = model_data['is_fitted']

            if self.verbose:
                logger.info(f"Model loaded from {filepath}")

            return self

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def clone(self) -> 'BaseModel':
        """
        Create a copy of the model

        Returns:
            BaseModel: Cloned model instance
        """
        # Create new instance of the same class
        cloned_model = self.__class__(
            model_name=f"{self.model_name}_clone",
            random_state=self.random_state,
            verbose=self.verbose
        )

        # Copy model parameters if available
        if hasattr(self.model, 'get_params'):
            cloned_model.model = self.model.__class__(**self.model.get_params())

        return cloned_model

    def __repr__(self) -> str:
        """String representation of the model"""
        status = "fitted" if self.is_fitted else "not fitted"
        return f"{self.model_name}({status}, features={self.n_features})"

    def __str__(self) -> str:
        """Human-readable string representation"""
        return self.__repr__()


class SklearnModelWrapper(BaseModel):
    """
    Wrapper for scikit-learn models to work with the BaseModel interface
    """

    def __init__(self, 
                 sklearn_model: BaseEstimator,
                 model_name: str = "SklearnModel",
                 **kwargs):
        """
        Initialize SklearnModelWrapper

        Args:
            sklearn_model: Scikit-learn model instance
            model_name: Name of the model
            **kwargs: Additional arguments for BaseModel
        """
        super().__init__(model_name=model_name, **kwargs)
        self.model = sklearn_model

        # Update metadata
        self.model_metadata.update({
            'model_type': 'sklearn_wrapper',
            'sklearn_model': str(type(sklearn_model).__name__)
        })

    def fit(self, 
            X: pd.DataFrame, 
            y: pd.Series,
            validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
            **kwargs) -> 'SklearnModelWrapper':
        """
        Train the sklearn model

        Args:
            X: Feature matrix
            y: Target vector
            validation_data: Optional validation data (not used for sklearn models)
            **kwargs: Additional training parameters

        Returns:
            Self: Trained model instance
        """
        # Call parent fit for validation and logging
        super().fit(X, y, validation_data, **kwargs)

        try:
            # Train the sklearn model
            self.model.fit(X, y, **kwargs)
            self.is_fitted = True

            # Calculate training scores
            train_predictions = self.model.predict(X)
            self.training_scores = self.evaluate(X, y)

            # Calculate validation scores if validation data provided
            if validation_data is not None:
                X_val, y_val = validation_data
                self.validation_scores = self.evaluate(X_val, y_val)

            self._log_training_step("training_completed", {
                'training_scores': self.training_scores,
                'validation_scores': self.validation_scores
            })

            return self

        except Exception as e:
            logger.error(f"Error training {self.model_name}: {str(e)}")
            raise

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the sklearn model

        Args:
            X: Feature matrix

        Returns:
            np.ndarray: Predictions
        """
        super().predict(X)  # Validation and checks
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities using the sklearn model

        Args:
            X: Feature matrix

        Returns:
            np.ndarray: Class probabilities
        """
        super().predict_proba(X)  # Validation and checks

        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        elif hasattr(self.model, 'decision_function'):
            # Convert decision function to probabilities
            decision_scores = self.model.decision_function(X)
            # Apply sigmoid for binary classification
            probabilities = 1 / (1 + np.exp(-decision_scores))
            return np.column_stack([1 - probabilities, probabilities])
        else:
            raise AttributeError(f"{self.model_name} does not support probability prediction")

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores from the sklearn model

        Returns:
            Dict[str, float]: Feature importance mapping
        """
        super().get_feature_importance()  # Check if fitted

        importance_scores = None

        # Try different ways to get feature importance
        if hasattr(self.model, 'feature_importances_'):
            importance_scores = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            # For linear models, use absolute coefficients
            coef = self.model.coef_
            if coef.ndim > 1:
                coef = coef[0]  # Take first class for binary classification
            importance_scores = np.abs(coef)
        else:
            logger.warning(f"{self.model_name} does not have feature importance")
            return {name: 0.0 for name in self.feature_names}

        # Create feature importance dictionary
        if importance_scores is not None:
            return dict(zip(self.feature_names, importance_scores))
        else:
            return {name: 0.0 for name in self.feature_names}


class EnsembleModelBase(BaseModel):
    """
    Base class for ensemble models
    """

    def __init__(self, 
                 base_models: List[BaseModel],
                 model_name: str = "EnsembleModel",
                 **kwargs):
        """
        Initialize EnsembleModelBase

        Args:
            base_models: List of base models to ensemble
            model_name: Name of the ensemble model
            **kwargs: Additional arguments for BaseModel
        """
        super().__init__(model_name=model_name, **kwargs)
        self.base_models = base_models
        self.model_weights = None

        # Update metadata
        self.model_metadata.update({
            'model_type': 'ensemble',
            'n_base_models': len(base_models),
            'base_model_names': [model.model_name for model in base_models]
        })

    def fit(self, 
            X: pd.DataFrame, 
            y: pd.Series,
            validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
            **kwargs) -> 'EnsembleModelBase':
        """
        Train all base models in the ensemble

        Args:
            X: Feature matrix
            y: Target vector
            validation_data: Optional validation data
            **kwargs: Additional training parameters

        Returns:
            Self: Trained ensemble model
        """
        # Call parent fit for validation and logging
        super().fit(X, y, validation_data, **kwargs)

        try:
            # Train each base model
            for i, model in enumerate(self.base_models):
                self._log_training_step(f"training_base_model_{i}", {
                    'model_name': model.model_name
                })
                model.fit(X, y, validation_data, **kwargs)

            self.is_fitted = True

            # Calculate ensemble training scores
            self.training_scores = self.evaluate(X, y)

            if validation_data is not None:
                X_val, y_val = validation_data
                self.validation_scores = self.evaluate(X_val, y_val)

            self._log_training_step("ensemble_training_completed", {
                'training_scores': self.training_scores,
                'validation_scores': self.validation_scores
            })

            return self

        except Exception as e:
            logger.error(f"Error training ensemble {self.model_name}: {str(e)}")
            raise

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make ensemble predictions (to be implemented by subclasses)

        Args:
            X: Feature matrix

        Returns:
            np.ndarray: Ensemble predictions
        """
        super().predict(X)  # Validation and checks
        raise NotImplementedError("Subclasses must implement ensemble prediction logic")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict ensemble probabilities (to be implemented by subclasses)

        Args:
            X: Feature matrix

        Returns:
            np.ndarray: Ensemble probabilities
        """
        super().predict_proba(X)  # Validation and checks
        raise NotImplementedError("Subclasses must implement ensemble probability prediction")

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get aggregated feature importance from all base models

        Returns:
            Dict[str, float]: Aggregated feature importance
        """
        super().get_feature_importance()  # Check if fitted

        # Aggregate feature importance from all base models
        aggregated_importance = {}

        for model in self.base_models:
            try:
                model_importance = model.get_feature_importance()
                for feature, importance in model_importance.items():
                    if feature not in aggregated_importance:
                        aggregated_importance[feature] = 0.0
                    aggregated_importance[feature] += importance
            except Exception as e:
                logger.warning(f"Could not get feature importance from {model.model_name}: {str(e)}")

        # Average the importance scores
        n_models = len(self.base_models)
        for feature in aggregated_importance:
            aggregated_importance[feature] /= n_models

        return aggregated_importance


# Utility functions for model management
def create_sklearn_model(sklearn_model: BaseEstimator, 
                        model_name: str = "SklearnModel",
                        **kwargs) -> SklearnModelWrapper:
    """
    Create a wrapped sklearn model

    Args:
        sklearn_model: Scikit-learn model instance
        model_name: Name for the model
        **kwargs: Additional arguments

    Returns:
        SklearnModelWrapper: Wrapped sklearn model
    """
    return SklearnModelWrapper(sklearn_model, model_name, **kwargs)

def load_model(filepath: str, **kwargs) -> BaseModel:
    """
    Load a model from disk

    Args:
        filepath: Path to the saved model
        **kwargs: Additional arguments for loading

    Returns:
        BaseModel: Loaded model instance
    """
    # Create a temporary BaseModel instance to use the load functionality
    temp_model = BaseModel()
    return temp_model.load_model(filepath, **kwargs)

def compare_models(models: List[BaseModel], 
                  X: pd.DataFrame, 
                  y: pd.Series,
                  metrics: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Compare multiple models on the same dataset

    Args:
        models: List of trained models
        X: Feature matrix
        y: Target vector
        metrics: List of metrics to compare

    Returns:
        pd.DataFrame: Comparison results
    """
    if metrics is None:
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']

    results = []

    for model in models:
        if not model.is_fitted:
            logger.warning(f"Model {model.model_name} is not fitted. Skipping.")
            continue

        try:
            model_scores = model.evaluate(X, y, metrics)
            model_scores['model_name'] = model.model_name
            results.append(model_scores)
        except Exception as e:
            logger.error(f"Error evaluating {model.model_name}: {str(e)}")

    if not results:
        raise ValueError("No models could be evaluated")

    # Create comparison DataFrame
    comparison_df = pd.DataFrame(results)

    # Reorder columns
    cols = ['model_name'] + [col for col in comparison_df.columns if col != 'model_name']
    comparison_df = comparison_df[cols]

    # Sort by primary metric (roc_auc if available, otherwise first metric)
    sort_metric = 'roc_auc' if 'roc_auc' in comparison_df.columns else metrics[0]
    comparison_df = comparison_df.sort_values(sort_metric, ascending=False)

    return comparison_df
