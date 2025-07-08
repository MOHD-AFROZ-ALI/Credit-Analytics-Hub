"""
Model Manager Module for Credit Analytics Hub

This module provides comprehensive model management functionality including
training, evaluation, saving, loading, deployment, and lifecycle management.
"""

import os
import json
import pickle
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from datetime import datetime, timedelta
import logging
from pathlib import Path
import hashlib
import shutil
from dataclasses import dataclass, asdict
from enum import Enum
import warnings

# Import model classes
from .base_model import BaseModel, ModelMetrics, ModelComparison
from .credit_model import (
    LogisticRegressionCreditModel,
    RandomForestCreditModel,
    GradientBoostingCreditModel,
    SVMCreditModel,
    NeuralNetworkCreditModel,
    EnsembleCreditModel,
    AutoMLCreditModel,
    CreditRiskPreprocessor
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelStatus(Enum):
    """Enumeration for model status."""
    UNTRAINED = "untrained"
    TRAINING = "training"
    TRAINED = "trained"
    VALIDATED = "validated"
    DEPLOYED = "deployed"
    DEPRECATED = "deprecated"
    FAILED = "failed"

@dataclass
class ModelMetadata:
    """Data class for storing model metadata."""
    model_id: str
    model_name: str
    model_type: str
    version: str
    status: ModelStatus
    created_at: datetime
    updated_at: datetime
    trained_at: Optional[datetime] = None
    deployed_at: Optional[datetime] = None
    training_data_hash: Optional[str] = None
    performance_metrics: Optional[Dict[str, float]] = None
    file_path: Optional[str] = None
    file_size: Optional[int] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    feature_names: Optional[List[str]] = None
    target_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        data = asdict(self)
        # Convert datetime objects to ISO format strings
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat() if value else None
            elif isinstance(value, ModelStatus):
                data[key] = value.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """Create metadata from dictionary."""
        # Convert ISO format strings back to datetime objects
        datetime_fields = ['created_at', 'updated_at', 'trained_at', 'deployed_at']
        for field in datetime_fields:
            if data.get(field):
                data[field] = datetime.fromisoformat(data[field])

        # Convert status string back to enum
        if 'status' in data:
            data['status'] = ModelStatus(data['status'])

        return cls(**data)

class ModelRegistry:
    """
    Registry for managing model metadata and lifecycle.

    Provides centralized storage and retrieval of model information,
    version control, and status tracking.
    """

    def __init__(self, registry_path: str = "/home/user/output/models/registry.json"):
        """
        Initialize the model registry.

        Args:
            registry_path: Path to the registry file
        """
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.models: Dict[str, ModelMetadata] = {}
        self._load_registry()

    def _load_registry(self) -> None:
        """Load registry from file."""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r') as f:
                    data = json.load(f)
                    self.models = {
                        model_id: ModelMetadata.from_dict(metadata)
                        for model_id, metadata in data.items()
                    }
                logger.info(f"Loaded {len(self.models)} models from registry")
            except Exception as e:
                logger.error(f"Error loading registry: {e}")
                self.models = {}
        else:
            logger.info("No existing registry found, starting fresh")

    def _save_registry(self) -> None:
        """Save registry to file."""
        try:
            data = {
                model_id: metadata.to_dict()
                for model_id, metadata in self.models.items()
            }
            with open(self.registry_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            logger.debug("Registry saved successfully")
        except Exception as e:
            logger.error(f"Error saving registry: {e}")

    def register_model(self, 
                      model_name: str,
                      model_type: str,
                      version: str = "1.0",
                      description: Optional[str] = None,
                      tags: Optional[List[str]] = None) -> str:
        """
        Register a new model in the registry.

        Args:
            model_name: Name of the model
            model_type: Type/class of the model
            version: Version of the model
            description: Optional description
            tags: Optional tags for categorization

        Returns:
            Generated model ID
        """
        model_id = self._generate_model_id(model_name, version)

        metadata = ModelMetadata(
            model_id=model_id,
            model_name=model_name,
            model_type=model_type,
            version=version,
            status=ModelStatus.UNTRAINED,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            description=description,
            tags=tags or []
        )

        self.models[model_id] = metadata
        self._save_registry()

        logger.info(f"Registered model: {model_id}")
        return model_id

    def update_model_status(self, model_id: str, status: ModelStatus) -> None:
        """Update model status."""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found in registry")

        self.models[model_id].status = status
        self.models[model_id].updated_at = datetime.now()

        if status == ModelStatus.TRAINED:
            self.models[model_id].trained_at = datetime.now()
        elif status == ModelStatus.DEPLOYED:
            self.models[model_id].deployed_at = datetime.now()

        self._save_registry()
        logger.info(f"Updated model {model_id} status to {status.value}")

    def update_model_metadata(self, model_id: str, **kwargs) -> None:
        """Update model metadata fields."""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found in registry")

        metadata = self.models[model_id]
        for key, value in kwargs.items():
            if hasattr(metadata, key):
                setattr(metadata, key, value)

        metadata.updated_at = datetime.now()
        self._save_registry()
        logger.info(f"Updated metadata for model {model_id}")

    def get_model_metadata(self, model_id: str) -> Optional[ModelMetadata]:
        """Get model metadata by ID."""
        return self.models.get(model_id)

    def list_models(self, 
                   status: Optional[ModelStatus] = None,
                   model_type: Optional[str] = None,
                   tags: Optional[List[str]] = None) -> List[ModelMetadata]:
        """
        List models with optional filtering.

        Args:
            status: Filter by status
            model_type: Filter by model type
            tags: Filter by tags (models must have all specified tags)

        Returns:
            List of matching model metadata
        """
        models = list(self.models.values())

        if status:
            models = [m for m in models if m.status == status]

        if model_type:
            models = [m for m in models if m.model_type == model_type]

        if tags:
            models = [m for m in models if m.tags and all(tag in m.tags for tag in tags)]

        return sorted(models, key=lambda x: x.updated_at, reverse=True)

    def delete_model(self, model_id: str, remove_files: bool = True) -> None:
        """
        Delete model from registry and optionally remove files.

        Args:
            model_id: Model ID to delete
            remove_files: Whether to remove associated files
        """
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found in registry")

        metadata = self.models[model_id]

        # Remove files if requested and path exists
        if remove_files and metadata.file_path and Path(metadata.file_path).exists():
            try:
                os.remove(metadata.file_path)
                logger.info(f"Removed model file: {metadata.file_path}")
            except Exception as e:
                logger.error(f"Error removing model file: {e}")

        # Remove from registry
        del self.models[model_id]
        self._save_registry()

        logger.info(f"Deleted model {model_id} from registry")

    def _generate_model_id(self, model_name: str, version: str) -> str:
        """Generate unique model ID."""
        base_id = f"{model_name}_{version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return base_id.replace(' ', '_').lower()

    def get_model_summary(self) -> pd.DataFrame:
        """Get summary of all models as DataFrame."""
        if not self.models:
            return pd.DataFrame()

        summary_data = []
        for metadata in self.models.values():
            summary_data.append({
                'Model ID': metadata.model_id,
                'Name': metadata.model_name,
                'Type': metadata.model_type,
                'Version': metadata.version,
                'Status': metadata.status.value,
                'Created': metadata.created_at.strftime('%Y-%m-%d %H:%M'),
                'Updated': metadata.updated_at.strftime('%Y-%m-%d %H:%M'),
                'Performance': metadata.performance_metrics.get('f1_score', 'N/A') if metadata.performance_metrics else 'N/A'
            })

        return pd.DataFrame(summary_data)

class ModelTrainer:
    """
    Comprehensive model training orchestrator.

    Handles training workflows, hyperparameter tuning, cross-validation,
    and performance evaluation for multiple model types.
    """

    def __init__(self, 
                 models_dir: str = "/home/user/output/models",
                 registry: Optional[ModelRegistry] = None):
        """
        Initialize the model trainer.

        Args:
            models_dir: Directory for storing trained models
            registry: Model registry instance
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.registry = registry or ModelRegistry()
        self.training_history: List[Dict[str, Any]] = []

        # Available model classes
        self.model_classes = {
            'logistic_regression': LogisticRegressionCreditModel,
            'random_forest': RandomForestCreditModel,
            'gradient_boosting': GradientBoostingCreditModel,
            'svm': SVMCreditModel,
            'neural_network': NeuralNetworkCreditModel,
            'ensemble': EnsembleCreditModel,
            'automl': AutoMLCreditModel
        }

    def train_model(self, 
                   model_type: str,
                   X: pd.DataFrame,
                   y: pd.Series,
                   model_name: Optional[str] = None,
                   version: str = "1.0",
                   hyperparameters: Optional[Dict[str, Any]] = None,
                   validation_split: float = 0.2,
                   cross_validation: bool = True,
                   save_model: bool = True) -> Tuple[str, BaseModel]:
        """
        Train a model with comprehensive evaluation and saving.

        Args:
            model_type: Type of model to train
            X: Training features
            y: Training targets
            model_name: Optional custom model name
            version: Model version
            hyperparameters: Custom hyperparameters
            validation_split: Validation data proportion
            cross_validation: Whether to perform cross-validation
            save_model: Whether to save the trained model

        Returns:
            Tuple of (model_id, trained_model)
        """
        if model_type not in self.model_classes:
            raise ValueError(f"Unknown model type: {model_type}. Available: {list(self.model_classes.keys())}")

        # Generate model name if not provided
        if model_name is None:
            model_name = f"{model_type.replace('_', ' ').title()} Model"

        # Register model
        model_id = self.registry.register_model(
            model_name=model_name,
            model_type=model_type,
            version=version,
            description=f"Trained {model_type} model for credit risk assessment"
        )

        try:
            # Update status to training
            self.registry.update_model_status(model_id, ModelStatus.TRAINING)

            # Create model instance
            model_class = self.model_classes[model_type]
            if hyperparameters:
                model = model_class(model_name=model_name, model_version=version, **hyperparameters)
            else:
                model = model_class(model_name=model_name, model_version=version)

            logger.info(f"Starting training for {model_id}")

            # Train the model
            model.fit(X, y, validation_split=validation_split)

            # Perform cross-validation if requested
            cv_results = {}
            if cross_validation:
                cv_results = model.cross_validate(X, y, cv_folds=5, scoring='f1_weighted')
                logger.info(f"Cross-validation F1 score: {cv_results.get('cv_f1_weighted_mean', 'N/A'):.4f}")

            # Calculate comprehensive metrics
            metrics_calc = ModelMetrics()
            y_pred = model.predict(X)
            y_pred_proba = model.predict_proba(X)[:, 1] if hasattr(model.model, 'predict_proba') else None

            performance_metrics = metrics_calc.calculate_credit_risk_metrics(y, y_pred, y_pred_proba)
            performance_metrics.update(cv_results)

            # Calculate data hash for tracking
            data_hash = self._calculate_data_hash(X, y)

            # Update model metadata
            self.registry.update_model_metadata(
                model_id,
                performance_metrics=performance_metrics,
                training_data_hash=data_hash,
                hyperparameters=model._get_model_params(),
                feature_names=X.columns.tolist(),
                target_name=y.name or 'target'
            )

            # Save model if requested
            if save_model:
                model_path = self.models_dir / f"{model_id}.pkl"
                model.save_model(str(model_path))

                # Update file information
                file_size = model_path.stat().st_size
                self.registry.update_model_metadata(
                    model_id,
                    file_path=str(model_path),
                    file_size=file_size
                )

            # Update status to trained
            self.registry.update_model_status(model_id, ModelStatus.TRAINED)

            # Record training history
            training_record = {
                'model_id': model_id,
                'model_type': model_type,
                'training_time': datetime.now(),
                'performance_metrics': performance_metrics,
                'data_shape': X.shape,
                'validation_split': validation_split,
                'cross_validation': cross_validation
            }
            self.training_history.append(training_record)

            logger.info(f"Training completed for {model_id}. F1 Score: {performance_metrics.get('f1_score', 'N/A'):.4f}")

            return model_id, model

        except Exception as e:
            logger.error(f"Training failed for {model_id}: {e}")
            self.registry.update_model_status(model_id, ModelStatus.FAILED)
            raise

    def train_multiple_models(self, 
                            model_types: List[str],
                            X: pd.DataFrame,
                            y: pd.Series,
                            base_name: str = "Credit Model",
                            hyperparameters_dict: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Tuple[str, BaseModel]]:
        """
        Train multiple models and compare their performance.

        Args:
            model_types: List of model types to train
            X: Training features
            y: Training targets
            base_name: Base name for models
            hyperparameters_dict: Dictionary of hyperparameters for each model type

        Returns:
            Dictionary mapping model_type to (model_id, model) tuples
        """
        results = {}
        hyperparameters_dict = hyperparameters_dict or {}

        for model_type in model_types:
            try:
                model_name = f"{base_name} - {model_type.replace('_', ' ').title()}"
                hyperparams = hyperparameters_dict.get(model_type, {})

                model_id, model = self.train_model(
                    model_type=model_type,
                    X=X,
                    y=y,
                    model_name=model_name,
                    hyperparameters=hyperparams
                )

                results[model_type] = (model_id, model)

            except Exception as e:
                logger.error(f"Failed to train {model_type}: {e}")
                continue

        logger.info(f"Successfully trained {len(results)} out of {len(model_types)} models")
        return results

    def hyperparameter_tuning(self, 
                            model_type: str,
                            X: pd.DataFrame,
                            y: pd.Series,
                            param_grid: Dict[str, List[Any]],
                            search_method: str = 'grid',
                            cv_folds: int = 5,
                            scoring: str = 'f1_weighted') -> Tuple[str, BaseModel, Dict[str, Any]]:
        """
        Perform hyperparameter tuning for a model.

        Args:
            model_type: Type of model to tune
            X: Training features
            y: Training targets
            param_grid: Parameter grid for tuning
            search_method: Search method ('grid' or 'random')
            cv_folds: Number of CV folds
            scoring: Scoring metric

        Returns:
            Tuple of (model_id, best_model, tuning_results)
        """
        from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

        if model_type not in self.model_classes:
            raise ValueError(f"Unknown model type: {model_type}")

        logger.info(f"Starting hyperparameter tuning for {model_type}")

        # Create base model
        model_class = self.model_classes[model_type]
        base_model = model_class()

        # Preprocess data
        preprocessor = CreditRiskPreprocessor()
        preprocessor.fit(X, y)
        X_processed = preprocessor.transform(X)

        # Set up search
        if search_method == 'grid':
            search = GridSearchCV(
                base_model._create_model(),
                param_grid,
                cv=cv_folds,
                scoring=scoring,
                n_jobs=-1,
                verbose=1
            )
        else:
            search = RandomizedSearchCV(
                base_model._create_model(),
                param_grid,
                cv=cv_folds,
                scoring=scoring,
                n_jobs=-1,
                n_iter=20,
                verbose=1,
                random_state=42
            )

        # Perform search
        search.fit(X_processed, y)

        # Create optimized model with best parameters
        best_params = search.best_params_
        model_name = f"Tuned {model_type.replace('_', ' ').title()} Model"

        model_id, tuned_model = self.train_model(
            model_type=model_type,
            X=X,
            y=y,
            model_name=model_name,
            hyperparameters=best_params,
            version="tuned_1.0"
        )

        tuning_results = {
            'best_score': search.best_score_,
            'best_params': best_params,
            'cv_results': search.cv_results_,
            'search_method': search_method
        }

        # Update metadata with tuning information
        self.registry.update_model_metadata(
            model_id,
            description=f"Hyperparameter tuned {model_type} model",
            tags=['tuned', 'optimized']
        )

        logger.info(f"Hyperparameter tuning completed. Best score: {search.best_score_:.4f}")

        return model_id, tuned_model, tuning_results

    def _calculate_data_hash(self, X: pd.DataFrame, y: pd.Series) -> str:
        """Calculate hash of training data for tracking."""
        data_string = f"{X.shape}_{X.columns.tolist()}_{y.shape}_{y.sum()}"
        return hashlib.md5(data_string.encode()).hexdigest()

    def get_training_summary(self) -> pd.DataFrame:
        """Get summary of training history."""
        if not self.training_history:
            return pd.DataFrame()

        summary_data = []
        for record in self.training_history:
            metrics = record['performance_metrics']
            summary_data.append({
                'Model ID': record['model_id'],
                'Model Type': record['model_type'],
                'Training Time': record['training_time'].strftime('%Y-%m-%d %H:%M'),
                'Data Shape': f"{record['data_shape'][0]}x{record['data_shape'][1]}",
                'Accuracy': metrics.get('accuracy', 'N/A'),
                'F1 Score': metrics.get('f1_score', 'N/A'),
                'ROC AUC': metrics.get('roc_auc', 'N/A'),
                'CV F1 Mean': metrics.get('cv_f1_weighted_mean', 'N/A')
            })

        return pd.DataFrame(summary_data)

class ModelManager:
    """
    Comprehensive model management system.

    Provides high-level interface for all model operations including
    training, evaluation, deployment, and lifecycle management.
    """

    def __init__(self, 
                 models_dir: str = "/home/user/output/models",
                 registry_path: Optional[str] = None):
        """
        Initialize the model manager.

        Args:
            models_dir: Directory for storing models
            registry_path: Path to model registry file
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        registry_path = registry_path or str(self.models_dir / "registry.json")
        self.registry = ModelRegistry(registry_path)
        self.trainer = ModelTrainer(str(self.models_dir), self.registry)
        self.comparison = ModelComparison()

        # Cache for loaded models
        self._model_cache: Dict[str, BaseModel] = {}

    def train_model(self, *args, **kwargs) -> Tuple[str, BaseModel]:
        """Train a model (delegates to trainer)."""
        return self.trainer.train_model(*args, **kwargs)

    def train_multiple_models(self, *args, **kwargs) -> Dict[str, Tuple[str, BaseModel]]:
        """Train multiple models (delegates to trainer)."""
        return self.trainer.train_multiple_models(*args, **kwargs)

    def load_model(self, model_id: str, use_cache: bool = True) -> BaseModel:
        """
        Load a trained model by ID.

        Args:
            model_id: Model ID to load
            use_cache: Whether to use cached model if available

        Returns:
            Loaded model instance
        """
        # Check cache first
        if use_cache and model_id in self._model_cache:
            return self._model_cache[model_id]

        # Get model metadata
        metadata = self.registry.get_model_metadata(model_id)
        if not metadata:
            raise ValueError(f"Model {model_id} not found in registry")

        if not metadata.file_path or not Path(metadata.file_path).exists():
            raise FileNotFoundError(f"Model file not found for {model_id}")

        # Load model
        try:
            model_data = joblib.load(metadata.file_path)

            # Reconstruct model based on type
            model_class = self.trainer.model_classes.get(metadata.model_type)
            if not model_class:
                raise ValueError(f"Unknown model type: {metadata.model_type}")

            # Create model instance and restore state
            model = model_class(
                model_name=metadata.model_name,
                model_version=metadata.version
            )

            # Restore model components
            model.model = model_data['model']
            model.preprocessor = model_data['preprocessor']
            model.training_metrics = model_data['training_metrics']
            model.feature_importance = model_data['feature_importance']
            model.training_timestamp = model_data['training_timestamp']
            model.is_trained = True

            # Cache the model
            if use_cache:
                self._model_cache[model_id] = model

            logger.info(f"Loaded model {model_id}")
            return model

        except Exception as e:
            logger.error(f"Error loading model {model_id}: {e}")
            raise

    def predict(self, 
               model_id: str, 
               X: pd.DataFrame,
               return_probabilities: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make predictions using a specific model.

        Args:
            model_id: Model ID to use for prediction
            X: Input features
            return_probabilities: Whether to return probabilities

        Returns:
            Predictions or tuple of (predictions, probabilities)
        """
        model = self.load_model(model_id)

        predictions = model.predict(X)

        if return_probabilities:
            if hasattr(model.model, 'predict_proba'):
                probabilities = model.predict_proba(X)
                return predictions, probabilities
            else:
                logger.warning(f"Model {model_id} does not support probability predictions")
                return predictions, None

        return predictions

    def evaluate_model(self, 
                      model_id: str,
                      X_test: pd.DataFrame,
                      y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate a model on test data.

        Args:
            model_id: Model ID to evaluate
            X_test: Test features
            y_test: Test targets

        Returns:
            Dictionary of evaluation metrics
        """
        model = self.load_model(model_id)

        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model.model, 'predict_proba') else None

        # Calculate metrics
        metrics_calc = ModelMetrics()
        metrics = metrics_calc.calculate_credit_risk_metrics(y_test, y_pred, y_pred_proba)

        # Update model metadata with evaluation results
        self.registry.update_model_metadata(
            model_id,
            performance_metrics=metrics
        )

        logger.info(f"Evaluated model {model_id}. F1 Score: {metrics.get('f1_score', 'N/A'):.4f}")

        return metrics

    def compare_models(self, 
                      model_ids: List[str],
                      X_test: pd.DataFrame,
                      y_test: pd.Series,
                      primary_metric: str = 'f1_score') -> pd.DataFrame:
        """
        Compare multiple models on test data.

        Args:
            model_ids: List of model IDs to compare
            X_test: Test features
            y_test: Test targets
            primary_metric: Primary metric for ranking

        Returns:
            DataFrame with comparison results
        """
        comparison = ModelComparison()

        for model_id in model_ids:
            try:
                model = self.load_model(model_id)
                metadata = self.registry.get_model_metadata(model_id)
                model_name = f"{metadata.model_name} ({model_id})" if metadata else model_id

                comparison.add_model(model_name, model, X_test, y_test)

            except Exception as e:
                logger.error(f"Error adding model {model_id} to comparison: {e}")
                continue

        results = comparison.compare_models(primary_metric)

        logger.info(f"Compared {len(model_ids)} models")
        return results

    def deploy_model(self, 
                    model_id: str,
                    deployment_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Deploy a model for production use.

        Args:
            model_id: Model ID to deploy
            deployment_config: Optional deployment configuration

        Returns:
            Deployment information
        """
        metadata = self.registry.get_model_metadata(model_id)
        if not metadata:
            raise ValueError(f"Model {model_id} not found")

        if metadata.status != ModelStatus.TRAINED:
            raise ValueError(f"Model {model_id} must be trained before deployment")

        # Load model to verify it works
        model = self.load_model(model_id)

        # Create deployment directory
        deployment_dir = self.models_dir / "deployed" / model_id
        deployment_dir.mkdir(parents=True, exist_ok=True)

        # Copy model file to deployment directory
        deployment_path = deployment_dir / f"{model_id}_deployed.pkl"
        if metadata.file_path:
            shutil.copy2(metadata.file_path, deployment_path)

        # Create deployment metadata
        deployment_info = {
            'model_id': model_id,
            'deployment_path': str(deployment_path),
            'deployment_time': datetime.now().isoformat(),
            'model_version': metadata.version,
            'performance_metrics': metadata.performance_metrics,
            'deployment_config': deployment_config or {}
        }

        # Save deployment info
        deployment_info_path = deployment_dir / "deployment_info.json"
        with open(deployment_info_path, 'w') as f:
            json.dump(deployment_info, f, indent=2, default=str)

        # Update model status
        self.registry.update_model_status(model_id, ModelStatus.DEPLOYED)

        logger.info(f"Deployed model {model_id} to {deployment_path}")

        return deployment_info

    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """
        Get comprehensive information about a model.

        Args:
            model_id: Model ID

        Returns:
            Dictionary with model information
        """
        metadata = self.registry.get_model_metadata(model_id)
        if not metadata:
            raise ValueError(f"Model {model_id} not found")

        info = metadata.to_dict()

        # Add file information if available
        if metadata.file_path and Path(metadata.file_path).exists():
            file_path = Path(metadata.file_path)
            info['file_exists'] = True
            info['file_size_mb'] = file_path.stat().st_size / (1024 * 1024)
            info['file_modified'] = datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
        else:
            info['file_exists'] = False

        return info

    def list_models(self, **kwargs) -> List[ModelMetadata]:
        """List models (delegates to registry)."""
        return self.registry.list_models(**kwargs)

    def get_model_summary(self) -> pd.DataFrame:
        """Get model summary (delegates to registry)."""
        return self.registry.get_model_summary()

    def delete_model(self, model_id: str, remove_files: bool = True) -> None:
        """
        Delete a model.

        Args:
            model_id: Model ID to delete
            remove_files: Whether to remove associated files
        """
        # Remove from cache if present
        if model_id in self._model_cache:
            del self._model_cache[model_id]

        # Delete from registry
        self.registry.delete_model(model_id, remove_files)

        logger.info(f"Deleted model {model_id}")

    def cleanup_old_models(self, 
                          days_old: int = 30,
                          keep_deployed: bool = True,
                          dry_run: bool = True) -> List[str]:
        """
        Clean up old models to save space.

        Args:
            days_old: Delete models older than this many days
            keep_deployed: Whether to keep deployed models
            dry_run: If True, only return what would be deleted

        Returns:
            List of model IDs that were (or would be) deleted
        """
        cutoff_date = datetime.now() - timedelta(days=days_old)
        models_to_delete = []

        for model_id, metadata in self.registry.models.items():
            # Skip if model is too recent
            if metadata.updated_at > cutoff_date:
                continue

            # Skip deployed models if requested
            if keep_deployed and metadata.status == ModelStatus.DEPLOYED:
                continue

            models_to_delete.append(model_id)

        if not dry_run:
            for model_id in models_to_delete:
                try:
                    self.delete_model(model_id, remove_files=True)
                except Exception as e:
                    logger.error(f"Error deleting model {model_id}: {e}")

        action = "Would delete" if dry_run else "Deleted"
        logger.info(f"{action} {len(models_to_delete)} old models")

        return models_to_delete

    def export_model_config(self, model_id: str, export_path: str) -> None:
        """
        Export model configuration for reproducibility.

        Args:
            model_id: Model ID to export
            export_path: Path to save configuration
        """
        metadata = self.registry.get_model_metadata(model_id)
        if not metadata:
            raise ValueError(f"Model {model_id} not found")

        config = {
            'model_metadata': metadata.to_dict(),
            'model_class': metadata.model_type,
            'hyperparameters': metadata.hyperparameters,
            'feature_names': metadata.feature_names,
            'performance_metrics': metadata.performance_metrics,
            'export_timestamp': datetime.now().isoformat()
        }

        with open(export_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)

        logger.info(f"Exported model config for {model_id} to {export_path}")

    def import_model_config(self, config_path: str) -> str:
        """
        Import model configuration and recreate model entry.

        Args:
            config_path: Path to configuration file

        Returns:
            New model ID
        """
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Extract metadata
        metadata_dict = config['model_metadata']

        # Register new model
        model_id = self.registry.register_model(
            model_name=f"Imported {metadata_dict['model_name']}",
            model_type=metadata_dict['model_type'],
            version=f"imported_{metadata_dict['version']}",
            description=f"Imported from config: {metadata_dict.get('description', '')}",
            tags=(metadata_dict.get('tags', []) + ['imported'])
        )

        # Update with imported metadata
        self.registry.update_model_metadata(
            model_id,
            hyperparameters=config.get('hyperparameters'),
            feature_names=config.get('feature_names'),
            performance_metrics=config.get('performance_metrics')
        )

        logger.info(f"Imported model config as {model_id}")

        return model_id

    def get_best_model(self, 
                      metric: str = 'f1_score',
                      model_type: Optional[str] = None,
                      status: ModelStatus = ModelStatus.TRAINED) -> Optional[str]:
        """
        Get the best performing model based on a metric.

        Args:
            metric: Metric to use for comparison
            model_type: Optional filter by model type
            status: Filter by model status

        Returns:
            Model ID of best model or None if no models found
        """
        models = self.registry.list_models(status=status, model_type=model_type)

        if not models:
            return None

        best_model = None
        best_score = -np.inf if metric != 'false_positive_rate' else np.inf

        for metadata in models:
            if not metadata.performance_metrics:
                continue

            score = metadata.performance_metrics.get(metric)
            if score is None:
                continue

            if metric == 'false_positive_rate':
                if score < best_score:
                    best_score = score
                    best_model = metadata.model_id
            else:
                if score > best_score:
                    best_score = score
                    best_model = metadata.model_id

        return best_model

    def create_ensemble_from_models(self, 
                                   model_ids: List[str],
                                   ensemble_name: str = "Ensemble Model",
                                   weights: Optional[List[float]] = None) -> Tuple[str, BaseModel]:
        """
        Create an ensemble model from existing trained models.

        Args:
            model_ids: List of model IDs to include in ensemble
            ensemble_name: Name for the ensemble model
            weights: Optional weights for each model

        Returns:
            Tuple of (ensemble_model_id, ensemble_model)
        """
        # Load all base models
        base_models = {}
        for model_id in model_ids:
            model = self.load_model(model_id)
            metadata = self.registry.get_model_metadata(model_id)
            base_models[f"{metadata.model_name}_{model_id}"] = model

        # Create ensemble model
        ensemble = EnsembleCreditModel(model_name=ensemble_name)

        # Add base models with weights
        weights = weights or [1.0] * len(model_ids)
        for (model_name, model), weight in zip(base_models.items(), weights):
            ensemble.add_base_model(model_name, model, weight)

        # Register ensemble model
        ensemble_id = self.registry.register_model(
            model_name=ensemble_name,
            model_type='ensemble',
            version="1.0",
            description=f"Ensemble of models: {', '.join(model_ids)}",
            tags=['ensemble', 'combined']
        )

        # Mark as trained (ensemble doesn't need separate training)
        ensemble.is_trained = True
        ensemble.training_timestamp = datetime.now()

        # Update registry
        self.registry.update_model_status(ensemble_id, ModelStatus.TRAINED)

        # Save ensemble model
        model_path = self.models_dir / f"{ensemble_id}.pkl"
        ensemble.save_model(str(model_path))

        # Update file information
        file_size = model_path.stat().st_size
        self.registry.update_model_metadata(
            ensemble_id,
            file_path=str(model_path),
            file_size=file_size,
            hyperparameters={'base_models': model_ids, 'weights': weights}
        )

        logger.info(f"Created ensemble model {ensemble_id} from {len(model_ids)} base models")

        return ensemble_id, ensemble


# Export main classes
__all__ = [
    'ModelStatus',
    'ModelMetadata',
    'ModelRegistry',
    'ModelTrainer',
    'ModelManager'
]
