"""
Credit Model Module for Credit Analytics Hub

This module provides credit-specific model implementations including
logistic regression, random forest, gradient boosting, and neural networks
specifically designed for credit risk assessment.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from datetime import datetime
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
import joblib
import os
from pathlib import Path

# Import base model classes
from .base_model import BaseModel, SklearnModelWrapper

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CreditLogisticRegression(SklearnModelWrapper):
    """
    Logistic Regression model specifically designed for credit risk assessment
    """

    def __init__(self, 
                 C: float = 1.0,
                 penalty: str = 'l2',
                 solver: str = 'liblinear',
                 max_iter: int = 1000,
                 class_weight: Optional[Union[str, Dict]] = 'balanced',
                 random_state: int = 42,
                 **kwargs):
        """
        Initialize Credit Logistic Regression

        Args:
            C: Regularization strength
            penalty: Regularization penalty ('l1', 'l2', 'elasticnet')
            solver: Solver algorithm
            max_iter: Maximum iterations
            class_weight: Class weight balancing
            random_state: Random state for reproducibility
            **kwargs: Additional arguments for BaseModel
        """
        # Create sklearn logistic regression model
        sklearn_model = LogisticRegression(
            C=C,
            penalty=penalty,
            solver=solver,
            max_iter=max_iter,
            class_weight=class_weight,
            random_state=random_state
        )

        super().__init__(
            sklearn_model=sklearn_model,
            model_name="CreditLogisticRegression",
            **kwargs
        )

        # Credit-specific parameters
        self.credit_params = {
            'C': C,
            'penalty': penalty,
            'solver': solver,
            'max_iter': max_iter,
            'class_weight': class_weight
        }

        # Update metadata
        self.model_metadata.update({
            'model_type': 'credit_logistic_regression',
            'credit_params': self.credit_params,
            'interpretable': True,
            'suitable_for': ['binary_classification', 'probability_estimation', 'feature_importance']
        })

    def get_credit_coefficients(self) -> Dict[str, float]:
        """
        Get model coefficients with credit interpretation

        Returns:
            Dict[str, float]: Feature coefficients
        """
        self._check_is_fitted()

        if hasattr(self.model, 'coef_'):
            coef = self.model.coef_[0] if self.model.coef_.ndim > 1 else self.model.coef_
            coefficients = dict(zip(self.feature_names, coef))

            # Sort by absolute value for importance
            sorted_coef = dict(sorted(coefficients.items(), key=lambda x: abs(x[1]), reverse=True))

            return sorted_coef
        else:
            return {}

    def interpret_coefficients(self) -> Dict[str, Dict[str, Any]]:
        """
        Interpret coefficients in credit context

        Returns:
            Dict[str, Dict[str, Any]]: Interpreted coefficients
        """
        coefficients = self.get_credit_coefficients()
        interpretations = {}

        for feature, coef in coefficients.items():
            interpretation = {
                'coefficient': coef,
                'odds_ratio': np.exp(coef),
                'impact': 'increases_default_risk' if coef > 0 else 'decreases_default_risk',
                'magnitude': 'high' if abs(coef) > 1 else 'medium' if abs(coef) > 0.5 else 'low'
            }

            # Add percentage change in odds
            interpretation['odds_change_percent'] = (np.exp(coef) - 1) * 100

            interpretations[feature] = interpretation

        return interpretations

    def get_default_probability_factors(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Get factors contributing to default probability for each sample

        Args:
            X: Feature matrix

        Returns:
            pd.DataFrame: Probability factors for each sample
        """
        self._check_is_fitted()

        coefficients = self.get_credit_coefficients()

        # Calculate contribution of each feature to log-odds
        contributions = pd.DataFrame(index=X.index)

        for feature in self.feature_names:
            if feature in coefficients:
                contributions[f'{feature}_contribution'] = X[feature] * coefficients[feature]

        # Add intercept contribution
        if hasattr(self.model, 'intercept_'):
            contributions['intercept_contribution'] = self.model.intercept_[0]

        # Calculate total log-odds and probability
        contributions['total_log_odds'] = contributions.sum(axis=1)
        contributions['predicted_probability'] = 1 / (1 + np.exp(-contributions['total_log_odds']))

        return contributions


class CreditRandomForest(SklearnModelWrapper):
    """
    Random Forest model specifically designed for credit risk assessment
    """

    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 max_features: str = 'sqrt',
                 class_weight: Optional[Union[str, Dict]] = 'balanced',
                 random_state: int = 42,
                 n_jobs: int = -1,
                 **kwargs):
        """
        Initialize Credit Random Forest

        Args:
            n_estimators: Number of trees
            max_depth: Maximum depth of trees
            min_samples_split: Minimum samples to split
            min_samples_leaf: Minimum samples in leaf
            max_features: Number of features to consider
            class_weight: Class weight balancing
            random_state: Random state for reproducibility
            n_jobs: Number of parallel jobs
            **kwargs: Additional arguments for BaseModel
        """
        # Create sklearn random forest model
        sklearn_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=n_jobs
        )

        super().__init__(
            sklearn_model=sklearn_model,
            model_name="CreditRandomForest",
            **kwargs
        )

        # Credit-specific parameters
        self.credit_params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'max_features': max_features,
            'class_weight': class_weight
        }

        # Update metadata
        self.model_metadata.update({
            'model_type': 'credit_random_forest',
            'credit_params': self.credit_params,
            'interpretable': True,
            'suitable_for': ['binary_classification', 'feature_importance', 'non_linear_patterns']
        })

    def get_tree_based_insights(self) -> Dict[str, Any]:
        """
        Get insights specific to tree-based models

        Returns:
            Dict[str, Any]: Tree-based insights
        """
        self._check_is_fitted()

        insights = {
            'n_trees': self.model.n_estimators,
            'max_depth_used': max([tree.tree_.max_depth for tree in self.model.estimators_]),
            'avg_depth': np.mean([tree.tree_.max_depth for tree in self.model.estimators_]),
            'total_nodes': sum([tree.tree_.node_count for tree in self.model.estimators_]),
            'avg_nodes_per_tree': np.mean([tree.tree_.node_count for tree in self.model.estimators_])
        }

        return insights

    def get_feature_interactions(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Identify potential feature interactions based on tree splits

        Args:
            top_n: Number of top interactions to return

        Returns:
            List[Dict[str, Any]]: Feature interactions
        """
        self._check_is_fitted()

        # This is a simplified approach - in practice, you'd analyze tree structures
        feature_importance = self.get_feature_importance()

        # Get top features
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]

        interactions = []
        for i, (feat1, imp1) in enumerate(top_features):
            for feat2, imp2 in top_features[i+1:]:
                interaction = {
                    'feature_1': feat1,
                    'feature_2': feat2,
                    'combined_importance': imp1 + imp2,
                    'interaction_strength': min(imp1, imp2) / max(imp1, imp2)
                }
                interactions.append(interaction)

        # Sort by interaction strength
        interactions.sort(key=lambda x: x['interaction_strength'], reverse=True)

        return interactions[:top_n]

    def get_prediction_paths(self, X: pd.DataFrame, sample_idx: int = 0) -> List[Dict[str, Any]]:
        """
        Get decision paths for a specific sample

        Args:
            X: Feature matrix
            sample_idx: Index of sample to analyze

        Returns:
            List[Dict[str, Any]]: Decision paths from different trees
        """
        self._check_is_fitted()

        if sample_idx >= len(X):
            raise ValueError(f"Sample index {sample_idx} out of range")

        sample = X.iloc[sample_idx:sample_idx+1]
        paths = []

        # Analyze first few trees (to avoid too much output)
        for i, tree in enumerate(self.model.estimators_[:5]):
            decision_path = tree.decision_path(sample)
            leaf_id = tree.apply(sample)

            path_info = {
                'tree_index': i,
                'leaf_id': int(leaf_id[0]),
                'path_length': decision_path.nnz,
                'prediction': tree.predict_proba(sample)[0].tolist()
            }

            paths.append(path_info)

        return paths


class CreditGradientBoosting(SklearnModelWrapper):
    """
    Gradient Boosting model specifically designed for credit risk assessment
    """

    def __init__(self,
                 n_estimators: int = 100,
                 learning_rate: float = 0.1,
                 max_depth: int = 3,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 subsample: float = 1.0,
                 random_state: int = 42,
                 **kwargs):
        """
        Initialize Credit Gradient Boosting

        Args:
            n_estimators: Number of boosting stages
            learning_rate: Learning rate
            max_depth: Maximum depth of trees
            min_samples_split: Minimum samples to split
            min_samples_leaf: Minimum samples in leaf
            subsample: Fraction of samples for fitting
            random_state: Random state for reproducibility
            **kwargs: Additional arguments for BaseModel
        """
        # Create sklearn gradient boosting model
        sklearn_model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            subsample=subsample,
            random_state=random_state
        )

        super().__init__(
            sklearn_model=sklearn_model,
            model_name="CreditGradientBoosting",
            **kwargs
        )

        # Credit-specific parameters
        self.credit_params = {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'subsample': subsample
        }

        # Update metadata
        self.model_metadata.update({
            'model_type': 'credit_gradient_boosting',
            'credit_params': self.credit_params,
            'interpretable': True,
            'suitable_for': ['binary_classification', 'feature_importance', 'sequential_learning']
        })

    def get_boosting_insights(self) -> Dict[str, Any]:
        """
        Get insights specific to gradient boosting

        Returns:
            Dict[str, Any]: Boosting insights
        """
        self._check_is_fitted()

        insights = {
            'n_estimators_used': self.model.n_estimators_,
            'train_score_': self.model.train_score_.tolist() if hasattr(self.model, 'train_score_') else None,
            'oob_improvement_': self.model.oob_improvement_.tolist() if hasattr(self.model, 'oob_improvement_') else None,
            'learning_rate': self.model.learning_rate,
            'max_depth': self.model.max_depth
        }

        return insights

    def plot_learning_curve(self) -> Dict[str, List[float]]:
        """
        Get data for plotting learning curve

        Returns:
            Dict[str, List[float]]: Learning curve data
        """
        self._check_is_fitted()

        if hasattr(self.model, 'train_score_'):
            return {
                'iterations': list(range(len(self.model.train_score_))),
                'train_scores': self.model.train_score_.tolist(),
                'oob_scores': self.model.oob_improvement_.tolist() if hasattr(self.model, 'oob_improvement_') else None
            }
        else:
            return {}

    def get_staged_predictions(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get predictions at each boosting stage

        Args:
            X: Feature matrix

        Returns:
            np.ndarray: Staged predictions
        """
        self._check_is_fitted()

        staged_proba = list(self.model.staged_predict_proba(X))
        return np.array(staged_proba)


class CreditSVM(SklearnModelWrapper):
    """
    Support Vector Machine model specifically designed for credit risk assessment
    """

    def __init__(self,
                 C: float = 1.0,
                 kernel: str = 'rbf',
                 gamma: str = 'scale',
                 class_weight: Optional[Union[str, Dict]] = 'balanced',
                 probability: bool = True,
                 random_state: int = 42,
                 **kwargs):
        """
        Initialize Credit SVM

        Args:
            C: Regularization parameter
            kernel: Kernel type
            gamma: Kernel coefficient
            class_weight: Class weight balancing
            probability: Enable probability estimates
            random_state: Random state for reproducibility
            **kwargs: Additional arguments for BaseModel
        """
        # Create sklearn SVM model
        sklearn_model = SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            class_weight=class_weight,
            probability=probability,
            random_state=random_state
        )

        super().__init__(
            sklearn_model=sklearn_model,
            model_name="CreditSVM",
            **kwargs
        )

        # Credit-specific parameters
        self.credit_params = {
            'C': C,
            'kernel': kernel,
            'gamma': gamma,
            'class_weight': class_weight,
            'probability': probability
        }

        # Update metadata
        self.model_metadata.update({
            'model_type': 'credit_svm',
            'credit_params': self.credit_params,
            'interpretable': False,
            'suitable_for': ['binary_classification', 'non_linear_patterns', 'high_dimensional']
        })

    def get_support_vector_info(self) -> Dict[str, Any]:
        """
        Get information about support vectors

        Returns:
            Dict[str, Any]: Support vector information
        """
        self._check_is_fitted()

        info = {
            'n_support_vectors': self.model.n_support_.tolist() if hasattr(self.model, 'n_support_') else None,
            'total_support_vectors': self.model.support_vectors_.shape[0] if hasattr(self.model, 'support_vectors_') else None,
            'support_vector_ratio': (self.model.support_vectors_.shape[0] / self.n_features) if hasattr(self.model, 'support_vectors_') else None
        }

        return info

    def get_decision_function_stats(self, X: pd.DataFrame) -> Dict[str, float]:
        """
        Get statistics about decision function values

        Args:
            X: Feature matrix

        Returns:
            Dict[str, float]: Decision function statistics
        """
        self._check_is_fitted()

        if hasattr(self.model, 'decision_function'):
            decision_values = self.model.decision_function(X)

            stats = {
                'mean_decision_value': float(np.mean(decision_values)),
                'std_decision_value': float(np.std(decision_values)),
                'min_decision_value': float(np.min(decision_values)),
                'max_decision_value': float(np.max(decision_values)),
                'median_decision_value': float(np.median(decision_values))
            }

            return stats
        else:
            return {}


class CreditNeuralNetwork(SklearnModelWrapper):
    """
    Neural Network model specifically designed for credit risk assessment
    """

    def __init__(self,
                 hidden_layer_sizes: Tuple[int, ...] = (100, 50),
                 activation: str = 'relu',
                 solver: str = 'adam',
                 alpha: float = 0.0001,
                 learning_rate: str = 'constant',
                 learning_rate_init: float = 0.001,
                 max_iter: int = 200,
                 random_state: int = 42,
                 **kwargs):
        """
        Initialize Credit Neural Network

        Args:
            hidden_layer_sizes: Sizes of hidden layers
            activation: Activation function
            solver: Solver for weight optimization
            alpha: L2 penalty parameter
            learning_rate: Learning rate schedule
            learning_rate_init: Initial learning rate
            max_iter: Maximum iterations
            random_state: Random state for reproducibility
            **kwargs: Additional arguments for BaseModel
        """
        # Create sklearn MLP model
        sklearn_model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            learning_rate=learning_rate,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            random_state=random_state
        )

        super().__init__(
            sklearn_model=sklearn_model,
            model_name="CreditNeuralNetwork",
            **kwargs
        )

        # Credit-specific parameters
        self.credit_params = {
            'hidden_layer_sizes': hidden_layer_sizes,
            'activation': activation,
            'solver': solver,
            'alpha': alpha,
            'learning_rate': learning_rate,
            'learning_rate_init': learning_rate_init,
            'max_iter': max_iter
        }

        # Update metadata
        self.model_metadata.update({
            'model_type': 'credit_neural_network',
            'credit_params': self.credit_params,
            'interpretable': False,
            'suitable_for': ['binary_classification', 'complex_patterns', 'non_linear_relationships']
        })

    def get_network_info(self) -> Dict[str, Any]:
        """
        Get information about the neural network

        Returns:
            Dict[str, Any]: Network information
        """
        self._check_is_fitted()

        info = {
            'n_layers': self.model.n_layers_,
            'n_outputs': self.model.n_outputs_,
            'hidden_layer_sizes': self.model.hidden_layer_sizes,
            'n_iter': self.model.n_iter_,
            'loss': self.model.loss_,
            'out_activation': self.model.out_activation_
        }

        # Add loss curve if available
        if hasattr(self.model, 'loss_curve_'):
            info['loss_curve'] = self.model.loss_curve_.tolist()

        return info

    def get_layer_weights_stats(self) -> List[Dict[str, float]]:
        """
        Get statistics about weights in each layer

        Returns:
            List[Dict[str, float]]: Weight statistics for each layer
        """
        self._check_is_fitted()

        if hasattr(self.model, 'coefs_'):
            layer_stats = []

            for i, layer_weights in enumerate(self.model.coefs_):
                stats = {
                    'layer': i,
                    'shape': layer_weights.shape,
                    'mean_weight': float(np.mean(layer_weights)),
                    'std_weight': float(np.std(layer_weights)),
                    'min_weight': float(np.min(layer_weights)),
                    'max_weight': float(np.max(layer_weights)),
                    'n_parameters': layer_weights.size
                }
                layer_stats.append(stats)

            return layer_stats
        else:
            return []
# credit_model_content_part2 =

class CreditEnsembleVoting(BaseModel):
    """
    Voting ensemble for credit models
    """
    
    def __init__(self,
                 models: List[BaseModel],
                 voting: str = 'soft',
                 weights: Optional[List[float]] = None,
                 model_name: str = "CreditEnsembleVoting",
                 **kwargs):
        """
        Initialize Credit Voting Ensemble
        
        Args:
            models: List of base models
            voting: Voting type ('hard' or 'soft')
            weights: Model weights for voting
            model_name: Name of the ensemble model
            **kwargs: Additional arguments for BaseModel
        """
        super().__init__(model_name=model_name, **kwargs)
        
        self.base_models = models
        self.voting = voting
        self.weights = weights
        
        # Update metadata
        self.model_metadata.update({
            'model_type': 'credit_ensemble_voting',
            'n_base_models': len(models),
            'base_model_names': [model.model_name for model in models],
            'voting_type': voting,
            'weighted': weights is not None
        })
    
    def fit(self, 
            X: pd.DataFrame, 
            y: pd.Series,
            validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
            **kwargs) -> 'CreditEnsembleVoting':
        """
        Train all base models
        
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
        Make ensemble predictions using voting
        
        Args:
            X: Feature matrix
            
        Returns:
            np.ndarray: Ensemble predictions
        """
        super().predict(X)  # Validation and checks
        
        if self.voting == 'hard':
            # Hard voting - majority vote
            predictions = np.array([model.predict(X) for model in self.base_models])
            
            if self.weights is not None:
                # Weighted hard voting
                weighted_predictions = []
                for i in range(len(X)):
                    votes = {}
                    for j, pred in enumerate(predictions[:, i]):
                        weight = self.weights[j]
                        votes[pred] = votes.get(pred, 0) + weight
                    weighted_predictions.append(max(votes.items(), key=lambda x: x[1])[0])
                return np.array(weighted_predictions)
            else:
                # Simple majority vote
                return np.array([np.bincount(predictions[:, i]).argmax() for i in range(len(X))])
        
        else:  # soft voting
            # Soft voting - average probabilities
            probabilities = np.array([model.predict_proba(X) for model in self.base_models])
            
            if self.weights is not None:
                # Weighted average
                weighted_proba = np.average(probabilities, axis=0, weights=self.weights)
            else:
                # Simple average
                weighted_proba = np.mean(probabilities, axis=0)
            
            return (weighted_proba[:, 1] > 0.5).astype(int)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict ensemble probabilities
        
        Args:
            X: Feature matrix
            
        Returns:
            np.ndarray: Ensemble probabilities
        """
        super().predict_proba(X)  # Validation and checks
        
        # Get probabilities from all models
        probabilities = np.array([model.predict_proba(X) for model in self.base_models])
        
        if self.weights is not None:
            # Weighted average
            ensemble_proba = np.average(probabilities, axis=0, weights=self.weights)
        else:
            # Simple average
            ensemble_proba = np.mean(probabilities, axis=0)
        
        return ensemble_proba
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get aggregated feature importance from all base models
        
        Returns:
            Dict[str, float]: Aggregated feature importance
        """
        super().get_feature_importance()  # Check if fitted
        
        # Aggregate feature importance from all base models
        aggregated_importance = {}
        total_weight = 0
        
        for i, model in enumerate(self.base_models):
            try:
                model_importance = model.get_feature_importance()
                weight = self.weights[i] if self.weights else 1.0
                total_weight += weight
                
                for feature, importance in model_importance.items():
                    if feature not in aggregated_importance:
                        aggregated_importance[feature] = 0.0
                    aggregated_importance[feature] += importance * weight
                    
            except Exception as e:
                logger.warning(f"Could not get feature importance from {model.model_name}: {str(e)}")
        
        # Normalize by total weight
        if total_weight > 0:
            for feature in aggregated_importance:
                aggregated_importance[feature] /= total_weight
        
        return aggregated_importance
    
    def get_model_contributions(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Get individual model contributions to ensemble predictions
        
        Args:
            X: Feature matrix
            
        Returns:
            Dict[str, np.ndarray]: Model contributions
        """
        self._check_is_fitted()
        
        contributions = {}
        
        for model in self.base_models:
            model_proba = model.predict_proba(X)[:, 1]  # Probability of positive class
            contributions[model.model_name] = model_proba
        
        return contributions
    
    def save_model(self, filepath: str, save_format: str = 'joblib') -> None:
        """
        Save the ensemble model
        
        Args:
            filepath: Path to save the model
            save_format: Format to save ('joblib', 'pickle')
        """
        self._check_is_fitted()
        
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare ensemble data
        ensemble_data = {
            'base_models': self.base_models,
            'voting': self.voting,
            'weights': self.weights,
            'model_name': self.model_name,
            'feature_names': self.feature_names,
            'n_features': self.n_features,
            'classes_': self.classes_,
            'model_metadata': self.model_metadata,
            'training_history': self.training_history,
            'is_fitted': self.is_fitted
        }
        
        # Save based on format
        if save_format == 'joblib':
            joblib.dump(ensemble_data, filepath)
        elif save_format == 'pickle':
            with open(filepath, 'wb') as f:
                pickle.dump(ensemble_data, f)
        else:
            raise ValueError(f"Unsupported save format: {save_format}")
        
        if self.verbose:
            logger.info(f"Ensemble model saved to {filepath}")
    
    def load_model(self, filepath: str, load_format: str = 'auto') -> 'CreditEnsembleVoting':
        """
        Load the ensemble model
        
        Args:
            filepath: Path to load the model from
            load_format: Format to load ('joblib', 'pickle', 'auto')
            
        Returns:
            CreditEnsembleVoting: Loaded ensemble model
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
                ensemble_data = joblib.load(filepath)
            elif load_format == 'pickle':
                with open(filepath, 'rb') as f:
                    ensemble_data = pickle.load(f)
            else:
                raise ValueError(f"Unsupported load format: {load_format}")
            
            # Restore ensemble state
            self.base_models = ensemble_data['base_models']
            self.voting = ensemble_data['voting']
            self.weights = ensemble_data['weights']
            self.model_name = ensemble_data['model_name']
            self.feature_names = ensemble_data['feature_names']
            self.n_features = ensemble_data['n_features']
            self.classes_ = ensemble_data['classes_']
            self.model_metadata = ensemble_data['model_metadata']
            self.training_history = ensemble_data['training_history']
            self.is_fitted = ensemble_data['is_fitted']
            
            if self.verbose:
                logger.info(f"Ensemble model loaded from {filepath}")
            
            return self
            
        except Exception as e:
            logger.error(f"Error loading ensemble model: {str(e)}")
            raise


class CreditEnsembleStacking(BaseModel):
    """
    Stacking ensemble for credit models
    """
    
    def __init__(self,
                 base_models: List[BaseModel],
                 meta_model: BaseModel,
                 cv_folds: int = 5,
                 model_name: str = "CreditEnsembleStacking",
                 **kwargs):
        """
        Initialize Credit Stacking Ensemble
        
        Args:
            base_models: List of base models
            meta_model: Meta-learner model
            cv_folds: Number of CV folds for generating meta-features
            model_name: Name of the ensemble model
            **kwargs: Additional arguments for BaseModel
        """
        super().__init__(model_name=model_name, **kwargs)
        
        self.base_models = base_models
        self.meta_model = meta_model
        self.cv_folds = cv_folds
        self.meta_features = None
        
        # Update metadata
        self.model_metadata.update({
            'model_type': 'credit_ensemble_stacking',
            'n_base_models': len(base_models),
            'base_model_names': [model.model_name for model in base_models],
            'meta_model_name': meta_model.model_name,
            'cv_folds': cv_folds
        })
    
    def fit(self, 
            X: pd.DataFrame, 
            y: pd.Series,
            validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
            **kwargs) -> 'CreditEnsembleStacking':
        """
        Train stacking ensemble
        
        Args:
            X: Feature matrix
            y: Target vector
            validation_data: Optional validation data
            **kwargs: Additional training parameters
            
        Returns:
            Self: Trained stacking ensemble
        """
        from sklearn.model_selection import KFold
        
        # Call parent fit for validation and logging
        super().fit(X, y, validation_data, **kwargs)
        
        try:
            # Generate meta-features using cross-validation
            self._log_training_step("generating_meta_features", {
                'cv_folds': self.cv_folds,
                'n_base_models': len(self.base_models)
            })
            
            kfold = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            meta_features = np.zeros((len(X), len(self.base_models)))
            
            # For each fold, train base models and generate predictions
            for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X)):
                X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
                y_fold_train = y.iloc[train_idx]
                
                # Train each base model on fold training data
                for model_idx, model in enumerate(self.base_models):
                    # Clone model to avoid interference
                    fold_model = model.clone()
                    fold_model.fit(X_fold_train, y_fold_train)
                    
                    # Generate predictions for fold validation data
                    fold_predictions = fold_model.predict_proba(X_fold_val)[:, 1]
                    meta_features[val_idx, model_idx] = fold_predictions
            
            # Train base models on full dataset
            self._log_training_step("training_base_models_full", {})
            for i, model in enumerate(self.base_models):
                self._log_training_step(f"training_base_model_{i}", {
                    'model_name': model.model_name
                })
                model.fit(X, y, validation_data, **kwargs)
            
            # Train meta-model on meta-features
            self._log_training_step("training_meta_model", {
                'meta_model_name': self.meta_model.model_name
            })
            
            meta_features_df = pd.DataFrame(
                meta_features, 
                columns=[f"{model.model_name}_pred" for model in self.base_models]
            )
            
            self.meta_model.fit(meta_features_df, y)
            self.meta_features = meta_features_df
            
            self.is_fitted = True
            
            # Calculate ensemble training scores
            self.training_scores = self.evaluate(X, y)
            
            if validation_data is not None:
                X_val, y_val = validation_data
                self.validation_scores = self.evaluate(X_val, y_val)
            
            self._log_training_step("stacking_training_completed", {
                'training_scores': self.training_scores,
                'validation_scores': self.validation_scores
            })
            
            return self
            
        except Exception as e:
            logger.error(f"Error training stacking ensemble {self.model_name}: {str(e)}")
            raise
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make stacking ensemble predictions
        
        Args:
            X: Feature matrix
            
        Returns:
            np.ndarray: Ensemble predictions
        """
        super().predict(X)  # Validation and checks
        
        # Generate meta-features from base models
        meta_features = self._generate_meta_features(X)
        
        # Use meta-model to make final predictions
        return self.meta_model.predict(meta_features)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict stacking ensemble probabilities
        
        Args:
            X: Feature matrix
            
        Returns:
            np.ndarray: Ensemble probabilities
        """
        super().predict_proba(X)  # Validation and checks
        
        # Generate meta-features from base models
        meta_features = self._generate_meta_features(X)
        
        # Use meta-model to make final probability predictions
        return self.meta_model.predict_proba(meta_features)
    
    def _generate_meta_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Generate meta-features from base models
        
        Args:
            X: Feature matrix
            
        Returns:
            pd.DataFrame: Meta-features
        """
        meta_features = []
        
        for model in self.base_models:
            model_proba = model.predict_proba(X)[:, 1]  # Probability of positive class
            meta_features.append(model_proba)
        
        meta_features_df = pd.DataFrame(
            np.column_stack(meta_features),
            columns=[f"{model.model_name}_pred" for model in self.base_models],
            index=X.index
        )
        
        return meta_features_df
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from meta-model
        
        Returns:
            Dict[str, float]: Meta-model feature importance
        """
        super().get_feature_importance()  # Check if fitted
        
        try:
            return self.meta_model.get_feature_importance()
        except Exception as e:
            logger.warning(f"Could not get feature importance from meta-model: {str(e)}")
            return {}
    
    def get_base_model_contributions(self) -> Dict[str, float]:
        """
        Get contributions of base models according to meta-model
        
        Returns:
            Dict[str, float]: Base model contributions
        """
        self._check_is_fitted()
        
        try:
            meta_importance = self.meta_model.get_feature_importance()
            return {
                model.model_name: meta_importance.get(f"{model.model_name}_pred", 0.0)
                for model in self.base_models
            }
        except Exception as e:
            logger.warning(f"Could not get base model contributions: {str(e)}")
            return {}


class CreditModelTrainer:
    """
    Utility class for training and comparing credit models
    """
    
    def __init__(self, 
                 random_state: int = 42,
                 verbose: bool = True):
        """
        Initialize Credit Model Trainer
        
        Args:
            random_state: Random state for reproducibility
            verbose: Whether to print verbose output
        """
        self.random_state = random_state
        self.verbose = verbose
        self.trained_models = {}
        self.training_results = {}
        
    def train_single_model(self,
                          model: BaseModel,
                          X_train: pd.DataFrame,
                          y_train: pd.Series,
                          X_val: Optional[pd.DataFrame] = None,
                          y_val: Optional[pd.Series] = None,
                          **kwargs) -> Dict[str, Any]:
        """
        Train a single model
        
        Args:
            model: Model to train
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            **kwargs: Additional training parameters
            
        Returns:
            Dict[str, Any]: Training results
        """
        try:
            if self.verbose:
                logger.info(f"Training {model.model_name}...")
            
            # Prepare validation data
            validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
            
            # Train the model
            start_time = datetime.now()
            model.fit(X_train, y_train, validation_data=validation_data, **kwargs)
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Evaluate the model
            train_metrics = model.evaluate(X_train, y_train)
            val_metrics = model.evaluate(X_val, y_val) if validation_data else {}
            
            # Store results
            results = {
                'model': model,
                'model_name': model.model_name,
                'training_time': training_time,
                'train_metrics': train_metrics,
                'validation_metrics': val_metrics,
                'model_info': model.get_model_info()
            }
            
            self.trained_models[model.model_name] = model
            self.training_results[model.model_name] = results
            
            if self.verbose:
                logger.info(f"✅ {model.model_name} training completed in {training_time:.2f}s")
                if val_metrics:
                    logger.info(f"   Validation ROC-AUC: {val_metrics.get('roc_auc', 'N/A'):.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error training {model.model_name}: {str(e)}")
            raise
    
    def train_multiple_models(self,
                             models: List[BaseModel],
                             X_train: pd.DataFrame,
                             y_train: pd.Series,
                             X_val: Optional[pd.DataFrame] = None,
                             y_val: Optional[pd.Series] = None,
                             **kwargs) -> Dict[str, Dict[str, Any]]:
        """
        Train multiple models
        
        Args:
            models: List of models to train
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            **kwargs: Additional training parameters
            
        Returns:
            Dict[str, Dict[str, Any]]: Training results for all models
        """
        results = {}
        
        for model in models:
            try:
                model_results = self.train_single_model(
                    model, X_train, y_train, X_val, y_val, **kwargs
                )
                results[model.model_name] = model_results
            except Exception as e:
                logger.error(f"Failed to train {model.model_name}: {str(e)}")
                continue
        
        return results
    
    def create_ensemble_models(self,
                              base_models: List[BaseModel],
                              ensemble_types: List[str] = ['voting', 'stacking']) -> List[BaseModel]:
        """
        Create ensemble models from base models
        
        Args:
            base_models: List of trained base models
            ensemble_types: Types of ensembles to create
            
        Returns:
            List[BaseModel]: List of ensemble models
        """
        ensemble_models = []
        
        if 'voting' in ensemble_types:
            # Create voting ensemble
            voting_ensemble = CreditEnsembleVoting(
                models=base_models,
                voting='soft',
                model_name="VotingEnsemble",
                random_state=self.random_state,
                verbose=self.verbose
            )
            ensemble_models.append(voting_ensemble)
        
        if 'stacking' in ensemble_types:
            # Create stacking ensemble with logistic regression as meta-model
            meta_model = CreditLogisticRegression(
                model_name="StackingMetaModel",
                random_state=self.random_state,
                verbose=False
            )
            
            stacking_ensemble = CreditEnsembleStacking(
                base_models=base_models,
                meta_model=meta_model,
                cv_folds=5,
                model_name="StackingEnsemble",
                random_state=self.random_state,
                verbose=self.verbose
            )
            ensemble_models.append(stacking_ensemble)
        
        return ensemble_models
    
    def hyperparameter_tuning(self,
                             model_class: type,
                             param_grid: Dict[str, List],
                             X_train: pd.DataFrame,
                             y_train: pd.Series,
                             cv: int = 5,
                             scoring: str = 'roc_auc',
                             search_type: str = 'grid') -> Dict[str, Any]:
        """
        Perform hyperparameter tuning
        
        Args:
            model_class: Model class to tune
            param_grid: Parameter grid for tuning
            X_train: Training features
            y_train: Training target
            cv: Cross-validation folds
            scoring: Scoring metric
            search_type: Search type ('grid' or 'random')
            
        Returns:
            Dict[str, Any]: Tuning results
        """
        try:
            if self.verbose:
                logger.info(f"Starting hyperparameter tuning for {model_class.__name__}...")
            
            # Create base model instance
            base_model = model_class(random_state=self.random_state, verbose=False)
            
            # Extract sklearn model for tuning
            if hasattr(base_model, 'model'):
                sklearn_model = base_model.model
            else:
                raise ValueError("Model does not have sklearn model attribute")
            
            # Perform search
            if search_type == 'grid':
                search = GridSearchCV(
                    sklearn_model,
                    param_grid,
                    cv=cv,
                    scoring=scoring,
                    n_jobs=-1,
                    verbose=1 if self.verbose else 0
                )
            else:  # random search
                search = RandomizedSearchCV(
                    sklearn_model,
                    param_grid,
                    cv=cv,
                    scoring=scoring,
                    n_jobs=-1,
                    n_iter=20,
                    verbose=1 if self.verbose else 0,
                    random_state=self.random_state
                )
            
            # Fit search
            search.fit(X_train, y_train)
            
            # Create best model
            best_model = model_class(**search.best_params_, 
                                   random_state=self.random_state,
                                   verbose=self.verbose)
            
            results = {
                'best_model': best_model,
                'best_params': search.best_params_,
                'best_score': search.best_score_,
                'cv_results': search.cv_results_,
                'search_type': search_type,
                'scoring': scoring
            }
            
            if self.verbose:
                logger.info(f"✅ Hyperparameter tuning completed")
                logger.info(f"   Best {scoring}: {search.best_score_:.4f}")
                logger.info(f"   Best params: {search.best_params_}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error during hyperparameter tuning: {str(e)}")
            raise
    
    def get_training_summary(self) -> pd.DataFrame:
        """
        Get summary of all trained models
        
        Returns:
            pd.DataFrame: Training summary
        """
        if not self.training_results:
            return pd.DataFrame()
        
        summary_data = []
        
        for model_name, results in self.training_results.items():
            row = {
                'model_name': model_name,
                'training_time': results['training_time']
            }
            
            # Add training metrics
            for metric, value in results['train_metrics'].items():
                row[f'train_{metric}'] = value
            
            # Add validation metrics
            for metric, value in results['validation_metrics'].items():
                row[f'val_{metric}'] = value
            
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        
        # Sort by validation ROC-AUC if available
        if 'val_roc_auc' in summary_df.columns:
            summary_df = summary_df.sort_values('val_roc_auc', ascending=False)
        
        return summary_df
    
    def save_all_models(self, output_dir: str) -> None:
        """
        Save all trained models
        
        Args:
            output_dir: Directory to save models
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for model_name, model in self.trained_models.items():
            model_path = output_path / f"{model_name}.joblib"
            model.save_model(str(model_path))
            
            if self.verbose:
                logger.info(f"Saved {model_name} to {model_path}")


# Convenience functions for model creation
def create_credit_logistic_regression(**kwargs) -> CreditLogisticRegression:
    """Create Credit Logistic Regression model"""
    return CreditLogisticRegression(**kwargs)

def create_credit_random_forest(**kwargs) -> CreditRandomForest:
    """Create Credit Random Forest model"""
    return CreditRandomForest(**kwargs)

def create_credit_gradient_boosting(**kwargs) -> CreditGradientBoosting:
    """Create Credit Gradient Boosting model"""
    return CreditGradientBoosting(**kwargs)

def create_credit_svm(**kwargs) -> CreditSVM:
    """Create Credit SVM model"""
    return CreditSVM(**kwargs)

def create_credit_neural_network(**kwargs) -> CreditNeuralNetwork:
    """Create Credit Neural Network model"""
    return CreditNeuralNetwork(**kwargs)

def create_voting_ensemble(models: List[BaseModel], **kwargs) -> CreditEnsembleVoting:
    """Create Voting Ensemble model"""
    return CreditEnsembleVoting(models=models, **kwargs)

def create_stacking_ensemble(base_models: List[BaseModel], meta_model: BaseModel, **kwargs) -> CreditEnsembleStacking:
    """Create Stacking Ensemble model"""
    return CreditEnsembleStacking(base_models=base_models, meta_model=meta_model, **kwargs)

def get_default_credit_models(random_state: int = 42) -> List[BaseModel]:
    """
    Get a list of default credit models with reasonable parameters
    
    Args:
        random_state: Random state for reproducibility
        
    Returns:
        List[BaseModel]: List of default credit models
    """
    models = [
        CreditLogisticRegression(random_state=random_state, verbose=False),
        CreditRandomForest(n_estimators=100, random_state=random_state, verbose=False),
        CreditGradientBoosting(n_estimators=100, random_state=random_state, verbose=False),
        CreditSVM(random_state=random_state, verbose=False),
        CreditNeuralNetwork(hidden_layer_sizes=(100, 50), random_state=random_state, verbose=False)
    ]
    
    return models

def train_credit_models_pipeline(self,
                                    X_train: pd.DataFrame,
                                    y_train: pd.Series,
                                    X_val: Optional[pd.DataFrame] = None,
                                    y_val: Optional[pd.Series] = None,
                                    include_ensembles: bool = True,
                                    include_sklearn_models: bool = True,
                                    random_state: int = 42) -> Dict[str, Any]:
        """
        Complete pipeline for training credit models

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            include_ensembles: Whether to include ensemble models
            include_sklearn_models: Whether to include sklearn models
            random_state: Random state for reproducibility

        Returns:
            Dict[str, Any]: Dictionary of trained models
        """
        results = {}

        # Prepare base models
        base_models = get_default_credit_models(random_state=random_state)

        # Optionally include sklearn models (wrapped)
        if include_sklearn_models:
            # Example: add sklearn LogisticRegression wrapped as BaseModel
            from sklearn.linear_model import LogisticRegression as SklearnLR
            sklearn_lr = SklearnLR(random_state=random_state)
            from .base_model import SklearnModelWrapper
            sklearn_lr_wrapper = SklearnModelWrapper(sklearn_model=sklearn_lr, model_name="SklearnLogisticRegression")
            base_models.append(sklearn_lr_wrapper)

        # Train base models
        base_results = self.train_multiple_models(base_models, X_train, y_train, X_val, y_val)
        results.update(base_results)

        # Optionally create and train ensemble models
        if include_ensembles:
            ensemble_models = self.create_ensemble_models(base_models)
            ensemble_results = self.train_multiple_models(ensemble_models, X_train, y_train, X_val, y_val)
            results.update(ensemble_results)

        return results

# def train_credit_models_pipeline(X_train: pd.DataFrame,
#                                 y_train: pd.Series,
#                                 X_val: Optional[pd.DataFrame] = None,
#                                 y_val: Optional[pd.Series] = None,
#                                 include_ensembles: bool = True,
#                                 random_state: int = 42) -> Dict[str, Any]:
#     """
#     Complete pipeline for training credit models
    
#     Args:
#         X_train: Training features
#         y_train: Training labels
#         X_val: Validation features (optional)
#         y_val: Validation labels (optional)
#         include_ensembles: Whether to include ensemble models
#         include_sklearn_models: Whether to include sklearn models
#         random_state: Random state for reproducibility
        
#     Returns:
#         Dict[str, Any]: Dictionary of trained models
#     """
    

   