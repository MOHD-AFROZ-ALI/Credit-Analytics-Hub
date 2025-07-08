"""
Credit Model Module for Credit Analytics Hub

This module provides specialized credit risk modeling classes that extend
the base model functionality with credit-specific features and algorithms.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
import warnings
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import logging
from datetime import datetime

# Import base classes
from .base_model import BaseModel, DataPreprocessor, ModelMetrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CreditRiskPreprocessor(DataPreprocessor):
    """
    Specialized data preprocessor for credit risk modeling.

    Extends the base preprocessor with credit-specific transformations
    and feature engineering capabilities.
    """

    def __init__(self, 
                 handle_missing: str = 'median',
                 encode_categorical: str = 'label',
                 scale_features: bool = True,
                 remove_outliers: bool = True,
                 outlier_threshold: float = 3.0,
                 create_risk_features: bool = True,
                 debt_to_income_threshold: float = 0.4):
        """
        Initialize the credit risk preprocessor.

        Args:
            handle_missing: Strategy for handling missing values
            encode_categorical: Method for encoding categorical variables
            scale_features: Whether to scale numerical features
            remove_outliers: Whether to remove outliers
            outlier_threshold: Z-score threshold for outlier detection
            create_risk_features: Whether to create additional risk features
            debt_to_income_threshold: Threshold for debt-to-income ratio risk
        """
        super().__init__(handle_missing, encode_categorical, scale_features, 
                        remove_outliers, outlier_threshold)
        self.create_risk_features = create_risk_features
        self.debt_to_income_threshold = debt_to_income_threshold
        self.risk_feature_names = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'CreditRiskPreprocessor':
        """
        Fit the credit risk preprocessor to the training data.

        Args:
            X: Input features
            y: Target variable (optional)

        Returns:
            self: Fitted preprocessor
        """
        X = X.copy()

        # Create risk features before fitting base preprocessor
        if self.create_risk_features:
            X = self._create_risk_features(X)

        # Fit base preprocessor
        super().fit(X, y)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input data using fitted preprocessor.

        Args:
            X: Input features to transform

        Returns:
            Transformed features with risk indicators
        """
        X = X.copy()

        # Create risk features
        if self.create_risk_features:
            X = self._create_risk_features(X)

        # Apply base transformation
        X = super().transform(X)

        return X

    def _create_risk_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional risk-related features.

        Args:
            X: Input features

        Returns:
            DataFrame with additional risk features
        """
        X = X.copy()

        # Debt-to-Income ratio risk indicator
        if 'annual_income' in X.columns and 'loan_amount' in X.columns:
            X['debt_to_income_ratio'] = X['loan_amount'] / (X['annual_income'] + 1e-6)
            X['high_debt_to_income'] = (X['debt_to_income_ratio'] > self.debt_to_income_threshold).astype(int)
            self.risk_feature_names.extend(['debt_to_income_ratio', 'high_debt_to_income'])

        # Credit utilization risk
        if 'credit_limit' in X.columns and 'current_balance' in X.columns:
            X['credit_utilization'] = X['current_balance'] / (X['credit_limit'] + 1e-6)
            X['high_credit_utilization'] = (X['credit_utilization'] > 0.8).astype(int)
            self.risk_feature_names.extend(['credit_utilization', 'high_credit_utilization'])

        # Employment stability indicator
        if 'employment_length' in X.columns:
            X['employment_stability'] = (X['employment_length'] >= 2).astype(int)
            self.risk_feature_names.append('employment_stability')

        # Age-based risk categories
        if 'age' in X.columns:
            X['age_risk_category'] = pd.cut(X['age'], 
                                          bins=[0, 25, 35, 50, 65, 100], 
                                          labels=[4, 3, 2, 1, 2],  # Higher numbers = higher risk
                                          ordered=False).astype(float)
            self.risk_feature_names.append('age_risk_category')

        # Loan amount risk categories
        if 'loan_amount' in X.columns:
            loan_percentiles = X['loan_amount'].quantile([0.25, 0.5, 0.75])
            X['loan_amount_risk'] = pd.cut(X['loan_amount'],
                                         bins=[0, loan_percentiles[0.25], 
                                              loan_percentiles[0.5], 
                                              loan_percentiles[0.75], 
                                              X['loan_amount'].max()],
                                         labels=[1, 2, 3, 4],
                                         ordered=False).astype(float)
            self.risk_feature_names.append('loan_amount_risk')

        # Payment history risk
        if 'payment_history_score' in X.columns:
            X['poor_payment_history'] = (X['payment_history_score'] < 600).astype(int)
            self.risk_feature_names.append('poor_payment_history')

        return X

class LogisticRegressionCreditModel(BaseModel):
    """
    Logistic Regression model specialized for credit risk assessment.

    Provides interpretable linear relationships between features and default probability.
    """

    def __init__(self, 
                 model_name: str = "Logistic Regression Credit Model",
                 model_version: str = "1.0",
                 random_state: int = 42,
                 regularization: str = 'l2',
                 C: float = 1.0,
                 max_iter: int = 1000):
        """
        Initialize the logistic regression credit model.

        Args:
            model_name: Name of the model
            model_version: Version of the model
            random_state: Random state for reproducibility
            regularization: Type of regularization ('l1', 'l2', 'elasticnet')
            C: Regularization strength
            max_iter: Maximum number of iterations
        """
        super().__init__(model_name, model_version, random_state)
        self.regularization = regularization
        self.C = C
        self.max_iter = max_iter

    def _create_model(self) -> LogisticRegression:
        """
        Create the logistic regression model.

        Returns:
            Initialized LogisticRegression instance
        """
        return LogisticRegression(
            penalty=self.regularization,
            C=self.C,
            max_iter=self.max_iter,
            random_state=self.random_state,
            solver='liblinear' if self.regularization == 'l1' else 'lbfgs'
        )

    def _get_model_params(self) -> Dict[str, Any]:
        """
        Get model-specific parameters.

        Returns:
            Dictionary of model parameters
        """
        return {
            'regularization': self.regularization,
            'C': self.C,
            'max_iter': self.max_iter,
            'solver': self.model.solver if self.model else None
        }

    def get_coefficients(self) -> Optional[pd.DataFrame]:
        """
        Get model coefficients with feature names.

        Returns:
            DataFrame with feature coefficients
        """
        if not self.is_trained or self.preprocessor is None:
            return None

        feature_names = self.preprocessor.feature_names
        if hasattr(self.preprocessor, 'risk_feature_names'):
            feature_names.extend(self.preprocessor.risk_feature_names)

        coefficients = pd.DataFrame({
            'feature': feature_names,
            'coefficient': self.model.coef_[0],
            'abs_coefficient': np.abs(self.model.coef_[0])
        }).sort_values('abs_coefficient', ascending=False)

        return coefficients

    def interpret_prediction(self, X: pd.DataFrame, top_features: int = 10) -> Dict[str, Any]:
        """
        Provide interpretation for predictions.

        Args:
            X: Input features for interpretation
            top_features: Number of top features to include in interpretation

        Returns:
            Dictionary with prediction interpretation
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before interpretation")

        # Get prediction and probability
        prediction = self.predict(X)[0]
        probability = self.predict_proba(X)[0, 1]

        # Get coefficients
        coefficients = self.get_coefficients()
        if coefficients is None:
            return {'error': 'Cannot interpret without trained model'}

        # Process input through preprocessor
        X_processed = self.preprocessor.transform(X)

        # Calculate feature contributions
        feature_contributions = []
        for idx, (feature, coef) in enumerate(zip(coefficients['feature'][:top_features], 
                                                 coefficients['coefficient'][:top_features])):
            if feature in X_processed.columns:
                value = X_processed[feature].iloc[0]
                contribution = coef * value
                feature_contributions.append({
                    'feature': feature,
                    'value': value,
                    'coefficient': coef,
                    'contribution': contribution
                })

        return {
            'prediction': int(prediction),
            'probability': float(probability),
            'risk_level': 'High' if probability > 0.7 else 'Medium' if probability > 0.3 else 'Low',
            'top_contributing_features': feature_contributions,
            'intercept': float(self.model.intercept_[0])
        }

class RandomForestCreditModel(BaseModel):
    """
    Random Forest model specialized for credit risk assessment.

    Provides robust ensemble predictions with feature importance rankings.
    """

    def __init__(self, 
                 model_name: str = "Random Forest Credit Model",
                 model_version: str = "1.0",
                 random_state: int = 42,
                 n_estimators: int = 100,
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 max_features: str = 'sqrt'):
        """
        Initialize the random forest credit model.

        Args:
            model_name: Name of the model
            model_version: Version of the model
            random_state: Random state for reproducibility
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees
            min_samples_split: Minimum samples required to split a node
            min_samples_leaf: Minimum samples required at a leaf node
            max_features: Number of features to consider for best split
        """
        super().__init__(model_name, model_version, random_state)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features

    def _create_model(self) -> RandomForestClassifier:
        """
        Create the random forest model.

        Returns:
            Initialized RandomForestClassifier instance
        """
        return RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=-1
        )

    def _get_model_params(self) -> Dict[str, Any]:
        """
        Get model-specific parameters.

        Returns:
            Dictionary of model parameters
        """
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'max_features': self.max_features
        }

    def get_tree_depths(self) -> List[int]:
        """
        Get the depth of each tree in the forest.

        Returns:
            List of tree depths
        """
        if not self.is_trained:
            return []

        return [tree.tree_.max_depth for tree in self.model.estimators_]

    def get_oob_score(self) -> Optional[float]:
        """
        Get out-of-bag score if available.

        Returns:
            OOB score or None
        """
        if not self.is_trained:
            return None

        # Retrain with OOB scoring enabled
        temp_model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            oob_score=True,
            n_jobs=-1
        )

        # This would require retraining, so return None for now
        return None

class GradientBoostingCreditModel(BaseModel):
    """
    Gradient Boosting model specialized for credit risk assessment.

    Provides sequential ensemble learning with strong predictive performance.
    """

    def __init__(self, 
                 model_name: str = "Gradient Boosting Credit Model",
                 model_version: str = "1.0",
                 random_state: int = 42,
                 n_estimators: int = 100,
                 learning_rate: float = 0.1,
                 max_depth: int = 3,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 subsample: float = 1.0):
        """
        Initialize the gradient boosting credit model.

        Args:
            model_name: Name of the model
            model_version: Version of the model
            random_state: Random state for reproducibility
            n_estimators: Number of boosting stages
            learning_rate: Learning rate shrinks contribution of each tree
            max_depth: Maximum depth of individual trees
            min_samples_split: Minimum samples required to split a node
            min_samples_leaf: Minimum samples required at a leaf node
            subsample: Fraction of samples used for fitting individual trees
        """
        super().__init__(model_name, model_version, random_state)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample

    def _create_model(self) -> GradientBoostingClassifier:
        """
        Create the gradient boosting model.

        Returns:
            Initialized GradientBoostingClassifier instance
        """
        return GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            subsample=self.subsample,
            random_state=self.random_state
        )

    def _get_model_params(self) -> Dict[str, Any]:
        """
        Get model-specific parameters.

        Returns:
            Dictionary of model parameters
        """
        return {
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'subsample': self.subsample
        }

    def get_staged_predictions(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get predictions at each stage of boosting.

        Args:
            X: Input features

        Returns:
            Array of staged predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting staged predictions")

        # Preprocess data
        X_processed = self.preprocessor.transform(X) if self.preprocessor else X

        # Get staged predictions
        staged_preds = list(self.model.staged_predict_proba(X_processed))
        return np.array([pred[:, 1] for pred in staged_preds])  # Return probabilities for positive class

    def plot_deviance(self) -> Optional[Dict[str, List[float]]]:
        """
        Get training and validation deviance for plotting.

        Returns:
            Dictionary with training deviance scores
        """
        if not self.is_trained:
            return None

        return {
            'train_deviance': self.model.train_score_.tolist(),
            'n_estimators': list(range(1, len(self.model.train_score_) + 1))
        }


class SVMCreditModel(BaseModel):
    """
    Support Vector Machine model specialized for credit risk assessment.

    Provides non-linear classification with kernel methods for complex patterns.
    """

    def __init__(self, 
                 model_name: str = "SVM Credit Model",
                 model_version: str = "1.0",
                 random_state: int = 42,
                 kernel: str = 'rbf',
                 C: float = 1.0,
                 gamma: str = 'scale',
                 probability: bool = True):
        """
        Initialize the SVM credit model.

        Args:
            model_name: Name of the model
            model_version: Version of the model
            random_state: Random state for reproducibility
            kernel: Kernel type ('linear', 'poly', 'rbf', 'sigmoid')
            C: Regularization parameter
            gamma: Kernel coefficient
            probability: Whether to enable probability estimates
        """
        super().__init__(model_name, model_version, random_state)
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.probability = probability

    def _create_model(self) -> SVC:
        """
        Create the SVM model.

        Returns:
            Initialized SVC instance
        """
        return SVC(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma,
            probability=self.probability,
            random_state=self.random_state
        )

    def _get_model_params(self) -> Dict[str, Any]:
        """
        Get model-specific parameters.

        Returns:
            Dictionary of model parameters
        """
        return {
            'kernel': self.kernel,
            'C': self.C,
            'gamma': self.gamma,
            'probability': self.probability
        }

    def get_support_vectors_info(self) -> Dict[str, Any]:
        """
        Get information about support vectors.

        Returns:
            Dictionary with support vector information
        """
        if not self.is_trained:
            return {}

        return {
            'n_support_vectors': self.model.n_support_.tolist(),
            'support_vector_indices': self.model.support_.tolist(),
            'total_support_vectors': len(self.model.support_)
        }

class NeuralNetworkCreditModel(BaseModel):
    """
    Multi-layer Perceptron model specialized for credit risk assessment.

    Provides deep learning capabilities for complex non-linear patterns.
    """

    def __init__(self, 
                 model_name: str = "Neural Network Credit Model",
                 model_version: str = "1.0",
                 random_state: int = 42,
                 hidden_layer_sizes: Tuple[int, ...] = (100, 50),
                 activation: str = 'relu',
                 solver: str = 'adam',
                 alpha: float = 0.0001,
                 learning_rate: str = 'constant',
                 max_iter: int = 500):
        """
        Initialize the neural network credit model.

        Args:
            model_name: Name of the model
            model_version: Version of the model
            random_state: Random state for reproducibility
            hidden_layer_sizes: Sizes of hidden layers
            activation: Activation function ('identity', 'logistic', 'tanh', 'relu')
            solver: Solver for weight optimization ('lbfgs', 'sgd', 'adam')
            alpha: L2 penalty parameter
            learning_rate: Learning rate schedule ('constant', 'invscaling', 'adaptive')
            max_iter: Maximum number of iterations
        """
        super().__init__(model_name, model_version, random_state)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.max_iter = max_iter

    def _create_model(self) -> MLPClassifier:
        """
        Create the neural network model.

        Returns:
            Initialized MLPClassifier instance
        """
        return MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            solver=self.solver,
            alpha=self.alpha,
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            random_state=self.random_state,
            early_stopping=True,
            validation_fraction=0.1
        )

    def _get_model_params(self) -> Dict[str, Any]:
        """
        Get model-specific parameters.

        Returns:
            Dictionary of model parameters
        """
        return {
            'hidden_layer_sizes': self.hidden_layer_sizes,
            'activation': self.activation,
            'solver': self.solver,
            'alpha': self.alpha,
            'learning_rate': self.learning_rate,
            'max_iter': self.max_iter
        }

    def get_training_info(self) -> Dict[str, Any]:
        """
        Get training information and convergence details.

        Returns:
            Dictionary with training information
        """
        if not self.is_trained:
            return {}

        info = {
            'n_iterations': self.model.n_iter_,
            'n_layers': self.model.n_layers_,
            'n_outputs': self.model.n_outputs_,
            'converged': self.model.n_iter_ < self.max_iter
        }

        if hasattr(self.model, 'loss_curve_'):
            info['loss_curve'] = self.model.loss_curve_.tolist()

        if hasattr(self.model, 'validation_scores_'):
            info['validation_scores'] = self.model.validation_scores_.tolist()

        return info

class EnsembleCreditModel(BaseModel):
    """
    Ensemble model that combines multiple credit risk models.

    Uses voting or stacking to combine predictions from multiple base models.
    """

    def __init__(self, 
                 model_name: str = "Ensemble Credit Model",
                 model_version: str = "1.0",
                 random_state: int = 42,
                 ensemble_method: str = 'voting',
                 voting_type: str = 'soft'):
        """
        Initialize the ensemble credit model.

        Args:
            model_name: Name of the model
            model_version: Version of the model
            random_state: Random state for reproducibility
            ensemble_method: Method for combining models ('voting', 'stacking')
            voting_type: Type of voting ('hard', 'soft')
        """
        super().__init__(model_name, model_version, random_state)
        self.ensemble_method = ensemble_method
        self.voting_type = voting_type
        self.base_models = {}
        self.model_weights = {}

    def add_base_model(self, model_name: str, model: BaseModel, weight: float = 1.0) -> None:
        """
        Add a base model to the ensemble.

        Args:
            model_name: Name identifier for the model
            model: Trained base model
            weight: Weight for the model in ensemble
        """
        if not model.is_trained:
            raise ValueError(f"Base model {model_name} must be trained before adding to ensemble")

        self.base_models[model_name] = model
        self.model_weights[model_name] = weight

    def _create_model(self) -> Any:
        """
        Create the ensemble model.

        Returns:
            Ensemble model (this is handled differently)
        """
        # Ensemble model doesn't use sklearn's ensemble classes directly
        # Instead, it manages multiple BaseModel instances
        return None

    def _get_model_params(self) -> Dict[str, Any]:
        """
        Get model-specific parameters.

        Returns:
            Dictionary of model parameters
        """
        return {
            'ensemble_method': self.ensemble_method,
            'voting_type': self.voting_type,
            'base_models': list(self.base_models.keys()),
            'model_weights': self.model_weights
        }

    def fit(self, 
            X: pd.DataFrame, 
            y: pd.Series,
            validation_split: float = 0.2,
            preprocess_data: bool = True) -> 'EnsembleCreditModel':
        """
        Train the ensemble model.

        Args:
            X: Input features
            y: Target variable
            validation_split: Proportion of data for validation
            preprocess_data: Whether to preprocess the data

        Returns:
            self: Trained ensemble model
        """
        if not self.base_models:
            raise ValueError("No base models added to ensemble")

        logger.info(f"Training ensemble with {len(self.base_models)} base models...")

        # Set training status
        self.is_trained = True
        self.training_timestamp = datetime.now()

        # Calculate ensemble metrics using base model predictions
        self._calculate_ensemble_metrics(X, y, validation_split)

        logger.info("Ensemble training completed")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make ensemble predictions.

        Args:
            X: Input features

        Returns:
            Ensemble predictions
        """
        if not self.is_trained or not self.base_models:
            raise ValueError("Ensemble must be trained with base models before prediction")

        predictions = []
        weights = []

        for model_name, model in self.base_models.items():
            pred = model.predict(X)
            predictions.append(pred)
            weights.append(self.model_weights[model_name])

        predictions = np.array(predictions)
        weights = np.array(weights)

        if self.voting_type == 'hard':
            # Weighted majority voting
            weighted_votes = np.average(predictions, axis=0, weights=weights)
            return (weighted_votes > 0.5).astype(int)
        else:
            # This shouldn't be reached for hard predictions, but included for completeness
            return np.round(np.average(predictions, axis=0, weights=weights)).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities using ensemble.

        Args:
            X: Input features

        Returns:
            Class probabilities
        """
        if not self.is_trained or not self.base_models:
            raise ValueError("Ensemble must be trained with base models before prediction")

        probabilities = []
        weights = []

        for model_name, model in self.base_models.items():
            if hasattr(model.model, 'predict_proba'):
                proba = model.predict_proba(X)
                probabilities.append(proba)
                weights.append(self.model_weights[model_name])

        if not probabilities:
            raise ValueError("No base models support probability prediction")

        probabilities = np.array(probabilities)
        weights = np.array(weights)

        # Weighted average of probabilities
        ensemble_proba = np.average(probabilities, axis=0, weights=weights)
        return ensemble_proba

    def _calculate_ensemble_metrics(self, X: pd.DataFrame, y: pd.Series, validation_split: float) -> None:
        """Calculate metrics for the ensemble model."""
        from sklearn.model_selection import train_test_split

        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=self.random_state, stratify=y
        )

        # Get ensemble predictions on validation set
        y_pred = self.predict(X_val)
        y_pred_proba = self.predict_proba(X_val)[:, 1] if len(self.base_models) > 0 else None

        # Calculate metrics
        metrics_calc = ModelMetrics()
        self.training_metrics = metrics_calc.calculate_credit_risk_metrics(
            y_val.values, y_pred, y_pred_proba
        )

    def get_base_model_contributions(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Get individual base model predictions for analysis.

        Args:
            X: Input features

        Returns:
            Dictionary with predictions from each base model
        """
        contributions = {}

        for model_name, model in self.base_models.items():
            pred = model.predict(X)
            proba = model.predict_proba(X)[:, 1] if hasattr(model.model, 'predict_proba') else None

            contributions[model_name] = {
                'predictions': pred,
                'probabilities': proba,
                'weight': self.model_weights[model_name]
            }

        return contributions

class AutoMLCreditModel(BaseModel):
    """
    Automated Machine Learning model for credit risk assessment.

    Automatically selects and tunes the best model from multiple algorithms.
    """

    def __init__(self, 
                 model_name: str = "AutoML Credit Model",
                 model_version: str = "1.0",
                 random_state: int = 42,
                 search_method: str = 'grid',
                 cv_folds: int = 5,
                 scoring: str = 'f1_weighted'):
        """
        Initialize the AutoML credit model.

        Args:
            model_name: Name of the model
            model_version: Version of the model
            random_state: Random state for reproducibility
            search_method: Search method ('grid', 'random')
            cv_folds: Number of cross-validation folds
            scoring: Scoring metric for model selection
        """
        super().__init__(model_name, model_version, random_state)
        self.search_method = search_method
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.best_model_name = None
        self.search_results = {}

    def _create_model(self) -> Any:
        """
        Create the AutoML model (will be determined during training).

        Returns:
            Best model found during search
        """
        # Define candidate models and their parameter grids
        models_and_params = {
            'logistic_regression': {
                'model': LogisticRegression(random_state=self.random_state),
                'params': {
                    'C': [0.1, 1.0, 10.0],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear']
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(random_state=self.random_state),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=self.random_state),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }
            },
            'svm': {
                'model': SVC(random_state=self.random_state, probability=True),
                'params': {
                    'C': [0.1, 1.0, 10.0],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto']
                }
            }
        }

        return models_and_params

    def _get_model_params(self) -> Dict[str, Any]:
        """
        Get model-specific parameters.

        Returns:
            Dictionary of model parameters
        """
        return {
            'search_method': self.search_method,
            'cv_folds': self.cv_folds,
            'scoring': self.scoring,
            'best_model_name': self.best_model_name,
            'search_results': self.search_results
        }

    def fit(self, 
            X: pd.DataFrame, 
            y: pd.Series,
            validation_split: float = 0.2,
            preprocess_data: bool = True) -> 'AutoMLCreditModel':
        """
        Train the AutoML model by searching for the best algorithm and parameters.

        Args:
            X: Input features
            y: Target variable
            validation_split: Proportion of data for validation
            preprocess_data: Whether to preprocess the data

        Returns:
            self: Trained AutoML model
        """
        logger.info("Starting AutoML model search...")

        # Preprocess data if required
        if preprocess_data:
            self.preprocessor = CreditRiskPreprocessor()
            self.preprocessor.fit(X, y)
            X_processed = self.preprocessor.transform(X)
        else:
            X_processed = X.copy()

        # Get candidate models
        models_and_params = self._create_model()

        best_score = -np.inf
        best_model = None
        best_params = None

        # Search through each model type
        for model_name, model_config in models_and_params.items():
            logger.info(f"Searching {model_name}...")

            # Perform hyperparameter search
            if self.search_method == 'grid':
                search = GridSearchCV(
                    model_config['model'],
                    model_config['params'],
                    cv=self.cv_folds,
                    scoring=self.scoring,
                    n_jobs=-1
                )
            else:  # random search
                search = RandomizedSearchCV(
                    model_config['model'],
                    model_config['params'],
                    cv=self.cv_folds,
                    scoring=self.scoring,
                    n_jobs=-1,
                    n_iter=20,
                    random_state=self.random_state
                )

            # Fit the search
            search.fit(X_processed, y)

            # Store results
            self.search_results[model_name] = {
                'best_score': search.best_score_,
                'best_params': search.best_params_,
                'cv_results': search.cv_results_
            }

            # Check if this is the best model so far
            if search.best_score_ > best_score:
                best_score = search.best_score_
                best_model = search.best_estimator_
                best_params = search.best_params_
                self.best_model_name = model_name

        # Set the best model as the final model
        self.model = best_model
        self.is_trained = True
        self.training_timestamp = datetime.now()

        # Calculate final metrics
        self.training_metrics = self.validator.evaluate_model(
            self.model, X_processed, y, test_size=validation_split
        )

        # Calculate feature importance
        self._calculate_feature_importance(X_processed)

        logger.info(f"AutoML completed. Best model: {self.best_model_name} (Score: {best_score:.4f})")

        return self

    def get_search_summary(self) -> pd.DataFrame:
        """
        Get a summary of the model search results.

        Returns:
            DataFrame with search results summary
        """
        if not self.search_results:
            return pd.DataFrame()

        summary_data = []
        for model_name, results in self.search_results.items():
            summary_data.append({
                'Model': model_name,
                'Best_Score': results['best_score'],
                'Best_Params': str(results['best_params']),
                'Is_Best': model_name == self.best_model_name
            })

        return pd.DataFrame(summary_data).sort_values('Best_Score', ascending=False)


# Export all credit model classes
__all__ = [
    'CreditRiskPreprocessor',
    'LogisticRegressionCreditModel',
    'RandomForestCreditModel', 
    'GradientBoostingCreditModel',
    'SVMCreditModel',
    'NeuralNetworkCreditModel',
    'EnsembleCreditModel',
    'AutoMLCreditModel'
]
