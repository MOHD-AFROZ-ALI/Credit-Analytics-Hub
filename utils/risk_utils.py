"""
Risk Assessment and Scoring Utilities Module for Credit Analytics Hub

This module provides comprehensive risk assessment utilities including
credit scoring, risk categorization, portfolio analysis, and regulatory
compliance calculations for credit risk management.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import logging
from datetime import datetime, timedelta
import warnings
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path

# Statistical and ML imports
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """Enumeration for risk levels."""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

class RiskCategory(Enum):
    """Enumeration for risk categories."""
    CREDIT_RISK = "credit_risk"
    OPERATIONAL_RISK = "operational_risk"
    MARKET_RISK = "market_risk"
    LIQUIDITY_RISK = "liquidity_risk"
    CONCENTRATION_RISK = "concentration_risk"

@dataclass
class RiskScore:
    """Data class for storing risk score information."""
    score: float
    level: RiskLevel
    category: RiskCategory
    confidence: float
    factors: Dict[str, float]
    timestamp: datetime
    model_version: str = "1.0"

@dataclass
class PortfolioRiskMetrics:
    """Data class for portfolio risk metrics."""
    total_exposure: float
    expected_loss: float
    unexpected_loss: float
    value_at_risk: float
    expected_shortfall: float
    concentration_ratio: float
    diversification_ratio: float
    risk_adjusted_return: float
    timestamp: datetime

class CreditScorer:
    """
    Comprehensive credit scoring utility.
    
    Provides multiple credit scoring methodologies including traditional
    scorecards, machine learning-based scores, and regulatory compliant scoring.
    """
    
    def __init__(self, 
                 scoring_method: str = 'traditional',
                 score_range: Tuple[int, int] = (300, 850),
                 default_threshold: float = 0.5):
        """
        Initialize the credit scorer.
        
        Args:
            scoring_method: Scoring methodology ('traditional', 'ml', 'hybrid')
            score_range: Range for credit scores (min, max)
            default_threshold: Threshold for default classification
        """
        self.scoring_method = scoring_method
        self.score_range = score_range
        self.default_threshold = default_threshold
        self.scorecard_weights = {}
        self.feature_importance = {}
        self.calibration_params = {}
        
        # Initialize default scorecard weights
        self._initialize_default_scorecard()
    
    def _initialize_default_scorecard(self):
        """Initialize default scorecard weights."""
        self.scorecard_weights = {
            'payment_history': 0.35,
            'credit_utilization': 0.30,
            'length_of_credit_history': 0.15,
            'credit_mix': 0.10,
            'new_credit': 0.10
        }
        
        # Feature mappings for traditional scoring
        self.feature_mappings = {
            'payment_history': ['delinquencies_2yrs', 'payment_history_score', 'late_payments'],
            'credit_utilization': ['credit_utilization', 'revolving_util', 'utilization_rate'],
            'length_of_credit_history': ['account_age_months', 'credit_history_length', 'oldest_account_age'],
            'credit_mix': ['total_accounts', 'account_types', 'credit_mix_score'],
            'new_credit': ['inquiries_6mths', 'new_accounts', 'recent_credit_activity']
        }
    
    def calculate_credit_score(self, 
                             customer_data: Union[pd.DataFrame, Dict[str, Any]],
                             method: Optional[str] = None) -> Union[RiskScore, List[RiskScore]]:
        """
        Calculate credit score for customer(s).
        
        Args:
            customer_data: Customer data (DataFrame for multiple, Dict for single)
            method: Scoring method override
            
        Returns:
            RiskScore object(s)
        """
        method = method or self.scoring_method
        
        if isinstance(customer_data, dict):
            return self._calculate_single_score(customer_data, method)
        elif isinstance(customer_data, pd.DataFrame):
            return [self._calculate_single_score(row.to_dict(), method) 
                   for _, row in customer_data.iterrows()]
        else:
            raise ValueError("customer_data must be DataFrame or dictionary")
    
    def _calculate_single_score(self, customer_data: Dict[str, Any], method: str) -> RiskScore:
        """Calculate credit score for a single customer."""
        if method == 'traditional':
            return self._traditional_scorecard(customer_data)
        elif method == 'ml':
            return self._ml_based_score(customer_data)
        elif method == 'hybrid':
            return self._hybrid_score(customer_data)
        else:
            raise ValueError(f"Unknown scoring method: {method}")
    
    def _traditional_scorecard(self, customer_data: Dict[str, Any]) -> RiskScore:
        """Calculate traditional scorecard-based credit score."""
        total_score = 0
        factor_scores = {}
        confidence = 1.0
        
        for category, weight in self.scorecard_weights.items():
            category_score = self._calculate_category_score(customer_data, category)
            factor_scores[category] = category_score
            total_score += category_score * weight
        
        # Normalize to score range
        normalized_score = self._normalize_score(total_score)
        risk_level = self._determine_risk_level(normalized_score)
        
        return RiskScore(
            score=normalized_score,
            level=risk_level,
            category=RiskCategory.CREDIT_RISK,
            confidence=confidence,
            factors=factor_scores,
            timestamp=datetime.now()
        )
    
    def _calculate_category_score(self, customer_data: Dict[str, Any], category: str) -> float:
        """Calculate score for a specific category."""
        relevant_features = self.feature_mappings.get(category, [])
        scores = []
        
        for feature in relevant_features:
            if feature in customer_data:
                value = customer_data[feature]
                if pd.notna(value):
                    # Normalize feature value to 0-1 scale
                    normalized_value = self._normalize_feature_value(feature, value)
                    scores.append(normalized_value)
        
        # Return average score for category
        return np.mean(scores) if scores else 0.5  # Default to neutral score
    
    def _normalize_feature_value(self, feature: str, value: float) -> float:
        """Normalize feature value to 0-1 scale."""
        # Define feature ranges and directions (higher is better/worse)
        feature_configs = {
            'delinquencies_2yrs': {'range': (0, 10), 'direction': 'lower_better'},
            'payment_history_score': {'range': (300, 850), 'direction': 'higher_better'},
            'credit_utilization': {'range': (0, 1), 'direction': 'lower_better'},
            'revolving_util': {'range': (0, 1), 'direction': 'lower_better'},
            'account_age_months': {'range': (0, 480), 'direction': 'higher_better'},
            'total_accounts': {'range': (1, 50), 'direction': 'moderate_better'},
            'inquiries_6mths': {'range': (0, 20), 'direction': 'lower_better'},
            'late_payments': {'range': (0, 10), 'direction': 'lower_better'}
        }
        
        config = feature_configs.get(feature, {'range': (0, 1), 'direction': 'higher_better'})
        min_val, max_val = config['range']
        direction = config['direction']
        
        # Clip value to range
        clipped_value = np.clip(value, min_val, max_val)
        
        # Normalize to 0-1
        if max_val == min_val:
            normalized = 0.5
        else:
            normalized = (clipped_value - min_val) / (max_val - min_val)
        
        # Adjust based on direction
        if direction == 'lower_better':
            normalized = 1 - normalized
        elif direction == 'moderate_better':
            # Optimal range is in the middle
            normalized = 1 - abs(normalized - 0.5) * 2
        
        return normalized
    
    def _ml_based_score(self, customer_data: Dict[str, Any]) -> RiskScore:
        """Calculate ML-based credit score (placeholder for actual ML model)."""
        # This would typically use a trained ML model
        # For now, we'll simulate with a weighted combination
        
        features = ['annual_income', 'loan_amount', 'credit_score', 'employment_length',
                   'debt_to_income_ratio', 'credit_utilization', 'age']
        
        feature_values = []
        factor_scores = {}
        
        for feature in features:
            if feature in customer_data:
                value = customer_data[feature]
                if pd.notna(value):
                    normalized_value = self._normalize_feature_value(feature, value)
                    feature_values.append(normalized_value)
                    factor_scores[feature] = normalized_value
                else:
                    feature_values.append(0.5)  # Default neutral value
                    factor_scores[feature] = 0.5
            else:
                feature_values.append(0.5)
                factor_scores[feature] = 0.5
        
        # Simulate ML prediction (would be replaced with actual model.predict())
        ml_score = np.mean(feature_values)
        confidence = 0.85  # Simulated confidence
        
        normalized_score = self._normalize_score(ml_score)
        risk_level = self._determine_risk_level(normalized_score)
        
        return RiskScore(
            score=normalized_score,
            level=risk_level,
            category=RiskCategory.CREDIT_RISK,
            confidence=confidence,
            factors=factor_scores,
            timestamp=datetime.now()
        )
    
    def _hybrid_score(self, customer_data: Dict[str, Any]) -> RiskScore:
        """Calculate hybrid score combining traditional and ML approaches."""
        traditional_score = self._traditional_scorecard(customer_data)
        ml_score = self._ml_based_score(customer_data)
        
        # Weighted combination (70% traditional, 30% ML)
        hybrid_score_value = 0.7 * traditional_score.score + 0.3 * ml_score.score
        hybrid_confidence = (traditional_score.confidence + ml_score.confidence) / 2
        
        # Combine factor scores
        combined_factors = {}
        all_factors = set(traditional_score.factors.keys()) | set(ml_score.factors.keys())
        
        for factor in all_factors:
            trad_score = traditional_score.factors.get(factor, 0)
            ml_score_val = ml_score.factors.get(factor, 0)
            combined_factors[factor] = 0.7 * trad_score + 0.3 * ml_score_val
        
        risk_level = self._determine_risk_level(hybrid_score_value)
        
        return RiskScore(
            score=hybrid_score_value,
            level=risk_level,
            category=RiskCategory.CREDIT_RISK,
            confidence=hybrid_confidence,
            factors=combined_factors,
            timestamp=datetime.now()
        )
    
    def _normalize_score(self, raw_score: float) -> float:
        """Normalize raw score to credit score range."""
        min_score, max_score = self.score_range
        # Assume raw_score is in 0-1 range
        normalized = min_score + (max_score - min_score) * raw_score
        return round(normalized)
    
    def _determine_risk_level(self, score: float) -> RiskLevel:
        """Determine risk level based on credit score."""
        min_score, max_score = self.score_range
        score_range = max_score - min_score
        
        if score >= min_score + 0.8 * score_range:
            return RiskLevel.VERY_LOW
        elif score >= min_score + 0.6 * score_range:
            return RiskLevel.LOW
        elif score >= min_score + 0.4 * score_range:
            return RiskLevel.MEDIUM
        elif score >= min_score + 0.2 * score_range:
            return RiskLevel.HIGH
        else:
            return RiskLevel.VERY_HIGH
    
    def calibrate_scorecard(self, 
                           training_data: pd.DataFrame,
                           target_column: str,
                           validation_data: Optional[pd.DataFrame] = None):
        """
        Calibrate scorecard weights using historical data.
        
        Args:
            training_data: Historical data with outcomes
            target_column: Name of target variable (default indicator)
            validation_data: Optional validation dataset
        """
        logger.info("Calibrating scorecard weights...")
        
        # Calculate feature importance using correlation with target
        numeric_features = training_data.select_dtypes(include=[np.number]).columns
        feature_correlations = {}
        
        for feature in numeric_features:
            if feature != target_column:
                correlation = abs(training_data[feature].corr(training_data[target_column]))
                if not pd.isna(correlation):
                    feature_correlations[feature] = correlation
        
        # Update feature importance
        self.feature_importance = feature_correlations
        
        # Recalibrate category weights based on predictive power
        category_importance = {}
        for category, features in self.feature_mappings.items():
            category_score = 0
            feature_count = 0
            
            for feature in features:
                if feature in feature_correlations:
                    category_score += feature_correlations[feature]
                    feature_count += 1
            
            if feature_count > 0:
                category_importance[category] = category_score / feature_count
        
        # Normalize weights to sum to 1
        total_importance = sum(category_importance.values())
        if total_importance > 0:
            for category in self.scorecard_weights:
                if category in category_importance:
                    self.scorecard_weights[category] = category_importance[category] / total_importance
        
        logger.info("Scorecard calibration completed")
        logger.info(f"Updated weights: {self.scorecard_weights}")
    
    def save_scorecard(self, filepath: str):
        """Save scorecard configuration."""
        config = {
            'scoring_method': self.scoring_method,
            'score_range': self.score_range,
            'default_threshold': self.default_threshold,
            'scorecard_weights': self.scorecard_weights,
            'feature_mappings': self.feature_mappings,
            'feature_importance': self.feature_importance,
            'calibration_params': self.calibration_params,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Scorecard saved to {filepath}")
    
    def load_scorecard(self, filepath: str):
        """Load scorecard configuration."""
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        self.scoring_method = config.get('scoring_method', self.scoring_method)
        self.score_range = tuple(config.get('score_range', self.score_range))
        self.default_threshold = config.get('default_threshold', self.default_threshold)
        self.scorecard_weights = config.get('scorecard_weights', self.scorecard_weights)
        self.feature_mappings = config.get('feature_mappings', self.feature_mappings)
        self.feature_importance = config.get('feature_importance', self.feature_importance)
        self.calibration_params = config.get('calibration_params', self.calibration_params)
        
        logger.info(f"Scorecard loaded from {filepath}")

class RiskSegmentation:
    """
    Risk-based customer segmentation utility.
    
    Provides clustering and segmentation capabilities for risk management
    and targeted strategies based on risk profiles.
    """
    
    def __init__(self, 
                 n_segments: int = 5,
                 segmentation_method: str = 'kmeans',
                 random_state: int = 42):
        """
        Initialize risk segmentation.
        
        Args:
            n_segments: Number of risk segments
            segmentation_method: Clustering method ('kmeans', 'hierarchical')
            random_state: Random state for reproducibility
        """
        self.n_segments = n_segments
        self.segmentation_method = segmentation_method
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self.segment_profiles = {}
        self.feature_names = []
    
    def fit_segments(self, 
                    data: pd.DataFrame,
                    risk_features: Optional[List[str]] = None) -> 'RiskSegmentation':
        """
        Fit segmentation model to data.
        
        Args:
            data: Customer data
            risk_features: List of features to use for segmentation
            
        Returns:
            Fitted segmentation model
        """
        # Select features for segmentation
        if risk_features is None:
            risk_features = data.select_dtypes(include=[np.number]).columns.tolist()
        
        self.feature_names = risk_features
        X = data[risk_features].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit clustering model
        if self.segmentation_method == 'kmeans':
            self.model = KMeans(n_clusters=self.n_segments, random_state=self.random_state)
            segments = self.model.fit_predict(X_scaled)
        else:
            raise ValueError(f"Unsupported segmentation method: {self.segmentation_method}")
        
        # Calculate segment profiles
        data_with_segments = data.copy()
        data_with_segments['risk_segment'] = segments
        
        self._calculate_segment_profiles(data_with_segments, risk_features)
        
        logger.info(f"Risk segmentation completed with {self.n_segments} segments")
        
        return self
    
    def predict_segments(self, data: pd.DataFrame) -> np.ndarray:
        """
        Predict risk segments for new data.
        
        Args:
            data: New customer data
            
        Returns:
            Array of segment predictions
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        X = data[self.feature_names].copy()
        X = X.fillna(X.median())
        X_scaled = self.scaler.transform(X)
        
        return self.model.predict(X_scaled)
    
    def _calculate_segment_profiles(self, data: pd.DataFrame, features: List[str]):
        """Calculate profiles for each segment."""
        for segment in range(self.n_segments):
            segment_data = data[data['risk_segment'] == segment]
            
            profile = {
                'size': len(segment_data),
                'percentage': len(segment_data) / len(data) * 100,
                'characteristics': {}
            }
            
            # Calculate feature statistics for segment
            for feature in features:
                if feature in segment_data.columns:
                    profile['characteristics'][feature] = {
                        'mean': segment_data[feature].mean(),
                        'median': segment_data[feature].median(),
                        'std': segment_data[feature].std(),
                        'min': segment_data[feature].min(),
                        'max': segment_data[feature].max()
                    }
            
            # Calculate risk level for segment
            if 'credit_score' in segment_data.columns:
                avg_score = segment_data['credit_score'].mean()
                profile['avg_credit_score'] = avg_score
                profile['risk_level'] = self._score_to_risk_level(avg_score)
            
            self.segment_profiles[f'segment_{segment}'] = profile
    
    def _score_to_risk_level(self, score: float) -> str:
        """Convert credit score to risk level."""
        if score >= 750:
            return 'Very Low Risk'
        elif score >= 700:
            return 'Low Risk'
        elif score >= 650:
            return 'Medium Risk'
        elif score >= 600:
            return 'High Risk'
        else:
            return 'Very High Risk'
    
    def get_segment_summary(self) -> pd.DataFrame:
        """Get summary of all segments."""
        summary_data = []
        
        for segment_name, profile in self.segment_profiles.items():
            summary_data.append({
                'Segment': segment_name,
                'Size': profile['size'],
                'Percentage': f"{profile['percentage']:.1f}%",
                'Risk Level': profile.get('risk_level', 'Unknown'),
                'Avg Credit Score': profile.get('avg_credit_score', 'N/A')
            })
        
        return pd.DataFrame(summary_data)

class PortfolioRiskAnalyzer:
    """
    Portfolio-level risk analysis utility.
    
    Provides comprehensive portfolio risk metrics including concentration,
    diversification, VaR, and expected loss calculations.
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize portfolio risk analyzer.
        
        Args:
            confidence_level: Confidence level for VaR calculations
        """
        self.confidence_level = confidence_level
        self.portfolio_data = None
        self.risk_metrics = None
    
    def analyze_portfolio(self, 
                         portfolio_data: pd.DataFrame,
                         exposure_column: str = 'loan_amount',
                         pd_column: str = 'probability_of_default',
                         lgd_column: str = 'loss_given_default',
                         segment_column: Optional[str] = None) -> PortfolioRiskMetrics:
        """
        Perform comprehensive portfolio risk analysis.
        
        Args:
            portfolio_data: Portfolio data with exposures and risk parameters
            exposure_column: Column name for exposure amounts
            pd_column: Column name for probability of default
            lgd_column: Column name for loss given default
            segment_column: Optional column for risk segmentation
            
        Returns:
            Portfolio risk metrics
        """
        self.portfolio_data = portfolio_data.copy()
        
        # Calculate basic portfolio metrics
        total_exposure = portfolio_data[exposure_column].sum()
        
        # Calculate expected loss
        if pd_column in portfolio_data.columns and lgd_column in portfolio_data.columns:
            expected_loss = (portfolio_data[exposure_column] * 
                           portfolio_data[pd_column] * 
                           portfolio_data[lgd_column]).sum()
        else:
            # Use default assumptions if PD/LGD not available
            default_pd = 0.05  # 5% default rate
            default_lgd = 0.45  # 45% loss given default
            expected_loss = total_exposure * default_pd * default_lgd
        
        # Calculate unexpected loss (simplified approach)
        unexpected_loss = self._calculate_unexpected_loss(portfolio_data, exposure_column, pd_column, lgd_column)
        
        # Calculate Value at Risk
        value_at_risk = self._calculate_var(portfolio_data, exposure_column, pd_column, lgd_column)
        
        # Calculate Expected Shortfall (Conditional VaR)
        expected_shortfall = self._calculate_expected_shortfall(portfolio_data, exposure_column, pd_column, lgd_column)
        
        # Calculate concentration metrics
        concentration_ratio = self._calculate_concentration_ratio(portfolio_data, exposure_column, segment_column)
        
        # Calculate diversification ratio
        diversification_ratio = self._calculate_diversification_ratio(portfolio_data, exposure_column, segment_column)
        
        # Calculate risk-adjusted return (simplified)
        risk_adjusted_return = self._calculate_risk_adjusted_return(total_exposure, expected_loss)
        
        self.risk_metrics = PortfolioRiskMetrics(
            total_exposure=total_exposure,
            expected_loss=expected_loss,
            unexpected_loss=unexpected_loss,
            value_at_risk=value_at_risk,
            expected_shortfall=expected_shortfall,
            concentration_ratio=concentration_ratio,
            diversification_ratio=diversification_ratio,
            risk_adjusted_return=risk_adjusted_return,
            timestamp=datetime.now()
        )
        
        return self.risk_metrics
    
    def _calculate_unexpected_loss(self, data: pd.DataFrame, exposure_col: str, pd_col: str, lgd_col: str) -> float:
        """Calculate unexpected loss using portfolio variance approach."""
        if pd_col in data.columns and lgd_col in data.columns:
            # Individual loan variances
            variances = (data[exposure_col] * data[lgd_col])**2 * data[pd_col] * (1 - data[pd_col])
            
            # Assume correlation of 0.2 between loans (simplified)
            correlation = 0.2
            portfolio_variance = variances.sum() + correlation * (variances.sum() - variances.sum())
            
            return np.sqrt(portfolio_variance)
        else:
            # Simplified calculation
            return data[exposure_col].sum() * 0.1  # 10% of total exposure
    
    def _calculate_var(self, data: pd.DataFrame, exposure_col: str, pd_col: str, lgd_col: str) -> float:
        """Calculate Value at Risk using Monte Carlo simulation (simplified)."""
        # Simplified VaR calculation
        expected_loss_rate = 0.05 if pd_col not in data.columns else data[pd_col].mean()
        volatility = 0.15  # Assumed portfolio volatility
        
        # Normal distribution assumption
        z_score = stats.norm.ppf(self.confidence_level)
        var = data[exposure_col].sum() * (expected_loss_rate + z_score * volatility)
        
        return var
    
    def _calculate_expected_shortfall(self, data: pd.DataFrame, exposure_col: str, pd_col: str, lgd_col: str) -> float:
        """Calculate Expected Shortfall (Conditional VaR)."""
        var = self._calculate_var(data, exposure_col, pd_col, lgd_col)
        
        # Simplified ES calculation (typically 1.3x VaR for normal distribution)
        expected_shortfall = var * 1.3
        
        return expected_shortfall
    
    def _calculate_concentration_ratio(self, data: pd.DataFrame, exposure_col: str, segment_col: Optional[str]) -> float:
        """Calculate concentration ratio (Herfindahl-Hirschman Index)."""
        if segment_col and segment_col in data.columns:
            # Concentration by segment
            segment_exposures = data.groupby(segment_col)[exposure_col].sum()
            total_exposure = data[exposure_col].sum()
            
            # Calculate HHI
            market_shares = segment_exposures / total_exposure
            hhi = (market_shares ** 2).sum()
            
            return hhi
        else:
            # Individual loan concentration
            total_exposure = data[exposure_col].sum()
            individual_shares = data[exposure_col] / total_exposure
            hhi = (individual_shares ** 2).sum()
            
            return hhi
    
    def _calculate_diversification_ratio(self, data: pd.DataFrame, exposure_col: str, segment_col: Optional[str]) -> float:
        """Calculate diversification ratio."""
        concentration_ratio = self._calculate_concentration_ratio(data, exposure_col, segment_col)
        
        # Diversification ratio is inverse of concentration
        # Perfect diversification = 1, No diversification = 0
        if segment_col and segment_col in data.columns:
            n_segments = data[segment_col].nunique()
            perfect_diversification = 1 / n_segments
        else:
            n_loans = len(data)
            perfect_diversification = 1 / n_loans
        
        diversification_ratio = perfect_diversification / concentration_ratio
        
        return min(diversification_ratio, 1.0)  # Cap at 1.0
    
    def _calculate_risk_adjusted_return(self, total_exposure: float, expected_loss: float) -> float:
        """Calculate risk-adjusted return (simplified RAROC)."""
        # Assume average interest rate of 8%
        gross_revenue = total_exposure * 0.08
        
        # Risk-adjusted return = (Revenue - Expected Loss) / Economic Capital
        # Simplified: use expected loss as proxy for economic capital
        if expected_loss > 0:
            risk_adjusted_return = (gross_revenue - expected_loss) / expected_loss
        else:
            risk_adjusted_return = 0.0
        
        return risk_adjusted_return
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary."""
        if self.risk_metrics is None:
            raise ValueError("Portfolio analysis must be performed first")
        
        return {
            'Portfolio Size': f"${self.risk_metrics.total_exposure:,.0f}",
            'Expected Loss': f"${self.risk_metrics.expected_loss:,.0f}",
            'Expected Loss Rate': f"{(self.risk_metrics.expected_loss / self.risk_metrics.total_exposure) * 100:.2f}%",
            'Unexpected Loss': f"${self.risk_metrics.unexpected_loss:,.0f}",
            'Value at Risk (95%)': f"${self.risk_metrics.value_at_risk:,.0f}",
            'Expected Shortfall': f"${self.risk_metrics.expected_shortfall:,.0f}",
            'Concentration Ratio': f"{self.risk_metrics.concentration_ratio:.4f}",
            'Diversification Ratio': f"{self.risk_metrics.diversification_ratio:.4f}",
            'Risk-Adjusted Return': f"{self.risk_metrics.risk_adjusted_return:.2f}x",
            'Analysis Date': self.risk_metrics.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        }

# class RegulatoryCompliance:
#     """
#     Regulatory compliance utility for credit risk management.
    
#     Provides calculations and checks for various regulatory requirements
#     including Basel III, CECL, and other credit risk regulations.
#     """
    
#     def __init__(self, regulation_type: str = 'basel_iii'):
#         """
#         Initialize regulatory compliance checker.
        
#         Args:
#             regulation_type: Type of regulation ('basel_iii', 'cecl', etc.).
#             confidence_level: Confidence level for VaR and ES calculations.
#         """
      
