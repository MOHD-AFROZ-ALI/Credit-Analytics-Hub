"""
Metrics Module for Credit Analytics Hub

This module provides comprehensive model evaluation and performance metrics
specifically designed for credit analytics and binary classification tasks.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, roc_curve, auc,
    confusion_matrix, classification_report, log_loss,
    average_precision_score, matthews_corrcoef, cohen_kappa_score,
    balanced_accuracy_score, brier_score_loss
)
from sklearn.calibration import calibration_curve
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CreditMetrics:
    """
    Comprehensive metrics calculator for credit analytics
    """

    def __init__(self, threshold: float = 0.5):
        """
        Initialize CreditMetrics

        Args:
            threshold: Default threshold for binary classification
        """
        self.threshold = threshold
        self.metrics_history = []

    def calculate_basic_metrics(self, 
                               y_true: np.ndarray, 
                               y_pred: np.ndarray,
                               y_prob: Optional[np.ndarray] = None,
                               threshold: Optional[float] = None) -> Dict[str, float]:
        """
        Calculate basic classification metrics

        Args:
            y_true: True binary labels
            y_pred: Predicted binary labels
            y_prob: Predicted probabilities (optional)
            threshold: Classification threshold

        Returns:
            Dict[str, float]: Dictionary of basic metrics
        """
        try:
            threshold = threshold or self.threshold

            # If probabilities provided, convert to predictions
            if y_prob is not None:
                y_pred = (y_prob >= threshold).astype(int)

            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1_score': f1_score(y_true, y_pred, zero_division=0),
                'specificity': self._calculate_specificity(y_true, y_pred),
                'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
                'matthews_corrcoef': matthews_corrcoef(y_true, y_pred),
                'cohen_kappa': cohen_kappa_score(y_true, y_pred)
            }

            # Add probability-based metrics if available
            if y_prob is not None:
                metrics.update({
                    'roc_auc': roc_auc_score(y_true, y_prob),
                    'pr_auc': average_precision_score(y_true, y_prob),
                    'log_loss': log_loss(y_true, y_prob),
                    'brier_score': brier_score_loss(y_true, y_prob)
                })

            logger.info(f"Calculated {len(metrics)} basic metrics")
            return metrics

        except Exception as e:
            logger.error(f"Error calculating basic metrics: {str(e)}")
            raise

    def calculate_credit_specific_metrics(self, 
                                        y_true: np.ndarray, 
                                        y_pred: np.ndarray,
                                        y_prob: Optional[np.ndarray] = None,
                                        loan_amounts: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate credit-specific metrics

        Args:
            y_true: True binary labels (1 = default, 0 = no default)
            y_pred: Predicted binary labels
            y_prob: Predicted probabilities
            loan_amounts: Loan amounts for financial impact calculation

        Returns:
            Dict[str, float]: Dictionary of credit-specific metrics
        """
        try:
            # Get confusion matrix components
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

            metrics = {
                # Credit-specific interpretations
                'default_detection_rate': tp / (tp + fn) if (tp + fn) > 0 else 0,  # Recall for defaults
                'false_default_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,  # FPR
                'precision_defaults': tp / (tp + fp) if (tp + fp) > 0 else 0,  # Precision for defaults
                'non_default_accuracy': tn / (tn + fp) if (tn + fp) > 0 else 0,  # Specificity

                # Business metrics
                'approval_rate': (tn + fn) / (tn + fp + fn + tp),  # Rate of approved loans
                'default_rate_approved': fn / (tn + fn) if (tn + fn) > 0 else 0,  # Defaults among approved
                'rejection_rate': (tp + fp) / (tn + fp + fn + tp),  # Rate of rejected loans

                # Risk metrics
                'type_i_error': fp / (fp + tn) if (fp + tn) > 0 else 0,  # False positive rate
                'type_ii_error': fn / (fn + tp) if (fn + tp) > 0 else 0,  # False negative rate
            }

            # Financial impact metrics (if loan amounts provided)
            if loan_amounts is not None:
                metrics.update(self._calculate_financial_metrics(
                    y_true, y_pred, loan_amounts, tn, fp, fn, tp
                ))

            # Probability-based credit metrics
            if y_prob is not None:
                metrics.update(self._calculate_probability_credit_metrics(y_true, y_prob))

            logger.info(f"Calculated {len(metrics)} credit-specific metrics")
            return metrics

        except Exception as e:
            logger.error(f"Error calculating credit-specific metrics: {str(e)}")
            raise

    def calculate_threshold_metrics(self, 
                                  y_true: np.ndarray, 
                                  y_prob: np.ndarray,
                                  thresholds: Optional[List[float]] = None) -> pd.DataFrame:
        """
        Calculate metrics across different thresholds

        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            thresholds: List of thresholds to evaluate

        Returns:
            pd.DataFrame: Metrics for each threshold
        """
        try:
            if thresholds is None:
                thresholds = np.arange(0.1, 1.0, 0.1)

            results = []

            for threshold in thresholds:
                y_pred = (y_prob >= threshold).astype(int)

                # Calculate metrics for this threshold
                basic_metrics = self.calculate_basic_metrics(y_true, y_pred, y_prob, threshold)
                credit_metrics = self.calculate_credit_specific_metrics(y_true, y_pred, y_prob)

                # Combine metrics
                threshold_metrics = {
                    'threshold': threshold,
                    **basic_metrics,
                    **credit_metrics
                }

                results.append(threshold_metrics)

            df_results = pd.DataFrame(results)
            logger.info(f"Calculated metrics for {len(thresholds)} thresholds")
            return df_results

        except Exception as e:
            logger.error(f"Error calculating threshold metrics: {str(e)}")
            raise

    def find_optimal_threshold(self, 
                             y_true: np.ndarray, 
                             y_prob: np.ndarray,
                             metric: str = "f1_score",
                             thresholds: Optional[List[float]] = None) -> Tuple[float, float]:
        """
        Find optimal threshold based on specified metric

        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            metric: Metric to optimize ('f1_score', 'precision', 'recall', etc.)
            thresholds: List of thresholds to evaluate

        Returns:
            Tuple[float, float]: Optimal threshold and corresponding metric value
        """
        try:
            # Calculate metrics for all thresholds
            threshold_df = self.calculate_threshold_metrics(y_true, y_prob, thresholds)

            # Find optimal threshold
            if metric not in threshold_df.columns:
                raise ValueError(f"Metric '{metric}' not found in calculated metrics")

            optimal_idx = threshold_df[metric].idxmax()
            optimal_threshold = threshold_df.loc[optimal_idx, 'threshold']
            optimal_value = threshold_df.loc[optimal_idx, metric]

            logger.info(f"Optimal threshold for {metric}: {optimal_threshold:.3f} (value: {optimal_value:.3f})")
            return optimal_threshold, optimal_value

        except Exception as e:
            logger.error(f"Error finding optimal threshold: {str(e)}")
            raise

    def calculate_roc_metrics(self, 
                            y_true: np.ndarray, 
                            y_prob: np.ndarray) -> Dict[str, Any]:
        """
        Calculate ROC curve metrics and data

        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities

        Returns:
            Dict[str, Any]: ROC metrics and curve data
        """
        try:
            # Calculate ROC curve
            fpr, tpr, thresholds = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)

            # Find optimal threshold using Youden's J statistic
            j_scores = tpr - fpr
            optimal_idx = np.argmax(j_scores)
            optimal_threshold = thresholds[optimal_idx]

            # Calculate additional ROC metrics
            metrics = {
                'roc_auc': roc_auc,
                'optimal_threshold_youden': optimal_threshold,
                'optimal_tpr': tpr[optimal_idx],
                'optimal_fpr': fpr[optimal_idx],
                'youden_j_score': j_scores[optimal_idx],
                'fpr': fpr,
                'tpr': tpr,
                'thresholds': thresholds
            }

            logger.info(f"ROC AUC: {roc_auc:.3f}, Optimal threshold: {optimal_threshold:.3f}")
            return metrics

        except Exception as e:
            logger.error(f"Error calculating ROC metrics: {str(e)}")
            raise

    def calculate_precision_recall_metrics(self, 
                                         y_true: np.ndarray, 
                                         y_prob: np.ndarray) -> Dict[str, Any]:
        """
        Calculate Precision-Recall curve metrics and data

        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities

        Returns:
            Dict[str, Any]: PR metrics and curve data
        """
        try:
            # Calculate PR curve
            precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
            pr_auc = auc(recall, precision)
            avg_precision = average_precision_score(y_true, y_prob)

            # Find optimal threshold using F1 score
            f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1])
            f1_scores = np.nan_to_num(f1_scores)  # Handle division by zero
            optimal_idx = np.argmax(f1_scores)
            optimal_threshold = thresholds[optimal_idx]

            metrics = {
                'pr_auc': pr_auc,
                'average_precision': avg_precision,
                'optimal_threshold_f1': optimal_threshold,
                'optimal_precision': precision[optimal_idx],
                'optimal_recall': recall[optimal_idx],
                'optimal_f1': f1_scores[optimal_idx],
                'precision': precision,
                'recall': recall,
                'thresholds': thresholds
            }

            logger.info(f"PR AUC: {pr_auc:.3f}, Average Precision: {avg_precision:.3f}")
            return metrics

        except Exception as e:
            logger.error(f"Error calculating PR metrics: {str(e)}")
            raise

    def calculate_calibration_metrics(self, 
                                    y_true: np.ndarray, 
                                    y_prob: np.ndarray,
                                    n_bins: int = 10) -> Dict[str, Any]:
        """
        Calculate calibration metrics

        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            n_bins: Number of bins for calibration curve

        Returns:
            Dict[str, Any]: Calibration metrics and data
        """
        try:
            # Calculate calibration curve
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_prob, n_bins=n_bins
            )

            # Calculate calibration metrics
            brier_score = brier_score_loss(y_true, y_prob)

            # Expected Calibration Error (ECE)
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]

            ece = 0
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
                prop_in_bin = in_bin.mean()

                if prop_in_bin > 0:
                    accuracy_in_bin = y_true[in_bin].mean()
                    avg_confidence_in_bin = y_prob[in_bin].mean()
                    ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

            # Maximum Calibration Error (MCE)
            mce = 0
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
                if in_bin.sum() > 0:
                    accuracy_in_bin = y_true[in_bin].mean()
                    avg_confidence_in_bin = y_prob[in_bin].mean()
                    mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))

            metrics = {
                'brier_score': brier_score,
                'expected_calibration_error': ece,
                'maximum_calibration_error': mce,
                'fraction_of_positives': fraction_of_positives,
                'mean_predicted_value': mean_predicted_value,
                'reliability': 1 - ece,  # Reliability score (1 - ECE)
            }

            logger.info(f"Brier Score: {brier_score:.3f}, ECE: {ece:.3f}, MCE: {mce:.3f}")
            return metrics

        except Exception as e:
            logger.error(f"Error calculating calibration metrics: {str(e)}")
            raise

    def _calculate_specificity(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate specificity (True Negative Rate)"""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0

    def _calculate_financial_metrics(self, 
                                   y_true: np.ndarray, 
                                   y_pred: np.ndarray,
                                   loan_amounts: np.ndarray,
                                   tn: int, fp: int, fn: int, tp: int) -> Dict[str, float]:
        """Calculate financial impact metrics"""
        try:
            # Create masks for each prediction type
            true_negatives_mask = (y_true == 0) & (y_pred == 0)
            false_positives_mask = (y_true == 0) & (y_pred == 1)
            false_negatives_mask = (y_true == 1) & (y_pred == 0)
            true_positives_mask = (y_true == 1) & (y_pred == 1)

            # Calculate financial impacts
            approved_loans_value = loan_amounts[true_negatives_mask].sum() + loan_amounts[false_negatives_mask].sum()
            rejected_loans_value = loan_amounts[false_positives_mask].sum() + loan_amounts[true_positives_mask].sum()

            # Potential losses (false negatives - approved but will default)
            potential_losses = loan_amounts[false_negatives_mask].sum()

            # Opportunity cost (false positives - rejected but would not default)
            opportunity_cost = loan_amounts[false_positives_mask].sum()

            # Saved losses (true positives - correctly rejected defaults)
            saved_losses = loan_amounts[true_positives_mask].sum()

            return {
                'approved_loans_value': approved_loans_value,
                'rejected_loans_value': rejected_loans_value,
                'potential_losses': potential_losses,
                'opportunity_cost': opportunity_cost,
                'saved_losses': saved_losses,
                'net_benefit': saved_losses - opportunity_cost,
                'loss_rate_approved': potential_losses / approved_loans_value if approved_loans_value > 0 else 0,
            }

        except Exception as e:
            logger.error(f"Error calculating financial metrics: {str(e)}")
            return {}

    def _calculate_probability_credit_metrics(self, 
                                            y_true: np.ndarray, 
                                            y_prob: np.ndarray) -> Dict[str, float]:
        """Calculate probability-based credit metrics"""
        try:
            # Gini coefficient (2 * AUC - 1)
            gini = 2 * roc_auc_score(y_true, y_prob) - 1

            # KS statistic (Kolmogorov-Smirnov)
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            ks_statistic = np.max(tpr - fpr)

            # Population Stability Index (PSI) - simplified version
            # This would typically compare two time periods, here we use deciles
            deciles = np.percentile(y_prob, np.arange(10, 100, 10))
            psi = 0  # Simplified - would need reference distribution for full PSI

            return {
                'gini_coefficient': gini,
                'ks_statistic': ks_statistic,
                'population_stability_index': psi,
                'mean_predicted_probability': np.mean(y_prob),
                'std_predicted_probability': np.std(y_prob),
            }

        except Exception as e:
            logger.error(f"Error calculating probability credit metrics: {str(e)}")
            return {}

    def generate_classification_report(self, 
                                     y_true: np.ndarray, 
                                     y_pred: np.ndarray,
                                     y_prob: Optional[np.ndarray] = None,
                                     model_name: str = "Model",
                                     include_plots: bool = False) -> Dict[str, Any]:
        """
        Generate comprehensive classification report

        Args:
            y_true: True binary labels
            y_pred: Predicted binary labels
            y_prob: Predicted probabilities (optional)
            model_name: Name of the model
            include_plots: Whether to include plot data

        Returns:
            Dict[str, Any]: Comprehensive classification report
        """
        try:
            report = {
                'model_name': model_name,
                'timestamp': pd.Timestamp.now().isoformat(),
                'data_summary': {
                    'total_samples': len(y_true),
                    'positive_samples': int(np.sum(y_true)),
                    'negative_samples': int(len(y_true) - np.sum(y_true)),
                    'positive_rate': float(np.mean(y_true)),
                    'class_balance': float(np.sum(y_true)) / float(len(y_true) - np.sum(y_true)) if np.sum(y_true) > 0 else 0
                }
            }

            # Basic metrics
            report['basic_metrics'] = self.calculate_basic_metrics(y_true, y_pred, y_prob)

            # Credit-specific metrics
            report['credit_metrics'] = self.calculate_credit_specific_metrics(y_true, y_pred, y_prob)

            # Confusion matrix details
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()
            report['confusion_matrix'] = {
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positives': int(tp),
                'matrix': cm.tolist()
            }

            # Probability-based metrics (if available)
            if y_prob is not None:
                report['roc_metrics'] = self.calculate_roc_metrics(y_true, y_prob)
                report['pr_metrics'] = self.calculate_precision_recall_metrics(y_true, y_prob)
                report['calibration_metrics'] = self.calculate_calibration_metrics(y_true, y_prob)

                # Optimal thresholds
                optimal_f1_threshold, optimal_f1_value = self.find_optimal_threshold(y_true, y_prob, 'f1_score')
                optimal_precision_threshold, optimal_precision_value = self.find_optimal_threshold(y_true, y_prob, 'precision')

                report['optimal_thresholds'] = {
                    'f1_score': {'threshold': optimal_f1_threshold, 'value': optimal_f1_value},
                    'precision': {'threshold': optimal_precision_threshold, 'value': optimal_precision_value},
                    'youden_j': {
                        'threshold': report['roc_metrics']['optimal_threshold_youden'],
                        'value': report['roc_metrics']['youden_j_score']
                    }
                }

            # Classification report from sklearn
            report['sklearn_classification_report'] = classification_report(y_true, y_pred, output_dict=True)

            # Store in history
            self.metrics_history.append(report)

            logger.info(f"Generated comprehensive classification report for {model_name}")
            return report

        except Exception as e:
            logger.error(f"Error generating classification report: {str(e)}")
            raise

    def compare_models(self, 
                      model_results: Dict[str, Dict[str, Any]],
                      primary_metric: str = "roc_auc",
                      secondary_metrics: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Compare multiple models based on their performance metrics

        Args:
            model_results: Dictionary of model names and their results
            primary_metric: Primary metric for ranking
            secondary_metrics: Additional metrics to include in comparison

        Returns:
            pd.DataFrame: Comparison table sorted by primary metric
        """
        try:
            if secondary_metrics is None:
                secondary_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'pr_auc']

            comparison_data = []

            for model_name, results in model_results.items():
                row = {'model_name': model_name}

                # Extract metrics from different sections
                for section in ['basic_metrics', 'credit_metrics', 'roc_metrics', 'pr_metrics']:
                    if section in results:
                        for metric, value in results[section].items():
                            if isinstance(value, (int, float)) and not isinstance(value, bool):
                                row[metric] = value

                comparison_data.append(row)

            # Create DataFrame
            comparison_df = pd.DataFrame(comparison_data)

            # Select relevant columns
            columns_to_include = ['model_name', primary_metric] + secondary_metrics
            available_columns = [col for col in columns_to_include if col in comparison_df.columns]
            comparison_df = comparison_df[available_columns]

            # Sort by primary metric (descending for most metrics)
            ascending = primary_metric in ['log_loss', 'brier_score', 'expected_calibration_error']
            comparison_df = comparison_df.sort_values(primary_metric, ascending=ascending)

            # Add ranking
            comparison_df['rank'] = range(1, len(comparison_df) + 1)

            # Reorder columns
            cols = ['rank', 'model_name'] + [col for col in comparison_df.columns if col not in ['rank', 'model_name']]
            comparison_df = comparison_df[cols]

            logger.info(f"Compared {len(model_results)} models based on {primary_metric}")
            return comparison_df

        except Exception as e:
            logger.error(f"Error comparing models: {str(e)}")
            raise

    def calculate_model_stability(self, 
                                model_results: List[Dict[str, Any]],
                                metric: str = "roc_auc") -> Dict[str, float]:
        """
        Calculate stability metrics across multiple model runs

        Args:
            model_results: List of model results from different runs
            metric: Metric to analyze for stability

        Returns:
            Dict[str, float]: Stability metrics
        """
        try:
            # Extract metric values
            metric_values = []
            for result in model_results:
                for section in ['basic_metrics', 'credit_metrics', 'roc_metrics', 'pr_metrics']:
                    if section in result and metric in result[section]:
                        metric_values.append(result[section][metric])
                        break

            if not metric_values:
                raise ValueError(f"Metric '{metric}' not found in model results")

            metric_values = np.array(metric_values)

            stability_metrics = {
                'mean': float(np.mean(metric_values)),
                'std': float(np.std(metric_values)),
                'min': float(np.min(metric_values)),
                'max': float(np.max(metric_values)),
                'range': float(np.max(metric_values) - np.min(metric_values)),
                'coefficient_of_variation': float(np.std(metric_values) / np.mean(metric_values)) if np.mean(metric_values) != 0 else 0,
                'stability_score': 1 - (np.std(metric_values) / np.mean(metric_values)) if np.mean(metric_values) != 0 else 0
            }

            logger.info(f"Calculated stability metrics for {metric} across {len(model_results)} runs")
            return stability_metrics

        except Exception as e:
            logger.error(f"Error calculating model stability: {str(e)}")
            raise

    def generate_performance_summary(self, 
                                   model_results: Dict[str, Dict[str, Any]],
                                   save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive performance summary for multiple models

        Args:
            model_results: Dictionary of model names and their results
            save_path: Path to save the summary (optional)

        Returns:
            Dict[str, Any]: Performance summary
        """
        try:
            summary = {
                'timestamp': pd.Timestamp.now().isoformat(),
                'total_models': len(model_results),
                'models_evaluated': list(model_results.keys())
            }

            # Model comparison
            comparison_df = self.compare_models(model_results)
            summary['model_ranking'] = comparison_df.to_dict('records')

            # Best performing model
            best_model = comparison_df.iloc[0]['model_name']
            summary['best_model'] = {
                'name': best_model,
                'results': model_results[best_model]
            }

            # Performance statistics across all models
            all_metrics = {}
            for model_name, results in model_results.items():
                for section in ['basic_metrics', 'credit_metrics']:
                    if section in results:
                        for metric, value in results[section].items():
                            if isinstance(value, (int, float)) and not isinstance(value, bool):
                                if metric not in all_metrics:
                                    all_metrics[metric] = []
                                all_metrics[metric].append(value)

            # Calculate statistics for each metric
            metric_statistics = {}
            for metric, values in all_metrics.items():
                if values:
                    metric_statistics[metric] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'best_model': model_results[comparison_df.iloc[0]['model_name']]['model_name'] if 'model_name' in model_results[comparison_df.iloc[0]['model_name']] else best_model
                    }

            summary['metric_statistics'] = metric_statistics

            # Recommendations
            summary['recommendations'] = self._generate_recommendations(model_results, comparison_df)

            # Save if path provided
            if save_path:
                with open(save_path, 'w') as f:
                    import json
                    json.dump(summary, f, indent=2, default=str)
                logger.info(f"Performance summary saved to {save_path}")

            logger.info(f"Generated performance summary for {len(model_results)} models")
            return summary

        except Exception as e:
            logger.error(f"Error generating performance summary: {str(e)}")
            raise

    def calculate_business_impact(self, 
                                y_true: np.ndarray, 
                                y_pred: np.ndarray,
                                loan_amounts: np.ndarray,
                                default_loss_rate: float = 0.6,
                                interest_rate: float = 0.05,
                                operational_cost_rate: float = 0.01) -> Dict[str, float]:
        """
        Calculate business impact metrics

        Args:
            y_true: True binary labels
            y_pred: Predicted binary labels
            loan_amounts: Loan amounts
            default_loss_rate: Loss rate when default occurs (0-1)
            interest_rate: Interest rate on loans
            operational_cost_rate: Operational cost rate

        Returns:
            Dict[str, float]: Business impact metrics
        """
        try:
            # Create masks for each prediction type
            true_negatives_mask = (y_true == 0) & (y_pred == 0)  # Correctly approved
            false_positives_mask = (y_true == 0) & (y_pred == 1)  # Incorrectly rejected
            false_negatives_mask = (y_true == 1) & (y_pred == 0)  # Incorrectly approved
            true_positives_mask = (y_true == 1) & (y_pred == 1)  # Correctly rejected

            # Calculate financial components
            approved_loans = loan_amounts[true_negatives_mask].sum() + loan_amounts[false_negatives_mask].sum()
            rejected_loans = loan_amounts[false_positives_mask].sum() + loan_amounts[true_positives_mask].sum()

            # Revenue from approved loans (interest)
            revenue_from_approved = approved_loans * interest_rate

            # Losses from defaults (false negatives)
            losses_from_defaults = loan_amounts[false_negatives_mask].sum() * default_loss_rate

            # Opportunity cost (false positives - good loans rejected)
            opportunity_cost = loan_amounts[false_positives_mask].sum() * interest_rate

            # Operational costs
            operational_costs = (approved_loans + rejected_loans) * operational_cost_rate

            # Net profit
            net_profit = revenue_from_approved - losses_from_defaults - opportunity_cost - operational_costs

            # Return on assets
            total_portfolio = approved_loans + rejected_loans
            roa = net_profit / total_portfolio if total_portfolio > 0 else 0

            business_metrics = {
                'total_portfolio_value': total_portfolio,
                'approved_loans_value': approved_loans,
                'rejected_loans_value': rejected_loans,
                'revenue_from_approved': revenue_from_approved,
                'losses_from_defaults': losses_from_defaults,
                'opportunity_cost': opportunity_cost,
                'operational_costs': operational_costs,
                'net_profit': net_profit,
                'return_on_assets': roa,
                'profit_margin': net_profit / revenue_from_approved if revenue_from_approved > 0 else 0,
                'loss_rate': losses_from_defaults / approved_loans if approved_loans > 0 else 0,
                'approval_rate': approved_loans / total_portfolio if total_portfolio > 0 else 0
            }

            logger.info(f"Calculated business impact: Net profit = ${net_profit:,.2f}, ROA = {roa:.2%}")
            return business_metrics

        except Exception as e:
            logger.error(f"Error calculating business impact: {str(e)}")
            raise

    def export_metrics_report(self, 
                            model_results: Dict[str, Dict[str, Any]],
                            output_path: str,
                            format: str = "excel") -> None:
        """
        Export comprehensive metrics report

        Args:
            model_results: Dictionary of model results
            output_path: Output file path
            format: Export format ('excel', 'csv', 'json')
        """
        try:
            if format == "excel":
                with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                    # Model comparison
                    comparison_df = self.compare_models(model_results)
                    comparison_df.to_excel(writer, sheet_name='Model_Comparison', index=False)

                    # Detailed metrics for each model
                    for model_name, results in model_results.items():
                        # Basic metrics
                        if 'basic_metrics' in results:
                            basic_df = pd.DataFrame([results['basic_metrics']])
                            basic_df.to_excel(writer, sheet_name=f'{model_name}_Basic', index=False)

                        # Credit metrics
                        if 'credit_metrics' in results:
                            credit_df = pd.DataFrame([results['credit_metrics']])
                            credit_df.to_excel(writer, sheet_name=f'{model_name}_Credit', index=False)

                        # Confusion matrix
                        if 'confusion_matrix' in results:
                            cm_df = pd.DataFrame(results['confusion_matrix']['matrix'])
                            cm_df.to_excel(writer, sheet_name=f'{model_name}_ConfMatrix', index=False)

            elif format == "csv":
                comparison_df = self.compare_models(model_results)
                comparison_df.to_csv(output_path, index=False)

            elif format == "json":
                with open(output_path, 'w') as f:
                    import json
                    json.dump(model_results, f, indent=2, default=str)

            logger.info(f"Metrics report exported to {output_path}")

        except Exception as e:
            logger.error(f"Error exporting metrics report: {str(e)}")
            raise

    def _generate_recommendations(self, 
                                model_results: Dict[str, Dict[str, Any]], 
                                comparison_df: pd.DataFrame) -> List[str]:
        """Generate recommendations based on model performance"""
        recommendations = []

        try:
            best_model = comparison_df.iloc[0]['model_name']
            best_results = model_results[best_model]

            # Performance recommendations
            if 'basic_metrics' in best_results:
                metrics = best_results['basic_metrics']

                if metrics.get('roc_auc', 0) > 0.8:
                    recommendations.append(f"✅ {best_model} shows excellent discriminative ability (AUC > 0.8)")
                elif metrics.get('roc_auc', 0) > 0.7:
                    recommendations.append(f"⚠️ {best_model} shows good discriminative ability (AUC > 0.7) but could be improved")
                else:
                    recommendations.append(f"❌ {best_model} shows poor discriminative ability (AUC < 0.7) - consider feature engineering or different algorithms")

                if metrics.get('precision', 0) < 0.6:
                    recommendations.append("⚠️ Low precision detected - consider adjusting threshold to reduce false positives")

                if metrics.get('recall', 0) < 0.6:
                    recommendations.append("⚠️ Low recall detected - consider adjusting threshold to reduce false negatives")

            # Calibration recommendations
            if 'calibration_metrics' in best_results:
                cal_metrics = best_results['calibration_metrics']
                if cal_metrics.get('expected_calibration_error', 1) > 0.1:
                    recommendations.append("⚠️ Model shows poor calibration - consider calibration techniques like Platt scaling")

            # Business recommendations
            if len(model_results) > 1:
                recommendations.append(f"💡 Consider ensemble methods combining top {min(3, len(model_results))} models")

            recommendations.append("📊 Monitor model performance regularly and retrain when performance degrades")
            recommendations.append("🔄 Consider A/B testing before deploying the model in production")

        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            recommendations.append("❌ Unable to generate specific recommendations due to missing data")

        return recommendations


# Convenience functions for direct use
def calculate_basic_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None, **kwargs) -> Dict[str, float]:
    """Convenience function to calculate basic metrics"""
    metrics_calculator = CreditMetrics()
    return metrics_calculator.calculate_basic_metrics(y_true, y_pred, y_prob, **kwargs)

def calculate_credit_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None, **kwargs) -> Dict[str, float]:
    """Convenience function to calculate credit-specific metrics"""
    metrics_calculator = CreditMetrics()
    return metrics_calculator.calculate_credit_specific_metrics(y_true, y_pred, y_prob, **kwargs)

def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray, metric: str = "f1_score", **kwargs) -> Tuple[float, float]:
    """Convenience function to find optimal threshold"""
    metrics_calculator = CreditMetrics()
    return metrics_calculator.find_optimal_threshold(y_true, y_prob, metric, **kwargs)

def generate_classification_report(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None, **kwargs) -> Dict[str, Any]:
    """Convenience function to generate classification report"""
    metrics_calculator = CreditMetrics()
    return metrics_calculator.generate_classification_report(y_true, y_pred, y_prob, **kwargs)

def compare_models(model_results: Dict[str, Dict[str, Any]], **kwargs) -> pd.DataFrame:
    """Convenience function to compare models"""
    metrics_calculator = CreditMetrics()
    return metrics_calculator.compare_models(model_results, **kwargs)

def calculate_business_impact(y_true: np.ndarray, y_pred: np.ndarray, loan_amounts: np.ndarray, **kwargs) -> Dict[str, float]:
    """Convenience function to calculate business impact"""
    metrics_calculator = CreditMetrics()
    return metrics_calculator.calculate_business_impact(y_true, y_pred, loan_amounts, **kwargs)

def export_metrics_report(model_results: Dict[str, Dict[str, Any]], output_path: str, **kwargs) -> None:
    """Convenience function to export metrics report"""
    metrics_calculator = CreditMetrics()
    metrics_calculator.export_metrics_report(model_results, output_path, **kwargs)


# Utility functions for metrics formatting and display
def format_metrics_for_display(metrics: Dict[str, float], decimal_places: int = 4) -> Dict[str, str]:
    """
    Format metrics for display with appropriate decimal places

    Args:
        metrics: Dictionary of metrics
        decimal_places: Number of decimal places

    Returns:
        Dict[str, str]: Formatted metrics
    """
    formatted = {}
    for key, value in metrics.items():
        if isinstance(value, float):
            if key in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'pr_auc']:
                formatted[key] = f"{value:.{decimal_places}f}"
            elif 'rate' in key.lower() or 'ratio' in key.lower():
                formatted[key] = f"{value:.2%}"
            elif 'amount' in key.lower() or 'value' in key.lower() or 'cost' in key.lower():
                formatted[key] = f"${value:,.2f}"
            else:
                formatted[key] = f"{value:.{decimal_places}f}"
        else:
            formatted[key] = str(value)

    return formatted

def create_metrics_summary_table(metrics: Dict[str, float]) -> pd.DataFrame:
    """
    Create a formatted summary table for metrics

    Args:
        metrics: Dictionary of metrics

    Returns:
        pd.DataFrame: Formatted metrics table
    """
    formatted_metrics = format_metrics_for_display(metrics)

    df = pd.DataFrame([
        {'Metric': key.replace('_', ' ').title(), 'Value': value}
        for key, value in formatted_metrics.items()
    ])

    return df

def calculate_metric_improvements(baseline_metrics: Dict[str, float], 
                                new_metrics: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    """
    Calculate improvements between baseline and new metrics

    Args:
        baseline_metrics: Baseline model metrics
        new_metrics: New model metrics

    Returns:
        Dict[str, Dict[str, float]]: Improvement analysis
    """
    improvements = {}

    for metric in baseline_metrics.keys():
        if metric in new_metrics:
            baseline_val = baseline_metrics[metric]
            new_val = new_metrics[metric]

            absolute_change = new_val - baseline_val
            relative_change = (absolute_change / baseline_val * 100) if baseline_val != 0 else 0

            improvements[metric] = {
                'baseline': baseline_val,
                'new': new_val,
                'absolute_change': absolute_change,
                'relative_change': relative_change,
                'improved': absolute_change > 0
            }

    return improvements
