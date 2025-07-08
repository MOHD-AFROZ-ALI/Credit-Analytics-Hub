"""
Visualizer Module for Credit Analytics Hub

This module provides comprehensive visualization functionality for credit analytics,
including statistical plots, model performance visualizations, and interactive dashboards.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style defaults
plt.style.use('default')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

class CreditVisualizer:
    """
    Comprehensive visualization class for credit analytics
    """

    def __init__(self, 
                 figsize: Tuple[int, int] = (12, 8),
                 color_palette: str = "viridis",
                 style: str = "whitegrid"):
        """
        Initialize CreditVisualizer

        Args:
            figsize: Default figure size for matplotlib plots
            color_palette: Default color palette
            style: Seaborn style
        """
        self.figsize = figsize
        self.color_palette = color_palette
        self.style = style

        # Set seaborn style
        sns.set_style(style)
        sns.set_palette(color_palette)

        # Color schemes for different plot types
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff7f0e',
            'info': '#17a2b8',
            'light': '#f8f9fa',
            'dark': '#343a40'
        }

        # Plotly template
        self.plotly_template = "plotly_white"

    def plot_distribution(self, 
                         data: pd.DataFrame,
                         column: str,
                         target: Optional[str] = None,
                         plot_type: str = "histogram",
                         bins: int = 30,
                         kde: bool = True,
                         title: Optional[str] = None,
                         save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot distribution of a variable

        Args:
            data: DataFrame containing the data
            column: Column name to plot
            target: Target column for comparison
            plot_type: Type of plot ('histogram', 'boxplot', 'violin')
            bins: Number of bins for histogram
            kde: Whether to show KDE
            title: Plot title
            save_path: Path to save the plot

        Returns:
            matplotlib.Figure: The created figure
        """
        try:
            fig, ax = plt.subplots(figsize=self.figsize)

            if plot_type == "histogram":
                if target and target in data.columns:
                    # Plot distribution by target
                    for target_value in data[target].unique():
                        subset = data[data[target] == target_value]
                        ax.hist(subset[column].dropna(), 
                               bins=bins, 
                               alpha=0.7, 
                               label=f'{target}={target_value}',
                               density=True)

                    if kde:
                        for target_value in data[target].unique():
                            subset = data[data[target] == target_value]
                            if len(subset[column].dropna()) > 1:
                                sns.kdeplot(data=subset, x=column, ax=ax, label=f'KDE {target}={target_value}')

                    ax.legend()
                else:
                    # Single distribution
                    ax.hist(data[column].dropna(), bins=bins, alpha=0.7, color=self.colors['primary'])

                    if kde and len(data[column].dropna()) > 1:
                        sns.kdeplot(data=data, x=column, ax=ax, color=self.colors['secondary'])

            elif plot_type == "boxplot":
                if target and target in data.columns:
                    sns.boxplot(data=data, x=target, y=column, ax=ax)
                else:
                    sns.boxplot(data=data, y=column, ax=ax)

            elif plot_type == "violin":
                if target and target in data.columns:
                    sns.violinplot(data=data, x=target, y=column, ax=ax)
                else:
                    sns.violinplot(data=data, y=column, ax=ax)

            # Set title and labels
            title = title or f'Distribution of {column}'
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel(column.replace('_', ' ').title())
            ax.set_ylabel('Frequency' if plot_type == "histogram" else column.replace('_', ' ').title())

            # Add grid
            ax.grid(True, alpha=0.3)

            plt.tight_layout()

            # Save if path provided
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Plot saved to {save_path}")

            return fig

        except Exception as e:
            logger.error(f"Error creating distribution plot: {str(e)}")
            raise

    def plot_correlation_matrix(self, 
                               data: pd.DataFrame,
                               method: str = "pearson",
                               annot: bool = True,
                               mask_upper: bool = True,
                               title: Optional[str] = None,
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot correlation matrix heatmap

        Args:
            data: DataFrame with numerical columns
            method: Correlation method ('pearson', 'spearman', 'kendall')
            annot: Whether to annotate cells
            mask_upper: Whether to mask upper triangle
            title: Plot title
            save_path: Path to save the plot

        Returns:
            matplotlib.Figure: The created figure
        """
        try:
            # Select only numerical columns
            numerical_data = data.select_dtypes(include=[np.number])

            if numerical_data.empty:
                raise ValueError("No numerical columns found for correlation analysis")

            # Calculate correlation matrix
            corr_matrix = numerical_data.corr(method=method)

            # Create mask for upper triangle if requested
            mask = None
            if mask_upper:
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

            # Create figure
            fig, ax = plt.subplots(figsize=(max(10, len(corr_matrix.columns)), 
                                          max(8, len(corr_matrix.columns))))

            # Create heatmap
            sns.heatmap(corr_matrix,
                       mask=mask,
                       annot=annot,
                       cmap='RdBu_r',
                       center=0,
                       square=True,
                       fmt='.2f',
                       cbar_kws={"shrink": .8},
                       ax=ax)

            # Set title
            title = title or f'{method.title()} Correlation Matrix'
            ax.set_title(title, fontsize=14, fontweight='bold')

            plt.tight_layout()

            # Save if path provided
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Correlation matrix saved to {save_path}")

            return fig

        except Exception as e:
            logger.error(f"Error creating correlation matrix: {str(e)}")
            raise

    def plot_feature_importance(self, 
                               feature_names: List[str],
                               importance_scores: List[float],
                               top_n: int = 20,
                               title: Optional[str] = None,
                               horizontal: bool = True,
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot feature importance scores

        Args:
            feature_names: List of feature names
            importance_scores: List of importance scores
            top_n: Number of top features to show
            title: Plot title
            horizontal: Whether to create horizontal bar plot
            save_path: Path to save the plot

        Returns:
            matplotlib.Figure: The created figure
        """
        try:
            # Create DataFrame and sort by importance
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance_scores
            }).sort_values('importance', ascending=False).head(top_n)

            # Create figure
            fig, ax = plt.subplots(figsize=self.figsize)

            if horizontal:
                bars = ax.barh(range(len(importance_df)), 
                              importance_df['importance'], 
                              color=self.colors['primary'])
                ax.set_yticks(range(len(importance_df)))
                ax.set_yticklabels(importance_df['feature'])
                ax.set_xlabel('Importance Score')
                ax.invert_yaxis()  # Highest importance at top
            else:
                bars = ax.bar(range(len(importance_df)), 
                             importance_df['importance'], 
                             color=self.colors['primary'])
                ax.set_xticks(range(len(importance_df)))
                ax.set_xticklabels(importance_df['feature'], rotation=45, ha='right')
                ax.set_ylabel('Importance Score')

            # Add value labels on bars
            for i, bar in enumerate(bars):
                height = bar.get_height() if not horizontal else bar.get_width()
                label_pos = height + max(importance_scores) * 0.01

                if horizontal:
                    ax.text(label_pos, bar.get_y() + bar.get_height()/2, 
                           f'{height:.3f}', ha='left', va='center')
                else:
                    ax.text(bar.get_x() + bar.get_width()/2, label_pos, 
                           f'{height:.3f}', ha='center', va='bottom')

            # Set title
            title = title or f'Top {top_n} Feature Importance'
            ax.set_title(title, fontsize=14, fontweight='bold')

            # Add grid
            ax.grid(True, alpha=0.3)

            plt.tight_layout()

            # Save if path provided
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Feature importance plot saved to {save_path}")

            return fig

        except Exception as e:
            logger.error(f"Error creating feature importance plot: {str(e)}")
            raise

    def plot_target_distribution(self, 
                                target: pd.Series,
                                title: Optional[str] = None,
                                show_percentages: bool = True,
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot target variable distribution

        Args:
            target: Target variable Series
            title: Plot title
            show_percentages: Whether to show percentages
            save_path: Path to save the plot

        Returns:
            matplotlib.Figure: The created figure
        """
        try:
            # Calculate value counts
            value_counts = target.value_counts()
            percentages = target.value_counts(normalize=True) * 100

            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Bar plot
            bars = ax1.bar(value_counts.index, value_counts.values, 
                          color=[self.colors['primary'], self.colors['secondary']])
            ax1.set_title('Target Distribution (Counts)')
            ax1.set_xlabel('Target Value')
            ax1.set_ylabel('Count')

            # Add count labels on bars
            for bar, count in zip(bars, value_counts.values):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + count * 0.01,
                        str(count), ha='center', va='bottom')

            # Pie chart
            colors = [self.colors['primary'], self.colors['secondary']]
            wedges, texts, autotexts = ax2.pie(value_counts.values, 
                                              labels=value_counts.index,
                                              autopct='%1.1f%%' if show_percentages else None,
                                              colors=colors,
                                              startangle=90)
            ax2.set_title('Target Distribution (Proportions)')

            # Overall title
            title = title or 'Target Variable Distribution'
            fig.suptitle(title, fontsize=16, fontweight='bold')

            plt.tight_layout()

            # Save if path provided
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Target distribution plot saved to {save_path}")

            return fig

        except Exception as e:
            logger.error(f"Error creating target distribution plot: {str(e)}")
            raise

    def plot_confusion_matrix(self, 
                             y_true: np.ndarray,
                             y_pred: np.ndarray,
                             labels: Optional[List[str]] = None,
                             normalize: str = "true",
                             title: Optional[str] = None,
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot confusion matrix

        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Label names
            normalize: Normalization method ('true', 'pred', 'all', None)
            title: Plot title
            save_path: Path to save the plot

        Returns:
            matplotlib.Figure: The created figure
        """
        try:
            from sklearn.metrics import confusion_matrix

            # Calculate confusion matrix
            cm = confusion_matrix(y_true, y_pred)

            # Normalize if requested
            if normalize == "true":
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                fmt = '.2%'
            elif normalize == "pred":
                cm = cm.astype('float') / cm.sum(axis=0)
                fmt = '.2%'
            elif normalize == "all":
                cm = cm.astype('float') / cm.sum()
                fmt = '.2%'
            else:
                fmt = 'd'

            # Create figure
            fig, ax = plt.subplots(figsize=(8, 6))

            # Create heatmap
            sns.heatmap(cm, 
                       annot=True, 
                       fmt=fmt, 
                       cmap='Blues',
                       xticklabels=labels or ['Class 0', 'Class 1'],
                       yticklabels=labels or ['Class 0', 'Class 1'],
                       ax=ax)

            # Set labels and title
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')
            title = title or 'Confusion Matrix'
            ax.set_title(title, fontsize=14, fontweight='bold')

            plt.tight_layout()

            # Save if path provided
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Confusion matrix saved to {save_path}")

            return fig

        except Exception as e:
            logger.error(f"Error creating confusion matrix: {str(e)}")
            raise

    def plot_roc_curve(self, 
                      y_true: np.ndarray,
                      y_prob: np.ndarray,
                      title: Optional[str] = None,
                      save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot ROC curve

        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities for positive class
            title: Plot title
            save_path: Path to save the plot

        Returns:
            matplotlib.Figure: The created figure
        """
        try:
            from sklearn.metrics import roc_curve, auc

            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)

            # Create figure
            fig, ax = plt.subplots(figsize=(8, 8))

            # Plot ROC curve
            ax.plot(fpr, tpr, color=self.colors['primary'], lw=2, 
                   label=f'ROC Curve (AUC = {roc_auc:.3f})')

            # Plot diagonal line (random classifier)
            ax.plot([0, 1], [0, 1], color=self.colors['danger'], lw=2, 
                   linestyle='--', label='Random Classifier')

            # Set labels and title
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            title = title or 'Receiver Operating Characteristic (ROC) Curve'
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.legend(loc="lower right")
            ax.grid(True, alpha=0.3)

            plt.tight_layout()

            # Save if path provided
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"ROC curve saved to {save_path}")

            return fig

        except Exception as e:
            logger.error(f"Error creating ROC curve: {str(e)}")
            raise

    def plot_precision_recall_curve(self, 
                                   y_true: np.ndarray,
                                   y_prob: np.ndarray,
                                   title: Optional[str] = None,
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot Precision-Recall curve

        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities for positive class
            title: Plot title
            save_path: Path to save the plot

        Returns:
            matplotlib.Figure: The created figure
        """
        try:
            from sklearn.metrics import precision_recall_curve, average_precision_score

            # Calculate precision-recall curve
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            avg_precision = average_precision_score(y_true, y_prob)

            # Create figure
            fig, ax = plt.subplots(figsize=(8, 8))

            # Plot PR curve
            ax.plot(recall, precision, color=self.colors['primary'], lw=2,
                   label=f'PR Curve (AP = {avg_precision:.3f})')

            # Plot baseline (random classifier)
            baseline = np.sum(y_true) / len(y_true)
            ax.axhline(y=baseline, color=self.colors['danger'], lw=2, 
                      linestyle='--', label=f'Random Classifier (AP = {baseline:.3f})')

            # Set labels and title
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            title = title or 'Precision-Recall Curve'
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.legend(loc="lower left")
            ax.grid(True, alpha=0.3)

            plt.tight_layout()

            # Save if path provided
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Precision-Recall curve saved to {save_path}")

            return fig

        except Exception as e:
            logger.error(f"Error creating Precision-Recall curve: {str(e)}")
            raise

    def plot_learning_curves(self, 
                            train_scores: List[float],
                            val_scores: List[float],
                            train_sizes: Optional[List[int]] = None,
                            metric_name: str = "Score",
                            title: Optional[str] = None,
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot learning curves

        Args:
            train_scores: Training scores
            val_scores: Validation scores
            train_sizes: Training set sizes
            metric_name: Name of the metric being plotted
            title: Plot title
            save_path: Path to save the plot

        Returns:
            matplotlib.Figure: The created figure
        """
        try:
            if train_sizes is None:
                train_sizes = list(range(1, len(train_scores) + 1))

            # Create figure
            fig, ax = plt.subplots(figsize=self.figsize)

            # Plot training and validation scores
            ax.plot(train_sizes, train_scores, 'o-', color=self.colors['primary'], 
                   label=f'Training {metric_name}', linewidth=2, markersize=6)
            ax.plot(train_sizes, val_scores, 'o-', color=self.colors['secondary'], 
                   label=f'Validation {metric_name}', linewidth=2, markersize=6)

            # Fill between curves to show gap
            ax.fill_between(train_sizes, train_scores, val_scores, 
                           alpha=0.1, color=self.colors['warning'])

            # Set labels and title
            ax.set_xlabel('Training Set Size' if 'size' in str(train_sizes[0]).lower() else 'Epoch')
            ax.set_ylabel(metric_name)
            title = title or f'Learning Curves - {metric_name}'
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3)

            plt.tight_layout()

            # Save if path provided
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Learning curves saved to {save_path}")

            return fig

        except Exception as e:
            logger.error(f"Error creating learning curves: {str(e)}")
            raise

    def plot_model_comparison(self, 
                             model_names: List[str],
                             metrics_dict: Dict[str, List[float]],
                             title: Optional[str] = None,
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot comparison of multiple models across different metrics

        Args:
            model_names: List of model names
            metrics_dict: Dictionary with metric names as keys and lists of scores as values
            title: Plot title
            save_path: Path to save the plot

        Returns:
            matplotlib.Figure: The created figure
        """
        try:
            # Create DataFrame for easier plotting
            df = pd.DataFrame(metrics_dict, index=model_names)

            # Create figure with subplots
            n_metrics = len(metrics_dict)
            n_cols = min(3, n_metrics)
            n_rows = (n_metrics + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
            if n_metrics == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = axes.reshape(1, -1)

            # Plot each metric
            for i, (metric, scores) in enumerate(metrics_dict.items()):
                row = i // n_cols
                col = i % n_cols
                ax = axes[row, col] if n_rows > 1 else axes[col]

                # Create bar plot
                bars = ax.bar(model_names, scores, color=self.colors['primary'], alpha=0.7)

                # Add value labels on bars
                for bar, score in zip(bars, scores):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(scores) * 0.01,
                           f'{score:.3f}', ha='center', va='bottom')

                ax.set_title(f'{metric}', fontweight='bold')
                ax.set_ylabel(metric)
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3)

            # Hide empty subplots
            for i in range(n_metrics, n_rows * n_cols):
                row = i // n_cols
                col = i % n_cols
                if n_rows > 1:
                    axes[row, col].set_visible(False)
                else:
                    axes[col].set_visible(False)

            # Overall title
            title = title or 'Model Performance Comparison'
            fig.suptitle(title, fontsize=16, fontweight='bold')

            plt.tight_layout()

            # Save if path provided
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Model comparison plot saved to {save_path}")

            return fig

        except Exception as e:
            logger.error(f"Error creating model comparison plot: {str(e)}")
            raise

    def plot_prediction_distribution(self, 
                                   y_true: np.ndarray,
                                   y_pred: np.ndarray,
                                   y_prob: Optional[np.ndarray] = None,
                                   title: Optional[str] = None,
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot distribution of predictions vs actual values

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (optional)
            title: Plot title
            save_path: Path to save the plot

        Returns:
            matplotlib.Figure: The created figure
        """
        try:
            # Create figure with subplots
            n_plots = 3 if y_prob is not None else 2
            fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 5))

            if n_plots == 2:
                axes = [axes[0], axes[1]]

            # Plot 1: Actual vs Predicted scatter
            ax1 = axes[0]
            ax1.scatter(y_true, y_pred, alpha=0.6, color=self.colors['primary'])
            ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                    'r--', lw=2, label='Perfect Prediction')
            ax1.set_xlabel('True Values')
            ax1.set_ylabel('Predicted Values')
            ax1.set_title('Actual vs Predicted')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Plot 2: Residuals
            ax2 = axes[1]
            residuals = y_true - y_pred
            ax2.scatter(y_pred, residuals, alpha=0.6, color=self.colors['secondary'])
            ax2.axhline(y=0, color='r', linestyle='--', lw=2)
            ax2.set_xlabel('Predicted Values')
            ax2.set_ylabel('Residuals')
            ax2.set_title('Residual Plot')
            ax2.grid(True, alpha=0.3)

            # Plot 3: Probability distribution (if available)
            if y_prob is not None:
                ax3 = axes[2]

                # Separate probabilities by true class
                prob_class_0 = y_prob[y_true == 0]
                prob_class_1 = y_prob[y_true == 1]

                ax3.hist(prob_class_0, bins=30, alpha=0.7, label='Class 0', 
                        color=self.colors['primary'], density=True)
                ax3.hist(prob_class_1, bins=30, alpha=0.7, label='Class 1', 
                        color=self.colors['secondary'], density=True)

                ax3.set_xlabel('Predicted Probability')
                ax3.set_ylabel('Density')
                ax3.set_title('Probability Distribution by True Class')
                ax3.legend()
                ax3.grid(True, alpha=0.3)

            # Overall title
            title = title or 'Prediction Analysis'
            fig.suptitle(title, fontsize=16, fontweight='bold')

            plt.tight_layout()

            # Save if path provided
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Prediction distribution plot saved to {save_path}")

            return fig

        except Exception as e:
            logger.error(f"Error creating prediction distribution plot: {str(e)}")
            raise

    def plot_calibration_curve(self, 
                              y_true: np.ndarray,
                              y_prob: np.ndarray,
                              n_bins: int = 10,
                              title: Optional[str] = None,
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot calibration curve (reliability diagram)

        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            n_bins: Number of bins for calibration
            title: Plot title
            save_path: Path to save the plot

        Returns:
            matplotlib.Figure: The created figure
        """
        try:
            from sklearn.calibration import calibration_curve

            # Calculate calibration curve
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_prob, n_bins=n_bins
            )

            # Create figure
            fig, ax = plt.subplots(figsize=(8, 8))

            # Plot calibration curve
            ax.plot(mean_predicted_value, fraction_of_positives, "s-",
                   color=self.colors['primary'], linewidth=2, markersize=8,
                   label=f'Model (Brier Score: {((y_prob - y_true) ** 2).mean():.3f})')

            # Plot perfect calibration line
            ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

            # Set labels and title
            ax.set_xlabel('Mean Predicted Probability')
            ax.set_ylabel('Fraction of Positives')
            title = title or 'Calibration Curve (Reliability Diagram)'
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.legend(loc="lower right")
            ax.grid(True, alpha=0.3)

            plt.tight_layout()

            # Save if path provided
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Calibration curve saved to {save_path}")

            return fig

        except Exception as e:
            logger.error(f"Error creating calibration curve: {str(e)}")
            raise

    def create_interactive_distribution(self, 
                                      data: pd.DataFrame,
                                      column: str,
                                      target: Optional[str] = None,
                                      plot_type: str = "histogram",
                                      title: Optional[str] = None) -> go.Figure:
        """
        Create interactive distribution plot using Plotly

        Args:
            data: DataFrame containing the data
            column: Column name to plot
            target: Target column for comparison
            plot_type: Type of plot ('histogram', 'box', 'violin')
            title: Plot title

        Returns:
            plotly.graph_objects.Figure: Interactive plot
        """
        try:
            title = title or f'Interactive Distribution of {column}'

            if plot_type == "histogram":
                if target and target in data.columns:
                    fig = px.histogram(data, x=column, color=target, 
                                     marginal="box", hover_data=data.columns,
                                     title=title, template=self.plotly_template)
                else:
                    fig = px.histogram(data, x=column, marginal="box",
                                     hover_data=data.columns,
                                     title=title, template=self.plotly_template)

            elif plot_type == "box":
                if target and target in data.columns:
                    fig = px.box(data, x=target, y=column, 
                               title=title, template=self.plotly_template)
                else:
                    fig = px.box(data, y=column, 
                               title=title, template=self.plotly_template)

            elif plot_type == "violin":
                if target and target in data.columns:
                    fig = px.violin(data, x=target, y=column, box=True,
                                  title=title, template=self.plotly_template)
                else:
                    fig = px.violin(data, y=column, box=True,
                                  title=title, template=self.plotly_template)

            # Update layout
            fig.update_layout(
                showlegend=True,
                hovermode='closest',
                height=600
            )

            return fig

        except Exception as e:
            logger.error(f"Error creating interactive distribution plot: {str(e)}")
            raise

    def create_interactive_correlation(self, 
                                     data: pd.DataFrame,
                                     method: str = "pearson",
                                     title: Optional[str] = None) -> go.Figure:
        """
        Create interactive correlation heatmap using Plotly

        Args:
            data: DataFrame with numerical columns
            method: Correlation method ('pearson', 'spearman', 'kendall')
            title: Plot title

        Returns:
            plotly.graph_objects.Figure: Interactive heatmap
        """
        try:
            # Select only numerical columns
            numerical_data = data.select_dtypes(include=[np.number])

            if numerical_data.empty:
                raise ValueError("No numerical columns found for correlation analysis")

            # Calculate correlation matrix
            corr_matrix = numerical_data.corr(method=method)

            # Create interactive heatmap
            fig = px.imshow(corr_matrix,
                          text_auto=True,
                          aspect="auto",
                          color_continuous_scale='RdBu_r',
                          color_continuous_midpoint=0,
                          title=title or f'{method.title()} Correlation Matrix',
                          template=self.plotly_template)

            # Update layout
            fig.update_layout(
                height=max(500, len(corr_matrix.columns) * 30),
                width=max(500, len(corr_matrix.columns) * 30)
            )

            return fig

        except Exception as e:
            logger.error(f"Error creating interactive correlation plot: {str(e)}")
            raise

    def create_interactive_scatter(self, 
                                 data: pd.DataFrame,
                                 x_col: str,
                                 y_col: str,
                                 color_col: Optional[str] = None,
                                 size_col: Optional[str] = None,
                                 title: Optional[str] = None) -> go.Figure:
        """
        Create interactive scatter plot using Plotly

        Args:
            data: DataFrame containing the data
            x_col: Column for x-axis
            y_col: Column for y-axis
            color_col: Column for color coding
            size_col: Column for size coding
            title: Plot title

        Returns:
            plotly.graph_objects.Figure: Interactive scatter plot
        """
        try:
            fig = px.scatter(data, 
                           x=x_col, 
                           y=y_col,
                           color=color_col,
                           size=size_col,
                           hover_data=data.columns,
                           title=title or f'{y_col} vs {x_col}',
                           template=self.plotly_template)

            # Update layout
            fig.update_layout(
                hovermode='closest',
                height=600
            )

            return fig

        except Exception as e:
            logger.error(f"Error creating interactive scatter plot: {str(e)}")
            raise

    def create_interactive_roc_curve(self, 
                                   y_true: np.ndarray,
                                   y_prob: np.ndarray,
                                   model_name: str = "Model",
                                   title: Optional[str] = None) -> go.Figure:
        """
        Create interactive ROC curve using Plotly

        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities for positive class
            model_name: Name of the model
            title: Plot title

        Returns:
            plotly.graph_objects.Figure: Interactive ROC curve
        """
        try:
            from sklearn.metrics import roc_curve, auc

            # Calculate ROC curve
            fpr, tpr, thresholds = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)

            # Create figure
            fig = go.Figure()

            # Add ROC curve
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'{model_name} (AUC = {roc_auc:.3f})',
                line=dict(width=3),
                hovertemplate='<b>FPR</b>: %{x:.3f}<br>' +
                            '<b>TPR</b>: %{y:.3f}<br>' +
                            '<b>Threshold</b>: %{customdata:.3f}<extra></extra>',
                customdata=thresholds
            ))

            # Add diagonal line (random classifier)
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Random Classifier',
                line=dict(dash='dash', width=2, color='red'),
                hovertemplate='<b>Random Classifier</b><extra></extra>'
            ))

            # Update layout
            fig.update_layout(
                title=title or 'Interactive ROC Curve',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                template=self.plotly_template,
                hovermode='closest',
                height=600,
                width=600
            )

            return fig

        except Exception as e:
            logger.error(f"Error creating interactive ROC curve: {str(e)}")
            raise

    def create_interactive_feature_importance(self, 
                                            feature_names: List[str],
                                            importance_scores: List[float],
                                            top_n: int = 20,
                                            title: Optional[str] = None) -> go.Figure:
        """
        Create interactive feature importance plot using Plotly

        Args:
            feature_names: List of feature names
            importance_scores: List of importance scores
            top_n: Number of top features to show
            title: Plot title

        Returns:
            plotly.graph_objects.Figure: Interactive bar plot
        """
        try:
            # Create DataFrame and sort by importance
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance_scores
            }).sort_values('importance', ascending=True).tail(top_n)  # Ascending for horizontal bar

            # Create horizontal bar plot
            fig = px.bar(importance_df, 
                        x='importance', 
                        y='feature',
                        orientation='h',
                        title=title or f'Top {top_n} Feature Importance',
                        template=self.plotly_template,
                        hover_data={'importance': ':.4f'})

            # Update layout
            fig.update_layout(
                height=max(400, top_n * 25),
                xaxis_title='Importance Score',
                yaxis_title='Features',
                hovermode='closest'
            )

            return fig

        except Exception as e:
            logger.error(f"Error creating interactive feature importance plot: {str(e)}")
            raise

    def create_dashboard_summary(self, 
                               data: pd.DataFrame,
                               target_col: Optional[str] = None) -> go.Figure:
        """
        Create a comprehensive dashboard summary with multiple subplots

        Args:
            data: DataFrame containing the data
            target_col: Target column name

        Returns:
            plotly.graph_objects.Figure: Dashboard with multiple subplots
        """
        try:
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Data Overview', 'Missing Values', 
                              'Numerical Distributions', 'Target Distribution'),
                specs=[[{"type": "table"}, {"type": "bar"}],
                       [{"type": "box"}, {"type": "pie"}]]
            )

            # 1. Data Overview Table
            overview_data = {
                'Metric': ['Total Rows', 'Total Columns', 'Numerical Columns', 
                          'Categorical Columns', 'Missing Values', 'Duplicate Rows'],
                'Value': [
                    len(data),
                    len(data.columns),
                    len(data.select_dtypes(include=[np.number]).columns),
                    len(data.select_dtypes(include=['object', 'category']).columns),
                    data.isnull().sum().sum(),
                    data.duplicated().sum()
                ]
            }

            fig.add_trace(
                go.Table(
                    header=dict(values=['Metric', 'Value']),
                    cells=dict(values=[overview_data['Metric'], overview_data['Value']])
                ),
                row=1, col=1
            )

            # 2. Missing Values Bar Chart
            missing_data = data.isnull().sum()
            missing_data = missing_data[missing_data > 0].sort_values(ascending=True)

            if not missing_data.empty:
                fig.add_trace(
                    go.Bar(x=missing_data.values, y=missing_data.index, 
                          orientation='h', name='Missing Values'),
                    row=1, col=2
                )

            # 3. Numerical Distributions (Box plots)
            numerical_cols = data.select_dtypes(include=[np.number]).columns[:5]  # Limit to 5
            for i, col in enumerate(numerical_cols):
                fig.add_trace(
                    go.Box(y=data[col], name=col, showlegend=False),
                    row=2, col=1
                )

            # 4. Target Distribution (if available)
            if target_col and target_col in data.columns:
                target_counts = data[target_col].value_counts()
                fig.add_trace(
                    go.Pie(labels=target_counts.index, values=target_counts.values,
                          name="Target Distribution", showlegend=False),
                    row=2, col=2
                )

            # Update layout
            fig.update_layout(
                title_text="Data Summary Dashboard",
                height=800,
                template=self.plotly_template
            )

            return fig

        except Exception as e:
            logger.error(f"Error creating dashboard summary: {str(e)}")
            raise

    def save_plot(self, 
                 fig: Union[plt.Figure, go.Figure], 
                 filename: str,
                 format: str = "png",
                 width: int = 1200,
                 height: int = 800) -> None:
        """
        Save plot to file (works for both matplotlib and plotly figures)

        Args:
            fig: Figure object (matplotlib or plotly)
            filename: Output filename
            format: Output format ('png', 'jpg', 'pdf', 'html', 'svg')
            width: Width in pixels (for plotly)
            height: Height in pixels (for plotly)
        """
        try:
            # Ensure output directory exists
            output_path = Path(filename)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if isinstance(fig, plt.Figure):
                # Matplotlib figure
                fig.savefig(filename, dpi=300, bbox_inches='tight', format=format)
            else:
                # Plotly figure
                if format.lower() == 'html':
                    fig.write_html(filename)
                elif format.lower() in ['png', 'jpg', 'jpeg', 'pdf', 'svg']:
                    fig.write_image(filename, width=width, height=height, format=format)
                else:
                    raise ValueError(f"Unsupported format for Plotly: {format}")

            logger.info(f"Plot saved to {filename}")

        except Exception as e:
            logger.error(f"Error saving plot: {str(e)}")
            raise


# Convenience functions for direct use
def plot_distribution(data: pd.DataFrame, column: str, **kwargs) -> plt.Figure:
    """Convenience function to plot distribution"""
    visualizer = CreditVisualizer()
    return visualizer.plot_distribution(data, column, **kwargs)

def plot_correlation_matrix(data: pd.DataFrame, **kwargs) -> plt.Figure:
    """Convenience function to plot correlation matrix"""
    visualizer = CreditVisualizer()
    return visualizer.plot_correlation_matrix(data, **kwargs)

def plot_feature_importance(feature_names: List[str], importance_scores: List[float], **kwargs) -> plt.Figure:
    """Convenience function to plot feature importance"""
    visualizer = CreditVisualizer()
    return visualizer.plot_feature_importance(feature_names, importance_scores, **kwargs)

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> plt.Figure:
    """Convenience function to plot confusion matrix"""
    visualizer = CreditVisualizer()
    return visualizer.plot_confusion_matrix(y_true, y_pred, **kwargs)

def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, **kwargs) -> plt.Figure:
    """Convenience function to plot ROC curve"""
    visualizer = CreditVisualizer()
    return visualizer.plot_roc_curve(y_true, y_prob, **kwargs)

def plot_precision_recall_curve(y_true: np.ndarray, y_prob: np.ndarray, **kwargs) -> plt.Figure:
    """Convenience function to plot Precision-Recall curve"""
    visualizer = CreditVisualizer()
    return visualizer.plot_precision_recall_curve(y_true, y_prob, **kwargs)

def create_interactive_distribution(data: pd.DataFrame, column: str, **kwargs) -> go.Figure:
    """Convenience function to create interactive distribution plot"""
    visualizer = CreditVisualizer()
    return visualizer.create_interactive_distribution(data, column, **kwargs)

def create_interactive_correlation(data: pd.DataFrame, **kwargs) -> go.Figure:
    """Convenience function to create interactive correlation heatmap"""
    visualizer = CreditVisualizer()
    return visualizer.create_interactive_correlation(data, **kwargs)

def create_interactive_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, **kwargs) -> go.Figure:
    """Convenience function to create interactive ROC curve"""
    visualizer = CreditVisualizer()
    return visualizer.create_interactive_roc_curve(y_true, y_prob, **kwargs)

def create_dashboard_summary(data: pd.DataFrame, **kwargs) -> go.Figure:
    """Convenience function to create dashboard summary"""
    visualizer = CreditVisualizer()
    return visualizer.create_dashboard_summary(data, **kwargs)

def save_plot(fig: Union[plt.Figure, go.Figure], filename: str, **kwargs) -> None:
    """Convenience function to save plots"""
    visualizer = CreditVisualizer()
    visualizer.save_plot(fig, filename, **kwargs)


# Additional utility functions for streamlit integration
def display_metrics_table(metrics_dict: Dict[str, float]) -> pd.DataFrame:
    """
    Create a formatted metrics table for display

    Args:
        metrics_dict: Dictionary of metric names and values

    Returns:
        pd.DataFrame: Formatted metrics table
    """
    df = pd.DataFrame(list(metrics_dict.items()), columns=['Metric', 'Value'])
    df['Value'] = df['Value'].round(4)
    return df

def create_model_summary_card(model_name: str, 
                            metrics: Dict[str, float],
                            best_params: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Create a summary card for model results

    Args:
        model_name: Name of the model
        metrics: Dictionary of performance metrics
        best_params: Best hyperparameters (if available)

    Returns:
        Dict: Summary card data
    """
    summary = {
        'model_name': model_name,
        'metrics': metrics,
        'best_params': best_params or {},
        'primary_metric': max(metrics.items(), key=lambda x: x[1]) if metrics else ('N/A', 0),
        'total_metrics': len(metrics)
    }

    return summary

def format_number(value: float, format_type: str = "decimal") -> str:
    """
    Format numbers for display

    Args:
        value: Number to format
        format_type: Format type ('decimal', 'percentage', 'currency')

    Returns:
        str: Formatted number string
    """
    if format_type == "percentage":
        return f"{value:.2%}"
    elif format_type == "currency":
        return f"${value:,.2f}"
    else:
        return f"{value:.4f}"
