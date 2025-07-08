"""Utility functions for CreditAnalyticsHub"""
"""
Utils package for machine learning utilities.

This package provides data loading, visualization, and metrics utilities
for machine learning projects.
"""

from .data_loader import DataLoader, load_dataset, preprocess_data
from .visualizer import Visualizer, plot_confusion_matrix, plot_roc_curve, plot_feature_importance
from .metrics import MetricsCalculator, calculate_accuracy, calculate_precision, calculate_recall, calculate_f1_score

__all__ = [
    # Data loading utilities
    'DataLoader',
    'load_dataset',
    'preprocess_data',
    
    # Visualization utilities
    'Visualizer',
    'plot_confusion_matrix',
    'plot_roc_curve',
    'plot_feature_importance',
    
    # Metrics utilities
    'MetricsCalculator',
    'calculate_accuracy',
    'calculate_precision',
    'calculate_recall',
    'calculate_f1_score'
]

__version__ = '1.0.0'
__author__ = 'ML Utils Team'