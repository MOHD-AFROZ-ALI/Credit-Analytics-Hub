"""Utility functions for CreditAnalyticsHub"""
"""
Utils Package for Credit Analytics Hub

This package provides comprehensive utilities for credit analytics including:
- Data processing and validation
- Visualization and reporting
- Risk assessment and scoring
- Regulatory compliance
"""

# Import main classes for easy access
try:
    from .data_utils import (
        DataLoader, DataCleaner, DataValidator, FeatureEngineer, DataSplitter,
        DataQualityLevel, DataQualityReport,
        generate_sample_credit_data, load_and_prepare_data
    )
except ImportError:
    pass

try:
    from .visualization_utils import (
        CreditVisualizationConfig, DataExplorer, ModelPerformanceVisualizer
    )
except ImportError:
    pass

try:
    from .risk_utils import (
        CreditScorer, RiskSegmentation, PortfolioRiskAnalyzer,
        RiskLevel, RiskCategory, RiskScore
    )
except ImportError:
    pass

try:
    from .compliance_utils import (
        ComplianceChecker, RegulatoryReporter, AuditTrail,
        ComplianceLevel, ComplianceReport
    )
except ImportError:
    pass

__version__ = "1.0.0"
__author__ = "Credit Analytics Hub Team"

# Package metadata
__all__ = [
    # Data utilities
    'DataLoader', 'DataCleaner', 'DataValidator', 'FeatureEngineer', 'DataSplitter',
    'DataQualityLevel', 'DataQualityReport',
    
    # Visualization utilities
    'CreditVisualizationConfig', 'DataExplorer', 'ModelPerformanceVisualizer',
    
    # Risk utilities
    'CreditScorer', 'RiskSegmentation', 'PortfolioRiskAnalyzer',
    'RiskLevel', 'RiskCategory', 'RiskScore',
    
    # Compliance utilities
    'ComplianceChecker', 'RegulatoryReporter', 'AuditTrail',
    'ComplianceLevel', 'ComplianceReport',
    
    # Utility functions
    'generate_sample_credit_data', 'load_and_prepare_data'
]