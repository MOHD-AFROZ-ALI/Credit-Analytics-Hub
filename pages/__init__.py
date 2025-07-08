"""
CreditAnalyticsHub Package Initialization
=======================================
This package contains all the page modules for the CreditAnalyticsHub application.
"""

__version__ = "2.0.0"
__author__ = "FinTech Solutions Inc."
__email__ = "support@fintechsolutions.com"

# Import all page modules for easy access
try:
    from . import dashboard
    from . import individual_prediction
    from . import batch_prediction
    from . import data_exploration
    from . import model_performance
    from . import shap_explainability
    from . import business_intelligence
    from . import compliance_report

    __all__ = [
        'dashboard',
        'individual_prediction', 
        'batch_prediction',
        'data_exploration',
        'model_performance',
        'shap_explainability',
        'business_intelligence',
        'compliance_report'
    ]

except ImportError as e:
    print(f"Warning: Could not import all page modules: {e}")
    __all__ = []

# Module metadata
MODULES = {
    'dashboard': 'Main system dashboard with KPIs and overview',
    'individual_prediction': 'Individual credit risk assessment',
    'batch_prediction': 'Batch processing for multiple applications',
    'data_exploration': 'Interactive data analysis and visualization',
    'model_performance': 'Model training and evaluation tools',
    'shap_explainability': 'AI-powered model explanations',
    'business_intelligence': 'Strategic insights and BI dashboard',
    'compliance_report': 'Regulatory compliance and governance'
}

def get_module_info():
    """Get information about all available modules"""
    return MODULES

def list_modules():
    """List all available page modules"""
    print("Available CreditAnalyticsHub Modules:")
    print("=" * 40)
    for module, description in MODULES.items():
        print(f"📄 {module}: {description}")
