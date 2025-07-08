"""
CreditAnalyticsHub Configuration File
====================================
Comprehensive configuration settings for the Credit Risk Analytics Platform
"""

from datetime import datetime
import os

# =============================================================================
# APPLICATION CONFIGURATION
# =============================================================================

APP_CONFIG = {
    'app_name': 'CreditAnalyticsHub',
    'company_name': 'FinTech Solutions Inc.',
    'version': '2.0.0',
    'environment': os.getenv('ENVIRONMENT', 'development'),
    'debug_mode': os.getenv('DEBUG', 'False').lower() == 'true',
    'max_file_size_mb': 50,
    'session_timeout_minutes': 30,
    'auto_save_interval_seconds': 300,
    'supported_file_formats': ['.csv', '.xlsx', '.xls', '.json'],
    'max_batch_size': 10000,
    'api_rate_limit': 1000,  # requests per hour
    'cache_ttl_seconds': 3600,
    'log_level': 'INFO'
}

# =============================================================================
# THEME AND UI CONFIGURATION
# =============================================================================

THEME_CONFIG = {
    'primary_color': '#00D4FF',
    'secondary_color': '#0099CC',
    'success_color': '#00FF88',
    'warning_color': '#FFB800',
    'error_color': '#FF4B4B',
    'background_color': '#0E1117',
    'secondary_background': '#1E1E1E',
    'card_background': '#2D2D2D',
    'text_primary': '#FFFFFF',
    'text_secondary': '#B0B0B0',
    'text_muted': '#808080',
    'border_color': '#333333',
    'accent_color': '#FF6B6B',
    'gradient_start': '#0E1117',
    'gradient_end': '#1E1E1E',
    'font_family': 'Inter, sans-serif',
    'font_sizes': {
        'small': '0.8rem',
        'normal': '1rem',
        'large': '1.2rem',
        'xlarge': '1.5rem',
        'xxlarge': '2rem'
    }
}

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

MODEL_CONFIG = {
    'default_model': 'xgboost',
    'available_models': {
        'xgboost': {
            'name': 'XGBoost Classifier',
            'type': 'ensemble',
            'params': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            }
        },
        'lightgbm': {
            'name': 'LightGBM Classifier',
            'type': 'ensemble',
            'params': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            }
        },
        'random_forest': {
            'name': 'Random Forest',
            'type': 'ensemble',
            'params': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42
            }
        },
        'logistic_regression': {
            'name': 'Logistic Regression',
            'type': 'linear',
            'params': {
                'C': 1.0,
                'max_iter': 1000,
                'random_state': 42
            }
        }
    },
    'feature_importance_threshold': 0.01,
    'cross_validation_folds': 5,
    'test_size': 0.2,
    'validation_size': 0.2,
    'random_state': 42,
    'model_retrain_threshold_days': 30,
    'performance_threshold': 0.75,
    'drift_detection_threshold': 0.1
}

# =============================================================================
# RISK ASSESSMENT CONFIGURATION
# =============================================================================

RISK_CONFIG = {
    'risk_categories': {
        'very_low': {'min': 0.0, 'max': 0.1, 'color': '#00FF88', 'label': 'Very Low Risk'},
        'low': {'min': 0.1, 'max': 0.3, 'color': '#90EE90', 'label': 'Low Risk'},
        'medium': {'min': 0.3, 'max': 0.6, 'color': '#FFB800', 'label': 'Medium Risk'},
        'high': {'min': 0.6, 'max': 0.8, 'color': '#FF8C00', 'label': 'High Risk'},
        'very_high': {'min': 0.8, 'max': 1.0, 'color': '#FF4B4B', 'label': 'Very High Risk'}
    },
    'default_threshold': 0.5,
    'approval_thresholds': {
        'auto_approve': 0.2,
        'manual_review': 0.7,
        'auto_reject': 0.9
    },
    'score_ranges': {
        'excellent': {'min': 800, 'max': 850},
        'very_good': {'min': 740, 'max': 799},
        'good': {'min': 670, 'max': 739},
        'fair': {'min': 580, 'max': 669},
        'poor': {'min': 300, 'max': 579}
    },
    'risk_factors': {
        'income_debt_ratio': {'weight': 0.25, 'threshold': 0.4},
        'credit_history_length': {'weight': 0.15, 'threshold': 24},
        'payment_history': {'weight': 0.35, 'threshold': 0.95},
        'credit_utilization': {'weight': 0.15, 'threshold': 0.3},
        'recent_inquiries': {'weight': 0.1, 'threshold': 3}
    }
}

# =============================================================================
# BUSINESS RULES CONFIGURATION
# =============================================================================

BUSINESS_RULES = {
    'minimum_age': 18,
    'maximum_age': 75,
    'minimum_income': 25000,
    'maximum_loan_amount': 1000000,
    'minimum_credit_score': 300,
    'maximum_credit_score': 850,
    'debt_to_income_max': 0.5,
    'employment_history_min_months': 6,
    'bankruptcy_exclusion_years': 7,
    'foreclosure_exclusion_years': 3,
    'max_recent_inquiries': 5,
    'min_credit_history_months': 12,
    'required_documents': [
        'identity_verification',
        'income_verification',
        'employment_verification',
        'bank_statements'
    ],
    'loan_purposes': {
        'home_purchase': {'max_ltv': 0.95, 'min_down_payment': 0.05},
        'refinance': {'max_ltv': 0.90, 'min_equity': 0.10},
        'personal': {'max_amount': 100000, 'max_term_months': 84},
        'auto': {'max_ltv': 1.0, 'max_term_months': 72},
        'business': {'max_amount': 500000, 'min_business_age_months': 24}
    },
    'geographic_restrictions': {
        'excluded_states': [],
        'high_risk_zip_codes': [],
        'international_allowed': False
    }
}

# =============================================================================
# FEATURE ENGINEERING CONFIGURATION
# =============================================================================

FEATURE_CONFIG = {
    'numerical_features': [
        'age', 'annual_income', 'credit_score', 'loan_amount', 
        'employment_length', 'debt_to_income_ratio', 'credit_utilization',
        'number_of_accounts', 'delinquencies_2yrs', 'inquiries_6mths',
        'open_accounts', 'total_accounts', 'revolving_balance'
    ],
    'categorical_features': [
        'loan_purpose', 'home_ownership', 'employment_status',
        'income_verification', 'loan_grade', 'state', 'education_level'
    ],
    'derived_features': {
        'income_per_person': ['annual_income', 'dependents'],
        'loan_to_income_ratio': ['loan_amount', 'annual_income'],
        'credit_age_score': ['credit_score', 'credit_history_length'],
        'stability_score': ['employment_length', 'residence_length'],
        'utilization_trend': ['current_utilization', 'avg_utilization_6m']
    },
    'scaling_methods': {
        'numerical': 'standard',  # standard, minmax, robust
        'categorical': 'onehot'   # onehot, label, target
    },
    'outlier_detection': {
        'method': 'iqr',  # iqr, zscore, isolation_forest
        'threshold': 3.0,
        'handle_method': 'cap'  # cap, remove, transform
    }
}

# =============================================================================
# DATA VALIDATION CONFIGURATION
# =============================================================================

VALIDATION_CONFIG = {
    'required_fields': [
        'loan_amount', 'annual_income', 'credit_score', 'employment_length',
        'loan_purpose', 'home_ownership', 'debt_to_income_ratio'
    ],
    'field_constraints': {
        'age': {'min': 18, 'max': 100, 'type': 'int'},
        'annual_income': {'min': 0, 'max': 10000000, 'type': 'float'},
        'credit_score': {'min': 300, 'max': 850, 'type': 'int'},
        'loan_amount': {'min': 1000, 'max': 1000000, 'type': 'float'},
        'debt_to_income_ratio': {'min': 0, 'max': 1, 'type': 'float'},
        'employment_length': {'min': 0, 'max': 50, 'type': 'float'}
    },
    'categorical_values': {
        'loan_purpose': ['debt_consolidation', 'home_improvement', 'major_purchase', 
                        'medical', 'car', 'vacation', 'wedding', 'moving', 'other'],
        'home_ownership': ['RENT', 'OWN', 'MORTGAGE', 'OTHER'],
        'employment_status': ['EMPLOYED', 'SELF_EMPLOYED', 'UNEMPLOYED', 'RETIRED'],
        'income_verification': ['VERIFIED', 'SOURCE_VERIFIED', 'NOT_VERIFIED'],
        'loan_grade': ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    },
    'data_quality_thresholds': {
        'missing_data_threshold': 0.1,
        'duplicate_threshold': 0.05,
        'outlier_threshold': 0.05,
        'consistency_threshold': 0.95
    }
}

# =============================================================================
# REPORTING AND COMPLIANCE CONFIGURATION
# =============================================================================

COMPLIANCE_CONFIG = {
    'regulations': {
        'fair_credit_reporting_act': True,
        'equal_credit_opportunity_act': True,
        'truth_in_lending_act': True,
        'gdpr_compliance': True,
        'ccpa_compliance': True
    },
    'audit_requirements': {
        'model_documentation': True,
        'decision_logging': True,
        'bias_testing': True,
        'performance_monitoring': True,
        'data_lineage': True
    },
    'protected_attributes': [
        'race', 'color', 'religion', 'national_origin', 'sex', 
        'marital_status', 'age', 'disability_status'
    ],
    'fairness_metrics': [
        'demographic_parity', 'equalized_odds', 'calibration',
        'individual_fairness', 'counterfactual_fairness'
    ],
    'reporting_schedule': {
        'daily_monitoring': True,
        'weekly_performance': True,
        'monthly_compliance': True,
        'quarterly_audit': True,
        'annual_review': True
    },
    'data_retention_days': 2555,  # 7 years
    'anonymization_required': True,
    'consent_tracking': True
}

# =============================================================================
# NOTIFICATION AND ALERT CONFIGURATION
# =============================================================================

ALERT_CONFIG = {
    'performance_degradation_threshold': 0.05,
    'data_drift_threshold': 0.1,
    'model_bias_threshold': 0.1,
    'system_error_threshold': 0.01,
    'notification_channels': {
        'email': True,
        'slack': False,
        'dashboard': True,
        'sms': False
    },
    'alert_levels': {
        'info': {'color': '#00D4FF', 'priority': 1},
        'warning': {'color': '#FFB800', 'priority': 2},
        'error': {'color': '#FF4B4B', 'priority': 3},
        'critical': {'color': '#FF0000', 'priority': 4}
    },
    'escalation_rules': {
        'auto_escalate_after_minutes': 30,
        'max_escalation_levels': 3,
        'business_hours_only': False
    }
}

# =============================================================================
# INTEGRATION CONFIGURATION
# =============================================================================

INTEGRATION_CONFIG = {
    'external_apis': {
        'credit_bureau': {
            'enabled': False,
            'timeout_seconds': 30,
            'retry_attempts': 3
        },
        'income_verification': {
            'enabled': False,
            'timeout_seconds': 15,
            'retry_attempts': 2
        },
        'fraud_detection': {
            'enabled': False,
            'timeout_seconds': 10,
            'retry_attempts': 1
        }
    },
    'database': {
        'connection_pool_size': 10,
        'query_timeout_seconds': 30,
        'backup_frequency_hours': 24
    },
    'file_storage': {
        'local_path': '/home/user/output/data/',
        'cloud_enabled': False,
        'encryption_enabled': True,
        'compression_enabled': True
    }
}

# =============================================================================
# DEVELOPMENT AND TESTING CONFIGURATION
# =============================================================================

DEV_CONFIG = {
    'sample_data_size': 1000,
    'test_data_split': 0.2,
    'mock_external_apis': True,
    'enable_profiling': False,
    'debug_logging': True,
    'performance_monitoring': True,
    'memory_limit_mb': 1024,
    'execution_timeout_seconds': 300
}

# =============================================================================
# EXPORT CONFIGURATION DICTIONARY
# =============================================================================

CONFIG = {
    'app': APP_CONFIG,
    'theme': THEME_CONFIG,
    'model': MODEL_CONFIG,
    'risk': RISK_CONFIG,
    'business_rules': BUSINESS_RULES,
    'features': FEATURE_CONFIG,
    'validation': VALIDATION_CONFIG,
    'compliance': COMPLIANCE_CONFIG,
    'alerts': ALERT_CONFIG,
    'integration': INTEGRATION_CONFIG,
    'development': DEV_CONFIG
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_config(section=None, key=None):
    """Get configuration value(s)"""
    if section is None:
        return CONFIG

    if section not in CONFIG:
        raise KeyError(f"Configuration section '{section}' not found")

    if key is None:
        return CONFIG[section]

    if key not in CONFIG[section]:
        raise KeyError(f"Configuration key '{key}' not found in section '{section}'")

    return CONFIG[section][key]

def update_config(section, key, value):
    """Update configuration value"""
    if section not in CONFIG:
        CONFIG[section] = {}
    CONFIG[section][key] = value

def validate_config():
    """Validate configuration settings"""
    errors = []

    # Validate required sections
    required_sections = ['app', 'theme', 'model', 'risk', 'business_rules']
    for section in required_sections:
        if section not in CONFIG:
            errors.append(f"Missing required configuration section: {section}")

    # Validate risk thresholds
    if 'risk' in CONFIG:
        thresholds = CONFIG['risk'].get('approval_thresholds', {})
        if 'auto_approve' in thresholds and 'auto_reject' in thresholds:
            if thresholds['auto_approve'] >= thresholds['auto_reject']:
                errors.append("Auto approve threshold must be less than auto reject threshold")

    return errors

# Initialize configuration validation
_validation_errors = validate_config()
if _validation_errors:
    print("Configuration validation errors:")
    for error in _validation_errors:
        print(f"  - {error}")
