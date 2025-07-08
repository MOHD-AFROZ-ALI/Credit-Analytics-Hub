import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
import sys
import os

# Add models directory to path for imports
sys.path.append('/home/user/output/models')

# Import our model management system
from model_manager import ModelManager
from credit_model import RandomForestCreditModel

# Generate synthetic credit data
def generate_synthetic_credit_data(n_samples=1000, random_state=42):
    """Generate realistic synthetic credit data for training."""
    np.random.seed(random_state)
    
    # Generate base features using make_classification
    X_base, y = make_classification(
        n_samples=n_samples,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_clusters_per_class=1,
        weights=[0.7, 0.3],  # 30% default rate
        flip_y=0.05,  # 5% label noise
        random_state=random_state
    )
    
    # Create realistic feature names and transform to meaningful ranges
    feature_names = [
        'annual_income', 'loan_amount', 'credit_score', 'employment_length',
        'debt_to_income_ratio', 'credit_utilization', 'payment_history_score',
        'age', 'number_of_accounts', 'recent_inquiries'
    ]
    
    # Transform features to realistic ranges
    X_transformed = np.zeros_like(X_base)
    
    # Annual income: 25k to 150k
    X_transformed[:, 0] = 25000 + (X_base[:, 0] - X_base[:, 0].min()) / (X_base[:, 0].max() - X_base[:, 0].min()) * 125000
    
    # Loan amount: 5k to 50k
    X_transformed[:, 1] = 5000 + (X_base[:, 1] - X_base[:, 1].min()) / (X_base[:, 1].max() - X_base[:, 1].min()) * 45000
    
    # Credit score: 300 to 850
    X_transformed[:, 2] = 300 + (X_base[:, 2] - X_base[:, 2].min()) / (X_base[:, 2].max() - X_base[:, 2].min()) * 550
    
    # Employment length: 0 to 20 years
    X_transformed[:, 3] = (X_base[:, 3] - X_base[:, 3].min()) / (X_base[:, 3].max() - X_base[:, 3].min()) * 20
    
    # Debt to income ratio: 0.1 to 0.8
    X_transformed[:, 4] = 0.1 + (X_base[:, 4] - X_base[:, 4].min()) / (X_base[:, 4].max() - X_base[:, 4].min()) * 0.7
    
    # Credit utilization: 0.0 to 1.0
    X_transformed[:, 5] = (X_base[:, 5] - X_base[:, 5].min()) / (X_base[:, 5].max() - X_base[:, 5].min())
    
    # Payment history score: 300 to 850
    X_transformed[:, 6] = 300 + (X_base[:, 6] - X_base[:, 6].min()) / (X_base[:, 6].max() - X_base[:, 6].min()) * 550
    
    # Age: 18 to 75
    X_transformed[:, 7] = 18 + (X_base[:, 7] - X_base[:, 7].min()) / (X_base[:, 7].max() - X_base[:, 7].min()) * 57
    
    # Number of accounts: 1 to 20
    X_transformed[:, 8] = 1 + (X_base[:, 8] - X_base[:, 8].min()) / (X_base[:, 8].max() - X_base[:, 8].min()) * 19
    
    # Recent inquiries: 0 to 10
    X_transformed[:, 9] = (X_base[:, 9] - X_base[:, 9].min()) / (X_base[:, 9].max() - X_base[:, 9].min()) * 10
    
    # Create DataFrame
    X_df = pd.DataFrame(X_transformed, columns=feature_names)
    y_series = pd.Series(y, name='default')
    
    # Add some categorical features
    np.random.seed(random_state)
    X_df['loan_purpose'] = np.random.choice(['debt_consolidation', 'home_improvement', 'major_purchase', 'other'], 
                                           size=n_samples, p=[0.4, 0.2, 0.2, 0.2])
    X_df['home_ownership'] = np.random.choice(['rent', 'own', 'mortgage'], 
                                            size=n_samples, p=[0.3, 0.3, 0.4])
    X_df['verification_status'] = np.random.choice(['verified', 'not_verified', 'source_verified'], 
                                                  size=n_samples, p=[0.4, 0.3, 0.3])
    
    return X_df, y_series

print("🔄 Generating synthetic credit data...")

# Generate training data
X_train, y_train = generate_synthetic_credit_data(n_samples=1000, random_state=42)

print(f"✅ Generated synthetic credit dataset:")
print(f"   - Shape: {X_train.shape}")
print(f"   - Features: {list(X_train.columns)}")
print(f"   - Default rate: {y_train.mean():.2%}")
print(f"   - Data types: {X_train.dtypes.value_counts().to_dict()}")

# Display sample data
print("\n📊 Sample data:")
print(X_train.head())
print(f"\nTarget distribution:")
print(y_train.value_counts())