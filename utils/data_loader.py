"""
Data Loader Module for Credit Analytics Hub

This module provides comprehensive data loading functionality for credit analytics,
including data validation, cleaning, and preprocessing utilities.
"""

import pandas as pd
import numpy as np
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings
from datetime import datetime
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """
    Comprehensive data loader for credit analytics with validation and preprocessing
    """

    def __init__(self, data_path: str = "data/", config: Optional[Dict] = None):
        """
        Initialize DataLoader

        Args:
            data_path: Path to data directory
            config: Configuration dictionary for data loading
        """
        self.data_path = Path(data_path)
        self.config = config or {}
        self.loaded_datasets = {}
        self.data_info = {}

    def load_data(self, 
                  file_path: Union[str, Path], 
                  file_type: str = "auto",
                  **kwargs) -> pd.DataFrame:
        """
        Load data from various file formats

        Args:
            file_path: Path to the data file
            file_type: Type of file ('csv', 'excel', 'json', 'parquet', 'auto')
            **kwargs: Additional arguments for pandas read functions

        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            file_path = Path(file_path)

            # Auto-detect file type if not specified
            if file_type == "auto":
                file_type = file_path.suffix.lower().lstrip('.')

            logger.info(f"Loading data from {file_path} (type: {file_type})")

            # Load based on file type
            if file_type in ['csv', 'txt']:
                df = pd.read_csv(file_path, **kwargs)
            elif file_type in ['xlsx', 'xls', 'excel']:
                df = pd.read_excel(file_path, **kwargs)
            elif file_type == 'json':
                df = pd.read_json(file_path, **kwargs)
            elif file_type == 'parquet':
                df = pd.read_parquet(file_path, **kwargs)
            elif file_type == 'pickle':
                with open(file_path, 'rb') as f:
                    df = pickle.load(f)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")

            # Store dataset info
            dataset_name = file_path.stem
            self.loaded_datasets[dataset_name] = df
            self.data_info[dataset_name] = {
                'shape': df.shape,
                'columns': list(df.columns),
                'dtypes': df.dtypes.to_dict(),
                'loaded_at': datetime.now().isoformat(),
                'file_path': str(file_path)
            }

            logger.info(f"Successfully loaded {df.shape[0]} rows and {df.shape[1]} columns")
            return df

        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {str(e)}")
            raise

    def load_credit_data(self, 
                        file_path: Union[str, Path],
                        target_column: str = "default",
                        **kwargs) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load credit data with automatic feature and target separation

        Args:
            file_path: Path to credit data file
            target_column: Name of target column
            **kwargs: Additional arguments for load_data

        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features and target
        """
        try:
            # Load the data
            df = self.load_data(file_path, **kwargs)

            # Separate features and target
            if target_column in df.columns:
                X = df.drop(columns=[target_column])
                y = df[target_column]
                logger.info(f"Separated features ({X.shape[1]} columns) and target")
            else:
                logger.warning(f"Target column '{target_column}' not found. Returning full dataset.")
                X = df
                y = pd.Series(dtype='float64')

            return X, y

        except Exception as e:
            logger.error(f"Error loading credit data: {str(e)}")
            raise

    def load_multiple_datasets(self, 
                             file_paths: Dict[str, str],
                             **kwargs) -> Dict[str, pd.DataFrame]:
        """
        Load multiple datasets at once

        Args:
            file_paths: Dictionary mapping dataset names to file paths
            **kwargs: Additional arguments for load_data

        Returns:
            Dict[str, pd.DataFrame]: Dictionary of loaded datasets
        """
        datasets = {}

        for name, path in file_paths.items():
            try:
                datasets[name] = self.load_data(path, **kwargs)
                logger.info(f"Loaded dataset '{name}' from {path}")
            except Exception as e:
                logger.error(f"Failed to load dataset '{name}' from {path}: {str(e)}")
                continue

        return datasets

    def get_data_info(self, dataset_name: Optional[str] = None) -> Dict:
        """
        Get information about loaded datasets

        Args:
            dataset_name: Specific dataset name, or None for all datasets

        Returns:
            Dict: Dataset information
        """
        if dataset_name:
            return self.data_info.get(dataset_name, {})
        return self.data_info

    def validate_data(self, 
                     df: pd.DataFrame,
                     required_columns: Optional[List[str]] = None,
                     min_rows: int = 1) -> Dict[str, Any]:
        """
        Validate loaded data

        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            min_rows: Minimum number of rows required

        Returns:
            Dict: Validation results
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'info': {}
        }

        try:
            # Check if DataFrame is empty
            if df.empty:
                validation_results['is_valid'] = False
                validation_results['errors'].append("DataFrame is empty")
                return validation_results

            # Check minimum rows
            if len(df) < min_rows:
                validation_results['is_valid'] = False
                validation_results['errors'].append(f"Dataset has {len(df)} rows, minimum required: {min_rows}")

            # Check required columns
            if required_columns:
                missing_columns = set(required_columns) - set(df.columns)
                if missing_columns:
                    validation_results['is_valid'] = False
                    validation_results['errors'].append(f"Missing required columns: {list(missing_columns)}")

            # Check for duplicate columns
            duplicate_columns = df.columns[df.columns.duplicated()].tolist()
            if duplicate_columns:
                validation_results['warnings'].append(f"Duplicate columns found: {duplicate_columns}")

            # Basic data quality checks
            null_counts = df.isnull().sum()
            high_null_columns = null_counts[null_counts > len(df) * 0.5].index.tolist()
            if high_null_columns:
                validation_results['warnings'].append(f"Columns with >50% null values: {high_null_columns}")

            # Store basic info
            validation_results['info'] = {
                'shape': df.shape,
                'dtypes': df.dtypes.value_counts().to_dict(),
                'null_counts': null_counts.to_dict(),
                'memory_usage': df.memory_usage(deep=True).sum()
            }

            logger.info(f"Data validation completed. Valid: {validation_results['is_valid']}")
            return validation_results

        except Exception as e:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Validation error: {str(e)}")
            logger.error(f"Error during data validation: {str(e)}")
            return validation_results

    def clean_data(self, 
                   df: pd.DataFrame,
                   remove_duplicates: bool = True,
                   handle_missing: str = "auto",
                   outlier_method: str = "iqr",
                   outlier_threshold: float = 1.5) -> pd.DataFrame:
        """
        Clean data by handling duplicates, missing values, and outliers

        Args:
            df: DataFrame to clean
            remove_duplicates: Whether to remove duplicate rows
            handle_missing: Method to handle missing values ('drop', 'fill', 'auto')
            outlier_method: Method for outlier detection ('iqr', 'zscore', 'none')
            outlier_threshold: Threshold for outlier detection

        Returns:
            pd.DataFrame: Cleaned data
        """
        try:
            df_cleaned = df.copy()
            cleaning_log = []

            # Remove duplicates
            if remove_duplicates:
                initial_rows = len(df_cleaned)
                df_cleaned = df_cleaned.drop_duplicates()
                removed_duplicates = initial_rows - len(df_cleaned)
                if removed_duplicates > 0:
                    cleaning_log.append(f"Removed {removed_duplicates} duplicate rows")

            # Handle missing values
            if handle_missing != "none":
                missing_counts = df_cleaned.isnull().sum()
                columns_with_missing = missing_counts[missing_counts > 0].index.tolist()

                if columns_with_missing:
                    if handle_missing == "drop":
                        df_cleaned = df_cleaned.dropna()
                        cleaning_log.append(f"Dropped rows with missing values")

                    elif handle_missing == "fill" or handle_missing == "auto":
                        for col in columns_with_missing:
                            if df_cleaned[col].dtype in ['int64', 'float64']:
                                # Fill numeric columns with median
                                fill_value = df_cleaned[col].median()
                                df_cleaned[col].fillna(fill_value, inplace=True)
                                cleaning_log.append(f"Filled missing values in {col} with median: {fill_value}")
                            else:
                                # Fill categorical columns with mode
                                fill_value = df_cleaned[col].mode().iloc[0] if not df_cleaned[col].mode().empty else "Unknown"
                                df_cleaned[col].fillna(fill_value, inplace=True)
                                cleaning_log.append(f"Filled missing values in {col} with mode: {fill_value}")

            # Handle outliers
            if outlier_method != "none":
                numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns

                for col in numeric_columns:
                    if outlier_method == "iqr":
                        Q1 = df_cleaned[col].quantile(0.25)
                        Q3 = df_cleaned[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - outlier_threshold * IQR
                        upper_bound = Q3 + outlier_threshold * IQR

                        outliers_mask = (df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)
                        outliers_count = outliers_mask.sum()

                        if outliers_count > 0:
                            # Cap outliers instead of removing them
                            df_cleaned.loc[df_cleaned[col] < lower_bound, col] = lower_bound
                            df_cleaned.loc[df_cleaned[col] > upper_bound, col] = upper_bound
                            cleaning_log.append(f"Capped {outliers_count} outliers in {col}")

                    elif outlier_method == "zscore":
                        z_scores = np.abs((df_cleaned[col] - df_cleaned[col].mean()) / df_cleaned[col].std())
                        outliers_mask = z_scores > outlier_threshold
                        outliers_count = outliers_mask.sum()

                        if outliers_count > 0:
                            # Cap outliers using percentiles
                            lower_percentile = df_cleaned[col].quantile(0.01)
                            upper_percentile = df_cleaned[col].quantile(0.99)
                            df_cleaned.loc[outliers_mask, col] = np.clip(
                                df_cleaned.loc[outliers_mask, col], 
                                lower_percentile, 
                                upper_percentile
                            )
                            cleaning_log.append(f"Capped {outliers_count} outliers in {col} using z-score")

            # Log cleaning results
            logger.info(f"Data cleaning completed: {'; '.join(cleaning_log)}")
            return df_cleaned

        except Exception as e:
            logger.error(f"Error during data cleaning: {str(e)}")
            raise

    def preprocess_features(self, 
                           df: pd.DataFrame,
                           categorical_encoding: str = "onehot",
                           numerical_scaling: str = "standard",
                           feature_selection: bool = False,
                           correlation_threshold: float = 0.95) -> pd.DataFrame:
        """
        Preprocess features for machine learning

        Args:
            df: DataFrame to preprocess
            categorical_encoding: Method for encoding categorical variables ('onehot', 'label', 'target')
            numerical_scaling: Method for scaling numerical variables ('standard', 'minmax', 'robust', 'none')
            feature_selection: Whether to perform feature selection
            correlation_threshold: Threshold for removing highly correlated features

        Returns:
            pd.DataFrame: Preprocessed data
        """
        try:
            df_processed = df.copy()

            # Separate categorical and numerical columns
            categorical_columns = df_processed.select_dtypes(include=['object', 'category']).columns.tolist()
            numerical_columns = df_processed.select_dtypes(include=[np.number]).columns.tolist()

            logger.info(f"Found {len(categorical_columns)} categorical and {len(numerical_columns)} numerical columns")

            # Handle categorical variables
            if categorical_columns and categorical_encoding != "none":
                if categorical_encoding == "onehot":
                    # One-hot encoding
                    df_encoded = pd.get_dummies(df_processed[categorical_columns], prefix=categorical_columns)
                    df_processed = df_processed.drop(columns=categorical_columns)
                    df_processed = pd.concat([df_processed, df_encoded], axis=1)
                    logger.info(f"Applied one-hot encoding to {len(categorical_columns)} categorical columns")

                elif categorical_encoding == "label":
                    # Label encoding
                    from sklearn.preprocessing import LabelEncoder
                    label_encoders = {}

                    for col in categorical_columns:
                        le = LabelEncoder()
                        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                        label_encoders[col] = le

                    logger.info(f"Applied label encoding to {len(categorical_columns)} categorical columns")

            # Handle numerical variables
            if numerical_columns and numerical_scaling != "none":
                if numerical_scaling == "standard":
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    df_processed[numerical_columns] = scaler.fit_transform(df_processed[numerical_columns])
                    logger.info("Applied standard scaling to numerical columns")

                elif numerical_scaling == "minmax":
                    from sklearn.preprocessing import MinMaxScaler
                    scaler = MinMaxScaler()
                    df_processed[numerical_columns] = scaler.fit_transform(df_processed[numerical_columns])
                    logger.info("Applied min-max scaling to numerical columns")

                elif numerical_scaling == "robust":
                    from sklearn.preprocessing import RobustScaler
                    scaler = RobustScaler()
                    df_processed[numerical_columns] = scaler.fit_transform(df_processed[numerical_columns])
                    logger.info("Applied robust scaling to numerical columns")

            # Feature selection based on correlation
            if feature_selection and len(df_processed.columns) > 1:
                # Calculate correlation matrix
                corr_matrix = df_processed.corr().abs()

                # Find highly correlated features
                upper_triangle = corr_matrix.where(
                    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
                )

                # Find features to drop
                to_drop = [column for column in upper_triangle.columns 
                          if any(upper_triangle[column] > correlation_threshold)]

                if to_drop:
                    df_processed = df_processed.drop(columns=to_drop)
                    logger.info(f"Removed {len(to_drop)} highly correlated features: {to_drop}")

            logger.info(f"Feature preprocessing completed. Final shape: {df_processed.shape}")
            return df_processed

        except Exception as e:
            logger.error(f"Error during feature preprocessing: {str(e)}")
            raise

    def create_feature_engineering(self, 
                                  df: pd.DataFrame,
                                  date_columns: Optional[List[str]] = None,
                                  create_interactions: bool = False,
                                  polynomial_features: bool = False,
                                  polynomial_degree: int = 2) -> pd.DataFrame:
        """
        Create engineered features

        Args:
            df: DataFrame to engineer features for
            date_columns: List of date columns to extract features from
            create_interactions: Whether to create interaction features
            polynomial_features: Whether to create polynomial features
            polynomial_degree: Degree for polynomial features

        Returns:
            pd.DataFrame: DataFrame with engineered features
        """
        try:
            df_engineered = df.copy()

            # Date feature engineering
            if date_columns:
                for col in date_columns:
                    if col in df_engineered.columns:
                        # Convert to datetime if not already
                        df_engineered[col] = pd.to_datetime(df_engineered[col], errors='coerce')

                        # Extract date components
                        df_engineered[f'{col}_year'] = df_engineered[col].dt.year
                        df_engineered[f'{col}_month'] = df_engineered[col].dt.month
                        df_engineered[f'{col}_day'] = df_engineered[col].dt.day
                        df_engineered[f'{col}_dayofweek'] = df_engineered[col].dt.dayofweek
                        df_engineered[f'{col}_quarter'] = df_engineered[col].dt.quarter

                        # Calculate days since epoch
                        df_engineered[f'{col}_days_since_epoch'] = (
                            df_engineered[col] - pd.Timestamp('1970-01-01')
                        ).dt.days

                        logger.info(f"Created date features for column: {col}")

            # Interaction features (for numerical columns only)
            if create_interactions:
                numerical_columns = df_engineered.select_dtypes(include=[np.number]).columns.tolist()

                if len(numerical_columns) >= 2:
                    from itertools import combinations

                    # Create pairwise interactions (limited to avoid explosion)
                    interaction_pairs = list(combinations(numerical_columns[:10], 2))  # Limit to first 10 columns

                    for col1, col2 in interaction_pairs:
                        interaction_name = f'{col1}_x_{col2}'
                        df_engineered[interaction_name] = df_engineered[col1] * df_engineered[col2]

                    logger.info(f"Created {len(interaction_pairs)} interaction features")

            # Polynomial features
            if polynomial_features:
                from sklearn.preprocessing import PolynomialFeatures

                numerical_columns = df_engineered.select_dtypes(include=[np.number]).columns.tolist()

                if numerical_columns:
                    # Limit to prevent feature explosion
                    selected_columns = numerical_columns[:5]  # Limit to first 5 numerical columns

                    poly = PolynomialFeatures(degree=polynomial_degree, include_bias=False)
                    poly_features = poly.fit_transform(df_engineered[selected_columns])

                    # Get feature names
                    feature_names = poly.get_feature_names_out(selected_columns)

                    # Create DataFrame with polynomial features
                    poly_df = pd.DataFrame(poly_features, columns=feature_names, index=df_engineered.index)

                    # Remove original columns to avoid duplication
                    poly_df = poly_df.drop(columns=selected_columns)

                    # Concatenate with original data
                    df_engineered = pd.concat([df_engineered, poly_df], axis=1)

                    logger.info(f"Created polynomial features of degree {polynomial_degree}")

            logger.info(f"Feature engineering completed. Final shape: {df_engineered.shape}")
            return df_engineered

        except Exception as e:
            logger.error(f"Error during feature engineering: {str(e)}")
            raise

    def split_data(self, 
                   X: pd.DataFrame, 
                   y: pd.Series,
                   test_size: float = 0.2,
                   validation_size: float = 0.1,
                   random_state: int = 42,
                   stratify: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Split data into train, validation, and test sets

        Args:
            X: Features DataFrame
            y: Target Series
            test_size: Proportion of data for test set
            validation_size: Proportion of data for validation set
            random_state: Random state for reproducibility
            stratify: Whether to stratify the split

        Returns:
            Tuple: X_train, X_val, X_test, y_train, y_val, y_test
        """
        try:
            from sklearn.model_selection import train_test_split

            # First split: separate test set
            stratify_param = y if stratify and len(y.unique()) > 1 else None

            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, 
                test_size=test_size, 
                random_state=random_state,
                stratify=stratify_param
            )

            # Second split: separate train and validation sets
            val_size_adjusted = validation_size / (1 - test_size)
            stratify_param_temp = y_temp if stratify and len(y_temp.unique()) > 1 else None

            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp,
                test_size=val_size_adjusted,
                random_state=random_state,
                stratify=stratify_param_temp
            )

            logger.info(f"Data split completed:")
            logger.info(f"  Train: {X_train.shape[0]} samples")
            logger.info(f"  Validation: {X_val.shape[0]} samples")
            logger.info(f"  Test: {X_test.shape[0]} samples")

            return X_train, X_val, X_test, y_train, y_val, y_test

        except Exception as e:
            logger.error(f"Error during data splitting: {str(e)}")
            raise

    def save_processed_data(self, 
                           data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                           output_path: str,
                           format: str = "csv") -> None:
        """
        Save processed data to file

        Args:
            data: Data to save (DataFrame or dict of DataFrames)
            output_path: Output file path
            format: Output format ('csv', 'parquet', 'pickle')
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if isinstance(data, dict):
                # Save multiple datasets
                for name, df in data.items():
                    file_path = output_path.parent / f"{output_path.stem}_{name}.{format}"
                    self._save_single_dataset(df, file_path, format)
            else:
                # Save single dataset
                self._save_single_dataset(data, output_path, format)

            logger.info(f"Data saved successfully to {output_path}")

        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            raise

    def _save_single_dataset(self, df: pd.DataFrame, file_path: Path, format: str) -> None:
        """Helper method to save a single dataset"""
        if format == "csv":
            df.to_csv(file_path, index=False)
        elif format == "parquet":
            df.to_parquet(file_path, index=False)
        elif format == "pickle":
            with open(file_path, 'wb') as f:
                pickle.dump(df, f)
        else:
            raise ValueError(f"Unsupported format: {format}")


# Convenience functions for direct use
def load_data(file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """Convenience function to load data"""
    loader = DataLoader()
    return loader.load_data(file_path, **kwargs)

def load_credit_data(file_path: Union[str, Path], target_column: str = "default", **kwargs) -> Tuple[pd.DataFrame, pd.Series]:
    """Convenience function to load credit data"""
    loader = DataLoader()
    return loader.load_credit_data(file_path, target_column, **kwargs)

def clean_data(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Convenience function to clean data"""
    loader = DataLoader()
    return loader.clean_data(df, **kwargs)

def preprocess_features(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Convenience function to preprocess features"""
    loader = DataLoader()
    return loader.preprocess_features(df, **kwargs)

def validate_data(df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    """Convenience function to validate data"""
    loader = DataLoader()
    return loader.validate_data(df, **kwargs)
