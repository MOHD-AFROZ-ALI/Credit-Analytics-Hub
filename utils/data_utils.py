"""
Data Utilities Module for Credit Analytics Hub

This module provides comprehensive data processing utilities for credit analytics,
including data loading, cleaning, transformation, validation, and feature engineering.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import logging
from datetime import datetime, timedelta
import warnings
from pathlib import Path
import json
import pickle
from dataclasses import dataclass
from enum import Enum
import re

# Statistical and ML imports
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataQualityLevel(Enum):
    """Enumeration for data quality levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"

@dataclass
class DataQualityReport:
    """Data class for storing data quality assessment results."""
    overall_score: float
    quality_level: DataQualityLevel
    total_records: int
    total_features: int
    missing_data_percentage: float
    duplicate_records: int
    outlier_percentage: float
    data_types_issues: int
    inconsistency_issues: int
    recommendations: List[str]
    detailed_metrics: Dict[str, Any]
    timestamp: datetime

class DataLoader:
    """
    Comprehensive data loading utility for various file formats.
    
    Supports CSV, Excel, JSON, Parquet, and database connections
    with automatic data type inference and validation.
    """
    
    def __init__(self, 
                 default_encoding: str = 'utf-8',
                 chunk_size: Optional[int] = None,
                 low_memory: bool = False):
        """
        Initialize the data loader.
        
        Args:
            default_encoding: Default encoding for text files
            chunk_size: Size of chunks for large file processing
            low_memory: Whether to use low memory mode
        """
        self.default_encoding = default_encoding
        self.chunk_size = chunk_size
        self.low_memory = low_memory
        self.supported_formats = ['.csv', '.xlsx', '.xls', '.json', '.parquet', '.pkl']
        
    def load_data(self, 
                  file_path: str,
                  file_type: Optional[str] = None,
                  **kwargs) -> pd.DataFrame:
        """
        Load data from various file formats.
        
        Args:
            file_path: Path to the data file
            file_type: Explicit file type (auto-detected if None)
            **kwargs: Additional arguments for specific loaders
            
        Returns:
            Loaded DataFrame
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Auto-detect file type if not provided
        if file_type is None:
            file_type = file_path.suffix.lower()
        
        if file_type not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_type}")
        
        logger.info(f"Loading data from {file_path} (format: {file_type})")
        
        try:
            if file_type == '.csv':
                return self._load_csv(file_path, **kwargs)
            elif file_type in ['.xlsx', '.xls']:
                return self._load_excel(file_path, **kwargs)
            elif file_type == '.json':
                return self._load_json(file_path, **kwargs)
            elif file_type == '.parquet':
                return self._load_parquet(file_path, **kwargs)
            elif file_type == '.pkl':
                return self._load_pickle(file_path, **kwargs)
            else:
                raise ValueError(f"Handler not implemented for {file_type}")
                
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {e}")
            raise
    
    def _load_csv(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Load CSV file with automatic delimiter detection."""
        default_params = {
            'encoding': self.default_encoding,
            'low_memory': self.low_memory
        }
        
        if self.chunk_size:
            default_params['chunksize'] = self.chunk_size
        
        # Merge with user parameters
        params = {**default_params, **kwargs}
        
        # Try to detect delimiter if not specified
        if 'sep' not in params and 'delimiter' not in params:
            with open(file_path, 'r', encoding=params['encoding']) as f:
                first_line = f.readline()
                if ',' in first_line:
                    params['sep'] = ','
                elif ';' in first_line:
                    params['sep'] = ';'
                elif '\\t' in first_line:
                    params['sep'] = '\\t'
        
        return pd.read_csv(file_path, **params)
    
    def _load_excel(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Load Excel file."""
        return pd.read_excel(file_path, **kwargs)
    
    def _load_json(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Load JSON file."""
        default_params = {'orient': 'records'}
        params = {**default_params, **kwargs}
        return pd.read_json(file_path, **params)
    
    def _load_parquet(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Load Parquet file."""
        return pd.read_parquet(file_path, **kwargs)
    
    def _load_pickle(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Load pickled DataFrame."""
        return pd.read_pickle(file_path, **kwargs)
    
    def load_multiple_files(self, 
                           file_paths: List[str],
                           combine_method: str = 'concat',
                           **kwargs) -> pd.DataFrame:
        """
        Load and combine multiple files.
        
        Args:
            file_paths: List of file paths to load
            combine_method: Method to combine files ('concat', 'merge')
            **kwargs: Additional arguments for combining
            
        Returns:
            Combined DataFrame
        """
        dataframes = []
        
        for file_path in file_paths:
            try:
                df = self.load_data(file_path, **kwargs)
                dataframes.append(df)
                logger.info(f"Loaded {len(df)} records from {file_path}")
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")
                continue
        
        if not dataframes:
            raise ValueError("No files were successfully loaded")
        
        if combine_method == 'concat':
            combined_df = pd.concat(dataframes, ignore_index=True)
        elif combine_method == 'merge':
            combined_df = dataframes[0]
            for df in dataframes[1:]:
                combined_df = pd.merge(combined_df, df, **kwargs)
        else:
            raise ValueError(f"Unknown combine method: {combine_method}")
        
        logger.info(f"Combined {len(dataframes)} files into {len(combined_df)} records")
        return combined_df
    
    def save_data(self, 
                  df: pd.DataFrame,
                  file_path: str,
                  file_type: Optional[str] = None,
                  **kwargs) -> None:
        """
        Save DataFrame to various formats.
        
        Args:
            df: DataFrame to save
            file_path: Output file path
            file_type: File format (auto-detected if None)
            **kwargs: Additional arguments for specific savers
        """
        file_path = Path(file_path)
        
        # Create directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Auto-detect file type if not provided
        if file_type is None:
            file_type = file_path.suffix.lower()
        
        logger.info(f"Saving {len(df)} records to {file_path} (format: {file_type})")
        
        try:
            if file_type == '.csv':
                df.to_csv(file_path, index=False, **kwargs)
            elif file_type in ['.xlsx', '.xls']:
                df.to_excel(file_path, index=False, **kwargs)
            elif file_type == '.json':
                df.to_json(file_path, orient='records', **kwargs)
            elif file_type == '.parquet':
                df.to_parquet(file_path, **kwargs)
            elif file_type == '.pkl':
                df.to_pickle(file_path, **kwargs)
            else:
                raise ValueError(f"Unsupported save format: {file_type}")
                
            logger.info(f"Data saved successfully to {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving data to {file_path}: {e}")
            raise

class DataCleaner:
    """
    Comprehensive data cleaning utility for credit analytics.
    
    Handles missing values, duplicates, outliers, data type conversions,
    and inconsistency detection with configurable strategies.
    """
    
    def __init__(self, 
                 missing_threshold: float = 0.5,
                 outlier_method: str = 'iqr',
                 outlier_threshold: float = 1.5):
        """
        Initialize the data cleaner.
        
        Args:
            missing_threshold: Threshold for dropping columns with missing values
            outlier_method: Method for outlier detection ('iqr', 'zscore', 'isolation')
            outlier_threshold: Threshold for outlier detection
        """
        self.missing_threshold = missing_threshold
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.cleaning_log = []
        
    def clean_data(self, 
                   df: pd.DataFrame,
                   target_column: Optional[str] = None,
                   preserve_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Perform comprehensive data cleaning.
        
        Args:
            df: Input DataFrame
            target_column: Name of target column to preserve
            preserve_columns: List of columns to preserve from dropping
            
        Returns:
            Cleaned DataFrame
        """
        df_cleaned = df.copy()
        preserve_columns = preserve_columns or []
        
        if target_column:
            preserve_columns.append(target_column)
        
        logger.info(f"Starting data cleaning for {len(df_cleaned)} records, {len(df_cleaned.columns)} columns")
        
        # Step 1: Handle duplicates
        df_cleaned = self._remove_duplicates(df_cleaned)
        
        # Step 2: Handle missing values
        df_cleaned = self._handle_missing_values(df_cleaned, preserve_columns)
        
        # Step 3: Fix data types
        df_cleaned = self._fix_data_types(df_cleaned)
        
        # Step 4: Handle outliers
        df_cleaned = self._handle_outliers(df_cleaned, preserve_columns)
        
        # Step 5: Standardize text data
        df_cleaned = self._standardize_text_data(df_cleaned)
        
        # Step 6: Validate data consistency
        df_cleaned = self._validate_data_consistency(df_cleaned)
        
        logger.info(f"Data cleaning completed: {len(df_cleaned)} records, {len(df_cleaned.columns)} columns")
        
        return df_cleaned
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate records."""
        initial_count = len(df)
        df_cleaned = df.drop_duplicates()
        duplicates_removed = initial_count - len(df_cleaned)
        
        if duplicates_removed > 0:
            self.cleaning_log.append(f"Removed {duplicates_removed} duplicate records")
            logger.info(f"Removed {duplicates_removed} duplicate records")
        
        return df_cleaned
    
    def _handle_missing_values(self, 
                              df: pd.DataFrame, 
                              preserve_columns: List[str]) -> pd.DataFrame:
        """Handle missing values with various strategies."""
        df_cleaned = df.copy()
        
        # Calculate missing percentages
        missing_percentages = df_cleaned.isnull().sum() / len(df_cleaned)
        
        # Drop columns with too many missing values (except preserved columns)
        columns_to_drop = []
        for col in df_cleaned.columns:
            if (missing_percentages[col] > self.missing_threshold and 
                col not in preserve_columns):
                columns_to_drop.append(col)
        
        if columns_to_drop:
            df_cleaned = df_cleaned.drop(columns=columns_to_drop)
            self.cleaning_log.append(f"Dropped {len(columns_to_drop)} columns with >{self.missing_threshold*100}% missing values")
            logger.info(f"Dropped columns with high missing values: {columns_to_drop}")
        
        # Handle remaining missing values
        for col in df_cleaned.columns:
            if df_cleaned[col].isnull().sum() > 0:
                if df_cleaned[col].dtype in ['object', 'category']:
                    # For categorical data, use mode or 'Unknown'
                    mode_value = df_cleaned[col].mode()
                    fill_value = mode_value[0] if len(mode_value) > 0 else 'Unknown'
                    df_cleaned[col] = df_cleaned[col].fillna(fill_value)
                else:
                    # For numerical data, use median
                    median_value = df_cleaned[col].median()
                    df_cleaned[col] = df_cleaned[col].fillna(median_value)
                
                self.cleaning_log.append(f"Filled missing values in column '{col}'")
        
        return df_cleaned
    
    def _fix_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix and optimize data types."""
        df_cleaned = df.copy()
        
        for col in df_cleaned.columns:
            # Try to convert string numbers to numeric
            if df_cleaned[col].dtype == 'object':
                # Check if it's numeric
                try:
                    # Remove common non-numeric characters
                    cleaned_series = df_cleaned[col].astype(str).str.replace('[,$%]', '', regex=True)
                    numeric_series = pd.to_numeric(cleaned_series, errors='coerce')
                    
                    # If most values are numeric, convert the column
                    if numeric_series.notna().sum() / len(numeric_series) > 0.8:
                        df_cleaned[col] = numeric_series
                        self.cleaning_log.append(f"Converted column '{col}' to numeric")
                except:
                    pass
                
                # Check if it's datetime
                try:
                    datetime_series = pd.to_datetime(df_cleaned[col], errors='coerce')
                    if datetime_series.notna().sum() / len(datetime_series) > 0.8:
                        df_cleaned[col] = datetime_series
                        self.cleaning_log.append(f"Converted column '{col}' to datetime")
                except:
                    pass
        
        # Optimize numeric types
        for col in df_cleaned.select_dtypes(include=[np.number]).columns:
            if df_cleaned[col].dtype == 'float64':
                if df_cleaned[col].isnull().sum() == 0 and df_cleaned[col] % 1 == 0:
                    # Convert to integer if no decimals
                    df_cleaned[col] = df_cleaned[col].astype('int64')
                else:
                    # Use float32 if possible
                    if df_cleaned[col].min() >= np.finfo(np.float32).min and df_cleaned[col].max() <= np.finfo(np.float32).max:
                        df_cleaned[col] = df_cleaned[col].astype('float32')
        
        return df_cleaned
    
    def _handle_outliers(self, 
                        df: pd.DataFrame, 
                        preserve_columns: List[str]) -> pd.DataFrame:
        """Handle outliers using specified method."""
        df_cleaned = df.copy()
        numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns
        
        outliers_removed = 0
        
        for col in numeric_columns:
            if col in preserve_columns:
                continue
                
            if self.outlier_method == 'iqr':
                Q1 = df_cleaned[col].quantile(0.25)
                Q3 = df_cleaned[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - self.outlier_threshold * IQR
                upper_bound = Q3 + self.outlier_threshold * IQR
                
                outlier_mask = (df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)
                
            elif self.outlier_method == 'zscore':
                z_scores = np.abs(stats.zscore(df_cleaned[col].dropna()))
                outlier_mask = z_scores > self.outlier_threshold
                
            elif self.outlier_method == 'isolation':
                isolation_forest = IsolationForest(contamination=0.1, random_state=42)
                outlier_mask = isolation_forest.fit_predict(df_cleaned[[col]].dropna()) == -1
            
            else:
                continue
            
            # Cap outliers instead of removing them
            if outlier_mask.sum() > 0:
                if self.outlier_method == 'iqr':
                    df_cleaned.loc[df_cleaned[col] < lower_bound, col] = lower_bound
                    df_cleaned.loc[df_cleaned[col] > upper_bound, col] = upper_bound
                else:
                    # For other methods, use percentile capping
                    lower_cap = df_cleaned[col].quantile(0.01)
                    upper_cap = df_cleaned[col].quantile(0.99)
                    df_cleaned.loc[df_cleaned[col] < lower_cap, col] = lower_cap
                    df_cleaned.loc[df_cleaned[col] > upper_cap, col] = upper_cap
                
                outliers_removed += outlier_mask.sum()
                self.cleaning_log.append(f"Capped {outlier_mask.sum()} outliers in column '{col}'")
        
        if outliers_removed > 0:
            logger.info(f"Handled {outliers_removed} outliers across numeric columns")
        
        return df_cleaned
    
    def _standardize_text_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize text data."""
        df_cleaned = df.copy()
        text_columns = df_cleaned.select_dtypes(include=['object']).columns
        
        for col in text_columns:
            # Convert to lowercase and strip whitespace
            df_cleaned[col] = df_cleaned[col].astype(str).str.lower().str.strip()
            
            # Remove extra whitespace
            df_cleaned[col] = df_cleaned[col].str.replace(r'\\s+', ' ', regex=True)
            
            # Standardize common variations
            standardizations = {
                r'\\b(yes|y|true|1)\\b': 'yes',
                r'\\b(no|n|false|0)\\b': 'no',
                r'\\b(male|m)\\b': 'male',
                r'\\b(female|f)\\b': 'female',
                r'\\b(unknown|unk|n/a|na|null|none)\\b': 'unknown'
            }
            
            for pattern, replacement in standardizations.items():
                df_cleaned[col] = df_cleaned[col].str.replace(pattern, replacement, regex=True)
        
        return df_cleaned
    
    def _validate_data_consistency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and fix data consistency issues."""
        df_cleaned = df.copy()
        
        # Check for negative values in columns that should be positive
        positive_columns = ['income', 'amount', 'balance', 'age', 'years', 'count']
        
        for col in df_cleaned.columns:
            if any(keyword in col.lower() for keyword in positive_columns):
                if df_cleaned[col].dtype in [np.number] and (df_cleaned[col] < 0).any():
                    negative_count = (df_cleaned[col] < 0).sum()
                    df_cleaned.loc[df_cleaned[col] < 0, col] = 0
                    self.cleaning_log.append(f"Fixed {negative_count} negative values in '{col}'")
        
        return df_cleaned
    
    def get_cleaning_report(self) -> Dict[str, Any]:
        """Get detailed cleaning report."""
        return {
            'cleaning_steps': self.cleaning_log,
            'timestamp': datetime.now(),
            'parameters': {
                'missing_threshold': self.missing_threshold,
                'outlier_method': self.outlier_method,
                'outlier_threshold': self.outlier_threshold
            }
        }

class DataValidator:
    """
    Data validation utility for credit analytics.
    
    Performs comprehensive data quality assessment, validation checks,
    and generates detailed quality reports.
    """
    
    def __init__(self):
        """Initialize the data validator."""
        self.validation_rules = {}
        self.quality_thresholds = {
            'missing_data': 0.1,      # 10% missing data threshold
            'duplicate_records': 0.05,  # 5% duplicate threshold
            'outlier_percentage': 0.1,  # 10% outlier threshold
            'data_consistency': 0.95    # 95% consistency threshold
        }
    
    def validate_data(self, 
                     df: pd.DataFrame,
                     target_column: Optional[str] = None) -> DataQualityReport:
        """
        Perform comprehensive data validation.
        
        Args:
            df: DataFrame to validate
            target_column: Name of target column
            
        Returns:
            Data quality report
        """
        logger.info(f"Starting data validation for {len(df)} records, {len(df.columns)} columns")
        
        # Initialize metrics
        metrics = {}
        recommendations = []
        issues_count = 0
        
        # Basic statistics
        metrics['total_records'] = len(df)
        metrics['total_features'] = len(df.columns)
        
        # Missing data analysis
        missing_analysis = self._analyze_missing_data(df)
        metrics['missing_data'] = missing_analysis
        missing_percentage = missing_analysis['overall_missing_percentage']
        
        if missing_percentage > self.quality_thresholds['missing_data']:
            recommendations.append(f"High missing data rate ({missing_percentage:.1%}). Consider data imputation or collection improvement.")
            issues_count += 1
        
        # Duplicate analysis
        duplicate_analysis = self._analyze_duplicates(df)
        metrics['duplicates'] = duplicate_analysis
        duplicate_percentage = duplicate_analysis['duplicate_percentage']
        
        if duplicate_percentage > self.quality_thresholds['duplicate_records']:
            recommendations.append(f"High duplicate rate ({duplicate_percentage:.1%}). Remove duplicate records.")
            issues_count += 1
        
        # Outlier analysis
        outlier_analysis = self._analyze_outliers(df)
        metrics['outliers'] = outlier_analysis
        outlier_percentage = outlier_analysis['overall_outlier_percentage']
        
        if outlier_percentage > self.quality_thresholds['outlier_percentage']:
            recommendations.append(f"High outlier rate ({outlier_percentage:.1%}). Review outlier handling strategy.")
            issues_count += 1
        
        # Data type analysis
        dtype_analysis = self._analyze_data_types(df)
        metrics['data_types'] = dtype_analysis
        
        if dtype_analysis['type_inconsistencies'] > 0:
            recommendations.append("Data type inconsistencies detected. Review and fix data types.")
            issues_count += 1
        
        # Data consistency analysis
        consistency_analysis = self._analyze_data_consistency(df)
        metrics['consistency'] = consistency_analysis
        
        if consistency_analysis['consistency_score'] < self.quality_thresholds['data_consistency']:
            recommendations.append("Data consistency issues detected. Review data validation rules.")
            issues_count += 1
        
        # Target variable analysis (if provided)
        if target_column and target_column in df.columns:
            target_analysis = self._analyze_target_variable(df, target_column)
            metrics['target_variable'] = target_analysis
            
            if target_analysis['class_imbalance'] > 0.8:
                recommendations.append("Severe class imbalance detected. Consider resampling techniques.")
        
        # Calculate overall quality score
        quality_score = self._calculate_quality_score(metrics)
        quality_level = self._determine_quality_level(quality_score)
        
        # Generate general recommendations
        if not recommendations:
            recommendations.append("Data quality is good. No major issues detected.")
        
        # Create quality report
        report = DataQualityReport(
            overall_score=quality_score,
            quality_level=quality_level,
            total_records=metrics['total_records'],
            total_features=metrics['total_features'],
            missing_data_percentage=missing_percentage,
            duplicate_records=duplicate_analysis['duplicate_count'],
            outlier_percentage=outlier_percentage,
            data_types_issues=dtype_analysis['type_inconsistencies'],
            inconsistency_issues=issues_count,
            recommendations=recommendations,
            detailed_metrics=metrics,
            timestamp=datetime.now()
        )
        
        logger.info(f"Data validation completed. Quality level: {quality_level.value}")
        
        return report
    def _analyze_missing_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing data patterns."""
        missing_counts = df.isnull().sum()
        missing_percentages = missing_counts / len(df)
        
        return {
            'missing_counts': missing_counts.to_dict(),
            'missing_percentages': missing_percentages.to_dict(),
            'columns_with_missing': missing_counts[missing_counts > 0].index.tolist(),
            'overall_missing_percentage': missing_percentages.mean(),
            'worst_column': missing_percentages.idxmax() if missing_percentages.max() > 0 else None,
            'worst_column_percentage': missing_percentages.max()
        }
    
    def _analyze_duplicates(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze duplicate records."""
        duplicate_count = df.duplicated().sum()
        duplicate_percentage = duplicate_count / len(df)
        
        return {
            'duplicate_count': duplicate_count,
            'duplicate_percentage': duplicate_percentage,
            'unique_records': len(df) - duplicate_count
        }
    
    def _analyze_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze outliers in numeric columns."""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        outlier_info = {}
        total_outliers = 0
        
        for col in numeric_columns:
            if df[col].notna().sum() > 0:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                outlier_percentage = outliers / df[col].notna().sum()
                
                outlier_info[col] = {
                    'outlier_count': outliers,
                    'outlier_percentage': outlier_percentage,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                }
                
                total_outliers += outliers
        
        overall_outlier_percentage = total_outliers / (len(df) * len(numeric_columns)) if len(numeric_columns) > 0 else 0
        
        return {
            'column_outliers': outlier_info,
            'total_outliers': total_outliers,
            'overall_outlier_percentage': overall_outlier_percentage,
            'numeric_columns_count': len(numeric_columns)
        }
    
    def _analyze_data_types(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data types and detect inconsistencies."""
        dtype_counts = df.dtypes.value_counts().to_dict()
        type_inconsistencies = 0
        
        # Check for potential type issues
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if numeric data is stored as string
                try:
                    numeric_conversion = pd.to_numeric(df[col], errors='coerce')
                    if numeric_conversion.notna().sum() / len(df[col]) > 0.8:
                        type_inconsistencies += 1
                except:
                    pass
        
        return {
            'dtype_distribution': dtype_counts,
            'type_inconsistencies': type_inconsistencies,
            'object_columns': df.select_dtypes(include=['object']).columns.tolist(),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'datetime_columns': df.select_dtypes(include=['datetime64']).columns.tolist()
        }
    
    def _analyze_data_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data consistency."""
        consistency_issues = 0
        total_checks = 0
        
        # Check for negative values in columns that should be positive
        positive_columns = ['income', 'amount', 'balance', 'age', 'years', 'count']
        for col in df.columns:
            if any(keyword in col.lower() for keyword in positive_columns):
                if df[col].dtype in [np.number]:
                    total_checks += 1
                    if (df[col] < 0).any():
                        consistency_issues += 1
        
        # Check for reasonable ranges
        range_checks = {
            'age': (0, 120),
            'income': (0, 10000000),
            'score': (0, 1000)
        }
        
        for col in df.columns:
            for keyword, (min_val, max_val) in range_checks.items():
                if keyword in col.lower() and df[col].dtype in [np.number]:
                    total_checks += 1
                    if (df[col] < min_val).any() or (df[col] > max_val).any():
                        consistency_issues += 1
        
        consistency_score = 1 - (consistency_issues / total_checks) if total_checks > 0 else 1
        
        return {
            'consistency_issues': consistency_issues,
            'total_checks': total_checks,
            'consistency_score': consistency_score
        }
    
    def _analyze_target_variable(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Analyze target variable characteristics."""
        target_series = df[target_column]
        
        # Basic statistics
        value_counts = target_series.value_counts()
        class_distribution = value_counts / len(target_series)
        
        # Calculate class imbalance
        if len(value_counts) == 2:
            class_imbalance = abs(class_distribution.iloc[0] - class_distribution.iloc[1])
        else:
            class_imbalance = 1 - (value_counts.min() / value_counts.max())
        
        return {
            'value_counts': value_counts.to_dict(),
            'class_distribution': class_distribution.to_dict(),
            'class_imbalance': class_imbalance,
            'unique_values': target_series.nunique(),
            'missing_values': target_series.isnull().sum()
        }
    
    def _calculate_quality_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall data quality score."""
        scores = []
        
        # Missing data score (0-1, higher is better)
        missing_score = 1 - min(metrics['missing_data']['overall_missing_percentage'], 1)
        scores.append(missing_score * 0.25)
        
        # Duplicate score (0-1, higher is better)
        duplicate_score = 1 - min(metrics['duplicates']['duplicate_percentage'], 1)
        scores.append(duplicate_score * 0.20)
        
        # Outlier score (0-1, higher is better)
        outlier_score = 1 - min(metrics['outliers']['overall_outlier_percentage'], 1)
        scores.append(outlier_score * 0.20)
        
        # Data type consistency score (0-1, higher is better)
        total_columns = metrics['total_features']
        type_score = 1 - (metrics['data_types']['type_inconsistencies'] / total_columns) if total_columns > 0 else 1
        scores.append(type_score * 0.15)
        
        # Data consistency score (0-1, higher is better)
        consistency_score = metrics['consistency']['consistency_score']
        scores.append(consistency_score * 0.20)
        
        return sum(scores)
    
    def _determine_quality_level(self, score: float) -> DataQualityLevel:
        """Determine quality level based on score."""
        if score >= 0.9:
            return DataQualityLevel.EXCELLENT
        elif score >= 0.8:
            return DataQualityLevel.GOOD
        elif score >= 0.6:
            return DataQualityLevel.FAIR
        elif score >= 0.4:
            return DataQualityLevel.POOR
        else:
            return DataQualityLevel.CRITICAL

class FeatureEngineer:
    """
    Advanced feature engineering utility for credit analytics.
    
    Provides comprehensive feature creation, transformation, selection,
    and engineering specifically designed for credit risk modeling.
    """
    
    def __init__(self, 
                 target_column: Optional[str] = None,
                 feature_selection_method: str = 'mutual_info',
                 max_features: Optional[int] = None):
        """
        Initialize the feature engineer.
        
        Args:
            target_column: Name of target column for supervised feature selection
            feature_selection_method: Method for feature selection
            max_features: Maximum number of features to select
        """
        self.target_column = target_column
        self.feature_selection_method = feature_selection_method
        self.max_features = max_features
        self.created_features = []
        self.selected_features = []
        self.feature_importance_scores = {}
        
    def engineer_features(self, 
                         df: pd.DataFrame,
                         create_interactions: bool = True,
                         create_ratios: bool = True,
                         create_aggregations: bool = True,
                         create_binning: bool = True) -> pd.DataFrame:
        """
        Perform comprehensive feature engineering.
        
        Args:
            df: Input DataFrame
            create_interactions: Whether to create interaction features
            create_ratios: Whether to create ratio features
            create_aggregations: Whether to create aggregation features
            create_binning: Whether to create binned features
            
        Returns:
            DataFrame with engineered features
        """
        df_engineered = df.copy()
        
        logger.info(f"Starting feature engineering for {len(df_engineered)} records, {len(df_engineered.columns)} columns")
        
        # Create credit-specific features
        df_engineered = self._create_credit_features(df_engineered)
        
        # Create interaction features
        if create_interactions:
            df_engineered = self._create_interaction_features(df_engineered)
        
        # Create ratio features
        if create_ratios:
            df_engineered = self._create_ratio_features(df_engineered)
        
        # Create aggregation features
        if create_aggregations:
            df_engineered = self._create_aggregation_features(df_engineered)
        
        # Create binned features
        if create_binning:
            df_engineered = self._create_binned_features(df_engineered)
        
        # Create polynomial features for key variables
        df_engineered = self._create_polynomial_features(df_engineered)
        
        # Create time-based features if datetime columns exist
        df_engineered = self._create_time_features(df_engineered)
        
        logger.info(f"Feature engineering completed: {len(df_engineered.columns)} total features")
        logger.info(f"Created {len(self.created_features)} new features")
        
        return df_engineered
    
    def _create_credit_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create credit-specific features."""
        df_new = df.copy()
        
        # Debt-to-income ratio
        if 'annual_income' in df_new.columns and 'loan_amount' in df_new.columns:
            df_new['debt_to_income_ratio'] = df_new['loan_amount'] / (df_new['annual_income'] + 1e-6)
            self.created_features.append('debt_to_income_ratio')
        
        # Credit utilization
        if 'credit_limit' in df_new.columns and 'current_balance' in df_new.columns:
            df_new['credit_utilization'] = df_new['current_balance'] / (df_new['credit_limit'] + 1e-6)
            self.created_features.append('credit_utilization')
        
        # Payment to income ratio
        if 'monthly_payment' in df_new.columns and 'annual_income' in df_new.columns:
            df_new['payment_to_income_ratio'] = (df_new['monthly_payment'] * 12) / (df_new['annual_income'] + 1e-6)
            self.created_features.append('payment_to_income_ratio')
        
        # Account age features
        if 'account_age_months' in df_new.columns:
            df_new['account_age_years'] = df_new['account_age_months'] / 12
            df_new['is_new_account'] = (df_new['account_age_months'] < 12).astype(int)
            self.created_features.extend(['account_age_years', 'is_new_account'])
        
        # Credit score categories
        if 'credit_score' in df_new.columns:
            df_new['credit_score_category'] = pd.cut(
                df_new['credit_score'],
                bins=[0, 580, 670, 740, 800, 850],
                labels=['Poor', 'Fair', 'Good', 'Very Good', 'Excellent'],
                ordered=True
            ).astype(str)
            self.created_features.append('credit_score_category')
        
        # Employment stability
        if 'employment_length' in df_new.columns:
            df_new['employment_stability'] = (df_new['employment_length'] >= 2).astype(int)
            self.created_features.append('employment_stability')
        
        return df_new
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between important variables."""
        df_new = df.copy()
        numeric_columns = df_new.select_dtypes(include=[np.number]).columns
        
        # Define important feature pairs for credit analysis
        important_pairs = [
            ('credit_score', 'annual_income'),
            ('credit_score', 'loan_amount'),
            ('annual_income', 'loan_amount'),
            ('age', 'employment_length'),
            ('credit_utilization', 'credit_score')
        ]
        
        for col1, col2 in important_pairs:
            if col1 in numeric_columns and col2 in numeric_columns:
                # Multiplicative interaction
                interaction_name = f'{col1}_x_{col2}'
                df_new[interaction_name] = df_new[col1] * df_new[col2]
                self.created_features.append(interaction_name)
        
        return df_new
    
    def _create_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create ratio features."""
        df_new = df.copy()
        numeric_columns = df_new.select_dtypes(include=[np.number]).columns
        
        # Define meaningful ratio pairs
        ratio_pairs = [
            ('loan_amount', 'annual_income'),
            ('monthly_payment', 'annual_income'),
            ('current_balance', 'credit_limit'),
            ('open_accounts', 'total_accounts'),
            ('delinquencies_2yrs', 'total_accounts')
        ]
        
        for numerator, denominator in ratio_pairs:
            if numerator in numeric_columns and denominator in numeric_columns:
                ratio_name = f'{numerator}_to_{denominator}_ratio'
                df_new[ratio_name] = df_new[numerator] / (df_new[denominator] + 1e-6)
                self.created_features.append(ratio_name)
        
        return df_new
    
    def _create_aggregation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create aggregation features."""
        df_new = df.copy()
        numeric_columns = df_new.select_dtypes(include=[np.number]).columns
        
        # Create sum features for related columns
        account_columns = [col for col in numeric_columns if 'account' in col.lower()]
        if len(account_columns) > 1:
            df_new['total_accounts_sum'] = df_new[account_columns].sum(axis=1)
            self.created_features.append('total_accounts_sum')
        
        # Create balance-related aggregations
        balance_columns = [col for col in numeric_columns if 'balance' in col.lower()]
        if len(balance_columns) > 1:
            df_new['total_balance'] = df_new[balance_columns].sum(axis=1)
            df_new['avg_balance'] = df_new[balance_columns].mean(axis=1)
            self.created_features.extend(['total_balance', 'avg_balance'])
        
        # Create delinquency aggregations
        delinq_columns = [col for col in numeric_columns if 'delinq' in col.lower()]
        if len(delinq_columns) > 1:
            df_new['total_delinquencies'] = df_new[delinq_columns].sum(axis=1)
            self.created_features.append('total_delinquencies')
        
        return df_new
    
    def _create_binned_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create binned categorical features from continuous variables."""
        df_new = df.copy()
        
        # Age bins
        if 'age' in df_new.columns:
            df_new['age_group'] = pd.cut(
                df_new['age'],
                bins=[0, 25, 35, 45, 55, 65, 100],
                labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+'],
                ordered=True
            ).astype(str)
            self.created_features.append('age_group')
        
        # Income bins
        if 'annual_income' in df_new.columns:
            df_new['income_group'] = pd.qcut(
                df_new['annual_income'],
                q=5,
                labels=['Low', 'Low-Med', 'Medium', 'Med-High', 'High'],
                duplicates='drop'
            ).astype(str)
            self.created_features.append('income_group')
        
        # Loan amount bins
        if 'loan_amount' in df_new.columns:
            df_new['loan_size'] = pd.qcut(
                df_new['loan_amount'],
                q=4,
                labels=['Small', 'Medium', 'Large', 'Very Large'],
                duplicates='drop'
            ).astype(str)
            self.created_features.append('loan_size')
        
        return df_new
    
    def _create_polynomial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create polynomial features for key variables."""
        df_new = df.copy()
        
        # Key variables for polynomial features
        poly_columns = ['credit_score', 'annual_income', 'age']
        
        for col in poly_columns:
            if col in df_new.columns and df_new[col].dtype in [np.number]:
                # Squared terms
                df_new[f'{col}_squared'] = df_new[col] ** 2
                self.created_features.append(f'{col}_squared')
                
                # Log transformation (for positive values)
                if (df_new[col] > 0).all():
                    df_new[f'{col}_log'] = np.log(df_new[col] + 1)
                    self.created_features.append(f'{col}_log')
        
        return df_new
    
    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features from datetime columns."""
        df_new = df.copy()
        datetime_columns = df_new.select_dtypes(include=['datetime64']).columns
        
        for col in datetime_columns:
            # Extract time components
            df_new[f'{col}_year'] = df_new[col].dt.year
            df_new[f'{col}_month'] = df_new[col].dt.month
            df_new[f'{col}_day'] = df_new[col].dt.day
            df_new[f'{col}_dayofweek'] = df_new[col].dt.dayofweek
            df_new[f'{col}_quarter'] = df_new[col].dt.quarter
            
            # Time since features
            df_new[f'days_since_{col}'] = (datetime.now() - df_new[col]).dt.days
            
            self.created_features.extend([
                f'{col}_year', f'{col}_month', f'{col}_day',
                f'{col}_dayofweek', f'{col}_quarter', f'days_since_{col}'
            ])
        
        return df_new
    
    def select_features(self, 
                       df: pd.DataFrame,
                       target: Optional[pd.Series] = None,
                       method: Optional[str] = None) -> pd.DataFrame:
        """
        Select best features using various methods.
        
        Args:
            df: DataFrame with features
            target: Target variable for supervised selection
            method: Feature selection method
            
        Returns:
            DataFrame with selected features
        """
        method = method or self.feature_selection_method
        
        if target is None and self.target_column and self.target_column in df.columns:
            target = df[self.target_column]
            df_features = df.drop(columns=[self.target_column])
        else:
            df_features = df.copy()
        
        if target is None:
            logger.warning("No target variable provided for feature selection")
            return df_features
        
        numeric_features = df_features.select_dtypes(include=[np.number])
        
        if len(numeric_features.columns) == 0:
            logger.warning("No numeric features found for selection")
            return df_features
        
        # Determine number of features to select
        n_features = self.max_features or min(50, len(numeric_features.columns))
        
        if method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_classif, k=n_features)
        elif method == 'f_classif':
            selector = SelectKBest(score_func=f_classif, k=n_features)
        else:
            logger.warning(f"Unknown selection method: {method}")
            return df_features
        
        # Fit selector on numeric features only
        selected_numeric = selector.fit_transform(numeric_features, target)
        selected_feature_names = numeric_features.columns[selector.get_support()].tolist()
        
        # Get feature scores
        feature_scores = dict(zip(numeric_features.columns, selector.scores_))
        self.feature_importance_scores = {
            name: score for name, score in feature_scores.items()
            if name in selected_feature_names
        }
        
        # Combine selected numeric features with non-numeric features
        non_numeric_features = df_features.select_dtypes(exclude=[np.number])
        
        selected_df = pd.concat([
            pd.DataFrame(selected_numeric, columns=selected_feature_names, index=df_features.index),
            non_numeric_features
        ], axis=1)
        
        self.selected_features = selected_df.columns.tolist()
        
        logger.info(f"Selected {len(selected_feature_names)} numeric features out of {len(numeric_features.columns)}")
        
        return selected_df
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores."""
        if not self.feature_importance_scores:
            return pd.DataFrame()
        
        importance_df = pd.DataFrame([
            {'feature': feature, 'importance': score}
            for feature, score in self.feature_importance_scores.items()
        ]).sort_values('importance', ascending=False)
        
        return importance_df
    
    def get_engineering_summary(self) -> Dict[str, Any]:
        """Get summary of feature engineering process."""
        return {
            'created_features': self.created_features,
            'selected_features': self.selected_features,
            'feature_importance_scores': self.feature_importance_scores,
            'total_created': len(self.created_features),
            'total_selected': len(self.selected_features),
            'selection_method': self.feature_selection_method
        }

class DataSplitter:
    """
    Advanced data splitting utility for machine learning workflows.
    
    Provides various splitting strategies including stratified, time-based,
    and custom splitting methods with proper validation.
    """
    
    def __init__(self, 
                 random_state: int = 42,
                 stratify: bool = True):
        """
        Initialize the data splitter.
        
        Args:
            random_state: Random state for reproducibility
            stratify: Whether to use stratified splitting by default
        """
        self.random_state = random_state
        self.stratify = stratify
        self.split_info = {}
    
    def train_test_split(self, 
                        df: pd.DataFrame,
                        target_column: str,
                        test_size: float = 0.2,
                        validation_size: Optional[float] = None,
                        stratify: Optional[bool] = None) -> Union[Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series], 
                                                                Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]]:
        """
        Split data into train/test or train/validation/test sets.
        
        Args:
            df: Input DataFrame
            target_column: Name of target column
            test_size: Proportion of test set
            validation_size: Proportion of validation set (optional)
            stratify: Whether to stratify split
            
        Returns:
            Split datasets
        """
        stratify = stratify if stratify is not None else self.stratify
        
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        stratify_target = y if stratify else None
        
        if validation_size is not None:
            # Three-way split
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state, stratify=stratify_target
            )
            
            # Adjust validation size for remaining data
            val_size_adjusted = validation_size / (1 - test_size)
            stratify_temp = y_temp if stratify else None
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size_adjusted, 
                random_state=self.random_state, stratify=stratify_temp
            )
            
            self.split_info = {
                'train_size': len(X_train),
                'validation_size': len(X_val),
                'test_size': len(X_test),
                'train_ratio': len(X_train) / len(df),
                'validation_ratio': len(X_val) / len(df),
                'test_ratio': len(X_test) / len(df)
            }
            
            return X_train, X_val, X_test, y_train, y_val, y_test
        
        else:
            # Two-way split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state, stratify=stratify_target
            )
            
            self.split_info = {
                'train_size': len(X_train),
                'test_size': len(X_test),
                'train_ratio': len(X_train) / len(df),
                'test_ratio': len(X_test) / len(df)
            }
            
            return X_train, X_test, y_train, y_test
    
    def time_based_split(self, 
                        df: pd.DataFrame,
                        target_column: str,
                        time_column: str,
                        test_months: int = 6) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data based on time for time series validation.
        
        Args:
            df: Input DataFrame
            target_column: Name of target column
            time_column: Name of time column
            test_months: Number of months for test set
            
        Returns:
            Time-based split datasets
        """
        df_sorted = df.sort_values(time_column)
        
        # Calculate split point
        max_date = df_sorted[time_column].max()
        split_date = max_date - pd.DateOffset(months=test_months)
        
        # Split data
        train_mask = df_sorted[time_column] <= split_date
        test_mask = df_sorted[time_column] > split_date
        
        X_train = df_sorted[train_mask].drop(columns=[target_column])
        X_test = df_sorted[test_mask].drop(columns=[target_column])
        y_train = df_sorted[train_mask][target_column]
        y_test = df_sorted[test_mask][target_column]
        
        self.split_info = {
            'split_date': split_date,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'train_date_range': (X_train[time_column].min(), X_train[time_column].max()),
            'test_date_range': (X_test[time_column].min(), X_test[time_column].max())
        }
        
        return X_train, X_test, y_train, y_test
    
    def get_split_summary(self) -> Dict[str, Any]:
        """Get summary of the last split operation."""
        return self.split_info

# Utility functions
def generate_sample_credit_data(n_samples: int = 1000, 
                              random_state: int = 42,
                              save_path: Optional[str] = None) -> pd.DataFrame:
    """
    Generate sample credit data for testing and development.
    
    Args:
        n_samples: Number of samples to generate
        random_state: Random state for reproducibility
        save_path: Optional path to save the data
        
    Returns:
        Generated DataFrame
    """
    np.random.seed(random_state)
    
    # Generate base features
    data = {
        'customer_id': range(1, n_samples + 1),
        'annual_income': np.random.lognormal(10.5, 0.5, n_samples),
        'loan_amount': np.random.lognormal(9.5, 0.7, n_samples),
        'credit_score': np.random.normal(650, 100, n_samples).clip(300, 850),
        'employment_length': np.random.exponential(5, n_samples).clip(0, 40),
        'age': np.random.normal(40, 12, n_samples).clip(18, 80),
        'debt_to_income_ratio': np.random.beta(2, 5, n_samples),
        'credit_utilization': np.random.beta(2, 3, n_samples),
        'open_accounts': np.random.poisson(8, n_samples),
        'total_accounts': np.random.poisson(15, n_samples),
        'delinquencies_2yrs': np.random.poisson(0.5, n_samples),
        'inquiries_6mths': np.random.poisson(1, n_samples),
        'public_records': np.random.poisson(0.1, n_samples),
        'loan_purpose': np.random.choice(
            ['debt_consolidation', 'credit_card', 'home_improvement', 'major_purchase', 'small_business'],
            n_samples, p=[0.4, 0.2, 0.15, 0.15, 0.1]
        ),
        'home_ownership': np.random.choice(
            ['RENT', 'OWN', 'MORTGAGE'], n_samples, p=[0.4, 0.2, 0.4]
        ),
        'verification_status': np.random.choice(
            ['Verified', 'Source Verified', 'Not Verified'], n_samples, p=[0.4, 0.3, 0.3]
        )
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Generate target variable based on features
    risk_score = (
        -0.3 * (df['credit_score'] - 650) / 100 +
        0.2 * df['debt_to_income_ratio'] +
        0.15 * df['credit_utilization'] +
        0.1 * df['delinquencies_2yrs'] +
        0.05 * df['inquiries_6mths'] +
        np.random.normal(0, 0.5, n_samples)
    )
    
    # Convert to binary target
    df['default'] = (risk_score > 0).astype(int)
    
    # Add some realistic constraints
    df['total_accounts'] = np.maximum(df['total_accounts'], df['open_accounts'])
    df['annual_income'] = df['annual_income'].clip(15000, 500000)
    df['loan_amount'] = df['loan_amount'].clip(1000, 100000)
    
    # Round numeric columns
    df['annual_income'] = df['annual_income'].round(0)
    df['loan_amount'] = df['loan_amount'].round(0)
    df['credit_score'] = df['credit_score'].round(0)
    df['employment_length'] = df['employment_length'].round(1)
    df['age'] = df['age'].round(0)
    
    if save_path:
        df.to_csv(save_path, index=False)
        logger.info(f"Sample data saved to {save_path}")
    
    return df

def load_and_prepare_data(file_path: str,
                         target_column: str,
                         test_size: float = 0.2,
                         validation_size: Optional[float] = None,
                         clean_data: bool = True,
                         engineer_features: bool = True,
                         select_features: bool = False,
                         max_features: Optional[int] = None) -> Dict[str, Any]:
    """
    Complete data loading and preparation pipeline.
    
    Args:
        file_path: Path to data file
        target_column: Name of target column
        test_size: Test set proportion
        validation_size: Validation set proportion (optional)
        clean_data: Whether to clean the data
        engineer_features: Whether to engineer features
        select_features: Whether to perform feature selection
        max_features: Maximum features to select
        
    Returns:
        Dictionary containing prepared datasets and metadata
    """
    logger.info("Starting complete data preparation pipeline")
    
    # Load data
    loader = DataLoader()
    df = loader.load_data(file_path)
    logger.info(f"Loaded data: {df.shape}")
    
    # Validate data
    validator = DataValidator()
    quality_report = validator.validate_data(df, target_column)
    logger.info(f"Data quality: {quality_report.quality_level.value} (score: {quality_report.overall_score:.3f})")
    
    # Clean data if requested
    if clean_data:
        cleaner = DataCleaner()
        df = cleaner.clean_data(df, target_column)
        logger.info(f"Data after cleaning: {df.shape}")
    
    # Engineer features if requested
    if engineer_features:
        engineer = FeatureEngineer(target_column=target_column, max_features=max_features)
        df = engineer.engineer_features(df)
        logger.info(f"Data after feature engineering: {df.shape}")
        
        # Select features if requested
        if select_features:
            df = engineer.select_features(df)
            logger.info(f"Data after feature selection: {df.shape}")
    
    # Split data
    splitter = DataSplitter()
    if validation_size:
        X_train, X_val, X_test, y_train, y_val, y_test = splitter.train_test_split(
            df, target_column, test_size, validation_size
        )
        datasets = {
            'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
            'y_train': y_train, 'y_val': y_val, 'y_test': y_test
        }
    else:
        X_train, X_test, y_train, y_test = splitter.train_test_split(
            df, target_column, test_size
        )
        datasets = {
            'X_train': X_train, 'X_test': X_test,
            'y_train': y_train, 'y_test': y_test
        }
    
    # Prepare metadata
    metadata = {
        'original_shape': (len(df), len(df.columns)),
        'final_shape': (len(X_train) + len(X_test), len(X_train.columns)),
        'quality_report': quality_report,
        'split_info': splitter.get_split_summary(),
        'target_column': target_column,
        'feature_names': X_train.columns.tolist()
    }
    
    if engineer_features:
        metadata['engineering_summary'] = engineer.get_engineering_summary()
    
    if clean_data:
        metadata['cleaning_report'] = cleaner.get_cleaning_report()
    
    result = {
        'datasets': datasets,
        'metadata': metadata,
        'full_data': df
    }
    
    logger.info("Data preparation pipeline completed successfully")
    
    return result

def save_prepared_data(prepared_data: Dict[str, Any],
                      output_dir: str,
                      prefix: str = 'credit_data') -> Dict[str, str]:
    """
    Save prepared datasets and metadata.
    
    Args:
        prepared_data: Output from load_and_prepare_data
        output_dir: Output directory
        prefix: File prefix
        
    Returns:
        Dictionary of saved file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    saved_files = {}
    
    # Save datasets
    datasets = prepared_data['datasets']
    for name, data in datasets.items():
        file_path = output_path / f"{prefix}_{name}.csv"
        data.to_csv(file_path, index=False)
        saved_files[name] = str(file_path)
    
    # Save full data
    full_data_path = output_path / f"{prefix}_full.csv"
    prepared_data['full_data'].to_csv(full_data_path, index=False)
    saved_files['full_data'] = str(full_data_path)
    
    # Save metadata
    metadata_path = output_path / f"{prefix}_metadata.json"
    metadata = prepared_data['metadata'].copy()
    
    # Convert non-serializable objects
    if 'quality_report' in metadata:
        quality_report = metadata['quality_report']
        metadata['quality_report'] = {
            'overall_score': quality_report.overall_score,
            'quality_level': quality_report.quality_level.value,
            'total_records': quality_report.total_records,
            'total_features': quality_report.total_features,
            'missing_data_percentage': quality_report.missing_data_percentage,
            'duplicate_records': quality_report.duplicate_records,
            'outlier_percentage': quality_report.outlier_percentage,
            'recommendations': quality_report.recommendations,
            'timestamp': quality_report.timestamp.isoformat()
        }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    saved_files['metadata'] = str(metadata_path)
    
    logger.info(f"Saved prepared data to {output_dir}")
    
    return saved_files

def load_prepared_data(metadata_path: str) -> Dict[str, Any]:
    """
    Load previously prepared and saved data.
    
    Args:
        metadata_path: Path to metadata JSON file
        
    Returns:
        Dictionary containing loaded datasets and metadata
    """
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Reconstruct file paths
    base_dir = Path(metadata_path).parent
    prefix = Path(metadata_path).stem.replace('_metadata', '')
    
    datasets = {}
    
    # Load datasets
    for dataset_name in ['X_train', 'X_test', 'y_train', 'y_test', 'X_val', 'y_val']:
        file_path = base_dir / f"{prefix}_{dataset_name}.csv"
        if file_path.exists():
            datasets[dataset_name] = pd.read_csv(file_path)
    
    # Load full data
    full_data_path = base_dir / f"{prefix}_full.csv"
    full_data = pd.read_csv(full_data_path) if full_data_path.exists() else None
    
    return {
        'datasets': datasets,
        'metadata': metadata,
        'full_data': full_data
    }

def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get comprehensive data summary.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with data summary
    """
    summary = {
        'shape': df.shape,
        'memory_usage': df.memory_usage(deep=True).sum(),
        'dtypes': df.dtypes.value_counts().to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'numeric_summary': df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {},
        'categorical_summary': {
            col: df[col].value_counts().head().to_dict()
            for col in df.select_dtypes(include=['object', 'category']).columns
        }
    }
    
    return summary

def detect_data_drift(reference_df: pd.DataFrame,
                     current_df: pd.DataFrame,
                     threshold: float = 0.1) -> Dict[str, Any]:
    """
    Detect data drift between reference and current datasets.
    
    Args:
        reference_df: Reference dataset
        current_df: Current dataset to compare
        threshold: Threshold for drift detection
        
    Returns:
        Dictionary with drift analysis results
    """
    drift_results = {
        'overall_drift': False,
        'drifted_columns': [],
        'drift_scores': {},
        'summary': {}
    }
    
    common_columns = set(reference_df.columns) & set(current_df.columns)
    
    for col in common_columns:
        if reference_df[col].dtype in [np.number]:
            # Numerical drift using KS test
            from scipy.stats import ks_2samp
            statistic, p_value = ks_2samp(reference_df[col].dropna(), current_df[col].dropna())
            drift_score = statistic
            is_drifted = p_value < 0.05
        else:
            # Categorical drift using distribution comparison
            ref_dist = reference_df[col].value_counts(normalize=True)
            curr_dist = current_df[col].value_counts(normalize=True)
            
            # Calculate Jensen-Shannon divergence
            all_categories = set(ref_dist.index) | set(curr_dist.index)
            ref_probs = [ref_dist.get(cat, 0) for cat in all_categories]
            curr_probs = [curr_dist.get(cat, 0) for cat in all_categories]
            
            # Simple drift score based on distribution difference
            drift_score = sum(abs(r - c) for r, c in zip(ref_probs, curr_probs)) / 2
            is_drifted = drift_score > threshold
        
        drift_results['drift_scores'][col] = drift_score
        
        if is_drifted:
            drift_results['drifted_columns'].append(col)
            drift_results['overall_drift'] = True
    
    drift_results['summary'] = {
        'total_columns_checked': len(common_columns),
        'drifted_columns_count': len(drift_results['drifted_columns']),
        'drift_percentage': len(drift_results['drifted_columns']) / len(common_columns) if common_columns else 0
    }
    
    return drift_results

def create_data_profile(df: pd.DataFrame,
                       output_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Create comprehensive data profile report.
    
    Args:
        df: Input DataFrame
        output_path: Optional path to save the profile
        
    Returns:
        Dictionary with comprehensive data profile
    """
    profile = {
        'basic_info': {
            'shape': df.shape,
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'creation_time': datetime.now().isoformat()
        },
        'data_types': df.dtypes.value_counts().to_dict(),
        'missing_data': {
            'total_missing': df.isnull().sum().sum(),
            'missing_percentage': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
            'columns_with_missing': df.columns[df.isnull().any()].tolist(),
            'missing_by_column': df.isnull().sum().to_dict()
        },
        'duplicates': {
            'duplicate_rows': df.duplicated().sum(),
            'duplicate_percentage': (df.duplicated().sum() / len(df)) * 100
        },
        'numerical_analysis': {},
        'categorical_analysis': {},
        'correlations': {}
    }
    
    # Numerical analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        profile['numerical_analysis'] = {
            'column_count': len(numeric_cols),
            'statistics': df[numeric_cols].describe().to_dict(),
            'skewness': df[numeric_cols].skew().to_dict(),
            'kurtosis': df[numeric_cols].kurtosis().to_dict()
        }
        
        # Correlation matrix
        if len(numeric_cols) > 1:
            profile['correlations'] = df[numeric_cols].corr().to_dict()
    
    # Categorical analysis
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        profile['categorical_analysis'] = {
            'column_count': len(categorical_cols),
            'unique_counts': df[categorical_cols].nunique().to_dict(),
            'value_counts': {
                col: df[col].value_counts().head(10).to_dict()
                for col in categorical_cols
            }
        }
    
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(profile, f, indent=2, default=str)
        logger.info(f"Data profile saved to {output_path}")
    
    return profile

# Export all classes and functions
__all__ = [
    # Enums and Data Classes
    'DataQualityLevel',
    'DataQualityReport',
    
    # Main Classes
    'DataLoader',
    'DataCleaner', 
    'DataValidator',
    'FeatureEngineer',
    'DataSplitter',
    
    # Utility Functions
    'generate_sample_credit_data',
    'load_and_prepare_data',
    'save_prepared_data',
    'load_prepared_data',
    'get_data_summary',
    'detect_data_drift',
    'create_data_profile'
]