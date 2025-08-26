"""
Data validation module for GA KKM Grouping
Validates CSV data format and content according to konteks-algen.md specifications
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
import re


class DataValidator:
    """Class for validating CSV data format and content"""
    
    # Expected columns according to konteks-algen.md
    EXPECTED_COLUMNS = ['Group', 'Gender', 'Fakultas', 'Jurusan', 'Htq']
    
    # Valid values for categorical columns
    VALID_GENDER = ['LK', 'PR', 'Laki-laki', 'Perempuan', 'L', 'P']
    VALID_HTQ = ['Ya', 'Tidak', 'YES', 'NO', 'Y', 'N', 'True', 'False', '1', '0']
    
    def __init__(self):
        """Initialize DataValidator"""
        pass
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive data validation
        
        Args:
            df: DataFrame to validate
            
        Returns:
            dict: Validation results
        """
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'summary': {}
        }
        
        # Run all validation checks
        self._validate_structure(df, validation_result)
        self._validate_content(df, validation_result)
        self._validate_data_quality(df, validation_result)
        self._generate_summary(df, validation_result)
        
        return validation_result
    
    def _validate_structure(self, df: pd.DataFrame, result: Dict[str, Any]):
        """Validate DataFrame structure"""
        # Check if DataFrame is empty
        if df.empty:
            result['errors'].append("Dataset is empty")
            result['is_valid'] = False
            return
        
        # Check column count
        if len(df.columns) != len(self.EXPECTED_COLUMNS):
            result['errors'].append(
                f"Expected {len(self.EXPECTED_COLUMNS)} columns, got {len(df.columns)}"
            )
            result['is_valid'] = False
        
        # Check column names (case insensitive)
        actual_columns = [col.lower() for col in df.columns]
        expected_columns = [col.lower() for col in self.EXPECTED_COLUMNS]
        
        for i, (expected, actual) in enumerate(zip(expected_columns, actual_columns)):
            if expected != actual:
                result['errors'].append(
                    f"Column {i+1} name mismatch: expected '{self.EXPECTED_COLUMNS[i]}', got '{df.columns[i]}'"
                )
                result['is_valid'] = False
    
    def _validate_content(self, df: pd.DataFrame, result: Dict[str, Any]):
        """Validate column content"""
        if df.empty:
            return
        
        # Validate Gender column
        self._validate_gender_column(df, result)
        
        # Validate HTQ column
        self._validate_htq_column(df, result)
        
        # Validate Fakultas column
        self._validate_fakultas_column(df, result)
        
        # Validate Jurusan column
        self._validate_jurusan_column(df, result)
        
        # Validate Group column
        self._validate_group_column(df, result)
    
    def _validate_gender_column(self, df: pd.DataFrame, result: Dict[str, Any]):
        """Validate Gender column"""
        if 'Gender' not in df.columns:
            return
        
        # Check for missing values
        missing_gender = df['Gender'].isna().sum()
        if missing_gender > 0:
            result['errors'].append(f"Found {missing_gender} missing values in Gender column")
            result['is_valid'] = False
        
        # Check for invalid gender values
        unique_genders = df['Gender'].dropna().unique()
        invalid_genders = []
        
        for gender in unique_genders:
            gender_str = str(gender).strip().upper()
            valid_gender_upper = [g.upper() for g in self.VALID_GENDER]
            if gender_str not in valid_gender_upper:
                invalid_genders.append(gender)
        
        if invalid_genders:
            result['warnings'].append(
                f"Found potentially invalid Gender values: {invalid_genders}. "
                f"Expected values: {self.VALID_GENDER}"
            )
    
    def _validate_htq_column(self, df: pd.DataFrame, result: Dict[str, Any]):
        """Validate HTQ column"""
        if 'Htq' not in df.columns:
            return
        
        # Check for missing values
        missing_htq = df['Htq'].isna().sum()
        if missing_htq > 0:
            result['errors'].append(f"Found {missing_htq} missing values in HTQ column")
            result['is_valid'] = False
        
        # Check for invalid HTQ values
        unique_htq = df['Htq'].dropna().unique()
        invalid_htq = []
        
        for htq in unique_htq:
            htq_str = str(htq).strip().upper()
            valid_htq_upper = [h.upper() for h in self.VALID_HTQ]
            if htq_str not in valid_htq_upper:
                invalid_htq.append(htq)
        
        if invalid_htq:
            result['warnings'].append(
                f"Found potentially invalid HTQ values: {invalid_htq}. "
                f"Expected values: {self.VALID_HTQ}"
            )
    
    def _validate_fakultas_column(self, df: pd.DataFrame, result: Dict[str, Any]):
        """Validate Fakultas column"""
        if 'Fakultas' not in df.columns:
            return
        
        # Check for missing values
        missing_fakultas = df['Fakultas'].isna().sum()
        if missing_fakultas > 0:
            result['errors'].append(f"Found {missing_fakultas} missing values in Fakultas column")
            result['is_valid'] = False
        
        # Check for empty strings
        empty_fakultas = (df['Fakultas'].astype(str).str.strip() == '').sum()
        if empty_fakultas > 0:
            result['warnings'].append(f"Found {empty_fakultas} empty Fakultas values")
    
    def _validate_jurusan_column(self, df: pd.DataFrame, result: Dict[str, Any]):
        """Validate Jurusan column"""
        if 'Jurusan' not in df.columns:
            return
        
        # Check for missing values
        missing_jurusan = df['Jurusan'].isna().sum()
        if missing_jurusan > 0:
            result['errors'].append(f"Found {missing_jurusan} missing values in Jurusan column")
            result['is_valid'] = False
        
        # Check for empty strings
        empty_jurusan = (df['Jurusan'].astype(str).str.strip() == '').sum()
        if empty_jurusan > 0:
            result['warnings'].append(f"Found {empty_jurusan} empty Jurusan values")
    
    def _validate_group_column(self, df: pd.DataFrame, result: Dict[str, Any]):
        """Validate Group column (diabaikan but should be numeric)"""
        if 'Group' not in df.columns:
            return
        
        # Check if Group column is numeric
        try:
            pd.to_numeric(df['Group'], errors='raise')
        except (ValueError, TypeError):
            result['warnings'].append(
                "Group column contains non-numeric values (will be ignored in processing)"
            )
    
    def _validate_data_quality(self, df: pd.DataFrame, result: Dict[str, Any]):
        """Validate overall data quality"""
        if df.empty:
            return
        
        # Check for duplicate rows
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            result['warnings'].append(f"Found {duplicates} duplicate rows")
        
        # Check dataset size
        total_rows = len(df)
        if total_rows < 100:
            result['warnings'].append(
                f"Small dataset: only {total_rows} records. "
                "GA performance may be suboptimal with small datasets."
            )
        
        # Check for extreme imbalances
        if 'Gender' in df.columns:
            gender_counts = df['Gender'].value_counts()
            if len(gender_counts) > 0:
                min_gender = gender_counts.min()
                max_gender = gender_counts.max()
                if min_gender / max_gender < 0.1:  # Less than 10%
                    result['warnings'].append(
                        f"Extreme gender imbalance detected: {dict(gender_counts)}"
                    )
        
        if 'Htq' in df.columns:
            htq_counts = df['Htq'].value_counts()
            if len(htq_counts) > 0:
                min_htq = htq_counts.min()
                max_htq = htq_counts.max()
                if min_htq / max_htq < 0.01:  # Less than 1%
                    result['warnings'].append(
                        f"Extreme HTQ imbalance detected: {dict(htq_counts)}"
                    )
    
    def _generate_summary(self, df: pd.DataFrame, result: Dict[str, Any]):
        """Generate validation summary"""
        if df.empty:
            result['summary'] = {'total_records': 0}
            return
        
        summary = {
            'total_records': len(df),
            'total_columns': len(df.columns),
            'missing_values_total': df.isna().sum().sum()
        }
        
        # Add column-specific summaries
        if 'Gender' in df.columns:
            summary['gender_distribution'] = df['Gender'].value_counts().to_dict()
        
        if 'Htq' in df.columns:
            summary['htq_distribution'] = df['Htq'].value_counts().to_dict()
        
        if 'Fakultas' in df.columns:
            summary['fakultas_count'] = df['Fakultas'].nunique()
        
        if 'Jurusan' in df.columns:
            summary['jurusan_count'] = df['Jurusan'].nunique()
        
        result['summary'] = summary
    
    def validate_for_ga(self, df: pd.DataFrame, num_groups: int) -> Dict[str, Any]:
        """
        Specialized validation for GA requirements
        
        Args:
            df: DataFrame to validate
            num_groups: Target number of groups
            
        Returns:
            dict: GA-specific validation results
        """
        result = {
            'is_suitable': True,
            'issues': [],
            'recommendations': []
        }
        
        total_students = len(df)
        
        # Check if dataset size is suitable for GA
        if total_students < num_groups:
            result['issues'].append(
                f"Not enough students ({total_students}) for target groups ({num_groups})"
            )
            result['is_suitable'] = False
        
        # Check average group size
        avg_group_size = total_students / num_groups
        if avg_group_size < 3:
            result['issues'].append(
                f"Very small average group size ({avg_group_size:.1f}). "
                "Consider reducing number of groups."
            )
        elif avg_group_size > 20:
            result['recommendations'].append(
                f"Large average group size ({avg_group_size:.1f}). "
                "Consider increasing number of groups for better balance."
            )
        
        # Check HTQ availability
        if 'Htq' in df.columns:
            htq_yes_count = df['Htq'].str.upper().isin(['YA', 'YES', 'Y', 'TRUE', '1']).sum()
            if htq_yes_count < num_groups:
                result['recommendations'].append(
                    f"Only {htq_yes_count} HTQ students for {num_groups} groups. "
                    f"{num_groups - htq_yes_count} groups will not have HTQ students."
                )
        
        return result


def validate_master_data(file_path: str = None) -> Dict[str, Any]:
    """
    Convenience function to validate master data file
    
    Args:
        file_path: Path to master data CSV
        
    Returns:
        dict: Validation results
    """
    import os
    from .data_loader import DataLoader
    
    if file_path is None:
        file_path = os.path.join('data', 'master_data.csv')
    
    try:
        loader = DataLoader()
        df = loader.load_csv(file_path)
        
        validator = DataValidator()
        return validator.validate_data(df)
    
    except Exception as e:
        return {
            'is_valid': False,
            'errors': [f"Failed to load/validate file: {str(e)}"],
            'warnings': [],
            'summary': {}
        }