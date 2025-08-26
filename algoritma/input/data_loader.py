"""
Data loading module for GA KKM Grouping
Handles CSV loading and basic data import operations
"""

import pandas as pd
import os
from typing import Optional
import sys


class DataLoader:
    """Class for loading CSV data files"""
    
    def __init__(self):
        """Initialize DataLoader"""
        self.supported_encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']
    
    def load_csv(self, file_path: str) -> pd.DataFrame:
        """
        Load CSV file with proper encoding detection
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            pd.DataFrame: Loaded data
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        # Try different encodings
        for encoding in self.supported_encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                print(f"Successfully loaded with encoding: {encoding}")
                return self._clean_dataframe(df)
            
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"Error with encoding {encoding}: {str(e)}")
                continue
        
        # If all encodings failed
        raise ValueError(f"Could not load file with any supported encoding: {self.supported_encodings}")
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and prepare dataframe
        
        Args:
            df: Raw dataframe
            
        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        # Remove BOM if present
        if df.columns[0].startswith('\ufeff'):
            df.columns = [df.columns[0].replace('\ufeff', '')] + list(df.columns[1:])
        
        # Strip whitespace from string columns
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.strip()
        
        # Remove completely empty rows
        df = df.dropna(how='all')
        
        # Reset index
        df = df.reset_index(drop=True)
        
        return df
    
    def get_file_info(self, file_path: str) -> dict:
        """
        Get basic information about CSV file
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            dict: File information
        """
        if not os.path.exists(file_path):
            return {"error": "File not found"}
        
        try:
            # Get file size
            file_size = os.path.getsize(file_path)
            
            # Try to load and get basic info
            df = self.load_csv(file_path)
            
            return {
                "file_path": file_path,
                "file_size_bytes": file_size,
                "file_size_mb": round(file_size / (1024*1024), 2),
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": list(df.columns),
                "dtypes": df.dtypes.to_dict()
            }
        
        except Exception as e:
            return {"error": str(e)}
    
    def preview_data(self, file_path: str, n_rows: int = 5) -> dict:
        """
        Preview first few rows of data
        
        Args:
            file_path: Path to CSV file
            n_rows: Number of rows to preview
            
        Returns:
            dict: Preview information
        """
        try:
            df = self.load_csv(file_path)
            
            return {
                "success": True,
                "total_rows": len(df),
                "preview_rows": n_rows,
                "data": df.head(n_rows).to_dict('records'),
                "columns": list(df.columns)
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def validate_csv_structure(self, file_path: str, expected_columns: list) -> dict:
        """
        Validate CSV has expected column structure
        
        Args:
            file_path: Path to CSV file
            expected_columns: List of expected column names
            
        Returns:
            dict: Validation result
        """
        try:
            df = self.load_csv(file_path)
            actual_columns = list(df.columns)
            
            # Check column count
            if len(actual_columns) != len(expected_columns):
                return {
                    "valid": False,
                    "error": f"Expected {len(expected_columns)} columns, got {len(actual_columns)}",
                    "expected": expected_columns,
                    "actual": actual_columns
                }
            
            # Check column names (case insensitive)
            expected_lower = [col.lower() for col in expected_columns]
            actual_lower = [col.lower() for col in actual_columns]
            
            if expected_lower != actual_lower:
                return {
                    "valid": False,
                    "error": "Column names don't match expected structure",
                    "expected": expected_columns,
                    "actual": actual_columns,
                    "mismatched": [
                        (exp, act) for exp, act in zip(expected_columns, actual_columns)
                        if exp.lower() != act.lower()
                    ]
                }
            
            return {
                "valid": True,
                "message": "CSV structure is valid",
                "columns": actual_columns,
                "rows": len(df)
            }
        
        except Exception as e:
            return {
                "valid": False,
                "error": str(e)
            }


def load_master_data(file_path: str = None) -> pd.DataFrame:
    """
    Convenience function to load master data
    
    Args:
        file_path: Path to master data CSV
        
    Returns:
        pd.DataFrame: Loaded master data
    """
    if file_path is None:
        file_path = os.path.join('data', 'master_data.csv')
    
    loader = DataLoader()
    return loader.load_csv(file_path)


def main():
    """Test function for data loader"""
    try:
        # Test with master data
        loader = DataLoader()
        
        # Get file info
        file_path = os.path.join('..', '..', 'data', 'master_data.csv')
        print("File info:")
        info = loader.get_file_info(file_path)
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # Preview data
        print(f"\nData preview:")
        preview = loader.preview_data(file_path, 3)
        if preview['success']:
            print(f"Total rows: {preview['total_rows']}")
            print(f"Columns: {preview['columns']}")
            print("Sample data:")
            for i, row in enumerate(preview['data']):
                print(f"  Row {i+1}: {row}")
        else:
            print(f"Error: {preview['error']}")
    
    except Exception as e:
        print(f"Error in test: {str(e)}")


if __name__ == "__main__":
    main()