"""
Data preprocessing module for GA KKM Grouping
Preprocesses and prepares data for genetic algorithm processing
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
from collections import Counter


class DataPreprocessor:
    """Class for preprocessing data for GA"""
    
    def __init__(self):
        """Initialize DataPreprocessor"""
        pass
    
    def preprocess(self, df: pd.DataFrame, num_groups: int) -> Dict[str, Any]:
        """
        Main preprocessing function
        
        Args:
            df: Raw DataFrame
            num_groups: Target number of groups
            
        Returns:
            dict: Preprocessed data structure
        """
        # Clean and normalize data
        clean_df = self._clean_data(df.copy())
        
        # Create preprocessed data structure
        preprocessed_data = {
            'data': clean_df,
            'total_students': len(clean_df),
            'num_groups': num_groups,
            'htq_indices': self._get_htq_indices(clean_df),
            'htq_count': self._count_htq_students(clean_df),
            'gender_ratio': self._calculate_gender_ratio(clean_df),
            'gender_indices': self._get_gender_indices(clean_df),
            'faculty_distribution': self._get_faculty_distribution(clean_df),
            'major_distribution': self._get_major_distribution(clean_df),
            'group_size_info': self._calculate_group_sizes(len(clean_df), num_groups),
            'duplicate_risk_majors': self._identify_duplicate_risk_majors(clean_df),
            'student_attributes': self._create_student_attributes(clean_df)
        }
        
        return preprocessed_data
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and normalize data values
        
        Args:
            df: Raw DataFrame
            
        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        # Normalize Gender values
        df['Gender'] = df['Gender'].apply(self._normalize_gender)
        
        # Normalize HTQ values
        df['Htq'] = df['Htq'].apply(self._normalize_htq)
        
        # Clean text fields
        df['Fakultas'] = df['Fakultas'].astype(str).str.strip().str.upper()
        df['Jurusan'] = df['Jurusan'].astype(str).str.strip().str.upper()
        
        # Remove any completely empty rows
        df = df.dropna(how='all')
        
        # Reset index
        df = df.reset_index(drop=True)
        
        return df
    
    def _normalize_gender(self, value: Any) -> str:
        """Normalize gender values to LK/PR"""
        if pd.isna(value):
            return 'UNKNOWN'
        
        value_str = str(value).strip().upper()
        
        # Map various gender representations to LK/PR
        lk_variants = ['LK', 'LAKI-LAKI', 'L', 'MALE', 'M', 'PRIA']
        pr_variants = ['PR', 'PEREMPUAN', 'P', 'FEMALE', 'F', 'WANITA']
        
        if value_str in lk_variants:
            return 'LK'
        elif value_str in pr_variants:
            return 'PR'
        else:
            return 'UNKNOWN'
    
    def _normalize_htq(self, value: Any) -> str:
        """Normalize HTQ values to Ya/Tidak"""
        if pd.isna(value):
            return 'Tidak'
        
        value_str = str(value).strip().upper()
        
        # Map various HTQ representations to Ya/Tidak
        ya_variants = ['YA', 'YES', 'Y', 'TRUE', '1', 'BENAR']
        tidak_variants = ['TIDAK', 'NO', 'N', 'FALSE', '0', 'SALAH']
        
        if value_str in ya_variants:
            return 'Ya'
        elif value_str in tidak_variants:
            return 'Tidak'
        else:
            return 'Tidak'  # Default to Tidak
    
    def _get_htq_indices(self, df: pd.DataFrame) -> List[int]:
        """Get indices of HTQ students"""
        return df[df['Htq'] == 'Ya'].index.tolist()
    
    def _count_htq_students(self, df: pd.DataFrame) -> int:
        """Count HTQ students"""
        return len(df[df['Htq'] == 'Ya'])
    
    def _calculate_gender_ratio(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate global gender ratio"""
        total = len(df)
        if total == 0:
            return {'LK': 0.0, 'PR': 0.0}
        
        gender_counts = df['Gender'].value_counts()
        
        return {
            'LK': gender_counts.get('LK', 0) / total,
            'PR': gender_counts.get('PR', 0) / total
        }
    
    def _get_gender_indices(self, df: pd.DataFrame) -> Dict[str, List[int]]:
        """Get indices grouped by gender"""
        return {
            'LK': df[df['Gender'] == 'LK'].index.tolist(),
            'PR': df[df['Gender'] == 'PR'].index.tolist()
        }
    
    def _get_faculty_distribution(self, df: pd.DataFrame) -> Dict[str, int]:
        """Get faculty distribution"""
        return df['Fakultas'].value_counts().to_dict()
    
    def _get_major_distribution(self, df: pd.DataFrame) -> Dict[str, int]:
        """Get major distribution"""
        return df['Jurusan'].value_counts().to_dict()
    
    def _calculate_group_sizes(self, total_students: int, num_groups: int) -> Dict[str, Any]:
        """Calculate target group sizes"""
        avg_size = total_students / num_groups
        min_size = int(avg_size)
        max_size = min_size + 1
        
        remainder = total_students % num_groups
        groups_max_size = remainder
        groups_min_size = num_groups - remainder
        
        return {
            'average_size': avg_size,
            'min_size': min_size,
            'max_size': max_size,
            'groups_with_min_size': groups_min_size,
            'groups_with_max_size': groups_max_size,
            'size_range': [min_size, max_size]
        }
    
    def _identify_duplicate_risk_majors(self, df: pd.DataFrame) -> List[str]:
        """Identify majors with high duplication risk (>= 10 students)"""
        major_counts = df['Jurusan'].value_counts()
        return major_counts[major_counts >= 10].index.tolist()
    
    def _create_student_attributes(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Create student attribute list for easy access"""
        attributes = []
        
        for idx, row in df.iterrows():
            attributes.append({
                'index': idx,
                'gender': row['Gender'],
                'fakultas': row['Fakultas'],
                'jurusan': row['Jurusan'],
                'htq': row['Htq'],
                'is_htq': row['Htq'] == 'Ya',
                'is_lk': row['Gender'] == 'LK',
                'is_pr': row['Gender'] == 'PR'
            })
        
        return attributes
    
    def analyze_constraints(self, preprocessed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze constraint satisfaction potential
        
        Args:
            preprocessed_data: Preprocessed data structure
            
        Returns:
            dict: Constraint analysis
        """
        analysis = {}
        
        # HTQ Constraint Analysis
        htq_count = preprocessed_data['htq_count']
        num_groups = preprocessed_data['num_groups']
        
        analysis['htq_constraint'] = {
            'htq_students': htq_count,
            'target_groups': num_groups,
            'coverage_possible': min(htq_count, num_groups),
            'coverage_percentage': min(htq_count / num_groups, 1.0) * 100,
            'groups_without_htq': max(0, num_groups - htq_count),
            'constraint_achievable': htq_count >= num_groups
        }
        
        # Gender Constraint Analysis
        gender_ratio = preprocessed_data['gender_ratio']
        avg_group_size = preprocessed_data['group_size_info']['average_size']
        
        analysis['gender_constraint'] = {
            'global_ratio_lk': gender_ratio['LK'],
            'global_ratio_pr': gender_ratio['PR'],
            'target_lk_per_group': avg_group_size * gender_ratio['LK'],
            'target_pr_per_group': avg_group_size * gender_ratio['PR'],
            'balance_achievable': True  # Usually achievable with proper algorithm
        }
        
        # Duplication Constraint Analysis
        duplicate_risk = preprocessed_data['duplicate_risk_majors']
        total_at_risk = sum([
            preprocessed_data['major_distribution'][major] 
            for major in duplicate_risk
        ])
        
        analysis['duplication_constraint'] = {
            'high_risk_majors': len(duplicate_risk),
            'students_at_risk': total_at_risk,
            'risk_percentage': total_at_risk / preprocessed_data['total_students'] * 100,
            'major_risk_list': duplicate_risk[:10]  # Top 10 risky majors
        }
        
        # Size Constraint Analysis
        size_info = preprocessed_data['group_size_info']
        
        analysis['size_constraint'] = {
            'average_size': size_info['average_size'],
            'size_variance': size_info['max_size'] - size_info['min_size'],
            'balance_achievable': size_info['size_variance'] <= 1,
            'distribution': {
                f"size_{size_info['min_size']}": size_info['groups_with_min_size'],
                f"size_{size_info['max_size']}": size_info['groups_with_max_size']
            }
        }
        
        return analysis
    
    def get_preprocessing_summary(self, preprocessed_data: Dict[str, Any]) -> str:
        """
        Generate preprocessing summary text
        
        Args:
            preprocessed_data: Preprocessed data structure
            
        Returns:
            str: Summary text
        """
        summary = []
        summary.append("=== DATA PREPROCESSING SUMMARY ===")
        summary.append(f"Total Students: {preprocessed_data['total_students']}")
        summary.append(f"Target Groups: {preprocessed_data['num_groups']}")
        summary.append(f"Average Group Size: {preprocessed_data['group_size_info']['average_size']:.1f}")
        summary.append("")
        
        # HTQ Information
        summary.append("HTQ Distribution:")
        summary.append(f"  - HTQ 'Ya': {preprocessed_data['htq_count']}")
        summary.append(f"  - HTQ 'Tidak': {preprocessed_data['total_students'] - preprocessed_data['htq_count']}")
        summary.append("")
        
        # Gender Information
        total = preprocessed_data['total_students']
        lk_count = int(preprocessed_data['gender_ratio']['LK'] * total)
        pr_count = int(preprocessed_data['gender_ratio']['PR'] * total)
        summary.append("Gender Distribution:")
        summary.append(f"  - LK: {lk_count} ({preprocessed_data['gender_ratio']['LK']:.1%})")
        summary.append(f"  - PR: {pr_count} ({preprocessed_data['gender_ratio']['PR']:.1%})")
        summary.append("")
        
        # Faculty Information
        summary.append(f"Faculties: {len(preprocessed_data['faculty_distribution'])}")
        summary.append(f"Majors: {len(preprocessed_data['major_distribution'])}")
        summary.append(f"High-risk Duplication Majors: {len(preprocessed_data['duplicate_risk_majors'])}")
        
        return "\n".join(summary)


def preprocess_master_data(file_path: str = None, num_groups: int = 190) -> Dict[str, Any]:
    """
    Convenience function to preprocess master data
    
    Args:
        file_path: Path to master data CSV
        num_groups: Target number of groups
        
    Returns:
        dict: Preprocessed data structure
    """
    import os
    from .data_loader import DataLoader
    
    if file_path is None:
        file_path = os.path.join('data', 'master_data.csv')
    
    loader = DataLoader()
    df = loader.load_csv(file_path)
    
    preprocessor = DataPreprocessor()
    return preprocessor.preprocess(df, num_groups)