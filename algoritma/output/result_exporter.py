"""
Result exporter module for GA KKM Grouping
Exports grouping results to CSV format
"""

import pandas as pd
import os
import csv
from typing import Dict, List, Any
from datetime import datetime
from collections import Counter


class ResultExporter:
    """Class for exporting GA results to various formats"""
    
    def __init__(self, output_dir: str):
        """
        Initialize result exporter
        
        Args:
            output_dir: Directory for output files
        """
        self.output_dir = output_dir
        
        # Create subfolder structure
        self.grouping_dir = os.path.join(output_dir, 'grouping_results')
        self.statistics_dir = os.path.join(output_dir, 'statistics')
        
        # Ensure all directories exist
        os.makedirs(self.grouping_dir, exist_ok=True)
        os.makedirs(self.statistics_dir, exist_ok=True)
    
    def export_grouping_results(self, original_data: pd.DataFrame, 
                              best_chromosome: List[int],
                              preprocessed_data: Dict[str, Any],
                              filename: str = None) -> str:
        """
        Export final grouping results to CSV
        
        Args:
            original_data: Original student data
            best_chromosome: Best chromosome (group assignments)
            preprocessed_data: Preprocessed data structure
            filename: Custom filename (optional)
            
        Returns:
            str: Path to exported file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"hasil_pengelompokan_{timestamp}.csv"
        
        filepath = os.path.join(self.grouping_dir, filename)
        
        # Create result DataFrame
        result_df = original_data.copy()
        result_df['Kelompok_Baru'] = best_chromosome
        
        # Add additional analysis columns for monitoring
        result_df['Is_HTQ'] = result_df['Htq'] == 'Ya'
        result_df['Gender_Code'] = result_df['Gender'].map({'LK': 1, 'PR': 0})
        
        # Calculate group statistics for validation
        group_stats = self._calculate_group_statistics(result_df, preprocessed_data)
        
        # Export main results
        result_df.to_csv(filepath, index=False, encoding='utf-8-sig')
        
        # Export group statistics
        stats_filename = filename.replace('.csv', '_group_stats.csv')
        stats_filepath = os.path.join(self.statistics_dir, stats_filename)
        group_stats.to_csv(stats_filepath, index=False, encoding='utf-8-sig')
        
        print(f"✅ Grouping results exported to: {filepath}")
        print(f"✅ Group statistics exported to: {stats_filepath}")
        
        return filepath
    
    def _calculate_group_statistics(self, result_df: pd.DataFrame, 
                                  preprocessed_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Calculate detailed statistics for each group
        
        Args:
            result_df: Results DataFrame with group assignments
            preprocessed_data: Preprocessed data structure
            
        Returns:
            pd.DataFrame: Group statistics
        """
        group_stats = []
        
        for group_id in sorted(result_df['Kelompok_Baru'].unique()):
            group_data = result_df[result_df['Kelompok_Baru'] == group_id]
            
            # Basic counts
            total_students = len(group_data)
            htq_count = len(group_data[group_data['Is_HTQ'] == True])
            lk_count = len(group_data[group_data['Gender'] == 'LK'])
            pr_count = len(group_data[group_data['Gender'] == 'PR'])
            
            # Faculty and major analysis
            faculties = group_data['Fakultas'].value_counts()
            majors = group_data['Jurusan'].value_counts()
            
            # Duplication analysis
            max_major_count = majors.max() if len(majors) > 0 else 0
            duplicate_majors = len(majors[majors > 1])
            
            # Gender ratio
            lk_ratio = lk_count / total_students if total_students > 0 else 0
            pr_ratio = pr_count / total_students if total_students > 0 else 0
            
            # Constraint satisfaction
            htq_satisfied = htq_count >= 1
            size_satisfied = (
                preprocessed_data['group_size_info']['min_size'] <= 
                total_students <= 
                preprocessed_data['group_size_info']['max_size']
            )
            
            # Gender balance check
            global_lk_ratio = preprocessed_data['gender_ratio']['LK']
            target_lk = total_students * global_lk_ratio
            lk_min = max(0, int(target_lk - 1))
            lk_max = min(total_students, int(target_lk + 1))
            gender_satisfied = lk_min <= lk_count <= lk_max
            
            # Duplication check
            duplication_satisfied = max_major_count <= (total_students // 2) if total_students > 0 else True
            
            group_stats.append({
                'Kelompok_ID': group_id,
                'Total_Mahasiswa': total_students,
                'HTQ_Count': htq_count,
                'HTQ_Percentage': (htq_count / total_students * 100) if total_students > 0 else 0,
                'LK_Count': lk_count,
                'PR_Count': pr_count,
                'LK_Ratio': lk_ratio,
                'PR_Ratio': pr_ratio,
                'Fakultas_Count': len(faculties),
                'Jurusan_Count': len(majors),
                'Duplicate_Majors': duplicate_majors,
                'Max_Major_Count': max_major_count,
                'HTQ_Satisfied': htq_satisfied,
                'Size_Satisfied': size_satisfied,
                'Gender_Satisfied': gender_satisfied,
                'Duplication_Satisfied': duplication_satisfied,
                'Overall_Satisfied': htq_satisfied and size_satisfied and gender_satisfied and duplication_satisfied,
                'Dominant_Fakultas': faculties.index[0] if len(faculties) > 0 else 'N/A',
                'Dominant_Jurusan': majors.index[0] if len(majors) > 0 else 'N/A'
            })
        
        return pd.DataFrame(group_stats)
    
    def export_detailed_grouping(self, original_data: pd.DataFrame, 
                               best_chromosome: List[int],
                               preprocessed_data: Dict[str, Any],
                               solution_analysis: Dict[str, Any],
                               filename: str = None) -> str:
        """
        Export detailed grouping with constraint analysis
        
        Args:
            original_data: Original student data
            best_chromosome: Best chromosome
            preprocessed_data: Preprocessed data structure
            solution_analysis: Solution analysis from GA
            filename: Custom filename
            
        Returns:
            str: Path to exported file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"detailed_grouping_{timestamp}.csv"
        
        filepath = os.path.join(self.grouping_dir, filename)
        
        # Create detailed result DataFrame
        result_df = original_data.copy()
        result_df['Kelompok_Baru'] = best_chromosome
        
        # Add constraint satisfaction indicators
        constraint_analysis = solution_analysis.get('constraint_analysis', {})
        
        # Add group-level constraint satisfaction
        group_htq_info = self._get_group_constraint_info(
            result_df, 'htq', constraint_analysis.get('htq_analysis', {})
        )
        group_gender_info = self._get_group_constraint_info(
            result_df, 'gender', constraint_analysis.get('gender_analysis', {})
        )
        group_size_info = self._get_group_constraint_info(
            result_df, 'size', constraint_analysis.get('size_analysis', {})
        )
        
        # Add constraint indicators to student records
        result_df['Group_HTQ_Satisfied'] = result_df['Kelompok_Baru'].map(group_htq_info)
        result_df['Group_Gender_Satisfied'] = result_df['Kelompok_Baru'].map(group_gender_info)
        result_df['Group_Size_Satisfied'] = result_df['Kelompok_Baru'].map(group_size_info)
        
        # Add fitness information
        fitness_info = solution_analysis.get('fitness_analysis', {})
        result_df['Solution_Fitness'] = fitness_info.get('total_fitness', 0.0)
        
        # Export detailed results
        result_df.to_csv(filepath, index=False, encoding='utf-8-sig')
        
        print(f"✅ Detailed grouping results exported to: {filepath}")
        return filepath
    
    def _get_group_constraint_info(self, result_df: pd.DataFrame, 
                                 constraint_type: str, 
                                 constraint_analysis: Dict[str, Any]) -> Dict[int, bool]:
        """Get constraint satisfaction info for each group"""
        group_info = {}
        
        if constraint_type == 'htq':
            htq_analysis = constraint_analysis.get('htq_distribution', [])
            for group_data in htq_analysis:
                group_id = group_data.get('group_id')
                htq_count = group_data.get('htq_count', 0)
                group_info[group_id] = htq_count >= 1
        
        elif constraint_type == 'gender':
            balanced_groups = constraint_analysis.get('balanced_groups', 0)
            imbalanced = constraint_analysis.get('imbalanced_groups', [])
            imbalanced_ids = [g['group_id'] for g in imbalanced]
            
            for group_id in result_df['Kelompok_Baru'].unique():
                group_info[group_id] = group_id not in imbalanced_ids
        
        elif constraint_type == 'size':
            balanced_groups = constraint_analysis.get('balanced_groups', 0)
            oversized = constraint_analysis.get('oversized_groups', [])
            undersized = constraint_analysis.get('undersized_groups', [])
            
            problem_ids = ([g['group_id'] for g in oversized] + 
                          [g['group_id'] for g in undersized])
            
            for group_id in result_df['Kelompok_Baru'].unique():
                group_info[group_id] = group_id not in problem_ids
        
        return group_info
    
    def export_constraint_violation_report(self, original_data: pd.DataFrame, 
                                         best_chromosome: List[int],
                                         solution_analysis: Dict[str, Any],
                                         filename: str = None) -> str:
        """
        Export detailed constraint violation report
        
        Args:
            original_data: Original student data
            best_chromosome: Best chromosome
            solution_analysis: Solution analysis
            filename: Custom filename
            
        Returns:
            str: Path to exported file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"constraint_violations_{timestamp}.csv"
        
        filepath = os.path.join(self.statistics_dir, filename)
        
        constraint_analysis = solution_analysis.get('constraint_analysis', {})
        violations = []
        
        # HTQ violations
        htq_analysis = constraint_analysis.get('htq_analysis', {})
        groups_without_htq = htq_analysis.get('groups_without_htq', 0)
        if groups_without_htq > 0:
            htq_distribution = htq_analysis.get('htq_distribution', [])
            for group_data in htq_distribution:
                if group_data.get('htq_count', 0) == 0:
                    violations.append({
                        'Violation_Type': 'HTQ',
                        'Group_ID': group_data.get('group_id'),
                        'Description': 'No HTQ student in group',
                        'Severity': 'High',
                        'Details': f"Group size: {group_data.get('group_size', 0)}"
                    })
        
        # Gender violations
        gender_analysis = constraint_analysis.get('gender_analysis', {})
        imbalanced_groups = gender_analysis.get('imbalanced_groups', [])
        for group in imbalanced_groups:
            violations.append({
                'Violation_Type': 'Gender',
                'Group_ID': group.get('group_id'),
                'Description': 'Gender imbalance',
                'Severity': 'Medium',
                'Details': f"LK count: {group.get('lk_count')}, Target range: {group.get('target_range')}"
            })
        
        # Size violations
        size_analysis = constraint_analysis.get('size_analysis', {})
        oversized = size_analysis.get('oversized_groups', [])
        undersized = size_analysis.get('undersized_groups', [])
        
        for group in oversized:
            violations.append({
                'Violation_Type': 'Size',
                'Group_ID': group.get('group_id'),
                'Description': 'Group too large',
                'Severity': 'Medium',
                'Details': f"Size: {group.get('size')}, Excess: {group.get('excess')}"
            })
        
        for group in undersized:
            violations.append({
                'Violation_Type': 'Size',
                'Group_ID': group.get('group_id'),
                'Description': 'Group too small',
                'Severity': 'Medium',
                'Details': f"Size: {group.get('size')}, Deficit: {group.get('deficit')}"
            })
        
        # Duplication violations
        duplication_analysis = constraint_analysis.get('duplication_analysis', {})
        worst_duplications = duplication_analysis.get('worst_duplications', [])
        
        for group in worst_duplications:
            duplications = group.get('duplications', [])
            for dup in duplications:
                if dup.get('count', 0) > 2:  # Only report severe duplications
                    violations.append({
                        'Violation_Type': 'Duplication',
                        'Group_ID': group.get('group_id'),
                        'Description': f"Major duplication: {dup.get('major')}",
                        'Severity': 'Low' if dup.get('count') <= 3 else 'Medium',
                        'Details': f"Count: {dup.get('count')}, Group size: {group.get('group_size')}"
                    })
        
        # Export violations
        if violations:
            violations_df = pd.DataFrame(violations)
            violations_df.to_csv(filepath, index=False, encoding='utf-8-sig')
            print(f"✅ Constraint violation report exported to: {filepath}")
        else:
            # Create empty report indicating no violations
            empty_df = pd.DataFrame([{
                'Violation_Type': 'None',
                'Group_ID': 'N/A',
                'Description': 'No constraint violations found',
                'Severity': 'None',
                'Details': 'All constraints satisfied'
            }])
            empty_df.to_csv(filepath, index=False, encoding='utf-8-sig')
            print(f"✅ No violations found - empty report exported to: {filepath}")
        
        return filepath
    
    def export_summary_statistics(self, solution_analysis: Dict[str, Any],
                                preprocessed_data: Dict[str, Any],
                                filename: str = None) -> str:
        """
        Export summary statistics to CSV
        
        Args:
            solution_analysis: Complete solution analysis
            preprocessed_data: Preprocessed data structure
            filename: Custom filename
            
        Returns:
            str: Path to exported file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"solution_summary_{timestamp}.csv"
        
        filepath = os.path.join(self.statistics_dir, filename)
        
        # Extract key metrics
        fitness_analysis = solution_analysis.get('fitness_analysis', {})
        constraint_analysis = solution_analysis.get('constraint_analysis', {})
        
        # Create summary data
        summary_data = [
            ['Metric', 'Value', 'Description'],
            ['Total Fitness', fitness_analysis.get('total_fitness', 0), 'Overall solution fitness'],
            ['HTQ Score', fitness_analysis.get('constraint_scores', {}).get('htq_score', 0), 'HTQ constraint score'],
            ['Duplication Score', fitness_analysis.get('constraint_scores', {}).get('duplication_score', 0), 'Duplication constraint score'],
            ['Gender Score', fitness_analysis.get('constraint_scores', {}).get('gender_score', 0), 'Gender balance score'],
            ['Size Score', fitness_analysis.get('constraint_scores', {}).get('size_score', 0), 'Group size balance score'],
            ['Groups with HTQ', constraint_analysis.get('htq_analysis', {}).get('groups_with_htq', 0), 'Groups having at least 1 HTQ student'],
            ['Groups without HTQ', constraint_analysis.get('htq_analysis', {}).get('groups_without_htq', 0), 'Groups without HTQ student'],
            ['Balanced Gender Groups', constraint_analysis.get('gender_analysis', {}).get('balanced_groups', 0), 'Groups with gender balance'],
            ['Balanced Size Groups', constraint_analysis.get('size_analysis', {}).get('balanced_groups', 0), 'Groups with proper size'],
            ['Total Duplications', constraint_analysis.get('duplication_analysis', {}).get('total_duplications', 0), 'Total major duplications'],
            ['Groups with Duplications', constraint_analysis.get('duplication_analysis', {}).get('groups_with_duplications', 0), 'Groups having major duplications']
        ]
        
        # Write to CSV
        with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerows(summary_data)
        
        print(f"✅ Solution summary statistics exported to: {filepath}")
        return filepath


def export_ga_results(output_dir: str, original_data: pd.DataFrame, 
                     best_chromosome: List[int], preprocessed_data: Dict[str, Any],
                     solution_analysis: Dict[str, Any]) -> Dict[str, str]:
    """
    Convenience function to export all GA results
    
    Args:
        output_dir: Output directory
        original_data: Original student data
        best_chromosome: Best solution chromosome
        preprocessed_data: Preprocessed data structure
        solution_analysis: Complete solution analysis
        
    Returns:
        dict: Paths to all exported files
    """
    exporter = ResultExporter(output_dir)
    
    exported_files = {}
    
    # Export main grouping results
    exported_files['grouping_results'] = exporter.export_grouping_results(
        original_data, best_chromosome, preprocessed_data
    )
    
    # Export detailed grouping
    exported_files['detailed_grouping'] = exporter.export_detailed_grouping(
        original_data, best_chromosome, preprocessed_data, solution_analysis
    )
    
    # Export constraint violations
    exported_files['constraint_violations'] = exporter.export_constraint_violation_report(
        original_data, best_chromosome, solution_analysis
    )
    
    # Export summary statistics
    exported_files['summary_statistics'] = exporter.export_summary_statistics(
        solution_analysis, preprocessed_data
    )
    
    return exported_files