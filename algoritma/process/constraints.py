"""
Constraints checking module for GA KKM Grouping
Contains functions for checking and scoring 4 main constraints
"""

import numpy as np
from typing import Dict, List, Any, Tuple
from collections import Counter
import math


class ConstraintChecker:
    """Class for checking and scoring constraints"""
    
    def __init__(self, preprocessed_data: Dict[str, Any]):
        """
        Initialize constraint checker
        
        Args:
            preprocessed_data: Preprocessed data structure
        """
        self.data = preprocessed_data
        self.total_students = preprocessed_data['total_students']
        self.num_groups = preprocessed_data['num_groups']
        self.student_attributes = preprocessed_data['student_attributes']
        self.gender_ratio = preprocessed_data['gender_ratio']
        self.group_size_info = preprocessed_data['group_size_info']
    
    def check_all_constraints(self, chromosome: List[int]) -> Dict[str, float]:
        """
        Check all constraints and return scores
        
        Args:
            chromosome: GA chromosome (group assignments)
            
        Returns:
            dict: Constraint scores
        """
        # Decode chromosome to groups
        groups = self.decode_chromosome_to_groups(chromosome)
        
        # Calculate constraint scores
        scores = {
            'htq_score': self.check_htq_constraint(groups),
            'duplication_score': self.check_duplication_constraint(groups),
            'gender_score': self.check_gender_constraint(groups),
            'size_score': self.check_size_constraint(groups)
        }
        
        return scores
    
    def decode_chromosome_to_groups(self, chromosome: List[int]) -> Dict[int, List[int]]:
        """
        Decode chromosome to group assignments
        
        Args:
            chromosome: List of group assignments for each student
            
        Returns:
            dict: Groups with student indices
        """
        groups = {}
        
        for student_idx, group_id in enumerate(chromosome):
            if group_id not in groups:
                groups[group_id] = []
            groups[group_id].append(student_idx)
        
        return groups
    
    def check_htq_constraint(self, groups: Dict[int, List[int]]) -> float:
        """
        Check HTQ constraint: each group should have at least 1 HTQ student
        
        Args:
            groups: Dictionary of group_id -> [student_indices]
            
        Returns:
            float: HTQ constraint score (0.0 to 1.0)
        """
        htq_satisfied_groups = 0
        
        for group_id, student_indices in groups.items():
            htq_count = 0
            
            for student_idx in student_indices:
                if self.student_attributes[student_idx]['is_htq']:
                    htq_count += 1
            
            # Group satisfies HTQ constraint if it has >= 1 HTQ student
            if htq_count >= 1:
                htq_satisfied_groups += 1
        
        # Calculate score as proportion of groups satisfying constraint
        score = htq_satisfied_groups / self.num_groups if self.num_groups > 0 else 0.0
        return min(score, 1.0)
    
    def check_duplication_constraint(self, groups: Dict[int, List[int]]) -> float:
        """
        Check duplication constraint: minimize major duplications within groups
        
        Args:
            groups: Dictionary of group_id -> [student_indices]
            
        Returns:
            float: Duplication constraint score (0.0 to 1.0)
        """
        satisfied_groups = 0
        
        for group_id, student_indices in groups.items():
            # Count majors in this group
            major_counts = Counter()
            
            for student_idx in student_indices:
                major = self.student_attributes[student_idx]['jurusan']
                major_counts[major] += 1
            
            # Check if duplications are acceptable
            group_size = len(student_indices)
            max_duplicates = max(1, group_size // 2)  # Allow up to half the group size
            
            # Find maximum duplication count
            max_major_count = max(major_counts.values()) if major_counts else 0
            
            # Group satisfies constraint if duplications <= threshold
            if max_major_count <= max_duplicates:
                satisfied_groups += 1
        
        # Calculate score
        score = satisfied_groups / self.num_groups if self.num_groups > 0 else 0.0
        return min(score, 1.0)
    
    def check_gender_constraint(self, groups: Dict[int, List[int]]) -> float:
        """
        Check gender constraint: maintain proportional gender distribution
        
        Args:
            groups: Dictionary of group_id -> [student_indices]
            
        Returns:
            float: Gender constraint score (0.0 to 1.0)
        """
        satisfied_groups = 0
        
        for group_id, student_indices in groups.items():
            group_size = len(student_indices)
            if group_size == 0:
                continue
            
            # Count genders in this group
            lk_count = sum(1 for idx in student_indices 
                          if self.student_attributes[idx]['is_lk'])
            pr_count = group_size - lk_count
            
            # Calculate target counts based on global ratio
            target_lk = group_size * self.gender_ratio['LK']
            target_pr = group_size * self.gender_ratio['PR']
            
            # Calculate acceptable range (±1 from target)
            lk_min = max(0, math.floor(target_lk - 1))
            lk_max = min(group_size, math.ceil(target_lk + 1))
            
            # Check if group satisfies gender balance
            if lk_min <= lk_count <= lk_max:
                satisfied_groups += 1
        
        # Calculate score
        score = satisfied_groups / self.num_groups if self.num_groups > 0 else 0.0
        return min(score, 1.0)
    
    def check_size_constraint(self, groups: Dict[int, List[int]]) -> float:
        """
        Check size constraint: groups should have balanced sizes
        
        Args:
            groups: Dictionary of group_id -> [student_indices]
            
        Returns:
            float: Size constraint score (0.0 to 1.0)
        """
        size_range = self.group_size_info['size_range']
        min_size, max_size = size_range[0], size_range[1]
        
        satisfied_groups = 0
        
        for group_id, student_indices in groups.items():
            group_size = len(student_indices)
            
            # Group satisfies constraint if size is within acceptable range
            if min_size <= group_size <= max_size:
                satisfied_groups += 1
        
        # Calculate score
        score = satisfied_groups / self.num_groups if self.num_groups > 0 else 0.0
        return min(score, 1.0)
    
    def get_detailed_constraint_analysis(self, chromosome: List[int]) -> Dict[str, Any]:
        """
        Get detailed analysis of constraint satisfaction
        
        Args:
            chromosome: GA chromosome
            
        Returns:
            dict: Detailed constraint analysis
        """
        groups = self.decode_chromosome_to_groups(chromosome)
        
        analysis = {
            'htq_analysis': self._analyze_htq_constraint(groups),
            'duplication_analysis': self._analyze_duplication_constraint(groups),
            'gender_analysis': self._analyze_gender_constraint(groups),
            'size_analysis': self._analyze_size_constraint(groups)
        }
        
        return analysis
    
    def _analyze_htq_constraint(self, groups: Dict[int, List[int]]) -> Dict[str, Any]:
        """Detailed HTQ constraint analysis"""
        analysis = {
            'groups_with_htq': 0,
            'groups_without_htq': 0,
            'htq_distribution': [],
            'total_htq_students': 0
        }
        
        for group_id, student_indices in groups.items():
            htq_count = sum(1 for idx in student_indices 
                           if self.student_attributes[idx]['is_htq'])
            
            analysis['htq_distribution'].append({
                'group_id': group_id,
                'htq_count': htq_count,
                'group_size': len(student_indices)
            })
            
            if htq_count > 0:
                analysis['groups_with_htq'] += 1
            else:
                analysis['groups_without_htq'] += 1
            
            analysis['total_htq_students'] += htq_count
        
        return analysis
    
    def _analyze_duplication_constraint(self, groups: Dict[int, List[int]]) -> Dict[str, Any]:
        """Detailed duplication constraint analysis"""
        analysis = {
            'groups_with_duplications': 0,
            'total_duplications': 0,
            'worst_duplications': []
        }
        
        for group_id, student_indices in groups.items():
            major_counts = Counter()
            
            for student_idx in student_indices:
                major = self.student_attributes[student_idx]['jurusan']
                major_counts[major] += 1
            
            # Find duplications in this group
            group_duplications = []
            for major, count in major_counts.items():
                if count > 1:
                    group_duplications.append({'major': major, 'count': count})
                    analysis['total_duplications'] += (count - 1)
            
            if group_duplications:
                analysis['groups_with_duplications'] += 1
                analysis['worst_duplications'].append({
                    'group_id': group_id,
                    'duplications': group_duplications,
                    'group_size': len(student_indices)
                })
        
        # Sort worst duplications by severity
        analysis['worst_duplications'].sort(
            key=lambda x: sum(d['count'] for d in x['duplications']), 
            reverse=True
        )
        
        return analysis
    
    def _analyze_gender_constraint(self, groups: Dict[int, List[int]]) -> Dict[str, Any]:
        """Detailed gender constraint analysis"""
        analysis = {
            'balanced_groups': 0,
            'imbalanced_groups': [],
            'gender_distribution': []
        }
        
        for group_id, student_indices in groups.items():
            group_size = len(student_indices)
            if group_size == 0:
                continue
            
            lk_count = sum(1 for idx in student_indices 
                          if self.student_attributes[idx]['is_lk'])
            pr_count = group_size - lk_count
            
            # Calculate target and actual ratios
            target_lk = group_size * self.gender_ratio['LK']
            actual_ratio_lk = lk_count / group_size
            
            analysis['gender_distribution'].append({
                'group_id': group_id,
                'group_size': group_size,
                'lk_count': lk_count,
                'pr_count': pr_count,
                'lk_ratio': actual_ratio_lk,
                'target_lk': target_lk
            })
            
            # Check if balanced
            lk_min = max(0, math.floor(target_lk - 1))
            lk_max = min(group_size, math.ceil(target_lk + 1))
            
            if lk_min <= lk_count <= lk_max:
                analysis['balanced_groups'] += 1
            else:
                analysis['imbalanced_groups'].append({
                    'group_id': group_id,
                    'lk_count': lk_count,
                    'target_range': [lk_min, lk_max],
                    'deviation': min(abs(lk_count - lk_min), abs(lk_count - lk_max))
                })
        
        return analysis
    
    def _analyze_size_constraint(self, groups: Dict[int, List[int]]) -> Dict[str, Any]:
        """Detailed size constraint analysis"""
        size_range = self.group_size_info['size_range']
        min_size, max_size = size_range[0], size_range[1]
        
        analysis = {
            'balanced_groups': 0,
            'size_distribution': Counter(),
            'oversized_groups': [],
            'undersized_groups': []
        }
        
        for group_id, student_indices in groups.items():
            group_size = len(student_indices)
            analysis['size_distribution'][group_size] += 1
            
            if min_size <= group_size <= max_size:
                analysis['balanced_groups'] += 1
            elif group_size > max_size:
                analysis['oversized_groups'].append({
                    'group_id': group_id,
                    'size': group_size,
                    'excess': group_size - max_size
                })
            else:  # group_size < min_size
                analysis['undersized_groups'].append({
                    'group_id': group_id,
                    'size': group_size,
                    'deficit': min_size - group_size
                })
        
        return analysis


def calculate_fitness_from_constraints(constraint_scores: Dict[str, float], 
                                     weights: Dict[str, float]) -> float:
    """
    Calculate total fitness from constraint scores and weights
    
    Args:
        constraint_scores: Dictionary of constraint scores
        weights: Dictionary of constraint weights
        
    Returns:
        float: Total fitness score
    """
    total_fitness = 0.0
    
    total_fitness += constraint_scores.get('htq_score', 0.0) * weights.get('weight_htq', 0.0)
    total_fitness += constraint_scores.get('duplication_score', 0.0) * weights.get('weight_duplikasi', 0.0)
    total_fitness += constraint_scores.get('gender_score', 0.0) * weights.get('weight_gender', 0.0)
    total_fitness += constraint_scores.get('size_score', 0.0) * weights.get('weight_jumlah', 0.0)
    
    return min(total_fitness, 1.0)  # Ensure fitness doesn't exceed 1.0