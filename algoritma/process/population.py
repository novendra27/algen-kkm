"""
Population initialization module for GA KKM Grouping
Implements 3-phase smart initialization as specified in konteks-algen.md
"""

import random
import numpy as np
from typing import List, Dict, Any
import math


class PopulationInitializer:
    """Class for initializing GA population with smart strategies"""
    
    def __init__(self, preprocessed_data: Dict[str, Any]):
        """
        Initialize population initializer
        
        Args:
            preprocessed_data: Preprocessed data structure
        """
        self.data = preprocessed_data
        self.total_students = preprocessed_data['total_students']
        self.num_groups = preprocessed_data['num_groups']
        self.htq_indices = preprocessed_data['htq_indices']
        self.gender_indices = preprocessed_data['gender_indices']
        self.student_attributes = preprocessed_data['student_attributes']
        self.group_size_info = preprocessed_data['group_size_info']
    
    def initialize_population(self, population_size: int) -> List[List[int]]:
        """
        Initialize population using 3-phase strategy
        
        Args:
            population_size: Size of population to create
            
        Returns:
            List[List[int]]: Population of chromosomes
        """
        population = []
        
        print(f"Initializing population with {population_size} chromosomes...")
        print("Using 3-phase initialization strategy:")
        print("  Phase 1: Smart HTQ assignment")
        print("  Phase 2: Size constraint setup")
        print("  Phase 3: Pure random assignment")
        
        for i in range(population_size):
            chromosome = self._create_chromosome()
            population.append(chromosome)
            
            if (i + 1) % (population_size // 10) == 0:
                print(f"  Progress: {i + 1}/{population_size} chromosomes created")
        
        print("Population initialization completed!")
        return population
    
    def _create_chromosome(self) -> List[int]:
        """
        Create single chromosome using 3-phase initialization
        
        Returns:
            List[int]: Single chromosome
        """
        # Initialize chromosome with zeros
        chromosome = [0] * self.total_students
        
        # Phase 1: Smart HTQ Assignment
        self._phase1_htq_assignment(chromosome)
        
        # Phase 2: Setup Size Constraints
        self._phase2_size_constraints(chromosome)
        
        # Phase 3: Pure Random Assignment (MUST be completely random)
        self._phase3_random_assignment(chromosome)
        
        return chromosome
    
    def _phase1_htq_assignment(self, chromosome: List[int]):
        """
        Phase 1: Smart HTQ Assignment
        Assign HTQ students to ensure maximum groups can have HTQ
        
        Args:
            chromosome: Chromosome to modify
        """
        htq_students = self.htq_indices.copy()
        available_groups = list(range(1, self.num_groups + 1))
        
        # Shuffle both lists for randomness
        random.shuffle(htq_students)
        random.shuffle(available_groups)
        
        # Assign HTQ students to groups (one per group if possible)
        groups_needing_htq = min(len(htq_students), len(available_groups))
        
        for i in range(groups_needing_htq):
            student_idx = htq_students[i]
            group_id = available_groups[i]
            chromosome[student_idx] = group_id
    
    def _phase2_size_constraints(self, chromosome: List[int]):
        """
        Phase 2: Setup Size Constraints
        Prepare for balanced group sizes but don't enforce strictly
        
        Args:
            chromosome: Chromosome to modify
        """
        # This phase is preparation for size balancing
        # We don't enforce strict constraints here as it would violate
        # the "Phase 3 must be pure random" requirement
        
        # Calculate target sizes for reference
        target_size = self.total_students / self.num_groups
        min_size = math.floor(target_size)
        max_size = math.ceil(target_size)
        
        # Store size info for potential use (but don't enforce in Phase 3)
        self.target_min_size = min_size
        self.target_max_size = max_size
    
    def _phase3_random_assignment(self, chromosome: List[int]):
        """
        Phase 3: Pure Random Assignment (NO RULES APPLIED)
        Assign all unassigned students randomly to groups
        
        Args:
            chromosome: Chromosome to modify
        """
        # Find unassigned students
        unassigned_indices = [
            i for i in range(self.total_students) 
            if chromosome[i] == 0
        ]
        
        # PURE RANDOM assignment - no rules whatsoever
        for student_idx in unassigned_indices:
            random_group = random.randint(1, self.num_groups)
            chromosome[student_idx] = random_group
    
    def create_random_chromosome(self) -> List[int]:
        """
        Create completely random chromosome (for comparison/testing)
        
        Returns:
            List[int]: Random chromosome
        """
        chromosome = []
        for _ in range(self.total_students):
            random_group = random.randint(1, self.num_groups)
            chromosome.append(random_group)
        return chromosome
    
    def create_balanced_chromosome(self) -> List[int]:
        """
        Create chromosome with focus on size balance
        
        Returns:
            List[int]: Balanced chromosome
        """
        chromosome = [0] * self.total_students
        
        # Calculate how many students per group
        base_size = self.total_students // self.num_groups
        remainder = self.total_students % self.num_groups
        
        # Create group size targets
        group_sizes = [base_size] * self.num_groups
        for i in range(remainder):
            group_sizes[i] += 1
        
        # Assign students to groups based on size targets
        student_indices = list(range(self.total_students))
        random.shuffle(student_indices)
        
        current_student = 0
        for group_id in range(1, self.num_groups + 1):
            target_size = group_sizes[group_id - 1]
            
            for _ in range(target_size):
                if current_student < len(student_indices):
                    student_idx = student_indices[current_student]
                    chromosome[student_idx] = group_id
                    current_student += 1
        
        return chromosome
    
    def create_htq_focused_chromosome(self) -> List[int]:
        """
        Create chromosome with focus on HTQ distribution
        
        Returns:
            List[int]: HTQ-focused chromosome
        """
        chromosome = [0] * self.total_students
        
        # First assign HTQ students (one per group)
        htq_students = self.htq_indices.copy()
        random.shuffle(htq_students)
        
        for i, student_idx in enumerate(htq_students[:self.num_groups]):
            chromosome[student_idx] = i + 1
        
        # Then assign remaining students randomly
        unassigned = [i for i in range(self.total_students) if chromosome[i] == 0]
        
        for student_idx in unassigned:
            chromosome[student_idx] = random.randint(1, self.num_groups)
        
        return chromosome
    
    def validate_chromosome(self, chromosome: List[int]) -> Dict[str, Any]:
        """
        Validate chromosome integrity
        
        Args:
            chromosome: Chromosome to validate
            
        Returns:
            dict: Validation results
        """
        validation = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        # Check chromosome length
        if len(chromosome) != self.total_students:
            validation['errors'].append(
                f"Chromosome length {len(chromosome)} != total students {self.total_students}"
            )
            validation['is_valid'] = False
        
        # Check group ID ranges
        invalid_groups = [g for g in chromosome if g < 1 or g > self.num_groups]
        if invalid_groups:
            validation['errors'].append(
                f"Found {len(invalid_groups)} invalid group IDs (must be 1-{self.num_groups})"
            )
            validation['is_valid'] = False
        
        # Calculate statistics
        if validation['is_valid']:
            from collections import Counter
            group_counts = Counter(chromosome)
            
            validation['stats'] = {
                'groups_used': len(group_counts),
                'empty_groups': self.num_groups - len(group_counts),
                'min_group_size': min(group_counts.values()) if group_counts else 0,
                'max_group_size': max(group_counts.values()) if group_counts else 0,
                'avg_group_size': sum(group_counts.values()) / len(group_counts) if group_counts else 0
            }
            
            # Check for empty groups
            if validation['stats']['empty_groups'] > 0:
                validation['warnings'].append(
                    f"{validation['stats']['empty_groups']} groups are empty"
                )
        
        return validation
    
    def get_population_diversity(self, population: List[List[int]]) -> float:
        """
        Calculate population diversity (average pairwise differences)
        
        Args:
            population: List of chromosomes
            
        Returns:
            float: Diversity score (0.0 to 1.0)
        """
        if len(population) < 2:
            return 0.0
        
        total_differences = 0
        comparisons = 0
        
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                differences = sum(
                    1 for a, b in zip(population[i], population[j]) if a != b
                )
                total_differences += differences
                comparisons += 1
        
        avg_differences = total_differences / comparisons
        max_possible_differences = self.total_students
        
        return avg_differences / max_possible_differences
    
    def analyze_population(self, population: List[List[int]]) -> Dict[str, Any]:
        """
        Analyze population characteristics
        
        Args:
            population: List of chromosomes
            
        Returns:
            dict: Population analysis
        """
        analysis = {
            'population_size': len(population),
            'diversity': self.get_population_diversity(population),
            'validation_summary': {
                'valid_chromosomes': 0,
                'invalid_chromosomes': 0,
                'total_errors': 0,
                'total_warnings': 0
            }
        }
        
        # Validate each chromosome
        for chromosome in population:
            validation = self.validate_chromosome(chromosome)
            if validation['is_valid']:
                analysis['validation_summary']['valid_chromosomes'] += 1
            else:
                analysis['validation_summary']['invalid_chromosomes'] += 1
            
            analysis['validation_summary']['total_errors'] += len(validation['errors'])
            analysis['validation_summary']['total_warnings'] += len(validation['warnings'])
        
        return analysis


def create_initial_population(preprocessed_data: Dict[str, Any], 
                            population_size: int) -> List[List[int]]:
    """
    Convenience function to create initial population
    
    Args:
        preprocessed_data: Preprocessed data structure
        population_size: Size of population
        
    Returns:
        List[List[int]]: Initial population
    """
    initializer = PopulationInitializer(preprocessed_data)
    return initializer.initialize_population(population_size)