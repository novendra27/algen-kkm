"""
Fitness calculation module for GA KKM Grouping
Calculates fitness based on 4 constraints with configurable weights
"""

import numpy as np
from typing import Dict, List, Any, Tuple
from .constraints import ConstraintChecker, calculate_fitness_from_constraints


class FitnessCalculator:
    """Class for calculating chromosome fitness"""
    
    def __init__(self, preprocessed_data: Dict[str, Any], weights: Dict[str, float]):
        """
        Initialize fitness calculator
        
        Args:
            preprocessed_data: Preprocessed data structure
            weights: Constraint weights dictionary
        """
        self.data = preprocessed_data
        self.weights = weights
        self.constraint_checker = ConstraintChecker(preprocessed_data)
        
        # Validate weights
        self._validate_weights()
    
    def _validate_weights(self):
        """Validate constraint weights"""
        required_weights = ['weight_htq', 'weight_duplikasi', 'weight_gender', 'weight_jumlah']
        
        for weight in required_weights:
            if weight not in self.weights:
                raise ValueError(f"Missing weight: {weight}")
        
        total_weight = sum(self.weights[w] for w in required_weights)
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")
    
    def calculate_fitness(self, chromosome: List[int]) -> float:
        """
        Calculate fitness for a single chromosome
        
        Args:
            chromosome: GA chromosome (group assignments)
            
        Returns:
            float: Fitness score (0.0 to 1.0)
        """
        try:
            # Get constraint scores
            constraint_scores = self.constraint_checker.check_all_constraints(chromosome)
            
            # Calculate weighted fitness
            fitness = calculate_fitness_from_constraints(constraint_scores, self.weights)
            
            return fitness
        
        except Exception as e:
            # Return very low fitness for invalid chromosomes
            print(f"Error calculating fitness: {str(e)}")
            return 0.0
    
    def calculate_population_fitness(self, population: List[List[int]]) -> List[float]:
        """
        Calculate fitness for entire population
        
        Args:
            population: List of chromosomes
            
        Returns:
            List[float]: Fitness scores for each chromosome
        """
        fitness_scores = []
        
        for i, chromosome in enumerate(population):
            fitness = self.calculate_fitness(chromosome)
            fitness_scores.append(fitness)
            
            # Progress indicator for large populations
            if len(population) > 100 and (i + 1) % (len(population) // 10) == 0:
                print(f"  Fitness calculation progress: {i + 1}/{len(population)}")
        
        return fitness_scores
    
    def get_detailed_fitness(self, chromosome: List[int]) -> Dict[str, Any]:
        """
        Get detailed fitness breakdown
        
        Args:
            chromosome: GA chromosome
            
        Returns:
            dict: Detailed fitness information
        """
        # Get constraint scores
        constraint_scores = self.constraint_checker.check_all_constraints(chromosome)
        
        # Calculate individual contributions
        contributions = {
            'htq_contribution': constraint_scores['htq_score'] * self.weights['weight_htq'],
            'duplication_contribution': constraint_scores['duplication_score'] * self.weights['weight_duplikasi'],
            'gender_contribution': constraint_scores['gender_score'] * self.weights['weight_gender'],
            'size_contribution': constraint_scores['size_score'] * self.weights['weight_jumlah']
        }
        
        # Calculate total fitness
        total_fitness = sum(contributions.values())
        
        detailed_fitness = {
            'total_fitness': total_fitness,
            'constraint_scores': constraint_scores,
            'weighted_contributions': contributions,
            'weights_used': self.weights.copy()
        }
        
        return detailed_fitness
    
    def get_population_statistics(self, population: List[List[int]]) -> Dict[str, Any]:
        """
        Get fitness statistics for population
        
        Args:
            population: List of chromosomes
            
        Returns:
            dict: Population fitness statistics
        """
        if not population:
            return {'error': 'Empty population'}
        
        # Calculate all fitness scores
        fitness_scores = self.calculate_population_fitness(population)
        
        # Calculate statistics
        stats = {
            'population_size': len(population),
            'best_fitness': max(fitness_scores),
            'worst_fitness': min(fitness_scores),
            'average_fitness': np.mean(fitness_scores),
            'median_fitness': np.median(fitness_scores),
            'std_deviation': np.std(fitness_scores),
            'variance': np.var(fitness_scores)
        }
        
        # Find best and worst chromosome indices
        best_idx = fitness_scores.index(stats['best_fitness'])
        worst_idx = fitness_scores.index(stats['worst_fitness'])
        
        stats['best_chromosome_index'] = best_idx
        stats['worst_chromosome_index'] = worst_idx
        
        # Calculate fitness distribution
        fitness_ranges = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
        distribution = {}
        
        for min_val, max_val in fitness_ranges:
            count = sum(1 for f in fitness_scores if min_val <= f < max_val)
            if max_val == 1.0:  # Include 1.0 in the last range
                count = sum(1 for f in fitness_scores if min_val <= f <= max_val)
            distribution[f"{min_val}-{max_val}"] = count
        
        stats['fitness_distribution'] = distribution
        
        return stats
    
    def analyze_best_solution(self, chromosome: List[int]) -> Dict[str, Any]:
        """
        Comprehensive analysis of best solution
        
        Args:
            chromosome: Best chromosome to analyze
            
        Returns:
            dict: Comprehensive analysis
        """
        # Get detailed fitness
        detailed_fitness = self.get_detailed_fitness(chromosome)
        
        # Get detailed constraint analysis
        constraint_analysis = self.constraint_checker.get_detailed_constraint_analysis(chromosome)
        
        # Combine analyses
        analysis = {
            'fitness_analysis': detailed_fitness,
            'constraint_analysis': constraint_analysis,
            'solution_quality': self._assess_solution_quality(detailed_fitness)
        }
        
        return analysis
    
    def _assess_solution_quality(self, detailed_fitness: Dict[str, Any]) -> str:
        """
        Assess solution quality based on fitness score
        
        Args:
            detailed_fitness: Detailed fitness information
            
        Returns:
            str: Quality assessment
        """
        total_fitness = detailed_fitness['total_fitness']
        
        if total_fitness >= 0.95:
            return "Excellent"
        elif total_fitness >= 0.85:
            return "Very Good"
        elif total_fitness >= 0.70:
            return "Good"
        elif total_fitness >= 0.50:
            return "Fair"
        else:
            return "Poor"
    
    def compare_solutions(self, chromosome1: List[int], chromosome2: List[int]) -> Dict[str, Any]:
        """
        Compare two solutions
        
        Args:
            chromosome1: First chromosome
            chromosome2: Second chromosome
            
        Returns:
            dict: Comparison results
        """
        fitness1 = self.get_detailed_fitness(chromosome1)
        fitness2 = self.get_detailed_fitness(chromosome2)
        
        comparison = {
            'chromosome1_fitness': fitness1['total_fitness'],
            'chromosome2_fitness': fitness2['total_fitness'],
            'winner': 1 if fitness1['total_fitness'] > fitness2['total_fitness'] else 2,
            'fitness_difference': abs(fitness1['total_fitness'] - fitness2['total_fitness']),
            'constraint_comparison': {}
        }
        
        # Compare each constraint
        for constraint in ['htq_score', 'duplication_score', 'gender_score', 'size_score']:
            score1 = fitness1['constraint_scores'][constraint]
            score2 = fitness2['constraint_scores'][constraint]
            
            comparison['constraint_comparison'][constraint] = {
                'chromosome1': score1,
                'chromosome2': score2,
                'difference': score1 - score2,
                'winner': 1 if score1 > score2 else (2 if score2 > score1 else 'tie')
            }
        
        return comparison
    
    def get_convergence_info(self, fitness_history: List[float]) -> Dict[str, Any]:
        """
        Analyze convergence from fitness history
        
        Args:
            fitness_history: List of best fitness scores per generation
            
        Returns:
            dict: Convergence analysis
        """
        if len(fitness_history) < 2:
            return {'error': 'Insufficient data for convergence analysis'}
        
        # Calculate improvement rates
        improvements = []
        for i in range(1, len(fitness_history)):
            improvement = fitness_history[i] - fitness_history[i-1]
            improvements.append(improvement)
        
        # Find plateaus (periods of no improvement)
        plateau_threshold = 1e-6
        current_plateau_length = 0
        max_plateau_length = 0
        
        for improvement in reversed(improvements):  # Check from end
            if abs(improvement) < plateau_threshold:
                current_plateau_length += 1
            else:
                break
        
        # Find maximum plateau in history
        plateau_length = 0
        for improvement in improvements:
            if abs(improvement) < plateau_threshold:
                plateau_length += 1
                max_plateau_length = max(max_plateau_length, plateau_length)
            else:
                plateau_length = 0
        
        convergence_info = {
            'total_generations': len(fitness_history),
            'initial_fitness': fitness_history[0],
            'final_fitness': fitness_history[-1],
            'total_improvement': fitness_history[-1] - fitness_history[0],
            'average_improvement_per_generation': np.mean(improvements),
            'current_plateau_length': current_plateau_length,
            'longest_plateau_length': max_plateau_length,
            'is_converged': current_plateau_length > 10,  # Simple convergence criterion
            'convergence_rate': self._calculate_convergence_rate(fitness_history)
        }
        
        return convergence_info
    
    def _calculate_convergence_rate(self, fitness_history: List[float]) -> str:
        """Calculate convergence rate category"""
        if len(fitness_history) < 10:
            return "Insufficient data"
        
        # Calculate improvement in first 25% of generations
        quarter_point = len(fitness_history) // 4
        early_improvement = fitness_history[quarter_point] - fitness_history[0]
        total_improvement = fitness_history[-1] - fitness_history[0]
        
        if total_improvement <= 0:
            return "No improvement"
        
        early_rate = early_improvement / total_improvement
        
        if early_rate >= 0.8:
            return "Fast"
        elif early_rate >= 0.5:
            return "Moderate"
        else:
            return "Slow"


def calculate_single_fitness(chromosome: List[int], 
                           preprocessed_data: Dict[str, Any], 
                           weights: Dict[str, float]) -> float:
    """
    Convenience function to calculate single chromosome fitness
    
    Args:
        chromosome: GA chromosome
        preprocessed_data: Preprocessed data structure
        weights: Constraint weights
        
    Returns:
        float: Fitness score
    """
    calculator = FitnessCalculator(preprocessed_data, weights)
    return calculator.calculate_fitness(chromosome)