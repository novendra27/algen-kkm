"""
Selection methods module for GA KKM Grouping
Implements elitism, tournament, and roulette wheel selection
"""

import random
import numpy as np
from typing import List, Tuple, Dict, Any
import heapq


class SelectionMethods:
    """Class containing various selection methods"""
    
    def __init__(self, population_size: int):
        """
        Initialize selection methods
        
        Args:
            population_size: Size of population to maintain
        """
        self.population_size = population_size
    
    def elitism_selection(self, population: List[List[int]], 
                         fitness_scores: List[float],
                         elite_percentage: float = 0.1) -> List[List[int]]:
        """
        Elitism selection: keep best individuals and fill rest randomly
        
        Args:
            population: Current population
            fitness_scores: Fitness scores for each individual
            elite_percentage: Percentage of elite individuals to keep
            
        Returns:
            List[List[int]]: Selected population
        """
        if not population:
            return []
        
        # Calculate number of elites
        num_elites = max(1, int(len(population) * elite_percentage))
        num_elites = min(num_elites, self.population_size)
        
        # Sort by fitness (descending)
        sorted_indices = sorted(range(len(population)), 
                              key=lambda i: fitness_scores[i], reverse=True)
        
        selected_population = []
        
        # Add elites
        for i in range(num_elites):
            idx = sorted_indices[i]
            selected_population.append(population[idx].copy())
        
        # Fill remaining slots with tournament selection
        remaining_slots = self.population_size - num_elites
        if remaining_slots > 0:
            additional = self.tournament_selection(
                population, fitness_scores, tournament_size=3, 
                population_size=remaining_slots
            )
            selected_population.extend(additional)
        
        return selected_population[:self.population_size]
    
    def tournament_selection(self, population: List[List[int]], 
                           fitness_scores: List[float],
                           tournament_size: int = 5,
                           population_size: int = None) -> List[List[int]]:
        """
        Tournament selection
        
        Args:
            population: Current population
            fitness_scores: Fitness scores for each individual
            tournament_size: Size of each tournament
            population_size: Target population size (default: self.population_size)
            
        Returns:
            List[List[int]]: Selected population
        """
        if not population:
            return []
        
        target_size = population_size or self.population_size
        selected_population = []
        
        for _ in range(target_size):
            # Run tournament
            winner = self._run_tournament(population, fitness_scores, tournament_size)
            selected_population.append(winner.copy())
        
        return selected_population
    
    def _run_tournament(self, population: List[List[int]], 
                       fitness_scores: List[float],
                       tournament_size: int) -> List[int]:
        """
        Run single tournament
        
        Args:
            population: Current population
            fitness_scores: Fitness scores
            tournament_size: Tournament size
            
        Returns:
            List[int]: Winner chromosome
        """
        # Select random contestants
        tournament_indices = random.sample(range(len(population)), 
                                         min(tournament_size, len(population)))
        
        # Find winner (highest fitness)
        winner_idx = max(tournament_indices, key=lambda i: fitness_scores[i])
        
        return population[winner_idx]
    
    def roulette_wheel_selection(self, population: List[List[int]], 
                               fitness_scores: List[float]) -> List[List[int]]:
        """
        Roulette wheel selection
        
        Args:
            population: Current population
            fitness_scores: Fitness scores for each individual
            
        Returns:
            List[List[int]]: Selected population
        """
        if not population:
            return []
        
        # Handle negative fitness scores by shifting
        min_fitness = min(fitness_scores)
        if min_fitness < 0:
            adjusted_scores = [f - min_fitness + 0.001 for f in fitness_scores]
        else:
            adjusted_scores = [max(f, 0.001) for f in fitness_scores]  # Ensure positive
        
        # Calculate selection probabilities
        total_fitness = sum(adjusted_scores)
        if total_fitness == 0:
            # Fallback to uniform random selection
            return self._uniform_random_selection(population)
        
        probabilities = [f / total_fitness for f in adjusted_scores]
        
        # Select individuals
        selected_population = []
        for _ in range(self.population_size):
            selected_idx = self._roulette_spin(probabilities)
            selected_population.append(population[selected_idx].copy())
        
        return selected_population
    
    def _roulette_spin(self, probabilities: List[float]) -> int:
        """
        Perform single roulette wheel spin
        
        Args:
            probabilities: Selection probabilities
            
        Returns:
            int: Selected index
        """
        r = random.random()
        cumulative = 0.0
        
        for i, prob in enumerate(probabilities):
            cumulative += prob
            if r <= cumulative:
                return i
        
        # Fallback (should not happen)
        return len(probabilities) - 1
    
    def _uniform_random_selection(self, population: List[List[int]]) -> List[List[int]]:
        """Fallback uniform random selection"""
        selected_population = []
        for _ in range(self.population_size):
            selected_idx = random.randint(0, len(population) - 1)
            selected_population.append(population[selected_idx].copy())
        return selected_population
    
    def rank_selection(self, population: List[List[int]], 
                      fitness_scores: List[float]) -> List[List[int]]:
        """
        Rank-based selection
        
        Args:
            population: Current population
            fitness_scores: Fitness scores for each individual
            
        Returns:
            List[List[int]]: Selected population
        """
        if not population:
            return []
        
        # Rank individuals by fitness
        ranked_indices = sorted(range(len(population)), 
                              key=lambda i: fitness_scores[i])
        
        # Assign selection probabilities based on rank
        ranks = list(range(1, len(population) + 1))
        total_rank = sum(ranks)
        probabilities = [rank / total_rank for rank in ranks]
        
        # Select based on rank probabilities
        selected_population = []
        for _ in range(self.population_size):
            selected_rank_idx = self._roulette_spin(probabilities)
            actual_idx = ranked_indices[selected_rank_idx]
            selected_population.append(population[actual_idx].copy())
        
        return selected_population
    
    def stochastic_universal_sampling(self, population: List[List[int]], 
                                    fitness_scores: List[float]) -> List[List[int]]:
        """
        Stochastic Universal Sampling (SUS)
        
        Args:
            population: Current population
            fitness_scores: Fitness scores for each individual
            
        Returns:
            List[List[int]]: Selected population
        """
        if not population:
            return []
        
        # Handle negative fitness scores
        min_fitness = min(fitness_scores)
        if min_fitness < 0:
            adjusted_scores = [f - min_fitness + 0.001 for f in fitness_scores]
        else:
            adjusted_scores = [max(f, 0.001) for f in fitness_scores]
        
        total_fitness = sum(adjusted_scores)
        if total_fitness == 0:
            return self._uniform_random_selection(population)
        
        # Calculate selection points
        step_size = total_fitness / self.population_size
        start_point = random.uniform(0, step_size)
        
        selection_points = [start_point + i * step_size 
                          for i in range(self.population_size)]
        
        # Select individuals based on points
        selected_population = []
        cumulative_fitness = 0.0
        current_individual = 0
        
        for point in selection_points:
            # Find individual corresponding to this point
            while cumulative_fitness < point and current_individual < len(population):
                cumulative_fitness += adjusted_scores[current_individual]
                current_individual += 1
            
            # Select the individual (wrap around if necessary)
            selected_idx = min(current_individual - 1, len(population) - 1)
            selected_idx = max(selected_idx, 0)
            selected_population.append(population[selected_idx].copy())
        
        return selected_population
    
    def select_population(self, population: List[List[int]], 
                         fitness_scores: List[float],
                         method: str,
                         **kwargs) -> List[List[int]]:
        """
        Select population using specified method
        
        Args:
            population: Current population
            fitness_scores: Fitness scores
            method: Selection method ('elitism', 'tournament', 'roulette', 'rank', 'sus')
            **kwargs: Additional parameters for selection methods
            
        Returns:
            List[List[int]]: Selected population
        """
        if method == 'elitism':
            elite_pct = kwargs.get('elite_percentage', 0.1)
            return self.elitism_selection(population, fitness_scores, elite_pct)
        
        elif method == 'tournament':
            tournament_size = kwargs.get('tournament_size', 5)
            return self.tournament_selection(population, fitness_scores, tournament_size)
        
        elif method == 'roulette':
            return self.roulette_wheel_selection(population, fitness_scores)
        
        elif method == 'rank':
            return self.rank_selection(population, fitness_scores)
        
        elif method == 'sus':
            return self.stochastic_universal_sampling(population, fitness_scores)
        
        else:
            # Default to tournament selection
            return self.tournament_selection(population, fitness_scores)
    
    def get_selection_statistics(self, original_population: List[List[int]], 
                               selected_population: List[List[int]],
                               fitness_scores: List[float]) -> Dict[str, Any]:
        """
        Calculate selection statistics
        
        Args:
            original_population: Population before selection
            selected_population: Population after selection
            fitness_scores: Original fitness scores
            
        Returns:
            dict: Selection statistics
        """
        if not original_population or not selected_population:
            return {'error': 'Empty populations'}
        
        # Calculate diversity metrics
        original_diversity = self._calculate_diversity(original_population)
        selected_diversity = self._calculate_diversity(selected_population)
        
        # Calculate fitness statistics
        original_fitness_stats = {
            'mean': np.mean(fitness_scores),
            'std': np.std(fitness_scores),
            'min': np.min(fitness_scores),
            'max': np.max(fitness_scores)
        }
        
        # Count unique individuals in selected population
        unique_selected = len(set(tuple(ind) for ind in selected_population))
        
        # Calculate selection pressure (how many times best individual was selected)
        best_idx = np.argmax(fitness_scores)
        best_individual = tuple(original_population[best_idx])
        best_count = sum(1 for ind in selected_population if tuple(ind) == best_individual)
        
        statistics = {
            'original_population_size': len(original_population),
            'selected_population_size': len(selected_population),
            'original_diversity': original_diversity,
            'selected_diversity': selected_diversity,
            'diversity_change': selected_diversity - original_diversity,
            'unique_individuals_selected': unique_selected,
            'selection_ratio': unique_selected / len(selected_population),
            'best_individual_copies': best_count,
            'selection_pressure': best_count / len(selected_population),
            'original_fitness_stats': original_fitness_stats
        }
        
        return statistics
    
    def _calculate_diversity(self, population: List[List[int]]) -> float:
        """
        Calculate population diversity
        
        Args:
            population: Population to analyze
            
        Returns:
            float: Diversity measure (0.0 to 1.0)
        """
        if len(population) < 2:
            return 0.0
        
        total_differences = 0
        comparisons = 0
        
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                differences = sum(1 for a, b in zip(population[i], population[j]) if a != b)
                total_differences += differences
                comparisons += 1
        
        if comparisons == 0:
            return 0.0
        
        avg_differences = total_differences / comparisons
        max_possible_differences = len(population[0]) if population else 1
        
        return avg_differences / max_possible_differences


def select_next_generation(population: List[List[int]], 
                         fitness_scores: List[float],
                         method: str,
                         population_size: int,
                         **kwargs) -> List[List[int]]:
    """
    Convenience function to select next generation
    
    Args:
        population: Current population
        fitness_scores: Fitness scores
        method: Selection method
        population_size: Target population size
        **kwargs: Additional parameters
        
    Returns:
        List[List[int]]: Selected population
    """
    selector = SelectionMethods(population_size)
    return selector.select_population(population, fitness_scores, method, **kwargs)