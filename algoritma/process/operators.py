"""
Genetic operators module for GA KKM Grouping
Implements crossover and mutation operators
"""

import random
import numpy as np
from typing import List, Tuple, Dict, Any
import copy


class GeneticOperators:
    """Class containing genetic operators for GA"""
    
    def __init__(self, preprocessed_data: Dict[str, Any]):
        """
        Initialize genetic operators
        
        Args:
            preprocessed_data: Preprocessed data structure
        """
        self.total_students = preprocessed_data['total_students']
        self.num_groups = preprocessed_data['num_groups']
    
    def uniform_crossover(self, parent1: List[int], parent2: List[int], 
                         crossover_rate: float) -> Tuple[List[int], List[int]]:
        """
        Uniform crossover operation
        
        Args:
            parent1: First parent chromosome
            parent2: Second parent chromosome
            crossover_rate: Probability of crossover
            
        Returns:
            Tuple[List[int], List[int]]: Two offspring chromosomes
        """
        # Check if crossover should occur
        if random.random() > crossover_rate:
            return parent1.copy(), parent2.copy()
        
        # Create offspring copies
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        # Uniform crossover: swap genes with 50% probability
        for i in range(len(parent1)):
            if random.random() < 0.5:
                child1[i], child2[i] = child2[i], child1[i]
        
        return child1, child2
    
    def single_point_crossover(self, parent1: List[int], parent2: List[int], 
                              crossover_rate: float) -> Tuple[List[int], List[int]]:
        """
        Single point crossover operation
        
        Args:
            parent1: First parent chromosome
            parent2: Second parent chromosome
            crossover_rate: Probability of crossover
            
        Returns:
            Tuple[List[int], List[int]]: Two offspring chromosomes
        """
        # Check if crossover should occur
        if random.random() > crossover_rate:
            return parent1.copy(), parent2.copy()
        
        # Choose crossover point
        crossover_point = random.randint(1, len(parent1) - 1)
        
        # Create offspring
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        
        return child1, child2
    
    def two_point_crossover(self, parent1: List[int], parent2: List[int], 
                           crossover_rate: float) -> Tuple[List[int], List[int]]:
        """
        Two point crossover operation
        
        Args:
            parent1: First parent chromosome
            parent2: Second parent chromosome
            crossover_rate: Probability of crossover
            
        Returns:
            Tuple[List[int], List[int]]: Two offspring chromosomes
        """
        # Check if crossover should occur
        if random.random() > crossover_rate:
            return parent1.copy(), parent2.copy()
        
        # Choose two crossover points
        length = len(parent1)
        point1 = random.randint(1, length - 2)
        point2 = random.randint(point1 + 1, length - 1)
        
        # Create offspring
        child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
        child2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
        
        return child1, child2
    
    def random_swap_mutation(self, chromosome: List[int], 
                           mutation_rate: float) -> List[int]:
        """
        Random swap mutation operation
        
        Args:
            chromosome: Chromosome to mutate
            mutation_rate: Probability of mutation
            
        Returns:
            List[int]: Mutated chromosome
        """
        # Check if mutation should occur
        if random.random() > mutation_rate:
            return chromosome.copy()
        
        mutated = chromosome.copy()
        
        # Perform random swaps
        num_swaps = random.randint(1, 5)
        
        for _ in range(num_swaps):
            # Choose two random positions
            pos1 = random.randint(0, len(chromosome) - 1)
            pos2 = random.randint(0, len(chromosome) - 1)
            
            # Swap the values
            mutated[pos1], mutated[pos2] = mutated[pos2], mutated[pos1]
        
        return mutated
    
    def random_reassignment_mutation(self, chromosome: List[int], 
                                   mutation_rate: float) -> List[int]:
        """
        Random reassignment mutation operation
        
        Args:
            chromosome: Chromosome to mutate
            mutation_rate: Probability of mutation
            
        Returns:
            List[int]: Mutated chromosome
        """
        # Check if mutation should occur
        if random.random() > mutation_rate:
            return chromosome.copy()
        
        mutated = chromosome.copy()
        
        # Randomly reassign some students to different groups
        num_reassignments = random.randint(1, max(1, len(chromosome) // 20))
        
        for _ in range(num_reassignments):
            # Choose random student
            student_pos = random.randint(0, len(chromosome) - 1)
            
            # Assign to random group
            new_group = random.randint(1, self.num_groups)
            mutated[student_pos] = new_group
        
        return mutated
    
    def group_swap_mutation(self, chromosome: List[int], 
                          mutation_rate: float) -> List[int]:
        """
        Group swap mutation: swap all students between two groups
        
        Args:
            chromosome: Chromosome to mutate
            mutation_rate: Probability of mutation
            
        Returns:
            List[int]: Mutated chromosome
        """
        # Check if mutation should occur
        if random.random() > mutation_rate:
            return chromosome.copy()
        
        mutated = chromosome.copy()
        
        # Choose two random groups to swap
        group1 = random.randint(1, self.num_groups)
        group2 = random.randint(1, self.num_groups)
        
        if group1 != group2:
            # Swap all students between the two groups
            for i in range(len(mutated)):
                if mutated[i] == group1:
                    mutated[i] = group2
                elif mutated[i] == group2:
                    mutated[i] = group1
        
        return mutated
    
    def adaptive_mutation(self, chromosome: List[int], mutation_rate: float, 
                         fitness_score: float) -> List[int]:
        """
        Adaptive mutation: mutation intensity based on fitness
        
        Args:
            chromosome: Chromosome to mutate
            mutation_rate: Base mutation rate
            fitness_score: Current fitness score
            
        Returns:
            List[int]: Mutated chromosome
        """
        # Adapt mutation rate based on fitness (lower fitness = higher mutation)
        adapted_rate = mutation_rate * (1.0 - fitness_score + 0.1)
        adapted_rate = min(adapted_rate, 0.5)  # Cap at 50%
        
        # Use random swap mutation with adapted rate
        return self.random_swap_mutation(chromosome, adapted_rate)
    
    def crossover_population(self, population: List[List[int]], 
                           fitness_scores: List[float],
                           crossover_rate: float,
                           crossover_method: str = 'uniform') -> List[List[int]]:
        """
        Apply crossover to entire population
        
        Args:
            population: Current population
            fitness_scores: Fitness scores for selection
            crossover_rate: Crossover probability
            crossover_method: Type of crossover ('uniform', 'single_point', 'two_point')
            
        Returns:
            List[List[int]]: New population after crossover
        """
        new_population = []
        
        # Select crossover method
        crossover_func = {
            'uniform': self.uniform_crossover,
            'single_point': self.single_point_crossover,
            'two_point': self.two_point_crossover
        }.get(crossover_method, self.uniform_crossover)
        
        # Generate offspring in pairs
        for i in range(0, len(population), 2):
            if i + 1 < len(population):
                parent1 = population[i]
                parent2 = population[i + 1]
                
                child1, child2 = crossover_func(parent1, parent2, crossover_rate)
                new_population.extend([child1, child2])
            else:
                # Odd population size - keep last individual
                new_population.append(population[i])
        
        return new_population
    
    def mutate_population(self, population: List[List[int]], 
                         mutation_rate: float,
                         fitness_scores: List[float] = None,
                         mutation_method: str = 'random_swap') -> List[List[int]]:
        """
        Apply mutation to entire population
        
        Args:
            population: Current population
            mutation_rate: Mutation probability
            fitness_scores: Fitness scores for adaptive mutation
            mutation_method: Type of mutation
            
        Returns:
            List[List[int]]: New population after mutation
        """
        new_population = []
        
        for i, chromosome in enumerate(population):
            if mutation_method == 'random_swap':
                mutated = self.random_swap_mutation(chromosome, mutation_rate)
            elif mutation_method == 'random_reassignment':
                mutated = self.random_reassignment_mutation(chromosome, mutation_rate)
            elif mutation_method == 'group_swap':
                mutated = self.group_swap_mutation(chromosome, mutation_rate)
            elif mutation_method == 'adaptive' and fitness_scores:
                mutated = self.adaptive_mutation(chromosome, mutation_rate, fitness_scores[i])
            else:
                mutated = self.random_swap_mutation(chromosome, mutation_rate)
            
            new_population.append(mutated)
        
        return new_population
    
    def repair_chromosome(self, chromosome: List[int]) -> List[int]:
        """
        Repair invalid chromosome (ensure all group IDs are valid)
        
        Args:
            chromosome: Chromosome to repair
            
        Returns:
            List[int]: Repaired chromosome
        """
        repaired = chromosome.copy()
        
        for i in range(len(repaired)):
            # Fix invalid group IDs
            if repaired[i] < 1 or repaired[i] > self.num_groups:
                repaired[i] = random.randint(1, self.num_groups)
        
        return repaired
    
    def get_operator_statistics(self, original_pop: List[List[int]], 
                              new_pop: List[List[int]]) -> Dict[str, Any]:
        """
        Calculate statistics about genetic operations
        
        Args:
            original_pop: Population before operations
            new_pop: Population after operations
            
        Returns:
            dict: Operation statistics
        """
        if len(original_pop) != len(new_pop):
            return {'error': 'Population sizes do not match'}
        
        total_changes = 0
        unchanged_chromosomes = 0
        
        for orig, new in zip(original_pop, new_pop):
            changes = sum(1 for a, b in zip(orig, new) if a != b)
            total_changes += changes
            
            if changes == 0:
                unchanged_chromosomes += 1
        
        avg_changes_per_chromosome = total_changes / len(original_pop)
        avg_change_percentage = (avg_changes_per_chromosome / len(original_pop[0])) * 100
        
        statistics = {
            'total_chromosomes': len(original_pop),
            'unchanged_chromosomes': unchanged_chromosomes,
            'modified_chromosomes': len(original_pop) - unchanged_chromosomes,
            'total_gene_changes': total_changes,
            'average_changes_per_chromosome': avg_changes_per_chromosome,
            'average_change_percentage': avg_change_percentage,
            'modification_rate': (len(original_pop) - unchanged_chromosomes) / len(original_pop)
        }
        
        return statistics


def apply_crossover_and_mutation(population: List[List[int]], 
                                fitness_scores: List[float],
                                crossover_rate: float,
                                mutation_rate: float,
                                preprocessed_data: Dict[str, Any],
                                crossover_method: str = 'uniform',
                                mutation_method: str = 'random_swap') -> List[List[int]]:
    """
    Convenience function to apply both crossover and mutation
    
    Args:
        population: Current population
        fitness_scores: Fitness scores
        crossover_rate: Crossover probability
        mutation_rate: Mutation probability
        preprocessed_data: Preprocessed data structure
        crossover_method: Crossover method to use
        mutation_method: Mutation method to use
        
    Returns:
        List[List[int]]: New population after operations
    """
    operators = GeneticOperators(preprocessed_data)
    
    # Apply crossover
    after_crossover = operators.crossover_population(
        population, fitness_scores, crossover_rate, crossover_method
    )
    
    # Apply mutation
    after_mutation = operators.mutate_population(
        after_crossover, mutation_rate, fitness_scores, mutation_method
    )
    
    return after_mutation