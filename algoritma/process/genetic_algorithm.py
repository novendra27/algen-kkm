"""
Main Genetic Algorithm implementation for KKM Grouping
Orchestrates the entire GA process following konteks-algen.md flow
"""

import time
import random
import numpy as np
from typing import Dict, List, Any, Optional
import copy

from .population import PopulationInitializer
from .fitness import FitnessCalculator
from .operators import GeneticOperators
from .selection import SelectionMethods
from .constraints import ConstraintChecker


class GeneticAlgorithm:
    """Main Genetic Algorithm class"""
    
    def __init__(self, preprocessed_data: Dict[str, Any], config, logger):
        """
        Initialize Genetic Algorithm
        
        Args:
            preprocessed_data: Preprocessed data structure
            config: Configuration object with GA parameters
            logger: Logger for tracking evolution
        """
        self.data = preprocessed_data
        self.config = config
        self.logger = logger
        
        # GA Parameters
        self.population_size = config.get_param('population_size')
        self.max_generation = config.get_param('max_generation')
        self.crossover_rate = config.get_param('crossover_rate')
        self.mutation_rate = config.get_param('mutation_rate')
        self.selection_method = config.get_param('selection_method')
        self.target_fitness = config.get_param('target_fitness')
        self.stagnation_limit = config.get_param('stagnation_limit')
        
        # Constraint weights
        self.weights = {
            'weight_htq': config.get_param('weight_htq'),
            'weight_duplikasi': config.get_param('weight_duplikasi'),
            'weight_gender': config.get_param('weight_gender'),
            'weight_jumlah': config.get_param('weight_jumlah')
        }
        
        # Initialize components
        self.population_initializer = PopulationInitializer(preprocessed_data)
        self.fitness_calculator = FitnessCalculator(preprocessed_data, self.weights)
        self.genetic_operators = GeneticOperators(preprocessed_data)
        self.selection_methods = SelectionMethods(self.population_size)
        self.constraint_checker = ConstraintChecker(preprocessed_data)
        
        # Evolution tracking
        self.current_generation = 0
        self.population = []
        self.fitness_scores = []
        self.best_solution = None
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.stagnation_counter = 0
        self.converged = False
        self.start_time = None
        
        print(f"Genetic Algorithm initialized:")
        print(f"  Population size: {self.population_size}")
        print(f"  Max generations: {self.max_generation}")
        print(f"  Selection method: {self.selection_method}")
        print(f"  Target fitness: {self.target_fitness}")
    
    def run(self) -> Dict[str, Any]:
        """
        Main GA execution loop
        
        Returns:
            dict: Best solution and statistics
        """
        print("\n" + "="*50)
        print("🚀 STARTING GENETIC ALGORITHM EXECUTION")
        print("="*50)
        
        self.start_time = time.time()
        
        try:
            # Initialize population
            self._initialize_population()
            
            # Main evolution loop
            while not self._should_stop():
                self._run_generation()
                self._update_statistics()
                self._check_convergence()
                
                # Progress reporting
                if self.current_generation % 50 == 0 or self.current_generation <= 10:
                    self._print_progress()
            
            # Finalize results
            final_solution = self._finalize_results()
            
            print(f"\n🏁 GA EXECUTION COMPLETED")
            print(f"Reason: {self._get_stop_reason()}")
            
            return final_solution
        
        except Exception as e:
            print(f"❌ Error during GA execution: {str(e)}")
            raise
    
    def _initialize_population(self):
        """Initialize population using smart 3-phase strategy"""
        print(f"\n📦 INITIALIZING POPULATION")
        print(f"Creating {self.population_size} chromosomes...")
        
        self.population = self.population_initializer.initialize_population(
            self.population_size
        )
        
        # Calculate initial fitness
        print("Calculating initial fitness scores...")
        self.fitness_scores = self.fitness_calculator.calculate_population_fitness(
            self.population
        )
        
        # Initialize best solution
        best_idx = np.argmax(self.fitness_scores)
        self.best_solution = {
            'chromosome': self.population[best_idx].copy(),
            'fitness': self.fitness_scores[best_idx],
            'generation': 0
        }
        
        # Log initial generation
        self.best_fitness_history.append(self.best_solution['fitness'])
        self.avg_fitness_history.append(np.mean(self.fitness_scores))
        
        self.logger.log_generation(
            0, self.best_solution['fitness'], 
            np.mean(self.fitness_scores), np.std(self.fitness_scores)
        )
        
        print(f"✅ Population initialized successfully")
        print(f"Initial best fitness: {self.best_solution['fitness']:.4f}")
        print(f"Initial avg fitness: {np.mean(self.fitness_scores):.4f}")
    
    def _run_generation(self):
        """Run single generation of GA"""
        self.current_generation += 1
        
        # Step 1: CROSSOVER
        offspring_population = self.genetic_operators.crossover_population(
            self.population, self.fitness_scores, self.crossover_rate
        )
        
        # Step 2: MUTATION
        mutated_population = self.genetic_operators.mutate_population(
            offspring_population, self.mutation_rate, self.fitness_scores
        )
        
        # Step 3: FITNESS CALCULATION
        new_fitness_scores = self.fitness_calculator.calculate_population_fitness(
            mutated_population
        )
        
        # Step 4: SELECTION
        combined_population = self.population + mutated_population
        combined_fitness = self.fitness_scores + new_fitness_scores
        
        selected_population = self.selection_methods.select_population(
            combined_population, combined_fitness, self.selection_method
        )
        
        # Update population
        self.population = selected_population
        self.fitness_scores = self.fitness_calculator.calculate_population_fitness(
            self.population
        )
        
        # Step 5: LOG BEST SOLUTION
        current_best_idx = np.argmax(self.fitness_scores)
        current_best_fitness = self.fitness_scores[current_best_idx]
        
        if current_best_fitness > self.best_solution['fitness']:
            self.best_solution = {
                'chromosome': self.population[current_best_idx].copy(),
                'fitness': current_best_fitness,
                'generation': self.current_generation
            }
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1
    
    def _update_statistics(self):
        """Update evolution statistics"""
        best_fitness = self.best_solution['fitness']
        avg_fitness = np.mean(self.fitness_scores)
        std_fitness = np.std(self.fitness_scores)
        
        self.best_fitness_history.append(best_fitness)
        self.avg_fitness_history.append(avg_fitness)
        
        # Log generation
        self.logger.log_generation(
            self.current_generation, best_fitness, avg_fitness, std_fitness
        )
    
    def _check_convergence(self):
        """Check if algorithm has converged"""
        # Target fitness reached
        if self.best_solution['fitness'] >= self.target_fitness:
            self.converged = True
        
        # Stagnation limit reached
        if self.stagnation_counter >= self.stagnation_limit:
            self.converged = True
    
    def _should_stop(self) -> bool:
        """Check if GA should stop"""
        # Max generations reached
        if self.current_generation >= self.max_generation:
            return True
        
        # Convergence achieved
        if self.converged:
            return True
        
        return False
    
    def _get_stop_reason(self) -> str:
        """Get reason for stopping"""
        if self.current_generation >= self.max_generation:
            return "Maximum generations reached"
        elif self.best_solution['fitness'] >= self.target_fitness:
            return "Target fitness achieved"
        elif self.stagnation_counter >= self.stagnation_limit:
            return "Stagnation limit reached"
        else:
            return "Unknown"
    
    def _print_progress(self):
        """Print generation progress"""
        elapsed_time = time.time() - self.start_time
        progress = (self.current_generation / self.max_generation) * 100
        
        print(f"\n📊 Generation {self.current_generation:4d} "
              f"({progress:5.1f}%) | "
              f"Best: {self.best_solution['fitness']:.4f} | "
              f"Avg: {np.mean(self.fitness_scores):.4f} | "
              f"Stagnation: {self.stagnation_counter:3d} | "
              f"Time: {elapsed_time:6.1f}s")
        
        # Show constraint breakdown for best solution
        if self.current_generation % 100 == 0:
            self._print_constraint_breakdown()
    
    def _print_constraint_breakdown(self):
        """Print detailed constraint analysis for best solution"""
        detailed_fitness = self.fitness_calculator.get_detailed_fitness(
            self.best_solution['chromosome']
        )
        
        scores = detailed_fitness['constraint_scores']
        contributions = detailed_fitness['weighted_contributions']
        
        print(f"    Constraint Breakdown:")
        print(f"      HTQ:         {scores['htq_score']:.3f} → {contributions['htq_contribution']:.3f}")
        print(f"      Duplication: {scores['duplication_score']:.3f} → {contributions['duplication_contribution']:.3f}")
        print(f"      Gender:      {scores['gender_score']:.3f} → {contributions['gender_contribution']:.3f}")
        print(f"      Size:        {scores['size_score']:.3f} → {contributions['size_contribution']:.3f}")
    
    def _finalize_results(self) -> Dict[str, Any]:
        """Finalize and return results"""
        end_time = time.time()
        total_runtime = end_time - self.start_time
        
        # Get comprehensive analysis of best solution
        best_analysis = self.fitness_calculator.analyze_best_solution(
            self.best_solution['chromosome']
        )
        
        # Get convergence information
        convergence_info = self.fitness_calculator.get_convergence_info(
            self.best_fitness_history
        )
        
        final_solution = {
            'chromosome': self.best_solution['chromosome'],
            'fitness': self.best_solution['fitness'],
            'generation_found': self.best_solution['generation'],
            'total_generations': self.current_generation,
            'total_runtime': total_runtime,
            'converged': self.converged,
            'stop_reason': self._get_stop_reason(),
            'fitness_history': {
                'best': self.best_fitness_history,
                'average': self.avg_fitness_history
            },
            'detailed_analysis': best_analysis,
            'convergence_info': convergence_info,
            'final_population_stats': self._get_final_population_stats()
        }
        
        return final_solution
    
    def _get_final_population_stats(self) -> Dict[str, Any]:
        """Get final population statistics"""
        return self.fitness_calculator.get_population_statistics(self.population)
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current GA status"""
        if self.start_time is None:
            return {'status': 'Not started'}
        
        elapsed_time = time.time() - self.start_time
        progress = (self.current_generation / self.max_generation) * 100 if self.max_generation > 0 else 0
        
        status = {
            'status': 'Running' if not self._should_stop() else 'Completed',
            'current_generation': self.current_generation,
            'max_generation': self.max_generation,
            'progress_percentage': progress,
            'best_fitness': self.best_solution['fitness'] if self.best_solution else 0.0,
            'average_fitness': np.mean(self.fitness_scores) if self.fitness_scores else 0.0,
            'stagnation_counter': self.stagnation_counter,
            'elapsed_time': elapsed_time,
            'converged': self.converged
        }
        
        return status
    
    def pause_execution(self):
        """Pause GA execution (for interactive control)"""
        print("⏸️  GA execution paused. Call resume_execution() to continue.")
        # Implementation for pause functionality could be added here
    
    def resume_execution(self):
        """Resume GA execution"""
        print("▶️  Resuming GA execution...")
        # Implementation for resume functionality could be added here
    
    def get_diversity_stats(self) -> Dict[str, Any]:
        """Get population diversity statistics"""
        if not self.population:
            return {'error': 'No population available'}
        
        diversity = self.population_initializer.get_population_diversity(self.population)
        
        return {
            'population_diversity': diversity,
            'unique_individuals': len(set(tuple(ind) for ind in self.population)),
            'population_size': len(self.population),
            'diversity_ratio': len(set(tuple(ind) for ind in self.population)) / len(self.population)
        }


def run_genetic_algorithm(preprocessed_data: Dict[str, Any], 
                         config, logger) -> Dict[str, Any]:
    """
    Convenience function to run GA
    
    Args:
        preprocessed_data: Preprocessed data structure
        config: Configuration object
        logger: Evolution logger
        
    Returns:
        dict: GA results
    """
    ga = GeneticAlgorithm(preprocessed_data, config, logger)
    return ga.run()