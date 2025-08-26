"""
Logging module for GA evolution tracking
Logs generation statistics and evolution progress
"""

import csv
import os
import time
from typing import List, Dict, Any
from datetime import datetime


class GALogger:
    """Logger for tracking GA evolution statistics"""
    
    def __init__(self):
        """Initialize GA Logger"""
        self.generation_logs = []
        self.start_time = None
        self.session_id = None
        self._initialize_session()
    
    def _initialize_session(self):
        """Initialize logging session"""
        self.start_time = time.time()
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize log headers
        self.generation_logs.append([
            'Generation', 'Best_Fitness', 'Avg_Fitness', 'Std_Dev', 'Time_Elapsed'
        ])
    
    def log_generation(self, generation: int, best_fitness: float, 
                      avg_fitness: float, std_dev: float):
        """
        Log generation statistics
        
        Args:
            generation: Generation number
            best_fitness: Best fitness in generation
            avg_fitness: Average fitness in generation
            std_dev: Standard deviation of fitness
        """
        elapsed_time = time.time() - self.start_time
        
        log_entry = [
            generation,
            round(best_fitness, 6),
            round(avg_fitness, 6),
            round(std_dev, 6),
            round(elapsed_time, 2)
        ]
        
        self.generation_logs.append(log_entry)
    
    def export_evolution_log(self, output_dir: str) -> str:
        """
        Export evolution log to CSV file
        
        Args:
            output_dir: Directory to save log file
            
        Returns:
            str: Path to exported file
        """
        # Create evolution_logs subfolder
        evolution_logs_dir = os.path.join(output_dir, 'evolution_logs')
        os.makedirs(evolution_logs_dir, exist_ok=True)
        
        filename = f"evolution_log_{self.session_id}.csv"
        filepath = os.path.join(evolution_logs_dir, filename)
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(self.generation_logs)
        
        print(f"Evolution log exported to: {filepath}")
        return filepath
    
    def get_session_summary(self) -> Dict[str, Any]:
        """
        Get summary of logging session
        
        Returns:
            dict: Session summary
        """
        if len(self.generation_logs) <= 1:
            return {'error': 'No generation data logged'}
        
        # Extract data (skip header)
        data = self.generation_logs[1:]
        generations = [row[0] for row in data]
        best_fitness_values = [row[1] for row in data]
        avg_fitness_values = [row[2] for row in data]
        times = [row[4] for row in data]
        
        summary = {
            'session_id': self.session_id,
            'total_generations': len(data),
            'start_time': self.start_time,
            'total_runtime': times[-1] if times else 0,
            'initial_best_fitness': best_fitness_values[0] if best_fitness_values else 0,
            'final_best_fitness': best_fitness_values[-1] if best_fitness_values else 0,
            'best_improvement': (best_fitness_values[-1] - best_fitness_values[0]) if best_fitness_values else 0,
            'initial_avg_fitness': avg_fitness_values[0] if avg_fitness_values else 0,
            'final_avg_fitness': avg_fitness_values[-1] if avg_fitness_values else 0,
            'avg_improvement': (avg_fitness_values[-1] - avg_fitness_values[0]) if avg_fitness_values else 0
        }
        
        return summary
    
    def print_session_summary(self):
        """Print session summary to console"""
        summary = self.get_session_summary()
        
        if 'error' in summary:
            print(f"❌ {summary['error']}")
            return
        
        print(f"\n📈 EVOLUTION LOG SUMMARY")
        print(f"="*40)
        print(f"Session ID: {summary['session_id']}")
        print(f"Total Generations: {summary['total_generations']}")
        print(f"Total Runtime: {summary['total_runtime']:.2f} seconds")
        print(f"\nFitness Evolution:")
        print(f"  Best Fitness:  {summary['initial_best_fitness']:.4f} → {summary['final_best_fitness']:.4f} (+{summary['best_improvement']:.4f})")
        print(f"  Avg Fitness:   {summary['initial_avg_fitness']:.4f} → {summary['final_avg_fitness']:.4f} (+{summary['avg_improvement']:.4f})")
    
    def get_fitness_trajectory(self) -> Dict[str, List]:
        """
        Get fitness trajectory data
        
        Returns:
            dict: Fitness trajectory data
        """
        if len(self.generation_logs) <= 1:
            return {'error': 'No generation data logged'}
        
        data = self.generation_logs[1:]
        
        return {
            'generations': [row[0] for row in data],
            'best_fitness': [row[1] for row in data],
            'avg_fitness': [row[2] for row in data],
            'std_dev': [row[3] for row in data],
            'time_elapsed': [row[4] for row in data]
        }
    
    def analyze_convergence(self) -> Dict[str, Any]:
        """
        Analyze convergence patterns from logged data
        
        Returns:
            dict: Convergence analysis
        """
        trajectory = self.get_fitness_trajectory()
        
        if 'error' in trajectory:
            return trajectory
        
        best_fitness = trajectory['best_fitness']
        avg_fitness = trajectory['avg_fitness']
        
        # Find improvement phases
        improvements = []
        for i in range(1, len(best_fitness)):
            improvement = best_fitness[i] - best_fitness[i-1]
            improvements.append(improvement)
        
        # Find plateaus
        plateau_threshold = 1e-6
        current_plateau = 0
        max_plateau = 0
        plateau_starts = []
        
        for i, improvement in enumerate(improvements):
            if abs(improvement) < plateau_threshold:
                if current_plateau == 0:
                    plateau_starts.append(i + 1)  # +1 for generation number
                current_plateau += 1
                max_plateau = max(max_plateau, current_plateau)
            else:
                current_plateau = 0
        
        # Calculate convergence metrics
        total_improvement = best_fitness[-1] - best_fitness[0] if best_fitness else 0
        avg_improvement_per_gen = total_improvement / len(best_fitness) if best_fitness else 0
        
        analysis = {
            'total_improvement': total_improvement,
            'avg_improvement_per_generation': avg_improvement_per_gen,
            'longest_plateau': max_plateau,
            'current_plateau': current_plateau,
            'plateau_starts': plateau_starts,
            'improvement_phases': len([imp for imp in improvements if imp > plateau_threshold]),
            'convergence_rate': self._calculate_convergence_rate(best_fitness)
        }
        
        return analysis
    
    def _calculate_convergence_rate(self, fitness_values: List[float]) -> str:
        """Calculate convergence rate classification"""
        if len(fitness_values) < 10:
            return "Insufficient data"
        
        # Check improvement in first quarter
        quarter_point = len(fitness_values) // 4
        early_improvement = fitness_values[quarter_point] - fitness_values[0]
        total_improvement = fitness_values[-1] - fitness_values[0]
        
        if total_improvement <= 0:
            return "No improvement"
        
        early_ratio = early_improvement / total_improvement
        
        if early_ratio >= 0.8:
            return "Fast convergence"
        elif early_ratio >= 0.5:
            return "Moderate convergence"
        else:
            return "Slow convergence"
    
    def create_detailed_log(self, output_dir: str, additional_info: Dict[str, Any] = None) -> str:
        """
        Create detailed log with additional information
        
        Args:
            output_dir: Output directory
            additional_info: Additional information to include
            
        Returns:
            str: Path to detailed log file
        """
        # Create evolution_logs subfolder
        evolution_logs_dir = os.path.join(output_dir, 'evolution_logs')
        os.makedirs(evolution_logs_dir, exist_ok=True)
        
        filename = f"detailed_evolution_log_{self.session_id}.txt"
        filepath = os.path.join(evolution_logs_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            # Write header
            f.write("="*60 + "\n")
            f.write("GENETIC ALGORITHM EVOLUTION LOG - DETAILED\n")
            f.write("="*60 + "\n\n")
            
            # Write session info
            f.write(f"Session ID: {self.session_id}\n")
            f.write(f"Start Time: {datetime.fromtimestamp(self.start_time)}\n")
            f.write(f"End Time: {datetime.now()}\n\n")
            
            # Write additional info if provided
            if additional_info:
                f.write("CONFIGURATION PARAMETERS:\n")
                f.write("-" * 30 + "\n")
                for key, value in additional_info.items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")
            
            # Write session summary
            summary = self.get_session_summary()
            if 'error' not in summary:
                f.write("SESSION SUMMARY:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Total Generations: {summary['total_generations']}\n")
                f.write(f"Total Runtime: {summary['total_runtime']:.2f} seconds\n")
                f.write(f"Best Fitness Improvement: {summary['best_improvement']:.6f}\n")
                f.write(f"Average Fitness Improvement: {summary['avg_improvement']:.6f}\n\n")
            
            # Write convergence analysis
            convergence = self.analyze_convergence()
            if 'error' not in convergence:
                f.write("CONVERGENCE ANALYSIS:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Total Improvement: {convergence['total_improvement']:.6f}\n")
                f.write(f"Average Improvement per Generation: {convergence['avg_improvement_per_generation']:.6f}\n")
                f.write(f"Longest Plateau: {convergence['longest_plateau']} generations\n")
                f.write(f"Current Plateau: {convergence['current_plateau']} generations\n")
                f.write(f"Convergence Rate: {convergence['convergence_rate']}\n\n")
            
            # Write generation-by-generation data
            f.write("GENERATION-BY-GENERATION DATA:\n")
            f.write("-" * 30 + "\n")
            f.write("Gen\tBest_Fit\tAvg_Fit\t\tStd_Dev\t\tTime\n")
            f.write("-" * 50 + "\n")
            
            for i, row in enumerate(self.generation_logs[1:], 1):  # Skip header
                f.write(f"{row[0]:3d}\t{row[1]:.4f}\t\t{row[2]:.4f}\t\t{row[3]:.4f}\t\t{row[4]:.1f}\n")
        
        print(f"Detailed evolution log created: {filepath}")
        return filepath
    
    def clear_logs(self):
        """Clear all logged data"""
        self.generation_logs = []
        self._initialize_session()
        print("Evolution logs cleared and new session initialized.")


def create_ga_logger() -> GALogger:
    """
    Create new GA Logger instance
    
    Returns:
        GALogger: New logger instance
    """
    return GALogger()