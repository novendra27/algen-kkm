"""
Configuration module for Genetic Algorithm KKM Grouping
Contains default parameters and configuration management
"""

import os
from typing import Dict, Any


class GAConfig:
    """Configuration class for Genetic Algorithm parameters"""
    
    # Default Parameters - sesuai konteks-algen.md
    DEFAULT_PARAMS = {
        # Grouping
        'num_groups': 190,                    # Variable untuk mudah diubah
        
        # GA Settings
        'population_size': 150,               # Ukuran populasi
        'max_generation': 1000,               # Maksimal generasi
        'crossover_rate': 0.8,                # 0.0 - 1.0
        'mutation_rate': 0.15,                # 0.0 - 1.0
        'selection_method': 'tournament',     # 'elitism', 'tournament', 'roulette'
        
        # Constraint Weights
        'weight_htq': 0.30,                   # Bobot constraint HTQ
        'weight_duplikasi': 0.20,             # Bobot duplikasi
        'weight_gender': 0.20,                # Bobot gender
        'weight_jumlah': 0.30,                # Bobot jumlah
        
        # Stopping Criteria
        'target_fitness': 0.95,               # Target fitness (0.0 - 1.0)
        'stagnation_limit': 100               # Generasi tanpa improvement
    }
    
    # File paths
    DATA_PATH = os.path.join('data', 'master_data.csv')
    RESULTS_PATH = 'results'
    
    def __init__(self, custom_params: Dict[str, Any] = None):
        """
        Initialize configuration with custom parameters
        
        Args:
            custom_params: Dictionary of custom parameters to override defaults
        """
        self.params = self.DEFAULT_PARAMS.copy()
        
        if custom_params:
            self.update_params(custom_params)
        
        self._validate_params()
    
    def update_params(self, new_params: Dict[str, Any]):
        """
        Update parameters with new values
        
        Args:
            new_params: Dictionary of parameters to update
        """
        for key, value in new_params.items():
            if key in self.params:
                self.params[key] = value
            else:
                raise ValueError(f"Unknown parameter: {key}")
        
        self._validate_params()
    
    def _validate_params(self):
        """Validate parameter values"""
        # Validate ranges
        if not (0.0 <= self.params['crossover_rate'] <= 1.0):
            raise ValueError("crossover_rate must be between 0.0 and 1.0")
        
        if not (0.0 <= self.params['mutation_rate'] <= 1.0):
            raise ValueError("mutation_rate must be between 0.0 and 1.0")
        
        if not (0.0 <= self.params['target_fitness'] <= 1.0):
            raise ValueError("target_fitness must be between 0.0 and 1.0")
        
        # Validate positive integers
        positive_int_params = ['num_groups', 'population_size', 'max_generation', 'stagnation_limit']
        for param in positive_int_params:
            if self.params[param] <= 0:
                raise ValueError(f"{param} must be positive integer")
        
        # Validate selection method
        valid_methods = ['elitism', 'tournament', 'roulette']
        if self.params['selection_method'] not in valid_methods:
            raise ValueError(f"selection_method must be one of: {valid_methods}")
        
        # Validate weights sum approximately to 1.0
        weight_sum = (self.params['weight_htq'] + self.params['weight_duplikasi'] + 
                     self.params['weight_gender'] + self.params['weight_jumlah'])
        if abs(weight_sum - 1.0) > 0.01:
            raise ValueError(f"Constraint weights must sum to 1.0, got {weight_sum}")
    
    def get_param(self, key: str) -> Any:
        """Get parameter value"""
        return self.params.get(key)
    
    def get_all_params(self) -> Dict[str, Any]:
        """Get all parameters"""
        return self.params.copy()
    
    def print_config(self):
        """Print current configuration"""
        print("=== GENETIC ALGORITHM CONFIGURATION ===")
        print(f"Target Groups: {self.params['num_groups']}")
        print(f"Population Size: {self.params['population_size']}")
        print(f"Max Generation: {self.params['max_generation']}")
        print(f"Crossover Rate: {self.params['crossover_rate']}")
        print(f"Mutation Rate: {self.params['mutation_rate']}")
        print(f"Selection Method: {self.params['selection_method']}")
        print(f"Target Fitness: {self.params['target_fitness']}")
        print(f"Stagnation Limit: {self.params['stagnation_limit']}")
        print("\nConstraint Weights:")
        print(f"  HTQ: {self.params['weight_htq']}")
        print(f"  Duplikasi: {self.params['weight_duplikasi']}")
        print(f"  Gender: {self.params['weight_gender']}")
        print(f"  Jumlah: {self.params['weight_jumlah']}")


def create_default_config() -> GAConfig:
    """Create default configuration"""
    return GAConfig()


def create_custom_config(**kwargs) -> GAConfig:
    """
    Create custom configuration
    
    Args:
        **kwargs: Custom parameters
        
    Returns:
        GAConfig: Configured GA parameters
    """
    return GAConfig(kwargs)


# Example configurations for different scenarios
SMALL_DATASET_CONFIG = {
    'population_size': 100,
    'max_generation': 500,
    'stagnation_limit': 50
}

LARGE_DATASET_CONFIG = {
    'population_size': 200,
    'max_generation': 2000,
    'stagnation_limit': 200
}

FAST_CONFIG = {
    'population_size': 50,
    'max_generation': 200,
    'stagnation_limit': 30,
    'target_fitness': 0.85
}

HIGH_QUALITY_CONFIG = {
    'population_size': 300,
    'max_generation': 3000,
    'stagnation_limit': 300,
    'target_fitness': 0.98
}