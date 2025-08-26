"""
Algoritma Genetika KKM Grouping Package
Main package for genetic algorithm implementation
"""

__version__ = "1.0.0"
__author__ = "KKM Grouping System"
__description__ = "Genetic Algorithm for KKM Student Grouping - UIN Malang"

# Import main components for easy access
from .main.main import KKMGroupingSystem
from .main.config import GAConfig, create_default_config, create_custom_config

# Import core GA components
from .process.genetic_algorithm import GeneticAlgorithm
from .process.population import PopulationInitializer
from .process.fitness import FitnessCalculator
from .process.operators import GeneticOperators
from .process.selection import SelectionMethods
from .process.constraints import ConstraintChecker

# Import data processing components
from .input.data_loader import DataLoader
from .input.data_validator import DataValidator
from .input.data_preprocessor import DataPreprocessor

# Import output components
from .output.result_exporter import ResultExporter
from .output.report_generator import ReportGenerator
from .output.logger import GALogger

__all__ = [
    # Main system
    'KKMGroupingSystem',
    'GAConfig',
    'create_default_config',
    'create_custom_config',
    
    # Core GA
    'GeneticAlgorithm',
    'PopulationInitializer',
    'FitnessCalculator',
    'GeneticOperators',
    'SelectionMethods',
    'ConstraintChecker',
    
    # Data processing
    'DataLoader',
    'DataValidator',
    'DataPreprocessor',
    
    # Output
    'ResultExporter',
    'ReportGenerator',
    'GALogger'
]