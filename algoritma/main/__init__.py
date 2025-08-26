"""
Main module for GA KKM Grouping
Contains main execution logic and configuration
"""

from .main import KKMGroupingSystem
from .config import (
    GAConfig, 
    create_default_config, 
    create_custom_config,
    SMALL_DATASET_CONFIG,
    LARGE_DATASET_CONFIG,
    FAST_CONFIG,
    HIGH_QUALITY_CONFIG
)

__all__ = [
    'KKMGroupingSystem',
    'GAConfig',
    'create_default_config',
    'create_custom_config',
    'SMALL_DATASET_CONFIG',
    'LARGE_DATASET_CONFIG', 
    'FAST_CONFIG',
    'HIGH_QUALITY_CONFIG'
]