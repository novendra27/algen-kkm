"""
Main execution file for Genetic Algorithm KKM Grouping
Entry point dan orchestration untuk seluruh proses GA
"""

import sys
import os
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from algoritma.main.config import GAConfig, create_default_config
from algoritma.input.data_loader import DataLoader
from algoritma.input.data_validator import DataValidator
from algoritma.input.data_preprocessor import DataPreprocessor
from algoritma.process.genetic_algorithm import GeneticAlgorithm
from algoritma.output.result_exporter import ResultExporter
from algoritma.output.report_generator import ReportGenerator
from algoritma.output.logger import GALogger


class KKMGroupingSystem:
    """Main system class for KKM Grouping using Genetic Algorithm"""
    
    def __init__(self, config: GAConfig = None):
        """
        Initialize KKM Grouping System
        
        Args:
            config: GA configuration object
        """
        self.config = config or create_default_config()
        self.data = None
        self.preprocessed_data = None
        self.best_solution = None
        self.runtime_stats = {}
        
        # Initialize components
        self.data_loader = DataLoader()
        self.data_validator = DataValidator()
        self.data_preprocessor = DataPreprocessor()
        self.logger = GALogger()
        
        print("KKM Grouping System initialized")
        self.config.print_config()
    
    def load_and_validate_data(self, data_path: str = None) -> bool:
        """
        Load and validate input data
        
        Args:
            data_path: Path to CSV data file
            
        Returns:
            bool: Success status
        """
        try:
            # Use default path if not provided
            if data_path is None:
                data_path = self.config.DATA_PATH
            
            print(f"\n=== LOADING DATA ===")
            print(f"Loading data from: {data_path}")
            
            # Load data
            self.data = self.data_loader.load_csv(data_path)
            print(f"Data loaded successfully: {len(self.data)} records")
            
            # Validate data
            print("Validating data format and content...")
            validation_result = self.data_validator.validate_data(self.data)
            
            if not validation_result['is_valid']:
                print("❌ Data validation failed:")
                for error in validation_result['errors']:
                    print(f"  - {error}")
                return False
            
            print("✅ Data validation passed")
            return True
            
        except Exception as e:
            print(f"❌ Error loading/validating data: {str(e)}")
            return False
    
    def preprocess_data(self) -> bool:
        """
        Preprocess data for GA
        
        Returns:
            bool: Success status
        """
        try:
            print(f"\n=== PREPROCESSING DATA ===")
            
            # Preprocess data
            self.preprocessed_data = self.data_preprocessor.preprocess(
                self.data, 
                self.config.get_param('num_groups')
            )
            
            # Print preprocessing summary
            print("Data preprocessing completed:")
            print(f"  - Total students: {self.preprocessed_data['total_students']}")
            print(f"  - HTQ students: {self.preprocessed_data['htq_count']}")
            print(f"  - Gender ratio LK:PR = {self.preprocessed_data['gender_ratio']['LK']:.3f}:{self.preprocessed_data['gender_ratio']['PR']:.3f}")
            print(f"  - Unique majors: {len(self.preprocessed_data['major_distribution'])}")
            print(f"  - Target groups: {self.config.get_param('num_groups')}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error preprocessing data: {str(e)}")
            return False
    
    def run_genetic_algorithm(self) -> bool:
        """
        Execute genetic algorithm
        
        Returns:
            bool: Success status
        """
        try:
            print(f"\n=== RUNNING GENETIC ALGORITHM ===")
            start_time = time.time()
            
            # Initialize GA
            ga = GeneticAlgorithm(
                self.preprocessed_data,
                self.config,
                self.logger
            )
            
            # Run optimization
            print("Starting GA optimization...")
            self.best_solution = ga.run()
            
            end_time = time.time()
            runtime = end_time - start_time
            
            # Store runtime stats
            self.runtime_stats = {
                'total_runtime': runtime,
                'generations_run': ga.current_generation,
                'final_fitness': self.best_solution['fitness'],
                'convergence_achieved': ga.converged
            }
            
            print(f"\n=== OPTIMIZATION COMPLETED ===")
            print(f"Runtime: {runtime:.2f} seconds")
            print(f"Generations: {ga.current_generation}")
            print(f"Final fitness: {self.best_solution['fitness']:.4f}")
            print(f"Convergence: {'Yes' if ga.converged else 'No'}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error running GA: {str(e)}")
            return False
    
    def export_results(self) -> bool:
        """
        Export results to files
        
        Returns:
            bool: Success status
        """
        try:
            print(f"\n=== EXPORTING RESULTS ===")
            
            # Ensure results directory exists
            results_dir = Path(self.config.RESULTS_PATH)
            results_dir.mkdir(exist_ok=True)
            
            # Initialize exporters
            result_exporter = ResultExporter(self.config.RESULTS_PATH)
            report_generator = ReportGenerator(self.config.RESULTS_PATH)
            
            # Export grouped results
            result_file = result_exporter.export_grouping_results(
                self.data,
                self.best_solution['chromosome'],
                self.preprocessed_data
            )
            print(f"✅ Grouping results exported: {result_file}")
            
            # Generate summary report
            report_file = report_generator.generate_summary_report(
                self.best_solution,
                self.config.get_all_params(),
                self.runtime_stats,
                self.preprocessed_data
            )
            print(f"✅ Summary report generated: {report_file}")
            
            # Export evolution log
            log_file = self.logger.export_evolution_log(self.config.RESULTS_PATH)
            print(f"✅ Evolution log exported: {log_file}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error exporting results: {str(e)}")
            return False
    
    def run_complete_process(self, data_path: str = None) -> bool:
        """
        Run complete KKM grouping process
        
        Args:
            data_path: Path to input CSV file
            
        Returns:
            bool: Success status
        """
        print("="*60)
        print("🧬 GENETIC ALGORITHM KKM GROUPING SYSTEM")
        print("="*60)
        
        # Step 1: Load and validate data
        if not self.load_and_validate_data(data_path):
            return False
        
        # Step 2: Preprocess data
        if not self.preprocess_data():
            return False
        
        # Step 3: Run genetic algorithm
        if not self.run_genetic_algorithm():
            return False
        
        # Step 4: Export results
        if not self.export_results():
            return False
        
        print("\n" + "="*60)
        print("🎉 KKM GROUPING PROCESS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        return True


def main():
    """Main execution function"""
    try:
        # Create system with default configuration
        system = KKMGroupingSystem()
        
        # Run complete process
        success = system.run_complete_process()
        
        if success:
            print("\n✅ Process completed successfully!")
            print("Check the 'results' folder for output files:")
            print("  - hasil_pengelompokan.csv")
            print("  - summary_report.txt")
            print("  - evolution_log.csv")
        else:
            print("\n❌ Process failed. Please check error messages above.")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n\n⚠️ Process interrupted by user")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()