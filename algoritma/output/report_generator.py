"""
Report generator module for GA KKM Grouping
Generates comprehensive summary reports
"""

import os
from typing import Dict, Any, List
from datetime import datetime
import json


class ReportGenerator:
    """Class for generating comprehensive GA reports"""
    
    def __init__(self, output_dir: str):
        """
        Initialize report generator
        
        Args:
            output_dir: Directory for output files
        """
        self.output_dir = output_dir
        
        # Create subfolder for reports
        self.reports_dir = os.path.join(output_dir, 'reports')
        
        # Ensure directory exists
        os.makedirs(self.reports_dir, exist_ok=True)
    
    def generate_summary_report(self, best_solution: Dict[str, Any],
                              ga_parameters: Dict[str, Any],
                              runtime_stats: Dict[str, Any],
                              preprocessed_data: Dict[str, Any],
                              filename: str = None) -> str:
        """
        Generate comprehensive summary report
        
        Args:
            best_solution: Best solution from GA
            ga_parameters: GA configuration parameters
            runtime_stats: Runtime statistics
            preprocessed_data: Preprocessed data structure
            filename: Custom filename
            
        Returns:
            str: Path to generated report
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"summary_report_{timestamp}.txt"
        
        filepath = os.path.join(self.reports_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            self._write_report_header(f)
            self._write_execution_summary(f, runtime_stats, ga_parameters)
            self._write_data_summary(f, preprocessed_data)
            self._write_solution_quality(f, best_solution)
            self._write_constraint_analysis(f, best_solution)
            self._write_evolution_statistics(f, best_solution)
            self._write_parameter_settings(f, ga_parameters)
            self._write_recommendations(f, best_solution, preprocessed_data)
            self._write_report_footer(f)
        
        print(f"✅ Summary report generated: {filepath}")
        return filepath
    
    def _write_report_header(self, f):
        """Write report header"""
        f.write("="*80 + "\n")
        f.write("           GENETIC ALGORITHM KKM GROUPING - SUMMARY REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"System: Algoritma Genetika Pengelompokan KKM UIN Malang\n\n")
    
    def _write_execution_summary(self, f, runtime_stats: Dict[str, Any], ga_parameters: Dict[str, Any]):
        """Write execution summary section"""
        f.write("EXECUTION SUMMARY\n")
        f.write("-" * 40 + "\n")
        
        # Basic execution info
        f.write(f"Total Runtime:           {runtime_stats.get('total_runtime', 0):.2f} seconds\n")
        f.write(f"Generations Executed:    {runtime_stats.get('generations_run', 0)}\n")
        f.write(f"Final Fitness Achieved:  {runtime_stats.get('final_fitness', 0):.6f}\n")
        f.write(f"Target Fitness:          {ga_parameters.get('target_fitness', 0):.6f}\n")
        f.write(f"Convergence Status:      {'Achieved' if runtime_stats.get('convergence_achieved', False) else 'Not Achieved'}\n")
        
        # Performance metrics
        avg_time_per_gen = runtime_stats.get('total_runtime', 0) / max(runtime_stats.get('generations_run', 1), 1)
        f.write(f"Average Time per Gen:    {avg_time_per_gen:.3f} seconds\n")
        
        # Success rate
        target_fitness = ga_parameters.get('target_fitness', 1.0)
        final_fitness = runtime_stats.get('final_fitness', 0)
        achievement_rate = (final_fitness / target_fitness) * 100 if target_fitness > 0 else 0
        f.write(f"Target Achievement:      {achievement_rate:.1f}%\n\n")
    
    def _write_data_summary(self, f, preprocessed_data: Dict[str, Any]):
        """Write data summary section"""
        f.write("DATA SUMMARY\n")
        f.write("-" * 40 + "\n")
        
        f.write(f"Total Students:          {preprocessed_data.get('total_students', 0)}\n")
        f.write(f"Target Groups:           {preprocessed_data.get('num_groups', 0)}\n")
        f.write(f"Average Group Size:      {preprocessed_data.get('group_size_info', {}).get('average_size', 0):.1f}\n")
        f.write(f"Size Range:              {preprocessed_data.get('group_size_info', {}).get('min_size', 0)} - {preprocessed_data.get('group_size_info', {}).get('max_size', 0)}\n\n")
        
        # Student demographics
        htq_count = preprocessed_data.get('htq_count', 0)
        total_students = preprocessed_data.get('total_students', 1)
        htq_percentage = (htq_count / total_students) * 100
        
        f.write("Student Demographics:\n")
        f.write(f"  HTQ Students:          {htq_count} ({htq_percentage:.1f}%)\n")
        f.write(f"  Non-HTQ Students:      {total_students - htq_count} ({100 - htq_percentage:.1f}%)\n")
        
        gender_ratio = preprocessed_data.get('gender_ratio', {})
        lk_ratio = gender_ratio.get('LK', 0) * 100
        pr_ratio = gender_ratio.get('PR', 0) * 100
        
        f.write(f"  Gender Ratio LK:PR:    {lk_ratio:.1f}% : {pr_ratio:.1f}%\n")
        f.write(f"  Total Faculties:       {len(preprocessed_data.get('faculty_distribution', {}))}\n")
        f.write(f"  Total Majors:          {len(preprocessed_data.get('major_distribution', {}))}\n")
        f.write(f"  High-Risk Majors:      {len(preprocessed_data.get('duplicate_risk_majors', []))}\n\n")
    
    def _write_solution_quality(self, f, best_solution: Dict[str, Any]):
        """Write solution quality section"""
        f.write("SOLUTION QUALITY ASSESSMENT\n")
        f.write("-" * 40 + "\n")
        
        fitness_analysis = best_solution.get('detailed_analysis', {}).get('fitness_analysis', {})
        total_fitness = fitness_analysis.get('total_fitness', 0)
        
        # Overall quality assessment
        quality = self._assess_solution_quality(total_fitness)
        f.write(f"Overall Quality:         {quality}\n")
        f.write(f"Total Fitness Score:     {total_fitness:.6f}\n\n")
        
        # Individual constraint scores
        constraint_scores = fitness_analysis.get('constraint_scores', {})
        contributions = fitness_analysis.get('weighted_contributions', {})
        
        f.write("Constraint Performance:\n")
        f.write("  Constraint          Score    Weight   Contribution\n")
        f.write("  " + "-" * 50 + "\n")
        
        constraints = [
            ('HTQ Distribution', 'htq_score', 'htq_contribution'),
            ('Major Duplication', 'duplication_score', 'duplication_contribution'),
            ('Gender Balance', 'gender_score', 'gender_contribution'),
            ('Group Size', 'size_score', 'size_contribution')
        ]
        
        for name, score_key, contrib_key in constraints:
            score = constraint_scores.get(score_key, 0)
            contrib = contributions.get(contrib_key, 0)
            weight = contrib / score if score > 0 else 0
            
            f.write(f"  {name:<18} {score:>6.3f}   {weight:>6.3f}   {contrib:>10.6f}\n")
        
        f.write("\n")
    
    def _write_constraint_analysis(self, f, best_solution: Dict[str, Any]):
        """Write detailed constraint analysis"""
        f.write("DETAILED CONSTRAINT ANALYSIS\n")
        f.write("-" * 40 + "\n")
        
        constraint_analysis = best_solution.get('detailed_analysis', {}).get('constraint_analysis', {})
        
        # HTQ Analysis
        f.write("1. HTQ Distribution:\n")
        htq_analysis = constraint_analysis.get('htq_analysis', {})
        f.write(f"   Groups with HTQ:      {htq_analysis.get('groups_with_htq', 0)}\n")
        f.write(f"   Groups without HTQ:   {htq_analysis.get('groups_without_htq', 0)}\n")
        f.write(f"   Total HTQ Students:   {htq_analysis.get('total_htq_students', 0)}\n\n")
        
        # Duplication Analysis
        f.write("2. Major Duplication:\n")
        dup_analysis = constraint_analysis.get('duplication_analysis', {})
        f.write(f"   Groups with Duplications: {dup_analysis.get('groups_with_duplications', 0)}\n")
        f.write(f"   Total Duplications:       {dup_analysis.get('total_duplications', 0)}\n")
        
        worst_dups = dup_analysis.get('worst_duplications', [])
        if worst_dups:
            f.write("   Worst Duplication Cases:\n")
            for i, group in enumerate(worst_dups[:5]):  # Top 5
                group_id = group.get('group_id', 'N/A')
                duplications = group.get('duplications', [])
                dup_count = sum(d.get('count', 0) for d in duplications)
                f.write(f"     Group {group_id}: {dup_count} duplications\n")
        f.write("\n")
        
        # Gender Analysis
        f.write("3. Gender Balance:\n")
        gender_analysis = constraint_analysis.get('gender_analysis', {})
        f.write(f"   Balanced Groups:      {gender_analysis.get('balanced_groups', 0)}\n")
        
        imbalanced = gender_analysis.get('imbalanced_groups', [])
        f.write(f"   Imbalanced Groups:    {len(imbalanced)}\n")
        if imbalanced:
            f.write("   Worst Imbalances:\n")
            sorted_imbalanced = sorted(imbalanced, key=lambda x: x.get('deviation', 0), reverse=True)
            for group in sorted_imbalanced[:5]:  # Top 5
                group_id = group.get('group_id', 'N/A')
                lk_count = group.get('lk_count', 0)
                deviation = group.get('deviation', 0)
                f.write(f"     Group {group_id}: LK={lk_count}, Deviation={deviation}\n")
        f.write("\n")
        
        # Size Analysis
        f.write("4. Group Size Balance:\n")
        size_analysis = constraint_analysis.get('size_analysis', {})
        f.write(f"   Properly Sized Groups: {size_analysis.get('balanced_groups', 0)}\n")
        f.write(f"   Oversized Groups:      {len(size_analysis.get('oversized_groups', []))}\n")
        f.write(f"   Undersized Groups:     {len(size_analysis.get('undersized_groups', []))}\n\n")
    
    def _write_evolution_statistics(self, f, best_solution: Dict[str, Any]):
        """Write evolution statistics"""
        f.write("EVOLUTION STATISTICS\n")
        f.write("-" * 40 + "\n")
        
        convergence_info = best_solution.get('convergence_info', {})
        fitness_history = best_solution.get('fitness_history', {})
        
        f.write(f"Solution Found at Gen:   {best_solution.get('generation_found', 0)}\n")
        f.write(f"Total Generations Run:   {best_solution.get('total_generations', 0)}\n")
        f.write(f"Initial Best Fitness:    {convergence_info.get('initial_fitness', 0):.6f}\n")
        f.write(f"Final Best Fitness:      {convergence_info.get('final_fitness', 0):.6f}\n")
        f.write(f"Total Improvement:       {convergence_info.get('total_improvement', 0):.6f}\n")
        f.write(f"Avg Improvement per Gen: {convergence_info.get('average_improvement_per_generation', 0):.6f}\n")
        f.write(f"Longest Plateau:         {convergence_info.get('longest_plateau_length', 0)} generations\n")
        f.write(f"Convergence Rate:        {convergence_info.get('convergence_rate', 'Unknown')}\n\n")
    
    def _write_parameter_settings(self, f, ga_parameters: Dict[str, Any]):
        """Write GA parameter settings"""
        f.write("ALGORITHM PARAMETERS\n")
        f.write("-" * 40 + "\n")
        
        f.write("Population & Evolution:\n")
        f.write(f"  Population Size:       {ga_parameters.get('population_size', 0)}\n")
        f.write(f"  Max Generations:       {ga_parameters.get('max_generation', 0)}\n")
        f.write(f"  Crossover Rate:        {ga_parameters.get('crossover_rate', 0):.3f}\n")
        f.write(f"  Mutation Rate:         {ga_parameters.get('mutation_rate', 0):.3f}\n")
        f.write(f"  Selection Method:      {ga_parameters.get('selection_method', 'Unknown')}\n\n")
        
        f.write("Constraint Weights:\n")
        f.write(f"  HTQ Weight:            {ga_parameters.get('weight_htq', 0):.3f}\n")
        f.write(f"  Duplication Weight:    {ga_parameters.get('weight_duplikasi', 0):.3f}\n")
        f.write(f"  Gender Weight:         {ga_parameters.get('weight_gender', 0):.3f}\n")
        f.write(f"  Size Weight:           {ga_parameters.get('weight_jumlah', 0):.3f}\n\n")
        
        f.write("Stopping Criteria:\n")
        f.write(f"  Target Fitness:        {ga_parameters.get('target_fitness', 0):.3f}\n")
        f.write(f"  Stagnation Limit:      {ga_parameters.get('stagnation_limit', 0)}\n\n")
    
    def _write_recommendations(self, f, best_solution: Dict[str, Any], preprocessed_data: Dict[str, Any]):
        """Write recommendations"""
        f.write("RECOMMENDATIONS\n")
        f.write("-" * 40 + "\n")
        
        fitness_analysis = best_solution.get('detailed_analysis', {}).get('fitness_analysis', {})
        total_fitness = fitness_analysis.get('total_fitness', 0)
        constraint_scores = fitness_analysis.get('constraint_scores', {})
        
        recommendations = []
        
        # Overall quality recommendations
        if total_fitness < 0.70:
            recommendations.append("Consider running the algorithm longer or adjusting parameters")
        elif total_fitness < 0.85:
            recommendations.append("Good solution achieved, minor improvements possible")
        else:
            recommendations.append("Excellent solution quality achieved")
        
        # Specific constraint recommendations
        if constraint_scores.get('htq_score', 0) < 0.80:
            htq_count = preprocessed_data.get('htq_count', 0)
            num_groups = preprocessed_data.get('num_groups', 1)
            if htq_count < num_groups:
                recommendations.append(f"HTQ constraint limited by data: only {htq_count} HTQ students for {num_groups} groups")
            else:
                recommendations.append("Consider increasing HTQ constraint weight")
        
        if constraint_scores.get('duplication_score', 0) < 0.80:
            risk_majors = len(preprocessed_data.get('duplicate_risk_majors', []))
            recommendations.append(f"High duplication risk from {risk_majors} majors with many students")
        
        if constraint_scores.get('gender_score', 0) < 0.80:
            recommendations.append("Consider adjusting gender balance constraints or weights")
        
        if constraint_scores.get('size_score', 0) < 0.80:
            recommendations.append("Group size imbalances detected, consider different group count")
        
        # Parameter tuning recommendations
        generations_run = best_solution.get('total_generations', 0)
        max_generations = 1000  # Default assumption
        if generations_run >= max_generations * 0.9:
            recommendations.append("Algorithm used most available generations - consider increasing limit")
        
        if not recommendations:
            recommendations.append("Solution quality is excellent - no specific recommendations")
        
        for i, rec in enumerate(recommendations, 1):
            f.write(f"{i}. {rec}\n")
        
        f.write("\n")
    
    def _write_report_footer(self, f):
        """Write report footer"""
        f.write("="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")
        f.write("\nGenerated by: Algoritma Genetika KKM Grouping System\n")
        f.write("UIN Maulana Malik Ibrahim Malang\n")
        f.write(f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    def _assess_solution_quality(self, fitness_score: float) -> str:
        """Assess solution quality based on fitness"""
        if fitness_score >= 0.95:
            return "Excellent"
        elif fitness_score >= 0.85:
            return "Very Good"
        elif fitness_score >= 0.70:
            return "Good"
        elif fitness_score >= 0.50:
            return "Fair"
        else:
            return "Poor"
    
    def generate_technical_report(self, best_solution: Dict[str, Any],
                                ga_parameters: Dict[str, Any],
                                preprocessed_data: Dict[str, Any],
                                filename: str = None) -> str:
        """
        Generate technical report with detailed algorithm analysis
        
        Args:
            best_solution: Best solution from GA
            ga_parameters: GA parameters
            preprocessed_data: Preprocessed data
            filename: Custom filename
            
        Returns:
            str: Path to technical report
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"technical_report_{timestamp}.txt"
        
        filepath = os.path.join(self.reports_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("           GENETIC ALGORITHM KKM GROUPING - TECHNICAL REPORT\n")
            f.write("="*80 + "\n\n")
            
            # Algorithm Analysis
            f.write("ALGORITHM PERFORMANCE ANALYSIS\n")
            f.write("-" * 40 + "\n")
            
            convergence_info = best_solution.get('convergence_info', {})
            fitness_history = best_solution.get('fitness_history', {})
            
            # Convergence analysis
            f.write("Convergence Analysis:\n")
            f.write(f"  Convergence Rate:      {convergence_info.get('convergence_rate', 'Unknown')}\n")
            f.write(f"  Plateau Analysis:      {convergence_info.get('longest_plateau_length', 0)} max plateau\n")
            f.write(f"  Improvement Pattern:   {convergence_info.get('total_improvement', 0):.6f} total\n\n")
            
            # Population statistics
            pop_stats = best_solution.get('final_population_stats', {})
            if pop_stats:
                f.write("Final Population Statistics:\n")
                f.write(f"  Population Size:       {pop_stats.get('population_size', 0)}\n")
                f.write(f"  Best Fitness:          {pop_stats.get('best_fitness', 0):.6f}\n")
                f.write(f"  Average Fitness:       {pop_stats.get('average_fitness', 0):.6f}\n")
                f.write(f"  Standard Deviation:    {pop_stats.get('std_deviation', 0):.6f}\n")
                f.write(f"  Population Diversity:  {pop_stats.get('variance', 0):.6f}\n\n")
            
            # Export solution data in JSON format for further analysis
            solution_data = {
                'fitness_history': fitness_history,
                'convergence_info': convergence_info,
                'final_solution': {
                    'fitness': best_solution.get('fitness', 0),
                    'generation_found': best_solution.get('generation_found', 0)
                },
                'parameters': ga_parameters
            }
            
            json_filename = filename.replace('.txt', '_data.json')
            json_filepath = os.path.join(self.reports_dir, json_filename)
            
            with open(json_filepath, 'w', encoding='utf-8') as json_f:
                json.dump(solution_data, json_f, indent=2)
            
            f.write(f"Detailed data exported to: {json_filename}\n")
        
        print(f"✅ Technical report generated: {filepath}")
        return filepath


def generate_comprehensive_report(output_dir: str, best_solution: Dict[str, Any],
                                ga_parameters: Dict[str, Any], runtime_stats: Dict[str, Any],
                                preprocessed_data: Dict[str, Any]) -> Dict[str, str]:
    """
    Generate all types of reports
    
    Args:
        output_dir: Output directory
        best_solution: Best solution from GA
        ga_parameters: GA parameters
        runtime_stats: Runtime statistics
        preprocessed_data: Preprocessed data
        
    Returns:
        dict: Paths to generated reports
    """
    generator = ReportGenerator(output_dir)
    
    reports = {}
    
    # Generate summary report
    reports['summary_report'] = generator.generate_summary_report(
        best_solution, ga_parameters, runtime_stats, preprocessed_data
    )
    
    # Generate technical report
    reports['technical_report'] = generator.generate_technical_report(
        best_solution, ga_parameters, preprocessed_data
    )
    
    return reports