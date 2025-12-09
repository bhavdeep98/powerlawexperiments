"""
Comprehensive System 2 Experiment Protocol
===========================================
Unified framework for running all System 2 experiments with proper
dataset loading, experiment orchestration, and result aggregation.

Integrates:
- Tree of Thought (multiple strategies)
- DSPy optimizations
- System 2 criticality experiments
- State tracking benchmarks
- Advanced architectures
- Power law analysis
"""

import os
import json
import time
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path

# Import all experiment modules
from tree_of_thought_enhanced import EnhancedGameOf24, SearchStrategy
from system2_criticality_experiment import System2CriticalityExperiment, ExperimentConfig
from state_tracking_benchmarks import run_state_tracking_experiments
from advanced_system2_architectures import (
    VerifierReasonerSystem, DebateSystem, MemoryAugmentedReasoner
)


# ==============================================================================
# DATASET LOADERS
# ==============================================================================

class DatasetLoader:
    """Load and manage datasets for System 2 experiments."""
    
    @staticmethod
    def load_game_of_24(difficulty: str = None) -> List[str]:
        """Load Game of 24 problems."""
        from system2_criticality_experiment import GameOf24Dataset
        dataset = GameOf24Dataset()
        return dataset.get_problems(difficulty) if difficulty else dataset.get_all_problems()
    
    @staticmethod
    def load_gsm8k_sample(num_problems: int = 10) -> List[Dict]:
        """Load sample GSM8K problems (simplified)."""
        # In practice, would load from actual GSM8K dataset
        sample_problems = [
            {
                'problem': 'Janet has 3 apples. She gives 1 to Bob. How many does she have?',
                'answer': '2',
                'type': 'arithmetic'
            },
            {
                'problem': 'A store has 20 items. They sell 5. How many remain?',
                'answer': '15',
                'type': 'arithmetic'
            }
        ]
        return sample_problems[:num_problems]
    
    @staticmethod
    def load_logic_puzzles() -> List[Dict]:
        """Load logic puzzle problems."""
        from system2_criticality_experiment import LogicPuzzleDataset
        dataset = LogicPuzzleDataset()
        return dataset.get_problems()
    
    @staticmethod
    def load_custom_problems(filepath: str) -> List[Dict]:
        """Load custom problems from JSON file."""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
        return []


# ==============================================================================
# EXPERIMENT ORCHESTRATOR
# ==============================================================================

@dataclass
class ExperimentSuite:
    """Configuration for a suite of experiments."""
    name: str
    run_tot_comparison: bool = True
    run_dspy_optimization: bool = True
    run_criticality: bool = True
    run_state_tracking: bool = True
    run_advanced_architectures: bool = True
    run_power_law_analysis: bool = True
    
    # Specific configurations
    tot_strategies: List[str] = None
    criticality_config: ExperimentConfig = None
    state_tracking_steps: List[int] = None
    
    def __post_init__(self):
        if self.tot_strategies is None:
            self.tot_strategies = ['BFS', 'DFS', 'BEST_FIRST', 'BEAM']
        if self.state_tracking_steps is None:
            self.state_tracking_steps = [5, 10, 15, 20]


class System2ExperimentOrchestrator:
    """Orchestrates all System 2 experiments."""
    
    def __init__(self, suite: ExperimentSuite):
        self.suite = suite
        self.results = {}
        self.start_time = None
    
    def run_tot_comparison(self) -> Dict:
        """Run Tree of Thought comparison experiment."""
        print(f"\n{'='*70}")
        print("EXPERIMENT: Tree of Thought Strategy Comparison")
        print(f"{'='*70}")
        
        tasks = DatasetLoader.load_game_of_24('hard')[:3]
        strategies = [SearchStrategy[s] for s in self.suite.tot_strategies 
                    if hasattr(SearchStrategy, s)]
        
        from tree_of_thought_enhanced import run_comparison_experiment
        results = run_comparison_experiment(tasks, strategies)
        
        self.results['tot_comparison'] = results
        return results
    
    def run_dspy_optimization(self) -> Dict:
        """Run DSPy optimization experiments."""
        print(f"\n{'='*70}")
        print("EXPERIMENT: DSPy Optimization Comparison")
        print(f"{'='*70}")
        
        from dspy_reasoning_enhanced import run_enhanced_experiment
        results = run_enhanced_experiment()
        
        self.results['dspy_optimization'] = results
        return results
    
    def run_criticality_experiment(self) -> Dict:
        """Run System 2 criticality experiment."""
        print(f"\n{'='*70}")
        print("EXPERIMENT: System 2 Criticality Analysis")
        print(f"{'='*70}")
        
        config = self.suite.criticality_config or ExperimentConfig.default()
        experiment = System2CriticalityExperiment(config)
        
        tasks = DatasetLoader.load_game_of_24('hard')[:3]
        results = experiment.run_scaling_experiment(tasks)
        output = experiment.save_results()
        
        self.results['criticality'] = output
        return output
    
    def run_state_tracking(self) -> Dict:
        """Run state tracking benchmarks."""
        print(f"\n{'='*70}")
        print("EXPERIMENT: State Tracking Fidelity")
        print(f"{'='*70}")
        
        results = run_state_tracking_experiments(
            num_steps_list=self.suite.state_tracking_steps
        )
        
        # Find critical points
        from state_tracking_benchmarks import find_critical_tracking_point
        critical_points = find_critical_tracking_point(results)
        
        output = {
            'results': results,
            'critical_points': critical_points
        }
        
        self.results['state_tracking'] = output
        return output
    
    def run_advanced_architectures(self) -> Dict:
        """Run advanced architecture comparisons."""
        print(f"\n{'='*70}")
        print("EXPERIMENT: Advanced System 2 Architectures")
        print(f"{'='*70}")
        
        problems = [
            f"Use numbers {nums} and arithmetic operations to get 24"
            for nums in DatasetLoader.load_game_of_24('hard')[:3]
        ]
        
        from advanced_system2_architectures import compare_architectures
        results = compare_architectures(problems)
        
        self.results['advanced_architectures'] = results
        return results
    
    def run_power_law_analysis(self) -> Dict:
        """Run power law analysis on System 2 results."""
        print(f"\n{'='*70}")
        print("EXPERIMENT: Power Law Analysis")
        print(f"{'='*70}")
        
        from system2_power_law_analysis import analyze_system2_scaling
        results = analyze_system2_scaling(self.results)
        
        self.results['power_law_analysis'] = results
        return results
    
    def run_all(self) -> Dict:
        """Run all experiments in the suite."""
        self.start_time = time.time()
        
        print(f"\n{'='*70}")
        print(f"SYSTEM 2 COMPREHENSIVE EXPERIMENT SUITE")
        print(f"Suite: {self.suite.name}")
        print(f"{'='*70}")
        
        if self.suite.run_tot_comparison:
            self.run_tot_comparison()
        
        if self.suite.run_dspy_optimization:
            self.run_dspy_optimization()
        
        if self.suite.run_criticality:
            self.run_criticality_experiment()
        
        if self.suite.run_state_tracking:
            self.run_state_tracking()
        
        if self.suite.run_advanced_architectures:
            self.run_advanced_architectures()
        
        if self.suite.run_power_law_analysis:
            self.run_power_law_analysis()
        
        total_time = time.time() - self.start_time
        
        # Save comprehensive results
        output = {
            'suite_name': self.suite.name,
            'config': asdict(self.suite),
            'results': self.results,
            'total_time': total_time,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        output_file = f'system2_comprehensive_results_{int(time.time())}.json'
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        print(f"\n{'='*70}")
        print("ALL EXPERIMENTS COMPLETED")
        print(f"{'='*70}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Results saved to: {output_file}")
        
        return output


# ==============================================================================
# QUICK START FUNCTIONS
# ==============================================================================

def run_quick_experiment():
    """Run a quick experiment suite for testing."""
    suite = ExperimentSuite(
        name="Quick Test",
        run_tot_comparison=True,
        run_dspy_optimization=False,  # Skip for speed
        run_criticality=False,  # Skip for speed
        run_state_tracking=False,  # Skip for speed
        run_advanced_architectures=True,
        run_power_law_analysis=False,
        tot_strategies=['BFS', 'BEAM'],
        state_tracking_steps=[5, 10]
    )
    
    orchestrator = System2ExperimentOrchestrator(suite)
    return orchestrator.run_all()


def run_full_experiment():
    """Run the full comprehensive experiment suite."""
    suite = ExperimentSuite(
        name="Full System 2 Analysis",
        run_tot_comparison=True,
        run_dspy_optimization=True,
        run_criticality=True,
        run_state_tracking=True,
        run_advanced_architectures=True,
        run_power_law_analysis=True
    )
    
    orchestrator = System2ExperimentOrchestrator(suite)
    return orchestrator.run_all()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='System 2 Comprehensive Experiments')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick test suite')
    parser.add_argument('--full', action='store_true',
                       help='Run full experiment suite')
    
    args = parser.parse_args()
    
    if args.quick:
        run_quick_experiment()
    elif args.full:
        run_full_experiment()
    else:
        print("Please specify --quick or --full")
        print("Example: python system2_experiment_protocol.py --quick")
