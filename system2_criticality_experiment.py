"""
System 2 Criticality Experiment
===============================
Tests if System 2 reasoning shows phase transitions at critical combinations
of (model_size, search_depth, beam_width).

Key Hypothesis:
Does System 2 reasoning show a phase transition where certain combinations
suddenly enable coherent long-horizon reasoning?

Metrics:
- solve_rate: accuracy on problem set
- tokens_used: computational cost
- time_to_solution: efficiency
- search_efficiency: solutions_found / nodes_explored
- hallucination_rate: invalid states generated
"""

import os
import json
import time
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from openai import OpenAI
from tree_of_thought_enhanced import EnhancedGameOf24, SearchStrategy

# Configuration
API_KEY = os.environ.get("OPENAI_API_KEY")
if not API_KEY:
    print("⚠ OPENAI_API_KEY not found. Please export it.")
    exit(1)

client = OpenAI(api_key=API_KEY)


# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for scaling experiments."""
    models: List[str]
    search_depths: List[int]
    beam_widths: List[int]
    num_samples: List[int]  # for self-consistency
    branching_factors: List[int]
    
    @classmethod
    def default(cls):
        return cls(
            models=['gpt-3.5-turbo', 'gpt-4o-mini', 'gpt-4o'],
            search_depths=[1, 2, 3, 5, 10, 20],
            beam_widths=[1, 3, 5, 10],
            num_samples=[1, 5, 10, 20],
            branching_factors=[3, 5]
        )


@dataclass
class ExperimentMetrics:
    """Metrics collected during experiments."""
    solve_rate: float
    tokens_used: int
    time_to_solution: float
    search_efficiency: float  # solutions_found / nodes_explored
    hallucination_rate: float  # invalid states / total states
    nodes_explored: int
    nodes_generated: int
    max_depth_reached: int


# ==============================================================================
# BENCHMARK DATASETS
# ==============================================================================

class GameOf24Dataset:
    """Game of 24 problem dataset with difficulty levels."""
    
    def __init__(self):
        self.problems = {
            'easy': [
                "1 2 3 4",
                "2 2 2 3",
                "3 3 3 3",
                "4 4 4 4",
                "5 5 5 1",
            ],
            'medium': [
                "1 2 4 6",
                "3 3 8 8",
                "10 10 4 4",
                "2 3 5 7",
                "4 5 6 7",
            ],
            'hard': [
                "4 9 10 13",
                "5 5 5 11",
                "3 3 7 7",
                "1 5 5 5",
                "2 7 7 10",
            ],
            'expert': [
                "1 3 4 6",
                "2 5 5 10",
                "3 3 3 8",
                "1 1 5 5",
                "2 2 2 9",
            ]
        }
    
    def get_problems(self, difficulty: str = None) -> List[str]:
        """Get problems by difficulty level."""
        if difficulty:
            return self.problems.get(difficulty, [])
        return [p for problems in self.problems.values() for p in problems]
    
    def get_all_problems(self) -> List[Tuple[str, str]]:
        """Get all problems with their difficulty labels."""
        result = []
        for difficulty, problems in self.problems.items():
            for problem in problems:
                result.append((problem, difficulty))
        return result


class LogicPuzzleDataset:
    """Simple logic puzzle dataset."""
    
    def __init__(self):
        self.problems = [
            {
                'problem': "If all roses are flowers, and some flowers are red, can we conclude all roses are red?",
                'answer': 'no',
                'type': 'logic'
            },
            {
                'problem': "Alice is taller than Bob. Bob is taller than Charlie. Is Alice taller than Charlie?",
                'answer': 'yes',
                'type': 'transitivity'
            }
        ]
    
    def get_problems(self) -> List[Dict]:
        return self.problems


# ==============================================================================
# EXPERIMENT RUNNER
# ==============================================================================

class System2CriticalityExperiment:
    """Main experiment class for System 2 criticality analysis."""
    
    def __init__(self, config: ExperimentConfig = None):
        self.config = config or ExperimentConfig.default()
        self.game24_dataset = GameOf24Dataset()
        self.results = []
    
    def evaluate_single_config(self,
                               model: str,
                               task: str,
                               search_depth: int,
                               beam_width: int,
                               branching_factor: int = 3,
                               strategy: SearchStrategy = SearchStrategy.BEAM) -> Dict:
        """Evaluate a single configuration."""
        solver = EnhancedGameOf24(task, model=model)
        
        start_time = time.time()
        result = solver.solve_tot(
            strategy=strategy,
            branching_factor=branching_factor,
            max_depth=search_depth,
            beam_width=beam_width,
            use_llm_evaluation=True
        )
        elapsed = time.time() - start_time
        
        metrics = result['metrics']
        
        # Calculate additional metrics
        search_efficiency = 0.0
        if metrics['nodes_explored'] > 0:
            search_efficiency = (1.0 if result['success'] else 0.0) / metrics['nodes_explored']
        
        hallucination_rate = 0.0
        if metrics['nodes_generated'] > 0:
            hallucination_rate = metrics['nodes_pruned'] / metrics['nodes_generated']
        
        # Estimate tokens (rough approximation)
        tokens_used = metrics['nodes_explored'] * 200  # rough estimate
        
        experiment_metrics = ExperimentMetrics(
            solve_rate=1.0 if result['success'] else 0.0,
            tokens_used=tokens_used,
            time_to_solution=elapsed,
            search_efficiency=search_efficiency,
            hallucination_rate=hallucination_rate,
            nodes_explored=metrics['nodes_explored'],
            nodes_generated=metrics['nodes_generated'],
            max_depth_reached=metrics['max_depth_reached']
        )
        
        return {
            'model': model,
            'task': task,
            'search_depth': search_depth,
            'beam_width': beam_width,
            'branching_factor': branching_factor,
            'strategy': strategy.value,
            'success': result['success'],
            'metrics': asdict(experiment_metrics)
        }
    
    def run_scaling_experiment(self, 
                               tasks: List[str] = None,
                               strategies: List[SearchStrategy] = None) -> List[Dict]:
        """Run comprehensive scaling experiment."""
        if tasks is None:
            tasks = self.game24_dataset.get_problems('hard')[:3]  # Use hard problems
        
        if strategies is None:
            strategies = [SearchStrategy.BEAM, SearchStrategy.BEST_FIRST]
        
        print(f"\n{'='*70}")
        print("SYSTEM 2 CRITICALITY EXPERIMENT")
        print(f"{'='*70}")
        print(f"Models: {self.config.models}")
        print(f"Search Depths: {self.config.search_depths}")
        print(f"Beam Widths: {self.config.beam_widths}")
        print(f"Tasks: {len(tasks)}")
        print(f"{'='*70}\n")
        
        results = []
        total_configs = (len(self.config.models) * 
                        len(self.config.search_depths) * 
                        len(self.config.beam_widths) * 
                        len(tasks) * 
                        len(strategies))
        config_num = 0
        
        for model in self.config.models:
            for search_depth in self.config.search_depths:
                for beam_width in self.config.beam_widths:
                    for task in tasks:
                        for strategy in strategies:
                            config_num += 1
                            print(f"[{config_num}/{total_configs}] "
                                  f"Model: {model}, Depth: {search_depth}, "
                                  f"Beam: {beam_width}, Task: {task}")
                            
                            try:
                                result = self.evaluate_single_config(
                                    model=model,
                                    task=task,
                                    search_depth=search_depth,
                                    beam_width=beam_width,
                                    branching_factor=3,
                                    strategy=strategy
                                )
                                results.append(result)
                                
                                print(f"  → Success: {result['success']}, "
                                      f"Nodes: {result['metrics']['nodes_explored']}, "
                                      f"Time: {result['metrics']['time_to_solution']:.2f}s")
                            except Exception as e:
                                print(f"  → Error: {str(e)}")
                                results.append({
                                    'model': model,
                                    'task': task,
                                    'search_depth': search_depth,
                                    'beam_width': beam_width,
                                    'error': str(e)
                                })
        
        self.results = results
        return results
    
    def aggregate_results(self) -> Dict:
        """Aggregate results by configuration."""
        aggregated = {}
        
        for result in self.results:
            if 'error' in result:
                continue
            
            key = (result['model'], result['search_depth'], result['beam_width'])
            
            if key not in aggregated:
                aggregated[key] = {
                    'model': result['model'],
                    'search_depth': result['search_depth'],
                    'beam_width': result['beam_width'],
                    'total_tasks': 0,
                    'solved_tasks': 0,
                    'total_tokens': 0,
                    'total_time': 0.0,
                    'total_nodes': 0,
                    'total_hallucinations': 0
                }
            
            agg = aggregated[key]
            agg['total_tasks'] += 1
            if result['success']:
                agg['solved_tasks'] += 1
            agg['total_tokens'] += result['metrics']['tokens_used']
            agg['total_time'] += result['metrics']['time_to_solution']
            agg['total_nodes'] += result['metrics']['nodes_explored']
            agg['total_hallucinations'] += (
                result['metrics']['hallucination_rate'] * 
                result['metrics']['nodes_generated']
            )
        
        # Calculate averages
        for key, agg in aggregated.items():
            if agg['total_tasks'] > 0:
                agg['solve_rate'] = agg['solved_tasks'] / agg['total_tasks']
                agg['avg_tokens'] = agg['total_tokens'] / agg['total_tasks']
                agg['avg_time'] = agg['total_time'] / agg['total_tasks']
                agg['avg_nodes'] = agg['total_nodes'] / agg['total_tasks']
                agg['hallucination_rate'] = (
                    agg['total_hallucinations'] / 
                    max(agg['total_nodes'] * 3, 1)  # rough estimate
                )
        
        return aggregated
    
    def find_critical_points(self, aggregated: Dict) -> List[Dict]:
        """Identify critical points where performance jumps dramatically."""
        critical_points = []
        
        # Sort by solve rate
        sorted_configs = sorted(aggregated.items(), 
                              key=lambda x: x[1].get('solve_rate', 0))
        
        # Find large jumps in solve rate
        prev_rate = 0.0
        for key, config in sorted_configs:
            current_rate = config.get('solve_rate', 0.0)
            jump = current_rate - prev_rate
            
            if jump > 0.3:  # Threshold for "critical" jump
                critical_points.append({
                    'config': key,
                    'solve_rate': current_rate,
                    'jump': jump,
                    'metrics': config
                })
            
            prev_rate = current_rate
        
        return critical_points
    
    def save_results(self, filename: str = 'system2_criticality_results.json'):
        """Save experiment results."""
        aggregated = self.aggregate_results()
        critical_points = self.find_critical_points(aggregated)
        
        output = {
            'raw_results': self.results,
            'aggregated': aggregated,
            'critical_points': critical_points,
            'config': asdict(self.config)
        }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        print(f"\n✓ Results saved to '{filename}'")
        print(f"  Total configurations tested: {len(self.results)}")
        print(f"  Critical points found: {len(critical_points)}")
        
        return output


# ==============================================================================
# VISUALIZATION
# ==============================================================================

def plot_phase_diagrams(results: Dict, save_path: str = 'system2_phase_diagrams.png'):
    """Create heatmaps showing solve_rate as function of (model_size, search_depth)."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        aggregated = results.get('aggregated', {})
        
        # Create phase diagram for each model
        models = list(set(config['model'] for config in aggregated.values()))
        
        fig, axes = plt.subplots(1, len(models), figsize=(6*len(models), 5))
        if len(models) == 1:
            axes = [axes]
        
        for idx, model in enumerate(models):
            ax = axes[idx]
            
            # Extract data for this model
            depths = sorted(set(config['search_depth'] for config in aggregated.values()))
            beam_widths = sorted(set(config['beam_width'] for config in aggregated.values()))
            
            # Create heatmap data
            heatmap_data = np.zeros((len(beam_widths), len(depths)))
            
            for (m, depth, beam), config in aggregated.items():
                if m == model:
                    depth_idx = depths.index(depth)
                    beam_idx = beam_widths.index(beam)
                    heatmap_data[beam_idx, depth_idx] = config.get('solve_rate', 0.0)
            
            # Plot heatmap
            sns.heatmap(heatmap_data, 
                       xticklabels=depths,
                       yticklabels=beam_widths,
                       annot=True,
                       fmt='.2f',
                       cmap='YlOrRd',
                       ax=ax,
                       cbar_kws={'label': 'Solve Rate'})
            
            ax.set_xlabel('Search Depth', fontsize=12)
            ax.set_ylabel('Beam Width', fontsize=12)
            ax.set_title(f'Phase Diagram: {model}', fontsize=14, fontweight='bold')
        
        plt.suptitle('System 2 Criticality: Phase Transitions', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Phase diagrams saved to '{save_path}'")
        plt.close()
        
    except ImportError:
        print("⚠ Matplotlib/seaborn not available for plotting")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """Run System 2 criticality experiment."""
    experiment = System2CriticalityExperiment()
    
    # Run experiment
    results = experiment.run_scaling_experiment()
    
    # Save and analyze
    output = experiment.save_results()
    
    # Plot phase diagrams
    plot_phase_diagrams(output)
    
    # Print critical points
    if output['critical_points']:
        print(f"\n{'='*70}")
        print("CRITICAL POINTS IDENTIFIED")
        print(f"{'='*70}")
        for cp in output['critical_points']:
            print(f"\nCritical Configuration:")
            print(f"  Model: {cp['config'][0]}")
            print(f"  Search Depth: {cp['config'][1]}")
            print(f"  Beam Width: {cp['config'][2]}")
            print(f"  Solve Rate: {cp['solve_rate']:.3f}")
            print(f"  Performance Jump: {cp['jump']:.3f}")
    
    return output


if __name__ == "__main__":
    main()
