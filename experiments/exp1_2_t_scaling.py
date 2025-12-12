"""
Agentic Scaling Laws Experiment 1.2: T-Scaling (Interaction Depth)
====================================================================

Tests the hypothesis that agentic system performance follows a power-law
relationship with interaction depth/episodes: E(T) ~ T^(-β_T)

This extends Experiment 1.1 by varying search depth instead of number of agents.
"""

import os
import json
import time
import numpy as np
from typing import List, Dict
from pathlib import Path

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Import existing infrastructure
# Import existing infrastructure
from experiments.base_experiment import (
    System2CriticalityExperiment, 
    ExperimentConfig,
    GameOf24Dataset
)
from agents.benchmarks import GameOf24Benchmark
from analysis.system2_power_law_analysis import fit_power_law, find_critical_exponent
from agents.tree_of_thought_enhanced import SearchStrategy

# Configuration
API_KEY = os.environ.get("OPENAI_API_KEY")
if not API_KEY:
    print("⚠ OPENAI_API_KEY not found. Please check .env file.")
    exit(1)


# ==============================================================================
# T-SCALING EXPERIMENT
# ==============================================================================

def run_t_scaling_experiment(
    T_values: List[int] = [1, 3, 5, 10, 20],
    model: str = "gpt-4o-mini",
    num_problems: int = 10,
    difficulty: int = 3,
    replications: int = 2,
    beam_width: int = 3,
    branching_factor: int = 3,
    output_dir: str = "results/agentic_scaling"
) -> Dict:
    """
    Run T-scaling experiment: vary interaction depth (search depth).
    
    Args:
        T_values: List of search depths to test
        model: LLM model to use
        num_problems: Number of problems per T value
        difficulty: Problem difficulty (1-5)
        replications: Number of replications per configuration
        beam_width: Beam search width (fixed)
        branching_factor: Branching factor for search (fixed)
        output_dir: Directory to save results
        
    Returns:
        Dictionary with experiment results and analysis
    """
    print(f"\n{'='*70}")
    print("EXPERIMENT 1.2: T-SCALING (Interaction Depth)")
    print(f"{'='*70}")
    print(f"T values (search depths): {T_values}")
    print(f"Model: {model}")
    print(f"Problems: {num_problems} (difficulty {difficulty})")
    print(f"Replications: {replications}")
    print(f"Beam width: {beam_width}, Branching: {branching_factor}")
    print(f"{'='*70}\n")
    
    # Get benchmark problems
    benchmark = GameOf24Benchmark()
    problems = benchmark.get_problems(difficulty=difficulty, num_problems=num_problems)
    
    if len(problems) < num_problems:
        print(f"⚠ Warning: Only {len(problems)} problems available, using all")
        num_problems = len(problems)
    
    print(f"Loaded {len(problems)} problems\n")
    
    # Create experiment config (focus on depth variation)
    config = ExperimentConfig(
        models=[model],
        search_depths=T_values,
        beam_widths=[beam_width],
        num_samples=[1],
        branching_factors=[branching_factor]
    )
    
    experiment = System2CriticalityExperiment(config)
    
    results = []
    all_traces = []  # Store individual traces for grammar analysis
    start_time = time.time()
    
    # Convert problems to task format
    tasks = [p.metadata['numbers'] for p in problems if 'numbers' in p.metadata]
    
    for T in T_values:
        print(f"\n{'─'*70}")
        print(f"Testing T={T} (search depth)...")
        print(f"{'─'*70}")
        
        # Define strategies to compare
        strategies = [
            SearchStrategy.BEAM,
            SearchStrategy.BEST_FIRST,
            # SearchStrategy.MCTS  # Skipped: excessively slow (>15m per problem) at T>=5
        ]
        
        for strategy in strategies:
            print(f"\n  Strategy: {strategy.value.upper()}")
            
            solve_rates = []
            search_efficiencies = []
            nodes_explored_list = []
            times_list = []
            
            for rep in range(replications):
                print(f"    Replication {rep+1}/{replications}:")
                
                for i, task in enumerate(tasks):
                    problem = problems[i]
                    print(f"      Problem {i+1}/{len(tasks)}: {problem.problem_id}", end=" ... ")
                    
                    try:
                        # Evaluate single configuration
                        result = experiment.evaluate_single_config(
                            model=model,
                            task=task,
                            search_depth=T,
                            beam_width=beam_width,
                            branching_factor=branching_factor,
                            strategy=strategy
                        )
                        
                        # Use the success flag from the result (which is now strictly verified)
                        # But we also double check with benchmark just in case
                        is_correct, partial_credit = benchmark.evaluate_solution(
                            problem, result.get('solution', '')
                        )
                        
                        # Trust the benchmark's check primarily
                        solve_rates.append(1.0 if is_correct else 0.0)
                        
                        metrics = result.get('metrics', {})
                        nodes_explored = metrics.get('nodes_explored', 0)
                        nodes_explored_list.append(nodes_explored)
                        
                        if nodes_explored > 0:
                            search_efficiency = (1.0 if is_correct else 0.0) / nodes_explored
                        else:
                            search_efficiency = 0.0
                        search_efficiencies.append(search_efficiency)
                        
                        times_list.append(metrics.get('time_to_solution', result.get('time', 0)))
                        
                        # Save trace
                        all_traces.append({
                            'T': T,
                            'strategy': strategy.value,
                            'problem_id': problem.problem_id,
                            'solution_text': result.get('solution', ''),
                            'success': is_correct,
                            'node_count': nodes_explored
                        })
                        
                        status = "✓" if is_correct else "✗"
                        print(f"{status} (nodes: {nodes_explored}, time: {metrics.get('time_to_solution', 0):.1f}s)")
                        
                    except Exception as e:
                        print(f"ERROR: {str(e)}")
                        solve_rates.append(0.0)
                        search_efficiencies.append(0.0)
                        nodes_explored_list.append(0)
                        times_list.append(0.0)
            
            # Aggregate results for this T value AND strategy
            result_summary = {
                'T': T,
                'strategy': strategy.value,
                'solve_rate': float(np.mean(solve_rates)),
                'solve_rate_std': float(np.std(solve_rates)),
                'search_efficiency': float(np.mean(search_efficiencies)),
                'search_efficiency_std': float(np.std(search_efficiencies)),
                'avg_nodes_explored': float(np.mean(nodes_explored_list)),
                'avg_nodes_std': float(np.std(nodes_explored_list)),
                'avg_time': float(np.mean(times_list)),
                'avg_time_std': float(np.std(times_list)),
                'num_problems': num_problems,
                'replications': replications,
                'total_runs': len(solve_rates)
            }
            
            results.append(result_summary)
            
            print(f"\n    Summary for T={T}, {strategy.value}:")
            print(f"      Solve Rate: {result_summary['solve_rate']:.3f} ± {result_summary['solve_rate_std']:.3f}")
            print(f"      Search Efficiency: {result_summary['search_efficiency']:.4f}")
    
    total_time = time.time() - start_time
    
    # Analyze results
    print(f"\n{'='*70}")
    print("ANALYZING RESULTS...")
    print(f"{'='*70}")
    
    analysis = analyze_t_scaling(results)
    
    # Create output
    output = {
        'experiment': 'T-scaling',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'T_values': T_values,
            'model': model,
            'num_problems': num_problems,
            'difficulty': difficulty,
            'replications': replications,
            'beam_width': beam_width,
            'branching_factor': branching_factor
        },
        'results': results,
        'analysis': analysis,
        'execution_time_seconds': total_time
    }
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    output_file = output_path / "exp1_2_t_scaling_results.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)
        
    # Save traces separately
    trace_file = output_path / "exp1_2_traces.json"
    with open(trace_file, 'w') as f:
        json.dump(all_traces, f, indent=2, default=str)
    
    print(f"\n✓ Results saved to {output_file}")
    print(f"✓ Traces saved to {trace_file}")
    print(f"✓ Total execution time: {total_time/60:.1f} minutes")
    
    return output


def analyze_t_scaling(results: List[Dict]) -> Dict:
    """
    Analyze T-scaling results for power-law fit.
    
    Fits: solve_rate = a × T^β_T
    """
    T_values = np.array([r['T'] for r in results])
    solve_rates = np.array([r['solve_rate'] for r in results])
    search_efficiencies = np.array([r['search_efficiency'] for r in results])
    
    # Fit power law: solve_rate ∝ T^β_T
    # We need to analyze per strategy now
    
    strategies = sorted(list(set(r['strategy'] for r in results)))
    analysis = {}
    
    for strategy in strategies:
        strat_results = [r for r in results if r['strategy'] == strategy]
        
        T_vals = np.array([r['T'] for r in strat_results])
        s_rates = np.array([r['solve_rate'] for r in strat_results])
        
        a_solve, beta_T_solve, r_squared_solve = fit_power_law(T_vals, s_rates)
        
        analysis[strategy] = {
            'solve_rate_power_law': {
                'coefficient': float(a_solve),
                'exponent': float(beta_T_solve),
                'r_squared': float(r_squared_solve),
                'formula': f"solve_rate = {a_solve:.4f} × T^{beta_T_solve:.4f}"
            }
        }
        
    return analysis
    
    # Fit power law for search efficiency
    a_eff, beta_T_eff, r_squared_eff = fit_power_law(T_values, search_efficiencies)
    
    # Find critical point (where solve_rate crosses 0.5)
    critical_T = find_critical_exponent(T_values, solve_rates, threshold=0.5)
    
    analysis = {
        'solve_rate_power_law': {
            'coefficient': float(a_solve),
            'exponent': float(beta_T_solve),
            'r_squared': float(r_squared_solve),
            'formula': f"solve_rate = {a_solve:.4f} × T^{beta_T_solve:.4f}"
        },
        'search_efficiency_power_law': {
            'coefficient': float(a_eff),
            'exponent': float(beta_T_eff),
            'r_squared': float(r_squared_eff),
            'formula': f"efficiency = {a_eff:.4f} × T^{beta_T_eff:.4f}"
        },
        'critical_T': float(critical_T) if critical_T is not None else None,
        'interpretation': {
            'solve_rate_scaling': 'positive' if beta_T_solve > 0 else 'negative',
            'efficiency_scaling': 'positive' if beta_T_eff > 0 else 'negative',
            'power_law_valid': r_squared_solve > 0.8
        }
    }
    
    print(f"\nPower Law Analysis:")
    print(f"  Solve Rate: {analysis['solve_rate_power_law']['formula']}")
    print(f"    R² = {analysis['solve_rate_power_law']['r_squared']:.3f}")
    print(f"  Search Efficiency: {analysis['search_efficiency_power_law']['formula']}")
    print(f"    R² = {analysis['search_efficiency_power_law']['r_squared']:.3f}")
    if critical_T:
        print(f"  Critical T: {critical_T:.2f}")
    
    return analysis


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """Run T-scaling experiment."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run T-scaling experiment')
    parser.add_argument('--T-values', type=int, nargs='+', default=[1, 3, 5, 10, 20],
                       help='Search depths to test')
    parser.add_argument('--model', type=str, default='gpt-4o-mini',
                       help='LLM model to use')
    parser.add_argument('--num-problems', type=int, default=10,
                       help='Number of problems per T value')
    parser.add_argument('--difficulty', type=int, default=3,
                       help='Problem difficulty (1-5)')
    parser.add_argument('--replications', type=int, default=2,
                       help='Number of replications')
    parser.add_argument('--beam-width', type=int, default=3,
                       help='Beam search width')
    parser.add_argument('--branching', type=int, default=3,
                       help='Branching factor')
    parser.add_argument('--pilot', action='store_true',
                       help='Run pilot (T=[1,3,5], 5 problems, 1 rep)')
    
    args = parser.parse_args()
    
    if args.pilot:
        print("Running PILOT experiment...")
        T_values = [1, 3, 5]
        num_problems = 5
        replications = 1
    else:
        T_values = args.T_values
        num_problems = args.num_problems
        replications = args.replications
    
    results = run_t_scaling_experiment(
        T_values=T_values,
        model=args.model,
        num_problems=num_problems,
        difficulty=args.difficulty,
        replications=replications,
        beam_width=args.beam_width,
        branching_factor=args.branching
    )
    
    print(f"\n{'='*70}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: results/agentic_scaling/exp1_2_t_scaling_results.json")
    print(f"Next steps:")
    print(f"  1. Review results")
    print(f"  2. Generate visualizations")
    print(f"  3. Proceed to Experiment 1.3 (K-Scaling)")


if __name__ == "__main__":
    main()
