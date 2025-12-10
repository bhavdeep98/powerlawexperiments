"""
Comprehensive System 2 Experiment Runner
=========================================
Runs all System 2 experiments with the expanded benchmark suite.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List

# Import benchmarks
from benchmarks import (
    GameOf24Benchmark,
    ArithmeticChainBenchmark,
    TowerOfHanoiBenchmark,
    VariableTrackingBenchmark,
    LogicPuzzleBenchmark
)

# Import experiment components
from tree_of_thought_enhanced import EnhancedGameOf24, SearchStrategy
from system2_criticality_experiment import System2CriticalityExperiment, ExperimentConfig
from advanced_system2_architectures import (
    VerifierReasonerSystem, DebateSystem, MemoryAugmentedReasoner
)
from state_tracking_benchmarks import run_state_tracking_experiments

# Configuration
API_KEY = os.environ.get("OPENAI_API_KEY")
if not API_KEY:
    print("⚠ OPENAI_API_KEY not found. Please export it.")
    exit(1)

# Create results directory
results_dir = Path("results/system2")
results_dir.mkdir(parents=True, exist_ok=True)


def run_baseline_evaluation():
    """Run baseline evaluation on all benchmarks."""
    print(f"\n{'='*70}")
    print("BASELINE EVALUATION")
    print(f"{'='*70}")
    
    benchmarks = {
        'game_of_24': GameOf24Benchmark(),
        'arithmetic_chains': ArithmeticChainBenchmark(),
        'tower_of_hanoi': TowerOfHanoiBenchmark(),
        'variable_tracking': VariableTrackingBenchmark(),
        'logic_puzzles': LogicPuzzleBenchmark()
    }
    
    results = {}
    model = "gpt-4o"
    
    for bench_name, benchmark in benchmarks.items():
        print(f"\nBenchmark: {bench_name}")
        print("-" * 70)
        
        # Get problems (sample for baseline)
        problems = benchmark.get_problems(difficulty=None, num_problems=5)
        
        bench_results = {
            'benchmark': bench_name,
            'model': model,
            'num_problems': len(problems),
            'results': []
        }
        
        for problem in problems:
            print(f"  Problem: {problem.problem_id}")
            
            # Simple zero-shot evaluation
            from openai import OpenAI
            client = OpenAI(api_key=API_KEY)
            
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": problem.problem_text}],
                    temperature=0.7
                )
                solution = response.choices[0].message.content
                
                # Evaluate
                is_correct, partial_credit = benchmark.evaluate_solution(problem, solution)
                
                bench_results['results'].append({
                    'problem_id': problem.problem_id,
                    'difficulty': problem.difficulty,
                    'solution': solution[:200],  # Truncate
                    'is_correct': is_correct,
                    'partial_credit': partial_credit
                })
                
                print(f"    Correct: {is_correct}, Credit: {partial_credit:.2f}")
                
            except Exception as e:
                print(f"    Error: {str(e)}")
                bench_results['results'].append({
                    'problem_id': problem.problem_id,
                    'error': str(e)
                })
        
        # Calculate summary
        correct = sum(1 for r in bench_results['results'] if r.get('is_correct', False))
        total = len(bench_results['results'])
        bench_results['accuracy'] = correct / total if total > 0 else 0.0
        
        results[bench_name] = bench_results
        print(f"\n  Accuracy: {bench_results['accuracy']:.3f} ({correct}/{total})")
    
    # Save results
    output_file = results_dir / "baseline_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✓ Baseline results saved to {output_file}")
    return results


def run_scaling_experiments():
    """Run comprehensive scaling experiments."""
    print(f"\n{'='*70}")
    print("SCALING EXPERIMENTS")
    print(f"{'='*70}")
    
    # Use Game of 24 for scaling (most established)
    benchmark = GameOf24Benchmark()
    problems = benchmark.get_problems(difficulty=3, num_problems=10)  # Hard problems
    
    # Convert to format expected by System2CriticalityExperiment
    tasks = [p.metadata['numbers'] for p in problems if 'numbers' in p.metadata]
    
    # Create experiment config (smaller for initial run)
    config = ExperimentConfig(
        models=['gpt-4o-mini', 'gpt-4o'],  # Start with 2 models
        search_depths=[1, 3, 5, 10],
        beam_widths=[1, 3],      # REMOVED 5 for speed
        num_samples=[1],         # REMOVED 5 for speed (single pass)
        branching_factors=[3]
    )
    
    experiment = System2CriticalityExperiment(config)
    
    # Run experiment
    results = experiment.run_scaling_experiment(tasks)
    
    # Save results
    output = experiment.save_results(str(results_dir / "scaling_results.json"))
    
    return output


def run_architecture_comparison():
    """Compare different System 2 architectures."""
    print(f"\n{'='*70}")
    print("ARCHITECTURE COMPARISON")
    print(f"{'='*70}")
    
    # Use Game of 24 problems
    benchmark = GameOf24Benchmark()
    problems = benchmark.get_problems(difficulty=3, num_problems=5)
    
    # Convert to problem format
    problem_texts = [
        f"Use numbers {p.metadata['numbers']} and arithmetic operations to get 24"
        for p in problems if 'numbers' in p.metadata
    ]
    
    results = {
        'verify_refine': [],
        'debate': [],
        'memory_augmented': []
    }
    
    # Test Verify-Refine
    print("\nTesting Verify-Refine System...")
    verifier = VerifierReasonerSystem(max_iterations=3)
    for i, problem in enumerate(problem_texts):
        print(f"  Problem {i+1}/{len(problem_texts)}")
        result = verifier.solve(problem)
        results['verify_refine'].append(result)
        print(f"    Success: {result['success']}, Iterations: {result['num_iterations']}")
    
    # Test Debate
    print("\nTesting Debate System...")
    debate = DebateSystem(debate_rounds=2)
    for i, problem in enumerate(problem_texts):
        print(f"  Problem {i+1}/{len(problem_texts)}")
        result = debate.solve(problem)
        results['debate'].append(result)
        print(f"    Success: {result['success']}, Selected Agent: {result['selected_agent']}")
    
    # Test Memory-Augmented
    print("\nTesting Memory-Augmented System...")
    memory = MemoryAugmentedReasoner()
    for i, problem in enumerate(problem_texts):
        print(f"  Problem {i+1}/{len(problem_texts)}")
        result = memory.solve(problem, max_steps=5)
        results['memory_augmented'].append(result)
        print(f"    Success: {result['success']}, Steps: {len(result['steps'])}")
    
    # Save results
    output_file = results_dir / "architecture_comparison.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✓ Architecture comparison saved to {output_file}")
    return results


def run_state_tracking_experiments_wrapper():
    """Run state tracking experiments."""
    print(f"\n{'='*70}")
    print("STATE TRACKING EXPERIMENTS")
    print(f"{'='*70}")
    
    # Run with smaller step counts for initial run
    results = run_state_tracking_experiments(
        model="gpt-4o",
        num_steps_list=[5, 10, 15, 20]
    )
    
    # Save results
    output_file = results_dir / "state_tracking_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✓ State tracking results saved to {output_file}")
    return results


def generate_summary_report():
    """Generate summary report of all experiments."""
    print(f"\n{'='*70}")
    print("GENERATING SUMMARY REPORT")
    print(f"{'='*70}")
    
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'experiments_run': [],
        'summary': {}
    }
    
    # Check what results exist
    result_files = {
        'baseline': results_dir / "baseline_results.json",
        'scaling': results_dir / "scaling_results.json",
        'architecture': results_dir / "architecture_comparison.json",
        'state_tracking': results_dir / "state_tracking_results.json"
    }
    
    for exp_name, file_path in result_files.items():
        if file_path.exists():
            report['experiments_run'].append(exp_name)
            with open(file_path, 'r') as f:
                data = json.load(f)
                report['summary'][exp_name] = {
                    'file': str(file_path),
                    'status': 'completed'
                }
    
    # Save report
    report_file = results_dir / "experiment_summary.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n✓ Summary report saved to {report_file}")
    print(f"\nExperiments completed: {len(report['experiments_run'])}")
    for exp in report['experiments_run']:
        print(f"  - {exp}")
    
    return report


def main():
    """Run all experiments."""
    print(f"\n{'='*70}")
    print("SYSTEM 2 COMPREHENSIVE EXPERIMENTS")
    print(f"{'='*70}")
    print("\nThis will run:")
    print("  1. Baseline evaluation on all benchmarks")
    print("  2. Scaling experiments (Game of 24)")
    print("  3. Architecture comparison")
    print("  4. State tracking experiments")
    print("\nNote: This may take a while and use API credits.")
    
    start_time = time.time()
    
    try:
        # 1. Baseline
        print("\n" + "="*70)
        print("STEP 1: BASELINE EVALUATION")
        print("="*70)
        baseline_results = run_baseline_evaluation()
        
        # 2. Scaling
        print("\n" + "="*70)
        print("STEP 2: SCALING EXPERIMENTS")
        print("="*70)
        scaling_results = run_scaling_experiments()
        
        # 3. Architecture comparison
        print("\n" + "="*70)
        print("STEP 3: ARCHITECTURE COMPARISON")
        print("="*70)
        arch_results = run_architecture_comparison()
        
        # 4. State tracking
        print("\n" + "="*70)
        print("STEP 4: STATE TRACKING EXPERIMENTS")
        print("="*70)
        state_results = run_state_tracking_experiments_wrapper()
        
        # 5. Generate summary
        summary = generate_summary_report()
        
        total_time = time.time() - start_time
        
        print(f"\n{'='*70}")
        print("ALL EXPERIMENTS COMPLETED")
        print(f"{'='*70}")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Results saved to: {results_dir}")
        print("\nNext steps:")
        print("  1. Review results in results/system2/")
        print("  2. Run power law analysis: python system2_power_law_analysis.py")
        print("  3. Generate visualizations")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Experiments interrupted by user.")
        print("Partial results may be available in results/system2/")
    except Exception as e:
        print(f"\n\n❌ Error during experiments: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
