
import os
import sys
from system2_criticality_experiment import System2CriticalityExperiment, ExperimentConfig
from benchmarks import GameOf24Benchmark
from tree_of_thought_enhanced import SearchStrategy

# Configuration
API_KEY = os.environ.get("OPENAI_API_KEY")
if not API_KEY:
    print("OPENAI_API_KEY not found")
    sys.exit(1)

def run_debug_trace():
    print("DEBUGGING EXPERIMENT 1.2 (T-Scaling) - SINGLE TRACE")
    print("===================================================")
    
    # 1. Setup
    model = "gpt-4o-mini"
    depth = 5
    
    # Get a problem
    benchmark = GameOf24Benchmark()
    problems = benchmark.get_problems(difficulty=3, num_problems=1)
    problem = problems[0]
    task = problem.metadata['numbers']
    
    print(f"Problem: {task}")
    print(f"Model: {model}")
    print(f"Depth: {depth}")
    
    # 2. Configure Experiment
    config = ExperimentConfig(
        models=[model],
        search_depths=[depth],
        beam_widths=[3],
        num_samples=[1],
        branching_factors=[3]
    )
    
    experiment = System2CriticalityExperiment(config)
    
    # 3. Run Single Config
    print("\nRunning search...")
    result = experiment.evaluate_single_config(
        model=model,
        task=task,
        search_depth=depth,
        beam_width=3,
        branching_factor=3,
        strategy=SearchStrategy.BEAM
    )
    
    # 4. Print Trace
    print("\nRESULT TRACE:")
    print("-------------")
    
    solution = result.get('solution', 'No solution found')
    print(f"\nFinal Solution:\n{solution}")
    
    metrics = result.get('metrics', {})
    print("\nMetrics:")
    print(json.dumps(metrics, indent=2))
    
    # If the system object has a trace or history, print it.
    # Looking at System2CriticalityExperiment, it might not return the full tree.
    # But evaluating `evaluate_single_config` usually calls `run_tree_of_thought` or similar.
    # Let's inspect what keys are in result.
    print("\nResult Keys:", result.keys())

    # We might need to inspect the `tree` object if returned.
    if 'tree' in result:
        print("\nTree Visualization (Partial):")
        # Custom simple visualizer if not present
        pass

if __name__ == "__main__":
    import json
    run_debug_trace()
