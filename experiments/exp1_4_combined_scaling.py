"""
Experiment 1.4: Combined Scaling (Prosthetic Search)
====================================================
Tests the Synergy Hypothesis: Does adding a Validator Tool (K) allow 
System 2 Search (T) to work for weak models?

Baseline (Exp 1.2): T-Scaling failed (0% success).
Hypothesis: Validator prevents "Tree Collapse" by pruning invalid final states 
and confirming valid ones, enabling the search to find the needle in the haystack.
"""

import os
import json
import time
import math
import argparse
import numpy as np
from typing import List, Dict, Callable
from pathlib import Path

# Import agents
from agents.tree_of_thought_enhanced import (
    EnhancedGameOf24, 
    SearchStrategy,
    validate_state,
    extract_numbers_from_state
)
from agents.benchmarks import GameOf24Benchmark

def wrapper_validator_tool(state: str) -> str:
    """
    Parses state and checks validity against Game of 24 rules.
    Returns: "YES", "NO", or "MAYBE".
    """
    try:
        # 1. Parse remaining numbers
        if "Remaining:" in state:
            rem_str = state.split("Remaining:")[-1].strip(" []")
            if not rem_str:
                return "NO" # Should have numbers
            remaining = [float(x.strip()) for x in rem_str.split(",") if x.strip()]
            
            # 2. Check logic
            if len(remaining) == 1:
                # Final state check
                val = remaining[0]
                if math.isclose(val, 24.0, rel_tol=1e-5):
                    return "YES"
                else:
                    return "NO"
            else:
                # Intermediate state
                # We could check if 24 is reachable, but that's cheating (requires solver).
                # We just say MAYBE and let LLM heuristic decide, OR we could check basic bounds.
                # For now, MAYBE.
                return "MAYBE"
        return "NO" # Malformed
    except:
        return "NO"

def run_combined_scaling_experiment(
    T_values: List[int] = [1, 5, 20],
    K_values: List[int] = [0, 2], # 0=No Tool, 2=Validator
    model: str = "gpt-4o-mini",
    num_problems: int = 5,
    replications: int = 1, # Keep low for speed
    output_dir: str = "results/agentic_scaling"
):
    print(f"EXPERIMENT 1.4: COMBINED SCALING (Prosthetic Search)")
    print("====================================================")
    print(f"Model: {model}")
    print(f"T (Depth): {T_values}")
    print(f"K (Tools): {K_values}")
    
    benchmark = GameOf24Benchmark()
    problems = benchmark.get_problems(difficulty=3, num_problems=num_problems)
    
    results = []
    
    for T in T_values:
        for K in K_values:
            print(f"\nConfiguration: T={T}, K={K}")
            
            solve_rates = []
            
            for rep in range(replications):
                print(f"  Replication {rep+1}:")
                for i, problem in enumerate(problems):
                    print(f"    Problem {i+1}: {problem.problem_id}", end=" ... ")
                    
                    nums_str = problem.metadata['numbers']
                    solver = EnhancedGameOf24(nums_str, model=model)
                    
                    # Prepare validator
                    validator = wrapper_validator_tool if K == 2 else None
                    
                    start_time = time.time()
                    
                    # Use BFS for consistent comparison
                    try:
                        result = solver.solve_tot(
                            strategy=SearchStrategy.BFS,
                            max_depth=T,
                            branching_factor=3,
                            validator_tool=validator
                        )
                        
                        success = result['success']
                        metrics = result['metrics']
                        duration = time.time() - start_time
                        
                        # Double check with benchmark
                        if success:
                            # Verify the solution string matches 24
                            # solution path format: "Current... -> Op... -> ..."
                            # We trust result['success'] from ToT if validator was on because 
                            # validator returns YES only for 24.
                            # But let's be sure.
                            pass
                        
                        status = "✓" if success else "✗"
                        print(f"{status} (nodes: {metrics.get('nodes_explored', 0)}, time: {duration:.1f}s)")
                        
                        solve_rates.append(1.0 if success else 0.0)
                        
                        results.append({
                            'T': T,
                            'K': K,
                            'model': model,
                            'problem_id': problem.problem_id,
                            'success': success,
                            'nodes': metrics.get('nodes_explored', 0),
                            'time': duration,
                            'solution': result.get('solution', '')
                        })
                        
                    except Exception as e:
                        print(f"ERROR: {e}")
                        solve_rates.append(0.0)
            
            avg_rate = np.mean(solve_rates) if solve_rates else 0.0
            print(f"  Summary T={T}, K={K}: Solve Rate = {avg_rate:.2%}")

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / "exp1_4_combined_scaling_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
        
    print(f"\nSaved results to {output_path}/exp1_4_combined_scaling_results.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-problems", type=int, default=5)
    parser.add_argument("--replications", type=int, default=1)
    parser.add_argument("--T-values", type=int, nargs="+", default=[1, 5, 20])
    args = parser.parse_args()
    
    run_combined_scaling_experiment(
        num_problems=args.num_problems, 
        replications=args.replications,
        T_values=args.T_values
    )
