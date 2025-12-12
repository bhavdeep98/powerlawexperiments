import os
import argparse
import time
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from agents.tool_augmented_agent import ToolAugmentedAgent, BASIC_TOOLS, VALIDATOR_TOOLS
from agents.benchmarks import GameOf24Benchmark

def run_k_scaling_experiment(
    models: list = ["gpt-4o-mini"],
    num_problems: int = 10,
    replications: int = 2,
    output_dir: str = "results/agentic_scaling"
):
    print(f"EXPERIMENT 1.3: K-SCALING (Tool Richness)")
    print("=========================================")
    
    benchmark = GameOf24Benchmark()
    problems = benchmark.get_problems(difficulty=3, num_problems=num_problems)
    
    # Define K levels
    k_configs = {
        0: [],  # No tools (Zero-shot Chain-of-Thought)
        1: BASIC_TOOLS, # Calculator
        2: VALIDATOR_TOOLS # Calculator + Validator
    }
    
    results = []
    
    for model in models:
        print(f"\nModel: {model}")
        
        for k, tools in k_configs.items():
            print(f"\n  K={k} (Tools: {[t.name for t in tools]})")
            
            solve_rates = []
            
            for rep in range(replications):
                print(f"    Replication {rep+1}/{replications}:")
                
                for i, problem in enumerate(problems):
                    print(f"      Problem {i+1}: {problem.problem_id}", end=" ... ")
                    
                    agent = ToolAugmentedAgent(
                        model=model,
                        tools=tools,
                        max_steps=10
                    )
                    
                    # Prompt needs to include the numbers
                    question = f"Use the numbers {problem.metadata['numbers']} to make 24 using +, -, *, /. You must use each number exactly once."
                    
                    start_time = time.time()
                    result = agent.solve(question)
                    duration = time.time() - start_time
                    
                    # Verify
                    is_correct = False
                    if result['success'] and result['final_answer']:
                         is_correct, _ = benchmark.evaluate_solution(problem, result['final_answer'])
                    
                    status = "✓" if is_correct else "✗"
                    print(f"{status} ({duration:.1f}s)")
                    
                    solve_rates.append(1.0 if is_correct else 0.0)
                    
                    results.append({
                        'K': k,
                        'model': model,
                        'problem_id': problem.problem_id,
                        'success': is_correct,
                        'steps': result.get('steps', 0),
                        'trace': result.get('trace', []),
                        'time': duration
                    })
            
            avg_rate = np.mean(solve_rates)
            print(f"    Summary K={k}: Solve Rate = {avg_rate:.2%}")

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / "exp1_3_k_scaling_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
        
    print(f"\nSaved results to {output_path}/exp1_3_k_scaling_results.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-problems", type=int, default=5)
    parser.add_argument("--replications", type=int, default=1)
    args = parser.parse_args()
    
    run_k_scaling_experiment(num_problems=args.num_problems, replications=args.replications)
