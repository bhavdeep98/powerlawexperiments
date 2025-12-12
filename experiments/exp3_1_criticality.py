"""
Experiment 3.1: Criticality in Multi-Agent Coordination (Bandwidth)
===================================================================
Tests the hypothesis that reliable consensus emerges only above a 
specific connectivity threshold.

Variables:
- Bandwidth (k): 0 to N-1
- N: Number of agents (3, 5)

Metric: Consensus Ratio (max fraction of agents agreeing on same answer)
"""

import os
import json
import time
import argparse
import numpy as np
from pathlib import Path
from agents.multi_agent_system import MultiAgentSystem
from agents.benchmarks import GameOf24Benchmark

def run_criticality_experiment(
    num_agents: int = 3,
    bandwidth_values: list = [0, 1, 2],
    num_problems: int = 5,
    rounds: int = 2,
    output_dir: str = "results/criticality"
):
    print(f"EXPERIMENT 3.1: MULTI-AGENT CRITICALITY")
    print("=======================================")
    print(f"Agents (N): {num_agents}")
    print(f"Bandwidth (k): {bandwidth_values}")
    
    benchmark = GameOf24Benchmark()
    problems = benchmark.get_problems(difficulty=3, num_problems=num_problems)
    
    results = []
    
    for k in bandwidth_values:
        print(f"\nConfiguration: N={num_agents}, Bandwidth={k}")
        
        consensus_ratios = []
        
        for i, problem in enumerate(problems):
            print(f"  Problem {i+1}: {problem.problem_id}", end=" ... ")
            
            # Formulate prompt
            nums = problem.metadata['numbers']
            prompt = f"Use numbers {nums} and basic arithmetic (+ - * /) to obtain 24. Return just the equation."
            
            mas = MultiAgentSystem(num_agents=num_agents)
            
            start_time = time.time()
            try:
                run_data = mas.run_debate(prompt, rounds=rounds, bandwidth=k)
                duration = time.time() - start_time
                
                ratio = run_data['consensus_ratio']
                answer = run_data['consensus_answer']
                
                print(f"Consensus: {ratio:.2f} on '{answer}' ({duration:.1f}s)")
                
                consensus_ratios.append(ratio)
                
                results.append({
                    'N': num_agents,
                    'bandwidth': k,
                    'problem_id': problem.problem_id,
                    'consensus_ratio': ratio,
                    'consensus_answer': answer,
                    'duration': duration,
                    'individual_answers': run_data['individual_answers']
                })
                
            except Exception as e:
                print(f"ERROR: {e}")
                consensus_ratios.append(0.0)
        
        avg_consensus = np.mean(consensus_ratios) if consensus_ratios else 0.0
        print(f"  Summary k={k}: Avg Consensus = {avg_consensus:.2%}")

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / f"exp3_1_criticality_N{num_agents}.json", 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"\nSaved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-problems", type=int, default=5)
    parser.add_argument("--num-agents", type=int, default=5)
    parser.add_argument("--bandwidths", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = parser.parse_args()
    
    run_criticality_experiment(
        num_problems=args.num_problems,
        num_agents=args.num_agents,
        bandwidth_values=args.bandwidths
    )
