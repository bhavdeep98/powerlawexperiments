"""
Experiment 3.2: Criticality in Multi-Agent Coordination (Topology)
==================================================================
Tests how different network topologies affect consensus emergence.

Variables:
- Topology: {random, ring, star, fully_connected}
- N: Number of agents (5)
"""

import json
import time
import argparse
import numpy as np
from pathlib import Path
from agents.multi_agent_system import MultiAgentSystem
from agents.benchmarks import GameOf24Benchmark

def run_topology_experiment(
    num_agents: int = 5,
    topologies: list = ["random", "ring", "star", "fully_connected"],
    num_problems: int = 5,
    rounds: int = 3,
    output_dir: str = "results/criticality"
):
    print(f"EXPERIMENT 3.2: MULTI-AGENT TOPOLOGY")
    print("====================================")
    
    benchmark = GameOf24Benchmark()
    problems = benchmark.get_problems(difficulty=3, num_problems=num_problems)
    
    results = []
    
    for topo in topologies:
        print(f"\nConfiguration: N={num_agents}, Topology={topo}")
        
        consensus_ratios = []
        
        for i, problem in enumerate(problems):
            print(f"  Problem {i+1}: {problem.problem_id}", end=" ... ")
            
            nums = problem.metadata['numbers']
            prompt = f"Use numbers {nums} and basic arithmetic (+ - * /) to obtain 24. Return just the equation."
            
            mas = MultiAgentSystem(num_agents=num_agents)
            
            start_time = time.time()
            try:
                # Bandwidth=2 is default for random, others ignore it
                run_data = mas.run_debate(prompt, rounds=rounds, bandwidth=2, topology=topo)
                duration = time.time() - start_time
                
                ratio = run_data['consensus_ratio']
                answer = run_data['consensus_answer']
                
                print(f"Consensus: {ratio:.2f} on '{answer}' ({duration:.1f}s)")
                consensus_ratios.append(ratio)
                
                results.append({
                    'N': num_agents,
                    'topology': topo,
                    'problem_id': problem.problem_id,
                    'consensus_ratio': ratio,
                    'consensus_answer': answer,
                    'duration': duration
                })
            except Exception as e:
                print(f"ERROR: {e}")
                consensus_ratios.append(0.0)
        
        avg_consensus = np.mean(consensus_ratios) if consensus_ratios else 0.0
        print(f"  Summary {topo}: Avg Consensus = {avg_consensus:.2%}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / f"exp3_2_topology_N{num_agents}.json", 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-problems", type=int, default=5)
    parser.add_argument("--num-agents", type=int, default=5)
    parser.add_argument("--topologies", type=str, nargs="+", default=["random", "ring", "star", "fully_connected"])
    args = parser.parse_args()
    
    run_topology_experiment(
        num_problems=args.num_problems,
        num_agents=args.num_agents,
        topologies=args.topologies
    )
