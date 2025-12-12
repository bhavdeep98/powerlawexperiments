"""
Experiment 4: Compute-Optimal Agent Architecture
================================================
Compares different architectures under a fixed compute budget.

Architectures:
1. Single Strong: 1x gpt-4o (simulated by high-reliability 4o-mini here to save cost, or actual 4o if key allows)
2. Ensemble Weak: 5x gpt-4o-mini (Voting)
3. Deep Search: 1x gpt-4o-mini with T=20 (Tree of Thought)

Budget Assumption: 
- 1x Strong ≈ 10x Weak
- 1x Deep Search (T=20) ≈ 20x Weak (linear in T)
- 1x Ensemble (N=5) ≈ 5x Weak

We will correct for this by analyzing Efficiency (Accuracy / Cost).
"""

import json
import time
import argparse
from pathlib import Path
from agents.multi_agent_system import MultiAgentSystem
from agents.benchmarks import GameOf24Benchmark
from agents.tree_of_thought_enhanced import EnhancedGameOf24, BFSSearcher

def run_compute_optimal_experiment(
    num_problems: int = 5,
    output_dir: str = "results/compute_optimal"
):
    print(f"EXPERIMENT 4: COMPUTE OPTIMAL ARCHITECTURE")
    print("==========================================")
    
    benchmark = GameOf24Benchmark()
    problems = benchmark.get_problems(difficulty=3, num_problems=num_problems)
    
    results = []
    
    for i, problem in enumerate(problems):
        print(f"\nProblem {i+1}: {problem.problem_id}")
        nums = problem.metadata['numbers']
        problem_str = f"Use {nums} to make 24."
        
        # 1. Ensemble Weak (N=5, 4o-mini)
        print("  Running Ensemble (N=5)...")
        start = time.time()
        mas = MultiAgentSystem(num_agents=5, model="gpt-4o-mini")
        ens_res = mas.run_debate(f"Use {nums} to get 24", rounds=2, bandwidth=4) # FC
        ens_time = time.time() - start
        ens_success = 1.0 if ens_res['consensus_ratio'] > 0.5 and "24" in str(ens_res['consensus_answer']) else 0.0
        
        # 2. Deep Search (T=20, 4o-mini)
        print("  Running Deep Search (T=20)...")
        start = time.time()
        input_str = " ".join(map(str, nums))
        # Correctly instantiate with numbers
        solver = EnhancedGameOf24(numbers=input_str)
        # Call solve_tot directly, default to BFS
        search_res = solver.solve_tot(max_depth=20)
        search_time = time.time() - start
        search_success = 1.0 if search_res['success'] else 0.0
        
        # 3. Single Strong (1x gpt-4o)
        # ONLY IF KEY SUPPORTS IT. Fallback to 4o-mini but treat as baseline.
        print("  Running Single Strong (Simulated)...")
        # simulation: standard react agent or just 1-agent MAS
        start = time.time()
        strong_mas = MultiAgentSystem(num_agents=1, model="gpt-4o") # Try 4o
        strong_res = strong_mas.run_debate(f"Use {nums} to get 24", rounds=1)
        strong_time = time.time() - start
        strong_success = 1.0 if strong_res['individual_answers'][0] and "24" in strong_res['individual_answers'][0] else 0.0
        
        results.append({
            "problem": problem.problem_id,
            "ensemble": {"success": ens_success, "time": ens_time, "cost_units": 5 * 2}, # N*Rounds
            "search": {"success": search_success, "time": search_time, "cost_units": 20}, # T
            "strong": {"success": strong_success, "time": strong_time, "cost_units": 10} # Assumed 10x cost
        })

    # Summary
    print("\nSummary Results:")
    ens_rate = sum(r['ensemble']['success'] for r in results) / num_problems
    search_rate = sum(r['search']['success'] for r in results) / num_problems
    strong_rate = sum(r['strong']['success'] for r in results) / num_problems
    
    print(f"Ensemble (N=5): {ens_rate:.0%}")
    print(f"Deep Search (T=20): {search_rate:.0%}")
    print(f"Strong (1x): {strong_rate:.0%}")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "exp4_results.json", 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-problems", type=int, default=5)
    args = parser.parse_args()
    
    run_compute_optimal_experiment(num_problems=args.num_problems)
