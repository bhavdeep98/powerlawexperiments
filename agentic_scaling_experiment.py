"""
Agentic Scaling Laws Experiment
===============================
Experiment 1.1: A-Scaling (Number of Agents)

Tests the hypothesis that agentic system performance follows a power-law
relationship with the number of agents: E(A) ~ A^(-β_A)

This is the first experiment in the unified framework for scaling laws and
critical phenomena in agentic systems.
"""

import os
import json
import time
import numpy as np
import re
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from collections import Counter

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv not installed, will use environment variables
    pass

# Import existing infrastructure
from advanced_system2_architectures import DebateAgent, DEFAULT_MODEL
from benchmarks import GameOf24Benchmark
from system2_power_law_analysis import fit_power_law, find_critical_exponent

# Configuration - Load API key from .env file or environment variable
API_KEY = os.environ.get("OPENAI_API_KEY")
if not API_KEY:
    print("⚠ OPENAI_API_KEY not found.")
    print("   Please either:")
    print("   1. Create a .env file with: OPENAI_API_KEY=your_key_here")
    print("   2. Or export it: export OPENAI_API_KEY=your_key_here")
    print("\n   You can copy .env.example to .env and add your key there.")
    exit(1)

from openai import OpenAI
client = OpenAI(api_key=API_KEY)


# ==============================================================================
# MULTI-AGENT SYSTEM (N Agents)
# ==============================================================================

class MultiAgentSystem:
    """
    Multi-agent system that extends DebateSystem to support N agents.
    
    This enables A-scaling experiments by varying the number of agents
    and measuring coordination and performance.
    """
    
    def __init__(self, n_agents: int, model: str = DEFAULT_MODEL):
        """
        Initialize multi-agent system.
        
        Args:
            n_agents: Number of agents (A in scaling law)
            model: LLM model to use for all agents
        """
        self.agents = [DebateAgent(f"Agent_{i+1}", model) 
                      for i in range(n_agents)]
        self.n_agents = n_agents
        self.model = model
    
    def solve_with_coordination(self, problem: str) -> Dict:
        """
        Solve problem with N agents and measure coordination.
        
        Args:
            problem: Problem text to solve
            
        Returns:
            Dictionary with solution, coordination metrics, and trace
        """
        start_time = time.time()
        
        # Get solutions from all agents (parallel, independent)
        solutions = []
        for agent in self.agents:
            solution = agent.solve(problem)
            solutions.append(solution)
        
        # Measure coordination (agreement among agents)
        coordination_accuracy = self._measure_agreement(solutions)
        
        # Reach consensus
        consensus_solution = self._reach_consensus(solutions, problem)
        
        elapsed_time = time.time() - start_time
        
        # Create trace for workflow extraction (Experiment 2)
        trace = self._create_trace(solutions, coordination_accuracy, elapsed_time)
        
        return {
            'solution': consensus_solution,
            'coordination_accuracy': coordination_accuracy,
            'n_agents': self.n_agents,
            'solutions': solutions,
            'consensus_time': elapsed_time,
            'trace': trace
        }
    
    def _measure_agreement(self, solutions: List[str]) -> float:
        """
        Measure how much agents agree (coordination accuracy).
        
        Returns value between 0 (no agreement) and 1 (perfect agreement).
        """
        if len(solutions) < 2:
            return 1.0  # Single agent always agrees with itself
        
        # Count pairs that agree
        agreements = 0
        total_pairs = 0
        
        for i in range(len(solutions)):
            for j in range(i+1, len(solutions)):
                total_pairs += 1
                if self._solutions_similar(solutions[i], solutions[j]):
                    agreements += 1
        
        return agreements / total_pairs if total_pairs > 0 else 0.0
    
    def _solutions_similar(self, sol1: str, sol2: str) -> bool:
        """
        Check if two solutions are similar.
        
        Uses multiple heuristics:
        1. Extract final answer numbers
        2. Check for common keywords
        3. Word overlap
        """
        # Extract numbers (especially final answers like "= 24")
        nums1 = set(re.findall(r'\d+', sol1))
        nums2 = set(re.findall(r'\d+', sol2))
        
        # If both have numbers, check overlap
        if len(nums1) > 0 and len(nums2) > 0:
            overlap = len(nums1 & nums2) / max(len(nums1 | nums2), 1)
            if overlap > 0.6:  # 60% number overlap
                return True
        
        # Fallback: word overlap for non-numeric solutions
        words1 = set(sol1.lower().split())
        words2 = set(sol2.lower().split())
        if len(words1) > 0 and len(words2) > 0:
            overlap = len(words1 & words2) / max(len(words1 | words2), 1)
            if overlap > 0.5:  # 50% word overlap
                return True
        
        return False
    
    def _reach_consensus(self, solutions: List[str], problem: str) -> str:
        """
        Reach consensus among agents.
        
        Strategy: Extract equations that evaluate to 24, use majority vote.
        Falls back to best equation found if no clear consensus.
        """
        # Extract equations that might equal 24
        equations = []
        for sol in solutions:
            # Pattern 1: Look for expressions in parentheses that might equal 24
            # e.g., "(10 - 4) * (13 - 9)" or "(10-4)*(13-9)"
            # Try to find complete expressions with parentheses
            paren_patterns = [
                r'\([^)]+\)\s*[*+\-/]\s*\([^)]+\)',  # (a) op (b)
                r'\([^)]+\)\s*[*+\-/]\s*\d+',  # (a) op n
                r'\d+\s*[*+\-/]\s*\([^)]+\)',  # n op (a)
            ]
            
            for pattern in paren_patterns:
                matches = re.findall(pattern, sol)
                for match in matches:
                    try:
                        # Clean and evaluate
                        clean_expr = match.strip()
                        result = eval(clean_expr)
                        if abs(result - 24.0) < 1e-5:
                            equations.append(clean_expr)
                            break
                    except:
                        continue
            
            # Pattern 2: Look for "= 24" and extract the left side
            eq_match = re.search(r'([\d+\-*/().\s]+)\s*=\s*24\b', sol, re.IGNORECASE)
            if eq_match:
                eq = eq_match.group(1).strip()
                # Clean up: remove LaTeX, markdown, etc.
                eq = re.sub(r'[\\[\]{}]', '', eq)  # Remove brackets
                eq = re.sub(r'[^\d+\-*/().\s]', '', eq)  # Keep only math chars
                eq = eq.strip()
                if eq:
                    try:
                        # Verify it evaluates to 24
                        result = eval(eq)
                        if abs(result - 24.0) < 1e-5:
                            equations.append(eq)
                            continue
                    except:
                        pass
            
            # Pattern 3: Look for expressions near "24" in the text
            # Find lines containing "24"
            lines = sol.split('\n')
            for line in lines:
                if '24' in line:
                    # Try to extract math expression from this line
                    # Look for patterns like "X * Y = 24" or "(X) op (Y) = 24"
                    expr_match = re.search(r'([\(]?[\d+\-*/().\s]+[\)]?)\s*[=:]\s*24', line)
                    if expr_match:
                        eq = expr_match.group(1).strip()
                        eq = re.sub(r'[\\[\]{}]', '', eq)
                        eq = re.sub(r'[^\d+\-*/().\s]', '', eq).strip()
                        if eq:
                            try:
                                result = eval(eq)
                                if abs(result - 24.0) < 1e-5:
                                    equations.append(eq)
                                    break
                            except:
                                pass
        
        if equations:
            # Count occurrences and use majority vote
            most_common = Counter(equations).most_common(1)[0][0]
            return most_common
        
        # Fallback: Try to construct equation from first solution
        # Look for the problem numbers and try common operations
        if solutions:
            first_sol = solutions[0]
            # Extract problem numbers
            prob_nums = re.findall(r'\d+', problem)
            if len(prob_nums) >= 4:
                # Try common patterns: (a-b)*(c-d), (a+b)*c/d, etc.
                # This is a last resort
                pass
        
        # Final fallback: return first solution (full text)
        # The evaluator will try to extract equation from it
        return solutions[0] if solutions else ""
    
    def _create_trace(self, solutions: List[str], coord_acc: float, 
                     elapsed_time: float) -> List[Dict]:
        """
        Create trace for workflow extraction (Experiment 2).
        
        Records agent actions, solutions, and coordination metrics.
        """
        trace = []
        for i, sol in enumerate(solutions):
            trace.append({
                'step': i+1,
                'agent': f'Agent_{i+1}',
                'action': 'solve',
                'action_type': 'reasoning',
                'solution': sol[:500],  # Truncate for storage
                'coordination_accuracy': coord_acc if i == 0 else None,
                'timestamp': elapsed_time * (i / len(solutions))
            })
        
        # Add consensus step
        trace.append({
            'step': len(solutions) + 1,
            'agent': 'consensus',
            'action': 'reach_consensus',
            'action_type': 'coordination',
            'coordination_accuracy': coord_acc,
            'timestamp': elapsed_time
        })
        
        return trace


# ==============================================================================
# A-SCALING EXPERIMENT
# ==============================================================================

def run_a_scaling_experiment(
    A_values: List[int] = [1, 2, 3, 5, 8],
    model: str = "gpt-4o-mini",
    num_problems: int = 10,
    difficulty: int = 3,
    replications: int = 3,
    output_dir: str = "results/agentic_scaling"
) -> Dict:
    """
    Run A-scaling experiment: vary number of agents.
    
    Args:
        A_values: List of agent counts to test
        model: LLM model to use
        num_problems: Number of problems per A value
        difficulty: Problem difficulty (1-5)
        replications: Number of replications per configuration
        output_dir: Directory to save results
        
    Returns:
        Dictionary with experiment results and analysis
    """
    print(f"\n{'='*70}")
    print("EXPERIMENT 1.1: A-SCALING (Number of Agents)")
    print(f"{'='*70}")
    print(f"A values: {A_values}")
    print(f"Model: {model}")
    print(f"Problems: {num_problems} (difficulty {difficulty})")
    print(f"Replications: {replications}")
    print(f"{'='*70}\n")
    
    # Get benchmark problems
    benchmark = GameOf24Benchmark()
    problems = benchmark.get_problems(difficulty=difficulty, num_problems=num_problems)
    
    if len(problems) < num_problems:
        print(f"⚠ Warning: Only {len(problems)} problems available, using all")
        num_problems = len(problems)
    
    print(f"Loaded {len(problems)} problems\n")
    
    results = []
    start_time = time.time()
    
    for A in A_values:
        print(f"\n{'─'*70}")
        print(f"Testing A={A} agents...")
        print(f"{'─'*70}")
        
        system = MultiAgentSystem(n_agents=A, model=model)
        
        solve_rates = []
        coord_accuracies = []
        consensus_times = []
        all_traces = []
        all_solutions = []
        
        for rep in range(replications):
            print(f"\n  Replication {rep+1}/{replications}:")
            
            for i, problem in enumerate(problems):
                print(f"    Problem {i+1}/{len(problems)}: {problem.problem_id}", end=" ... ")
                
                try:
                    result = system.solve_with_coordination(problem.problem_text)
                    
                    # Evaluate solution
                    is_correct, partial_credit = benchmark.evaluate_solution(
                        problem, result['solution']
                    )
                    
                    solve_rates.append(1.0 if is_correct else 0.0)
                    coord_accuracies.append(result['coordination_accuracy'])
                    consensus_times.append(result['consensus_time'])
                    all_traces.append(result['trace'])
                    all_solutions.append({
                        'problem_id': problem.problem_id,
                        'solution': result['solution'],
                        'is_correct': is_correct,
                        'partial_credit': partial_credit,
                        'all_agent_solutions': result['solutions']
                    })
                    
                    status = "✓" if is_correct else "✗"
                    print(f"{status} (coord: {result['coordination_accuracy']:.2f})")
                    
                except Exception as e:
                    print(f"ERROR: {str(e)}")
                    solve_rates.append(0.0)
                    coord_accuracies.append(0.0)
                    consensus_times.append(0.0)
        
        # Aggregate results for this A value
        result_summary = {
            'A': A,
            'solve_rate': float(np.mean(solve_rates)),
            'solve_rate_std': float(np.std(solve_rates)),
            'coordination_accuracy': float(np.mean(coord_accuracies)),
            'coordination_accuracy_std': float(np.std(coord_accuracies)),
            'consensus_time': float(np.mean(consensus_times)),
            'consensus_time_std': float(np.std(consensus_times)),
            'num_problems': num_problems,
            'replications': replications,
            'total_runs': len(solve_rates),
            'traces': all_traces,
            'solutions': all_solutions
        }
        
        results.append(result_summary)
        
        print(f"\n  Summary for A={A}:")
        print(f"    Solve Rate: {result_summary['solve_rate']:.3f} ± {result_summary['solve_rate_std']:.3f}")
        print(f"    Coordination: {result_summary['coordination_accuracy']:.3f} ± {result_summary['coordination_accuracy_std']:.3f}")
        print(f"    Consensus Time: {result_summary['consensus_time']:.2f}s ± {result_summary['consensus_time_std']:.2f}s")
    
    total_time = time.time() - start_time
    
    # Analyze results
    print(f"\n{'='*70}")
    print("ANALYZING RESULTS...")
    print(f"{'='*70}")
    
    analysis = analyze_a_scaling(results)
    
    # Create output
    output = {
        'experiment': 'A-scaling',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'A_values': A_values,
            'model': model,
            'num_problems': num_problems,
            'difficulty': difficulty,
            'replications': replications
        },
        'results': results,
        'analysis': analysis,
        'execution_time_seconds': total_time
    }
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    output_file = output_path / "exp1_1_a_scaling_results.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n✓ Results saved to {output_file}")
    print(f"✓ Total execution time: {total_time/60:.1f} minutes")
    
    return output


def analyze_a_scaling(results: List[Dict]) -> Dict:
    """
    Analyze A-scaling results for power-law fit.
    
    Fits: solve_rate = a × A^β_A
    """
    A_values = np.array([r['A'] for r in results])
    solve_rates = np.array([r['solve_rate'] for r in results])
    coord_accs = np.array([r['coordination_accuracy'] for r in results])
    
    # Fit power law: solve_rate ∝ A^β_A
    a_solve, beta_A_solve, r_squared_solve = fit_power_law(A_values, solve_rates)
    
    # Fit power law for coordination (if applicable)
    a_coord, beta_A_coord, r_squared_coord = fit_power_law(A_values, coord_accs)
    
    # Find critical point (where solve_rate crosses 0.5)
    critical_A = find_critical_exponent(A_values, solve_rates, threshold=0.5)
    
    analysis = {
        'solve_rate_power_law': {
            'coefficient': float(a_solve),
            'exponent': float(beta_A_solve),
            'r_squared': float(r_squared_solve),
            'formula': f"solve_rate = {a_solve:.4f} × A^{beta_A_solve:.4f}"
        },
        'coordination_power_law': {
            'coefficient': float(a_coord),
            'exponent': float(beta_A_coord),
            'r_squared': float(r_squared_coord),
            'formula': f"coordination = {a_coord:.4f} × A^{beta_A_coord:.4f}"
        },
        'critical_A': float(critical_A) if critical_A is not None else None,
        'interpretation': {
            'solve_rate_scaling': 'positive' if beta_A_solve > 0 else 'negative',
            'coordination_scaling': 'positive' if beta_A_coord > 0 else 'negative',
            'power_law_valid': r_squared_solve > 0.8
        }
    }
    
    print(f"\nPower Law Analysis:")
    print(f"  Solve Rate: {analysis['solve_rate_power_law']['formula']}")
    print(f"    R² = {analysis['solve_rate_power_law']['r_squared']:.3f}")
    print(f"  Coordination: {analysis['coordination_power_law']['formula']}")
    print(f"    R² = {analysis['coordination_power_law']['r_squared']:.3f}")
    if critical_A:
        print(f"  Critical A: {critical_A:.2f}")
    
    return analysis


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """Run A-scaling experiment."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run A-scaling experiment')
    parser.add_argument('--A-values', type=int, nargs='+', default=[1, 2, 3, 5],
                       help='Agent counts to test')
    parser.add_argument('--model', type=str, default='gpt-4o-mini',
                       help='LLM model to use')
    parser.add_argument('--num-problems', type=int, default=5,
                       help='Number of problems per A value')
    parser.add_argument('--difficulty', type=int, default=3,
                       help='Problem difficulty (1-5)')
    parser.add_argument('--replications', type=int, default=1,
                       help='Number of replications')
    parser.add_argument('--pilot', action='store_true',
                       help='Run pilot (A=[1,2,3], 3 problems, 1 rep)')
    
    args = parser.parse_args()
    
    if args.pilot:
        print("Running PILOT experiment...")
        A_values = [1, 2, 3]
        num_problems = 3
        replications = 1
    else:
        A_values = args.A_values
        num_problems = args.num_problems
        replications = args.replications
    
    results = run_a_scaling_experiment(
        A_values=A_values,
        model=args.model,
        num_problems=num_problems,
        difficulty=args.difficulty,
        replications=replications
    )
    
    print(f"\n{'='*70}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: results/agentic_scaling/exp1_1_a_scaling_results.json")
    print(f"Next steps:")
    print(f"  1. Review results")
    print(f"  2. Generate visualizations")
    print(f"  3. Proceed to Experiment 1.2 (T-Scaling)")


if __name__ == "__main__":
    main()
