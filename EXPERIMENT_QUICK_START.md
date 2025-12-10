# Experiment Quick Start Guide

This guide helps you get started implementing the four experiments from the research plan.

## Prerequisites

- Existing codebase infrastructure (✅ you have this)
- OpenAI API key set in environment
- Python 3.8+ with dependencies installed

## Experiment 1: Agentic Scaling Laws

### Step 1: Extend DebateSystem for N Agents

Create `agentic_scaling_experiment.py`:

```python
from advanced_system2_architectures import DebateSystem, DebateAgent
from typing import List, Dict
import numpy as np

class MultiAgentSystem:
    """Extends DebateSystem to support N agents."""
    
    def __init__(self, n_agents: int, model: str = "gpt-4o"):
        self.agents = [DebateAgent(f"Agent_{i}", model) for i in range(n_agents)]
        self.n_agents = n_agents
    
    def solve_with_coordination(self, problem: str) -> Dict:
        """Solve with N agents, measure coordination."""
        # Initial solutions from all agents
        solutions = [agent.solve(problem) for agent in self.agents]
        
        # Measure coordination (agreement)
        coordination_accuracy = self._measure_agreement(solutions)
        
        # Consensus mechanism (simple: majority vote)
        consensus_solution = self._reach_consensus(solutions, problem)
        
        return {
            'solution': consensus_solution,
            'coordination_accuracy': coordination_accuracy,
            'n_agents': self.n_agents,
            'solutions': solutions
        }
    
    def _measure_agreement(self, solutions: List[str]) -> float:
        """Measure how much agents agree."""
        # Simple: check if solutions are similar
        # In practice: use embeddings or structured comparison
        if len(solutions) < 2:
            return 1.0
        
        # Count pairs that agree (simplified)
        agreements = 0
        total_pairs = 0
        for i in range(len(solutions)):
            for j in range(i+1, len(solutions)):
                total_pairs += 1
                if self._solutions_similar(solutions[i], solutions[j]):
                    agreements += 1
        
        return agreements / total_pairs if total_pairs > 0 else 0.0
    
    def _solutions_similar(self, sol1: str, sol2: str) -> bool:
        """Check if two solutions are similar."""
        # Simplified: check for common keywords/numbers
        # In practice: use semantic similarity
        words1 = set(sol1.lower().split())
        words2 = set(sol2.lower().split())
        overlap = len(words1 & words2) / max(len(words1 | words2), 1)
        return overlap > 0.5
    
    def _reach_consensus(self, solutions: List[str], problem: str) -> str:
        """Reach consensus among agents."""
        # Simple: return most common solution
        # In practice: use judge or voting mechanism
        from collections import Counter
        # For now, return first solution (extend with voting)
        return solutions[0] if solutions else ""
```

### Step 2: Run A-Scaling Experiment

```python
from benchmarks import GameOf24Benchmark
from agentic_scaling_experiment import MultiAgentSystem

def run_a_scaling_experiment():
    """Vary number of agents A."""
    benchmark = GameOf24Benchmark()
    problems = benchmark.get_problems(difficulty=3, num_problems=10)
    
    A_values = [1, 2, 3, 5, 8]  # Number of agents
    results = []
    
    for A in A_values:
        print(f"Testing A={A} agents...")
        system = MultiAgentSystem(n_agents=A)
        
        solve_rates = []
        coord_accuracies = []
        
        for problem in problems:
            result = system.solve_with_coordination(problem.problem_text)
            
            # Evaluate solution
            is_correct, _ = benchmark.evaluate_solution(problem, result['solution'])
            solve_rates.append(1.0 if is_correct else 0.0)
            coord_accuracies.append(result['coordination_accuracy'])
        
        results.append({
            'A': A,
            'solve_rate': np.mean(solve_rates),
            'coordination_accuracy': np.mean(coord_accuracies),
            'std': np.std(solve_rates)
        })
    
    return results

# Run and plot
results = run_a_scaling_experiment()
# Plot: solve_rate vs A (log-log)
```

### Step 3: Add Tool Richness (K-Scaling)

Create `tool_augmented_agent.py`:

```python
class ToolAugmentedAgent:
    """Agent with access to tools."""
    
    def __init__(self, available_tools: List[str]):
        self.tools = available_tools
        self.tool_usage_history = []
    
    def solve(self, problem: str) -> Dict:
        """Solve with tool access."""
        # Use tools as needed
        solution = self._reason_with_tools(problem)
        return {
            'solution': solution,
            'tools_used': self.tool_usage_history,
            'tool_count': len(self.tools)
        }
    
    def _reason_with_tools(self, problem: str) -> str:
        """Reason using available tools."""
        # Simplified: agent decides which tools to use
        # In practice: LLM decides tool usage
        if 'calculator' in self.tools:
            # Use calculator for arithmetic
            pass
        if 'search' in self.tools:
            # Use search for information
            pass
        # ... implement tool usage logic
        return "solution"
```

## Experiment 2: Workflow Grammar Extraction

### Step 1: Create Workflow Extractor

Create `workflow_grammar_extractor.py`:

```python
from typing import List, Dict
import re
from collections import Counter

class WorkflowExtractor:
    """Extract production rules from agent traces."""
    
    def __init__(self):
        self.production_rules = []
        self.skill_frequencies = Counter()
    
    def extract_from_trace(self, trace: List[Dict]) -> List[str]:
        """Extract production rules from agent trace."""
        rules = []
        
        for step in trace:
            action_type = step.get('action_type', 'unknown')
            
            if action_type == 'tool_call':
                tool = step.get('tool', 'unknown')
                rules.append(f"Action → ToolCall({tool})")
                self.skill_frequencies[f"tool_{tool}"] += 1
            
            elif action_type == 'reasoning':
                reasoning_type = step.get('reasoning_type', 'unknown')
                rules.append(f"Action → Reasoning({reasoning_type})")
                self.skill_frequencies[f"reasoning_{reasoning_type}"] += 1
            
            elif action_type == 'validation':
                rules.append("Action → Validation")
                self.skill_frequencies["validation"] += 1
        
        return rules
    
    def analyze_skill_distribution(self) -> Dict:
        """Analyze skill frequency distribution."""
        total = sum(self.skill_frequencies.values())
        frequencies = {k: v/total for k, v in self.skill_frequencies.items()}
        
        # Rank by frequency
        ranked = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'frequencies': frequencies,
            'ranked': ranked,
            'rare_skills': [k for k, v in frequencies.items() if v < 0.05],
            'common_skills': [k for k, v in frequencies.items() if v > 0.2]
        }
    
    def test_zipf_law(self) -> Dict:
        """Test if distribution follows Zipf's law."""
        ranked = self.analyze_skill_distribution()['ranked']
        
        # Extract ranks and frequencies
        ranks = np.arange(1, len(ranked) + 1)
        freqs = np.array([v for _, v in ranked])
        
        # Fit power law: frequency ∝ rank^-α
        from system2_power_law_analysis import fit_power_law
        a, alpha, r_squared = fit_power_law(ranks, freqs)
        
        return {
            'zipf_exponent': alpha,
            'r_squared': r_squared,
            'follows_zipf': abs(alpha - 1.0) < 0.3 and r_squared > 0.8
        }
```

### Step 2: Extract Workflows from Existing Traces

```python
# Load existing experiment results
import json

with open('results/system2/scaling_results.json', 'r') as f:
    results = json.load(f)

extractor = WorkflowExtractor()

# Extract workflows from all traces
for result in results.get('raw_results', []):
    if 'trace' in result:
        rules = extractor.extract_from_trace(result['trace'])

# Analyze
skill_analysis = extractor.analyze_skill_distribution()
zipf_test = extractor.test_zipf_law()

print(f"Zipf exponent: {zipf_test['zipf_exponent']:.3f}")
print(f"R²: {zipf_test['r_squared']:.3f}")
print(f"Follows Zipf: {zipf_test['follows_zipf']}")
```

## Experiment 3: Criticality in Coordination

### Step 1: Extend Coordination Game

Modify `game_theory_mas.py` or create `multi_agent_criticality.py`:

```python
from game_theory_mas import CoordinationGame
import numpy as np

class CriticalityCoordinationExperiment:
    """Test criticality in multi-agent coordination."""
    
    def __init__(self):
        self.results = []
    
    def vary_communication_bandwidth(self, 
                                    n_agents: int = 5,
                                    bandwidths: List[int] = [0, 1, 3, 5, 10, 20]):
        """Vary communication budget."""
        results = []
        
        for bandwidth in bandwidths:
            print(f"Testing bandwidth={bandwidth}...")
            
            coord_accuracies = []
            consensus_times = []
            
            for trial in range(5):  # Replications
                game = CoordinationGame(n_agents=n_agents, n_tasks=10)
                
                # Run with communication budget
                result = self._run_with_bandwidth(game, bandwidth)
                
                coord_accuracies.append(result['coordination_accuracy'])
                consensus_times.append(result['consensus_time'])
            
            results.append({
                'bandwidth': bandwidth,
                'coordination_accuracy': np.mean(coord_accuracies),
                'consensus_time': np.mean(consensus_times),
                'std': np.std(coord_accuracies)
            })
        
        return results
    
    def _run_with_bandwidth(self, game, bandwidth: int) -> Dict:
        """Run coordination game with limited bandwidth."""
        # Simplified: agents can send 'bandwidth' messages
        # In practice: implement message passing protocol
        
        messages_sent = 0
        coordination_accuracy = 0.0
        
        # Simulate coordination with message budget
        # (Implement actual coordination logic)
        
        return {
            'coordination_accuracy': coordination_accuracy,
            'consensus_time': messages_sent,
            'messages_used': messages_sent
        }
    
    def find_critical_bandwidth(self, results: List[Dict]) -> float:
        """Find critical bandwidth where coordination emerges."""
        from system2_power_law_analysis import find_critical_exponent
        
        bandwidths = np.array([r['bandwidth'] for r in results])
        coord_accs = np.array([r['coordination_accuracy'] for r in results])
        
        # Find where coordination crosses 0.5 threshold
        critical = find_critical_exponent(bandwidths, coord_accs, threshold=0.5)
        
        return critical
```

### Step 2: Run Criticality Experiment

```python
experiment = CriticalityCoordinationExperiment()
results = experiment.vary_communication_bandwidth()
critical_bandwidth = experiment.find_critical_bandwidth(results)

print(f"Critical bandwidth: {critical_bandwidth}")
# Plot phase diagram: coordination_accuracy vs bandwidth
```

## Experiment 4: Compute-Optimal Architecture

### Step 1: Create Compute Budget System

Create `compute_optimal_architecture.py`:

```python
from dataclasses import dataclass
from typing import List, Dict
import numpy as np

@dataclass
class ComputeBudget:
    """Tracks compute budget and usage."""
    total_tokens: int
    tokens_used: int = 0
    
    def can_afford(self, cost: int) -> bool:
        return self.tokens_used + cost <= self.total_tokens
    
    def spend(self, cost: int):
        self.tokens_used += cost
    
    def remaining(self) -> int:
        return self.total_tokens - self.tokens_used

class ArchitectureComparison:
    """Compare different agent architectures under compute constraints."""
    
    def __init__(self, budget: int = 1000000):
        self.budget = ComputeBudget(budget)
    
    def compare_agent_sizes(self, 
                            task: str,
                            agent_configs: List[Dict]) -> Dict:
        """Compare different (size, count) configurations."""
        results = []
        
        for config in agent_configs:
            size = config['size']  # 'small', 'medium', 'large'
            count = config['count']
            
            # Estimate cost per agent
            cost_per_agent = self._estimate_cost(size)
            total_cost = cost_per_agent * count
            
            if total_cost > self.budget.total_tokens:
                continue  # Skip if over budget
            
            # Run with this configuration
            result = self._run_architecture(task, size, count)
            
            results.append({
                'size': size,
                'count': count,
                'solve_rate': result['solve_rate'],
                'cost': total_cost,
                'efficiency': result['solve_rate'] / total_cost
            })
        
        return results
    
    def _estimate_cost(self, size: str) -> int:
        """Estimate token cost per agent."""
        costs = {
            'small': 10000,   # GPT-3.5-turbo
            'medium': 20000,  # GPT-4o-mini
            'large': 50000    # GPT-4o
        }
        return costs.get(size, 20000)
    
    def _run_architecture(self, task: str, size: str, count: int) -> Dict:
        """Run architecture and measure performance."""
        # Simplified: use existing systems
        # In practice: implement actual architecture execution
        
        from agentic_scaling_experiment import MultiAgentSystem
        
        # Map size to model
        model_map = {
            'small': 'gpt-3.5-turbo',
            'medium': 'gpt-4o-mini',
            'large': 'gpt-4o'
        }
        
        system = MultiAgentSystem(n_agents=count, model=model_map[size])
        result = system.solve_with_coordination(task)
        
        # Evaluate (simplified)
        solve_rate = result.get('coordination_accuracy', 0.5)
        
        return {'solve_rate': solve_rate}
```

### Step 2: Find Optimal Configuration

```python
comparison = ArchitectureComparison(budget=1000000)

configs = [
    {'size': 'large', 'count': 1},
    {'size': 'medium', 'count': 2},
    {'size': 'small', 'count': 5}
]

task = "Use numbers 4 9 10 13 to get 24"
results = comparison.compare_agent_sizes(task, configs)

# Find optimal (highest efficiency)
optimal = max(results, key=lambda x: x['efficiency'])
print(f"Optimal: {optimal['size']} × {optimal['count']}")
print(f"Efficiency: {optimal['efficiency']:.2e}")
```

## Running All Experiments

Create `run_all_experiments.py`:

```python
"""Run all four experiments sequentially."""

from agentic_scaling_experiment import run_a_scaling_experiment
from workflow_grammar_extractor import extract_workflows
from multi_agent_criticality import CriticalityCoordinationExperiment
from compute_optimal_architecture import ArchitectureComparison

def main():
    print("="*70)
    print("RUNNING ALL EXPERIMENTS")
    print("="*70)
    
    # Experiment 1: Scaling Laws
    print("\n[1/4] Agentic Scaling Laws...")
    scaling_results = run_a_scaling_experiment()
    # Save results...
    
    # Experiment 2: Workflow Grammar
    print("\n[2/4] Workflow Grammar Extraction...")
    workflow_results = extract_workflows()
    # Save results...
    
    # Experiment 3: Criticality
    print("\n[3/4] Multi-Agent Criticality...")
    criticality_exp = CriticalityCoordinationExperiment()
    criticality_results = criticality_exp.vary_communication_bandwidth()
    # Save results...
    
    # Experiment 4: Compute-Optimal
    print("\n[4/4] Compute-Optimal Architecture...")
    comparison = ArchitectureComparison()
    optimal_results = comparison.compare_agent_sizes(...)
    # Save results...
    
    print("\n✓ All experiments completed!")
    print("Results saved to results/agentic_scaling/")

if __name__ == "__main__":
    main()
```

## Next Steps

1. **Start Small**: Run pilot experiments with 2-3 configurations
2. **Validate**: Check that metrics make sense
3. **Scale Up**: Expand to full experimental design
4. **Analyze**: Use existing analysis tools from `system2_power_law_analysis.py`

## Tips

- **Cache Results**: Save intermediate results frequently
- **Parallelize**: Run independent configurations in parallel
- **Monitor Costs**: Track API usage to avoid overruns
- **Incremental**: Build on existing infrastructure rather than rewriting

Good luck! 🚀
