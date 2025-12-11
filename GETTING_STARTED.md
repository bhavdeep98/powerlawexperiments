# Getting Started: Experiment Implementation Guide

Based on `EXPERIMENT_MAPPING.md` and `EXPERIMENT_READINESS.md`, this guide shows exactly what to do to start running experiments.

---

## Quick Status

✅ **Ready to Start**: ~40% infrastructure exists  
🟡 **Needs Extension**: Core systems exist but need N-agent support  
🔴 **Needs Creation**: Experiment runners and specialized systems

---

## Recommended Starting Point: Experiment 1.1 (A-Scaling)

**Why Start Here?**
- Builds on existing `DebateSystem` (just needs extension)
- Simple to implement (extend to N agents)
- Quick to test (can run with A=1,2,3 immediately)
- Provides foundation for other experiments

**Time Estimate**: 2-3 hours for basic implementation

---

## Step-by-Step: Implement A-Scaling

### Step 1: Create MultiAgentSystem (30 min)

Create `agentic_scaling_experiment.py` with:

```python
from advanced_system2_architectures import DebateAgent, DebateSystem
from typing import List, Dict
import numpy as np

class MultiAgentSystem:
    """Extends DebateSystem to support N agents for A-scaling experiments."""
    
    def __init__(self, n_agents: int, model: str = "gpt-4o"):
        self.agents = [DebateAgent(f"Agent_{i+1}", model) 
                      for i in range(n_agents)]
        self.n_agents = n_agents
        self.model = model
    
    def solve_with_coordination(self, problem: str) -> Dict:
        """Solve with N agents, measure coordination."""
        # Get solutions from all agents
        solutions = [agent.solve(problem) for agent in self.agents]
        
        # Measure coordination (agreement)
        coordination_accuracy = self._measure_agreement(solutions)
        
        # Reach consensus
        consensus_solution = self._reach_consensus(solutions, problem)
        
        return {
            'solution': consensus_solution,
            'coordination_accuracy': coordination_accuracy,
            'n_agents': self.n_agents,
            'solutions': solutions,
            'trace': self._create_trace(solutions, coordination_accuracy)
        }
    
    def _measure_agreement(self, solutions: List[str]) -> float:
        """Measure how much agents agree (coordination accuracy)."""
        if len(solutions) < 2:
            return 1.0
        
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
        """Check if two solutions are similar."""
        # Extract numbers/answers from solutions
        import re
        nums1 = set(re.findall(r'\d+', sol1))
        nums2 = set(re.findall(r'\d+', sol2))
        
        # Check overlap
        if len(nums1) == 0 or len(nums2) == 0:
            # Fallback: word overlap
            words1 = set(sol1.lower().split())
            words2 = set(sol2.lower().split())
            overlap = len(words1 & words2) / max(len(words1 | words2), 1)
            return overlap > 0.5
        
        overlap = len(nums1 & nums2) / max(len(nums1 | nums2), 1)
        return overlap > 0.6
    
    def _reach_consensus(self, solutions: List[str], problem: str) -> str:
        """Reach consensus among agents (simple: majority vote on answer)."""
        # Extract answers/numbers from solutions
        import re
        answers = []
        for sol in solutions:
            # Try to extract final answer (e.g., "24" or "= 24")
            match = re.search(r'(?:=|is|equals?)\s*(\d+)', sol, re.IGNORECASE)
            if match:
                answers.append(match.group(1))
        
        if answers:
            # Majority vote
            from collections import Counter
            most_common = Counter(answers).most_common(1)[0][0]
            return most_common
        
        # Fallback: return first solution
        return solutions[0] if solutions else ""
    
    def _create_trace(self, solutions: List[str], coord_acc: float) -> List[Dict]:
        """Create trace for workflow extraction."""
        trace = []
        for i, sol in enumerate(solutions):
            trace.append({
                'step': i+1,
                'agent': f'Agent_{i+1}',
                'action': 'solve',
                'solution': sol[:200],  # Truncate
                'coordination_accuracy': coord_acc if i == 0 else None
            })
        return trace
```

### Step 2: Create A-Scaling Experiment Runner (1 hour)

Add to same file:

```python
from benchmarks import GameOf24Benchmark
import json
from pathlib import Path

def run_a_scaling_experiment(
    A_values: List[int] = [1, 2, 3, 5, 8],
    model: str = "gpt-4o",
    num_problems: int = 10,
    difficulty: int = 3
) -> Dict:
    """Run A-scaling experiment: vary number of agents."""
    
    print(f"\n{'='*70}")
    print("EXPERIMENT 1.1: A-SCALING (Number of Agents)")
    print(f"{'='*70}")
    print(f"A values: {A_values}")
    print(f"Model: {model}")
    print(f"Problems: {num_problems} (difficulty {difficulty})")
    print(f"{'='*70}\n")
    
    # Get benchmark problems
    benchmark = GameOf24Benchmark()
    problems = benchmark.get_problems(difficulty=difficulty, num_problems=num_problems)
    
    results = []
    
    for A in A_values:
        print(f"\nTesting A={A} agents...")
        system = MultiAgentSystem(n_agents=A, model=model)
        
        solve_rates = []
        coord_accuracies = []
        all_traces = []
        
        for i, problem in enumerate(problems):
            print(f"  Problem {i+1}/{len(problems)}: {problem.problem_id}")
            
            result = system.solve_with_coordination(problem.problem_text)
            
            # Evaluate solution
            is_correct, partial_credit = benchmark.evaluate_solution(
                problem, result['solution']
            )
            
            solve_rates.append(1.0 if is_correct else 0.0)
            coord_accuracies.append(result['coordination_accuracy'])
            all_traces.append(result['trace'])
            
            print(f"    Correct: {is_correct}, Coordination: {result['coordination_accuracy']:.2f}")
        
        results.append({
            'A': A,
            'solve_rate': np.mean(solve_rates),
            'coordination_accuracy': np.mean(coord_accuracies),
            'std': np.std(solve_rates),
            'num_problems': len(problems),
            'traces': all_traces
        })
        
        print(f"  A={A}: Solve Rate = {np.mean(solve_rates):.3f} ± {np.std(solve_rates):.3f}")
        print(f"         Coordination = {np.mean(coord_accuracies):.3f}")
    
    return {
        'experiment': 'A-scaling',
        'config': {
            'A_values': A_values,
            'model': model,
            'num_problems': num_problems,
            'difficulty': difficulty
        },
        'results': results
    }

def analyze_a_scaling(results: Dict) -> Dict:
    """Analyze A-scaling results for power-law fit."""
    from system2_power_law_analysis import fit_power_law
    
    A_values = np.array([r['A'] for r in results['results']])
    solve_rates = np.array([r['solve_rate'] for r in results['results']])
    
    # Fit power law: solve_rate ∝ A^β_A
    a, beta_A, r_squared = fit_power_law(A_values, solve_rates)
    
    return {
        'power_law_coefficient': a,
        'power_law_exponent': beta_A,
        'r_squared': r_squared,
        'critical_A': None  # Can add critical point detection
    }

if __name__ == "__main__":
    # Run experiment
    results = run_a_scaling_experiment(
        A_values=[1, 2, 3, 5],  # Start small
        model="gpt-4o-mini",     # Use cheaper model for testing
        num_problems=5,          # Start with few problems
        difficulty=3
    )
    
    # Analyze
    analysis = analyze_a_scaling(results)
    
    print(f"\n{'='*70}")
    print("ANALYSIS")
    print(f"{'='*70}")
    print(f"Power law: solve_rate ∝ A^{analysis['power_law_exponent']:.3f}")
    print(f"R² = {analysis['r_squared']:.3f}")
    
    # Save results
    output_dir = Path("results/agentic_scaling")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "exp1_1_a_scaling_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'results': results,
            'analysis': analysis
        }, f, indent=2, default=str)
    
    print(f"\n✓ Results saved to {output_file}")
```

### Step 3: Test It (15 min)

```bash
# Run the experiment
python agentic_scaling_experiment.py

# Should output:
# - Progress for each A value
# - Solve rates and coordination accuracies
# - Power-law fit results
# - Saved JSON file
```

---

## Next Steps After A-Scaling Works

### Immediate Next (Same Day)
1. ✅ Verify power-law fit (R² > 0.8)
2. ✅ Check coordination metrics make sense
3. ✅ Review traces for workflow extraction

### This Week
1. **Experiment 1.2 (T-Scaling)**: Mostly exists, just need to aggregate
2. **Experiment 1.3 (K-Scaling)**: Create `ToolAugmentedAgent`
3. **Experiment 2.1 (Workflow Extraction)**: Create `WorkflowExtractor`

### Next Week
1. **Experiment 3.1 (Criticality)**: Create `multi_agent_criticality.py`
2. **Experiment 4.1 (Compute-Optimal)**: Create `compute_optimal_architecture.py`

---

## File Creation Checklist

### Week 1 (Critical)
- [x] `EXPERIMENT_READINESS.md` - This assessment
- [ ] `agentic_scaling_experiment.py` - **CREATE THIS FIRST**
- [ ] `tool_augmented_agent.py` - For K-scaling
- [ ] `workflow_grammar_extractor.py` - For workflow extraction

### Week 2
- [ ] `multi_agent_criticality.py` - For criticality experiments
- [ ] `compute_optimal_architecture.py` - For compute-optimal experiments
- [ ] `network_topology.py` - For network experiments

### Week 3+
- [ ] `trace_collector.py` - Shared trace utilities
- [ ] `result_aggregator.py` - Result analysis
- [ ] `optimization_framework.py` - Multi-objective optimization

---

## Testing Strategy

### Unit Tests
```python
# Test MultiAgentSystem
def test_multi_agent_coordination():
    system = MultiAgentSystem(n_agents=3)
    result = system.solve_with_coordination("test problem")
    assert 'coordination_accuracy' in result
    assert 0 <= result['coordination_accuracy'] <= 1
```

### Integration Tests
```python
# Test A-scaling experiment
def test_a_scaling():
    results = run_a_scaling_experiment(A_values=[1, 2, 3], num_problems=2)
    assert len(results['results']) == 3
    assert all('solve_rate' in r for r in results['results'])
```

---

## Common Issues & Solutions

### Issue: Coordination accuracy always 0 or 1
**Solution**: Improve `_solutions_similar()` - may need semantic similarity

### Issue: Power-law fit has low R²
**Solution**: 
- Check if A values span enough range
- Verify solve rates vary meaningfully
- May need more problems or replications

### Issue: API costs too high
**Solution**:
- Start with `gpt-4o-mini` instead of `gpt-4o`
- Reduce num_problems to 3-5 for testing
- Use smaller A values [1, 2, 3] first

---

## Success Criteria

### For A-Scaling (Experiment 1.1)
- ✅ Can run with A ∈ {1, 2, 3, 5}
- ✅ Coordination accuracy calculated correctly
- ✅ Power-law fit has R² > 0.8
- ✅ Results saved to JSON
- ✅ Traces collected for workflow extraction

### For Full Experiment 1
- ✅ All three scaling variables (A, T, K) work
- ✅ Combined experiment runs
- ✅ All metrics collected
- ✅ Power-law exponents extracted

---

## Resources

- **Detailed Plan**: `EXPERIMENTAL_PLAN.md`
- **Quick Start**: `EXPERIMENT_QUICK_START.md`
- **Readiness**: `EXPERIMENT_READINESS.md`
- **Mapping**: `EXPERIMENT_MAPPING.md`

---

**Ready to start?** Create `agentic_scaling_experiment.py` and run your first A-scaling experiment!
