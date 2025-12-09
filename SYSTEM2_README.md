# System 2 Reasoning Experiments

This directory contains comprehensive experiments for System 2 reasoning, implementing the roadmap for exploring criticality in deliberate, multi-step reasoning.

## Overview

System 2 reasoning represents the frontier of AI capabilities - the ability to perform deliberate, multi-step reasoning that goes beyond pattern matching. These experiments test the hypothesis that System 2 reasoning shows **phase transitions** at critical combinations of model size, search depth, and beam width.

## Key Components

### 1. Enhanced Tree of Thought (`tree_of_thought_enhanced.py`)

Implements multiple search strategies with comprehensive metrics:
- **Search Strategies**: BFS, DFS, Best-First, Beam Search, MCTS
- **State Validation**: Catches hallucinated numbers/operations
- **Metrics Tracking**: Branching factor, depth, pruning efficiency, search efficiency
- **Comparison Framework**: Compare strategies on same problems

**Usage:**
```python
from tree_of_thought_enhanced import EnhancedGameOf24, SearchStrategy

solver = EnhancedGameOf24("4 9 10 13")
result = solver.solve_tot(
    strategy=SearchStrategy.BEAM,
    branching_factor=3,
    max_depth=5,
    beam_width=3
)
```

### 2. Enhanced DSPy Reasoning (`dspy_reasoning_enhanced.py`)

Expands DSPy with advanced optimizations:
- **Multiple Optimizers**: BootstrapFewShot, MIPROv2, COPRO
- **Self-Consistency**: Multiple samples with voting
- **Multi-Stage Reasoning**: Separate modules for planning, execution, verification
- **Automatic Prompt Engineering**: Optimize across problem types

**Usage:**
```python
from dspy_reasoning_enhanced import compare_optimizers, test_self_consistency

# Compare optimizers
results = compare_optimizers(trainset, testset, optimizers=['bootstrap', 'mipro'])

# Test self-consistency
consistency_results = test_self_consistency(trainset, testset, num_samples_list=[1, 5, 10])
```

### 3. System 2 Criticality Experiment (`system2_criticality_experiment.py`)

Tests phase transitions in System 2 reasoning:
- **Scaling Variables**: Model size, search depth, beam width, branching factor
- **Metrics**: Solve rate, tokens used, time to solution, search efficiency, hallucination rate
- **Phase Diagrams**: Heatmaps showing solve_rate as function of (model_size, search_depth)
- **Critical Point Detection**: Identifies where performance jumps dramatically

**Usage:**
```python
from system2_criticality_experiment import System2CriticalityExperiment, ExperimentConfig

config = ExperimentConfig.default()
experiment = System2CriticalityExperiment(config)
results = experiment.run_scaling_experiment()
output = experiment.save_results()
```

### 4. State Tracking Benchmarks (`state_tracking_benchmarks.py`)

Measures state tracking fidelity - a critical bottleneck:
- **Tasks**: Stack tracking, variable tracking, counter tracking
- **Metrics**: Per-step accuracy, catastrophic failure point
- **Critical Analysis**: Finds where state tracking breaks down

**Usage:**
```python
from state_tracking_benchmarks import run_state_tracking_experiments

results = run_state_tracking_experiments(
    model="gpt-4o",
    num_steps_list=[5, 10, 15, 20, 25, 30]
)
```

### 5. Advanced System 2 Architectures (`advanced_system2_architectures.py`)

Three advanced reasoning architectures:

#### A. Verify-and-Refine Loop
Iterative refinement with verification:
```python
from advanced_system2_architectures import VerifierReasonerSystem

verifier = VerifierReasonerSystem(max_iterations=5)
result = verifier.solve(problem)
```

#### B. Debate/Multi-Agent System
Multiple agents debate solutions:
```python
from advanced_system2_architectures import DebateSystem

debate = DebateSystem(agent_models=['gpt-4o', 'gpt-4o'], debate_rounds=3)
result = debate.solve(problem)
```

#### C. Memory-Augmented Reasoning
Explicit working memory and long-term memory:
```python
from advanced_system2_architectures import MemoryAugmentedReasoner

memory = MemoryAugmentedReasoner()
result = memory.solve(problem, max_steps=10)
```

### 6. Comprehensive Experiment Protocol (`system2_experiment_protocol.py`)

Unified framework for running all experiments:
- **Dataset Loaders**: Game of 24, GSM8K, logic puzzles
- **Experiment Orchestration**: Run all experiments in sequence
- **Result Aggregation**: Combine results from all components

**Usage:**
```python
from system2_experiment_protocol import run_quick_experiment, run_full_experiment

# Quick test
results = run_quick_experiment()

# Full comprehensive suite
results = run_full_experiment()
```

**Command Line:**
```bash
# Quick test
python system2_experiment_protocol.py --quick

# Full experiment suite
python system2_experiment_protocol.py --full
```

### 7. Power Law Analysis (`system2_power_law_analysis.py`)

Analyzes System 2 results for power law relationships:
- **Solve Rate Scaling**: Tests S ∝ (M × D)^α
- **Hallucination Phase Transition**: Detects sharp drops
- **Search Efficiency**: Compares strategies
- **Visualization**: Generates power law plots

**Usage:**
```python
from system2_power_law_analysis import analyze_system2_scaling

# Load results
with open('system2_comprehensive_results.json', 'r') as f:
    results = json.load(f)

# Analyze
analysis = analyze_system2_scaling(results['results'])
```

## Quick Start

1. **Install Dependencies:**
```bash
pip install -r requirements.txt
```

2. **Set API Key:**
```bash
export OPENAI_API_KEY=your_key_here
```

3. **Run Quick Test:**
```bash
python system2_experiment_protocol.py --quick
```

4. **Run Full Experiment Suite:**
```bash
python system2_experiment_protocol.py --full
```

## Key Hypotheses Tested

1. **Phase Transitions**: Do certain (model_size, search_depth, beam_width) combinations suddenly enable coherent reasoning?

2. **Power Law Scaling**: Does solve_rate follow S ∝ (M × D)^α where M = model_size, D = search_depth?

3. **State Tracking Criticality**: At what point does state tracking break down?

4. **Architecture Comparison**: Which System 2 architecture performs best?

5. **Hallucination Phase Transition**: Does hallucination rate show a sharp drop at critical model size?

## Expected Outputs

- `tot_comparison_results.json`: Tree of Thought strategy comparison
- `dspy_enhanced_results.json`: DSPy optimization results
- `system2_criticality_results.json`: Criticality experiment results
- `state_tracking_results.json`: State tracking benchmarks
- `advanced_system2_results.json`: Advanced architecture comparisons
- `system2_power_law_analysis.json`: Power law analysis
- `system2_comprehensive_results_*.json`: Combined results
- `system2_phase_diagrams.png`: Phase transition heatmaps
- `system2_power_laws.png`: Power law visualizations

## Integration with Power Law Thesis

These experiments connect System 2 reasoning to the broader power law thesis:

1. **Ising Model**: Shows phase transitions in physical systems
2. **Neural Scaling Laws**: Shows power law scaling in deep learning
3. **System 2 Reasoning**: Tests if reasoning shows similar criticality

The goal is to demonstrate that System 2 reasoning follows the same criticality principles, with phase transitions at critical resource combinations.

## Next Steps

1. **Expand Datasets**: Add more problem types (GSM8K, MATH, logic puzzles)
2. **Run Scaling Experiments**: Test across multiple model sizes and search depths
3. **Measure Critical Points**: Identify exact thresholds for phase transitions
4. **Compare Architectures**: Determine which System 2 architecture is most effective
5. **Publish Results**: Document findings showing System 2 criticality

## References

- "Tree of Thoughts: Deliberate Problem Solving with Large Language Models" (Yao et al., 2023)
- "Self-Consistency Improves Chain of Thought Reasoning" (Wang et al., 2023)
- "Let's Verify Step by Step" (Lightman et al., 2023)
- "DSPy: Compiling Declarative Language Model Calls" (Khattab et al., 2023)
