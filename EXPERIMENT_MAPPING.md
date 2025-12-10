# Experiment Mapping: Theory → Experiments

This document provides a visual mapping of how each theoretical contribution maps to specific experiments.

---

## Contribution 1: Agentic Scaling Laws

```
Theory: E(A, T, K) ~ A^(-β_A) T^(-β_T) K^(-β_K)
         ↓
Experiments:
├─ 1.1 A-Scaling: Vary number of agents {1,2,3,5,8,13}
│   └─ Measure: solve_rate, coordination_accuracy
│   └─ Output: β_A, A* (critical threshold)
│
├─ 1.2 T-Scaling: Vary interaction depth {1,3,5,10,20,50}
│   └─ Measure: solve_rate, search_efficiency
│   └─ Output: β_T, T*
│
├─ 1.3 K-Scaling: Vary tool richness {0,1,2,3,5}
│   └─ Measure: solve_rate, tool_effectiveness
│   └─ Output: β_K, K*
│
└─ 1.4 Combined: Full factorial (subset)
    └─ Measure: solve_rate, efficiency
    └─ Output: Multi-variate power-law fit
```

**Implementation**:
- Extend `DebateSystem` → N agents
- Create `ToolAugmentedAgent` → K tools
- Use existing `System2CriticalityExperiment` → T depth

---

## Contribution 2: Agent Workflows as Hierarchical Grammars

```
Theory: Heavy-tailed skill frequency → Power-law learning curves
         ↓
Experiments:
├─ 2.1 Grammar Extraction: Parse agent traces → production rules
│   └─ Output: PCFG structure, rule frequencies
│
├─ 2.2 Skill Distribution: Analyze frequency distribution
│   └─ Test: Zipf's law (frequency ∝ rank^-α)
│   └─ Output: α (Zipf exponent), rare skill identification
│
├─ 2.3 Learning Curves: Track error vs episodes
│   └─ Test: error ∝ episodes^-γ (power-law)
│   └─ Output: γ (learning exponent), rare skill error contribution
│
└─ 2.4 Skill Importance: Ablation study
    └─ Output: Skill importance ranking
```

**Implementation**:
- Create `WorkflowExtractor` → parse traces
- Use existing trace collection from Experiment 1
- Analyze with `fit_power_law()` from existing code

---

## Contribution 3: Criticality in Multi-Agent Coordination

```
Theory: Phase transitions at critical connectivity thresholds
         ↓
Experiments:
├─ 3.1 Communication Bandwidth: {0,1,3,5,10,20,50} messages
│   └─ Measure: coordination_accuracy, consensus_time
│   └─ Output: Critical bandwidth b*, phase diagram
│
├─ 3.2 Network Topology: {FC, ring, star, tree, random}
│   └─ Measure: coordination_accuracy, propagation_speed
│   └─ Output: Optimal topology, critical connectivity
│
├─ 3.3 Agent Reliability: {0%,10%,20%,30%,50%} failure rate
│   └─ Measure: solve_rate, resilience
│   └─ Output: Critical failure threshold
│
├─ 3.4 Prompt/LLM Capability: {gpt-3.5, gpt-4-mini, gpt-4}
│   └─ Measure: coordination_accuracy
│   └─ Output: Critical capability threshold
│
└─ 3.5 Order Parameters: Extract φ_coord, τ_consensus, ψ_coherence
    └─ Measure: Critical exponents (β, γ, δ)
    └─ Output: Critical exponent values, universality class
```

**Implementation**:
- Extend `CoordinationGame` from `game_theory_mas.py`
- Create `CriticalityCoordinationExperiment`
- Use existing `find_critical_points()` for detection

---

## Contribution 4: Compute-Optimal Agent Architecture

```
Theory: Optimal allocation under compute budget
         ↓
Experiments:
├─ 4.1 Agent Size vs Count: (large,1) vs (medium,2) vs (small,5)
│   └─ Measure: solve_rate, efficiency
│   └─ Output: Optimal (size, count) configuration
│
├─ 4.2 Depth vs Width: Vary (depth, width) pairs
│   └─ Measure: solve_rate, efficiency
│   └─ Output: Optimal (depth, width) ratio
│
├─ 4.3 Parallelism: {sequential, parallel, hybrid}
│   └─ Measure: solve_rate, latency, coordination_cost
│   └─ Output: Optimal workflow pattern
│
└─ 4.4 Full Optimization: All dimensions
    └─ Measure: solve_rate, efficiency, latency
    └─ Output: Optimal 5D surface, recommendations
```

**Implementation**:
- Create `ComputeBudget` constraint system
- Create `ArchitectureComparison` class
- Use existing agent systems with compute tracking

---

## Data Flow

```
Existing Infrastructure
├─ system2_power_law_analysis.py
│   └─ fit_power_law() → Used by all experiments
│   └─ find_critical_exponent() → Used by Exps 1, 3
│
├─ system2_criticality_experiment.py
│   └─ find_critical_points() → Used by Exp 3
│   └─ plot_phase_diagrams() → Used by Exp 3
│
├─ advanced_system2_architectures.py
│   └─ DebateSystem → Extend for Exp 1.1 (A-scaling)
│
├─ game_theory_mas.py
│   └─ CoordinationGame → Extend for Exp 3
│
└─ tree_of_thought_enhanced.py
    └─ Search strategies → Used by Exp 1.2 (T-scaling)
         ↓
New Experiments
├─ Experiment 1 → agentic_scaling_experiment.py
├─ Experiment 2 → workflow_grammar_extractor.py
├─ Experiment 3 → multi_agent_criticality.py
└─ Experiment 4 → compute_optimal_architecture.py
         ↓
Results & Analysis
├─ results/agentic_scaling/
│   ├─ exp1_scaling_results.json
│   ├─ exp2_workflow_results.json
│   ├─ exp3_criticality_results.json
│   └─ exp4_compute_optimal_results.json
└─ Analysis
    ├─ Power-law fits (β_A, β_T, β_K)
    ├─ Critical points (A*, T*, K*, b*)
    ├─ Zipf exponents (α)
    ├─ Learning exponents (γ)
    └─ Optimal configurations
```

---

## Dependencies

```
Experiment 1 (Scaling Laws)
├─ Requires: Multi-agent system, tool system
├─ Uses: Existing benchmarks, power-law analysis
└─ Produces: Scaling exponents, critical thresholds

Experiment 2 (Workflows)
├─ Requires: Agent traces from Experiment 1
├─ Uses: Trace parsing, frequency analysis
└─ Produces: Grammar, skill distributions, learning curves

Experiment 3 (Criticality)
├─ Requires: Multi-agent coordination system
├─ Uses: Existing coordination games, criticality detection
└─ Produces: Phase diagrams, critical exponents

Experiment 4 (Compute-Optimal)
├─ Requires: All agent architectures
├─ Uses: Compute tracking, efficiency measurement
└─ Produces: Optimal configurations, efficiency curves
```

---

## Timeline

```
Week 1-2: Infrastructure
├─ Extend DebateSystem (N agents)
├─ Create ToolAugmentedAgent
├─ Create WorkflowExtractor
└─ Add network topology support
         ↓
Week 3-4: Experiments 1 & 2
├─ Run A, T, K scaling
├─ Extract workflows
└─ Analyze skill distributions
         ↓
Week 5-6: Experiment 3
├─ Run bandwidth/topology/reliability experiments
└─ Extract order parameters
         ↓
Week 7: Experiment 4
├─ Run architecture comparisons
└─ Find optimal configurations
         ↓
Week 8: Analysis
├─ Integrate results
├─ Generate visualizations
└─ Prepare paper
```

---

## Key Metrics Summary

| Metric | Experiment | Purpose |
|--------|------------|---------|
| **solve_rate** | All | Primary performance metric |
| **β_A, β_T, β_K** | 1 | Scaling exponents |
| **A*, T*, K*** | 1 | Critical thresholds |
| **α (Zipf)** | 2 | Skill distribution exponent |
| **γ (Learning)** | 2 | Learning curve exponent |
| **φ_coord** | 3 | Coordination order parameter |
| **b*** | 3 | Critical bandwidth |
| **Efficiency** | 4 | solve_rate / compute_cost |
| **Optimal config** | 4 | Best (size, count, depth, width) |

---

## Success Validation

For each experiment, validate:

1. **Statistical Significance**: R² > 0.8 for power-law fits
2. **Critical Points**: Clear phase transitions or thresholds
3. **Universality**: Exponents consistent across tasks
4. **Reproducibility**: Results stable across replications

---

**Version**: 1.0  
**Purpose**: Quick reference for experiment-theory mapping
