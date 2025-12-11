# Experiment Readiness Assessment

Based on `EXPERIMENT_MAPPING.md`, this document assesses what exists and what's needed to start each experiment.

**Last Updated**: December 2024  
**Status**: Gap Analysis Complete

---

## Summary

| Experiment | Status | Readiness | Missing Components |
|------------|--------|-----------|-------------------|
| **1. Agentic Scaling Laws** | 🟡 Partial | 60% | Multi-agent system (N agents), Tool system, Combined experiment runner |
| **2. Workflow Grammar** | 🔴 Not Started | 20% | WorkflowExtractor, Trace collection infrastructure |
| **3. Multi-Agent Criticality** | 🟡 Partial | 50% | Criticality experiment runner, Network topology, Order parameter extraction |
| **4. Compute-Optimal** | 🔴 Not Started | 30% | ComputeBudget system, ArchitectureComparison, Optimization framework |

**Overall Readiness**: ~40% - Core infrastructure exists, but experiment-specific implementations needed.

---

## Experiment 1: Agentic Scaling Laws

### Required Components (from EXPERIMENT_MAPPING.md)

```
├─ 1.1 A-Scaling: Vary number of agents {1,2,3,5,8,13}
├─ 1.2 T-Scaling: Vary interaction depth {1,3,5,10,20,50}
├─ 1.3 K-Scaling: Vary tool richness {0,1,2,3,5}
└─ 1.4 Combined: Full factorial (subset)
```

### ✅ What Exists

1. **DebateSystem** (`advanced_system2_architectures.py`)
   - ✅ `DebateAgent` class exists
   - ✅ `DebateSystem` class exists
   - ⚠️ **LIMITATION**: Only supports 2 agents (hardcoded)
   - ✅ Has judge mechanism for consensus

2. **T-Scaling Infrastructure** (`system2_criticality_experiment.py`)
   - ✅ `System2CriticalityExperiment` class
   - ✅ Supports varying search depth
   - ✅ `ExperimentConfig` dataclass
   - ✅ Power-law fitting utilities

3. **Power-Law Analysis** (`system2_power_law_analysis.py`)
   - ✅ `fit_power_law()` function
   - ✅ `find_critical_exponent()` function
   - ✅ Visualization framework

4. **Benchmarks** (`benchmarks/`)
   - ✅ `GameOf24Benchmark`
   - ✅ `ArithmeticChainBenchmark`
   - ✅ `LogicPuzzleBenchmark`
   - ✅ `TowerOfHanoiBenchmark`
   - ✅ `VariableTrackingBenchmark`

5. **Search Strategies** (`tree_of_thought_enhanced.py`)
   - ✅ Multiple search strategies (BFS, DFS, Beam, etc.)
   - ✅ Depth control
   - ✅ Metrics tracking

### ❌ What's Missing

1. **Multi-Agent System (N agents)**
   - ❌ `MultiAgentSystem` class (extend DebateSystem to N agents)
   - ❌ Coordination accuracy measurement
   - ❌ Consensus mechanism for N agents
   - ❌ Per-agent solution tracking

2. **Tool-Augmented Agent System**
   - ❌ `ToolAugmentedAgent` class
   - ❌ Tool registry (calculator, search, validator, planner, debugger)
   - ❌ Tool usage tracking
   - ❌ Tool effectiveness measurement

3. **Combined Experiment Runner**
   - ❌ `agentic_scaling_experiment.py` (main experiment file)
   - ❌ A-scaling experiment runner
   - ❌ K-scaling experiment runner
   - ❌ Combined factorial design runner

4. **Metrics Collection**
   - ❌ Coordination accuracy calculation for N agents
   - ❌ Tool usage frequency tracking
   - ❌ Combined scaling metrics aggregation

### 📋 Implementation Checklist

- [ ] Create `agentic_scaling_experiment.py`
- [ ] Extend `DebateSystem` → `MultiAgentSystem` (N agents)
- [ ] Implement `_measure_agreement()` for N agents
- [ ] Implement `_reach_consensus()` for N agents
- [ ] Create `ToolAugmentedAgent` class
- [ ] Implement tool registry and usage tracking
- [ ] Create A-scaling experiment function
- [ ] Create K-scaling experiment function
- [ ] Create combined experiment function
- [ ] Add coordination metrics to results
- [ ] Test with small A values (1, 2, 3)

### 🎯 Priority Actions

1. **HIGH**: Extend DebateSystem to N agents (needed for 1.1)
2. **HIGH**: Create ToolAugmentedAgent (needed for 1.3)
3. **MEDIUM**: Create agentic_scaling_experiment.py runner
4. **LOW**: Combined factorial design (can start with individual experiments)

---

## Experiment 2: Workflow Grammar Extraction

### Required Components (from EXPERIMENT_MAPPING.md)

```
├─ 2.1 Grammar Extraction: Parse traces → production rules
├─ 2.2 Skill Distribution: Analyze frequency distribution
├─ 2.3 Learning Curves: Track error vs episodes
└─ 2.4 Skill Importance: Ablation study
```

### ✅ What Exists

1. **Power-Law Analysis** (`system2_power_law_analysis.py`)
   - ✅ `fit_power_law()` for learning curves
   - ✅ Statistical analysis utilities

2. **Trace Collection** (from Experiment 1)
   - ⚠️ **DEPENDENCY**: Needs Experiment 1 to run first
   - ✅ Results saved to JSON (can parse traces)

3. **Action Types** (implicit in existing code)
   - ✅ Tool calls (from tool usage)
   - ✅ Reasoning steps (from ToT)
   - ✅ Validation (from verify-refine)

### ❌ What's Missing

1. **WorkflowExtractor Class**
   - ❌ `workflow_grammar_extractor.py` file
   - ❌ `WorkflowExtractor` class
   - ❌ Trace parsing logic
   - ❌ Production rule extraction
   - ❌ Grammar structure definition

2. **Trace Infrastructure**
   - ❌ Structured trace format
   - ❌ Trace collection in Experiment 1
   - ❌ Action type taxonomy
   - ❌ Step-by-step action logging

3. **Skill Analysis**
   - ❌ Skill frequency calculation
   - ❌ Zipf's law testing
   - ❌ Rare skill identification
   - ❌ Skill importance ranking

4. **Learning Curve Analysis**
   - ❌ Error tracking over episodes
   - ❌ Rare skill error attribution
   - ❌ Learning curve fitting

### 📋 Implementation Checklist

- [ ] Create `workflow_grammar_extractor.py`
- [ ] Define action taxonomy (tool_call, reasoning, validation, refinement)
- [ ] Implement trace parsing from Experiment 1 results
- [ ] Implement production rule extraction
- [ ] Create grammar structure (PCFG)
- [ ] Implement skill frequency analysis
- [ ] Implement Zipf's law test
- [ ] Implement learning curve extraction
- [ ] Implement skill importance (ablation) analysis
- [ ] Add trace collection to Experiment 1

### 🎯 Priority Actions

1. **HIGH**: Define action taxonomy and trace format
2. **HIGH**: Create WorkflowExtractor class
3. **MEDIUM**: Add trace collection to Experiment 1
4. **LOW**: Ablation study (can do after basic extraction works)

**Note**: Experiment 2 depends on Experiment 1 for traces. Can start with mock traces for testing.

---

## Experiment 3: Multi-Agent Criticality

### Required Components (from EXPERIMENT_MAPPING.md)

```
├─ 3.1 Communication Bandwidth: {0,1,3,5,10,20,50} messages
├─ 3.2 Network Topology: {FC, ring, star, tree, random}
├─ 3.3 Agent Reliability: {0%,10%,20%,30%,50%} failure rate
├─ 3.4 Prompt/LLM Capability: {gpt-3.5, gpt-4-mini, gpt-4}
└─ 3.5 Order Parameters: Extract φ_coord, τ_consensus, ψ_coherence
```

### ✅ What Exists

1. **CoordinationGame** (`game_theory_mas.py`)
   - ✅ `CoordinationGame` class
   - ✅ Supports N agents and M tasks
   - ✅ Task allocation mechanisms
   - ⚠️ **LIMITATION**: No communication bandwidth control
   - ⚠️ **LIMITATION**: No network topology support

2. **Criticality Detection** (`system2_criticality_experiment.py`)
   - ✅ `find_critical_points()` function
   - ✅ `plot_phase_diagrams()` function
   - ✅ Phase transition detection

3. **Model Variation** (`system2_criticality_experiment.py`)
   - ✅ Support for multiple models (gpt-3.5, gpt-4-mini, gpt-4)
   - ✅ Model size mapping

4. **Power-Law Analysis** (`system2_power_law_analysis.py`)
   - ✅ Critical exponent extraction
   - ✅ Power-law fitting near criticality

### ❌ What's Missing

1. **Criticality Experiment Runner**
   - ❌ `multi_agent_criticality.py` file
   - ❌ `CriticalityCoordinationExperiment` class
   - ❌ Communication bandwidth control
   - ❌ Message passing protocol

2. **Network Topology**
   - ❌ Network topology classes (FC, ring, star, tree, random)
   - ❌ Message routing based on topology
   - ❌ Connectivity measurement

3. **Agent Reliability**
   - ❌ Failure injection mechanism
   - ❌ Failure modes (no_response, wrong_answer, timeout)
   - ❌ Resilience measurement

4. **Order Parameters**
   - ❌ Coordination accuracy (φ_coord) calculation
   - ❌ Consensus lag (τ_consensus) measurement
   - ❌ Plan coherence (ψ_coherence) measurement
   - ❌ Critical exponent extraction (β, γ, δ)

5. **Communication Infrastructure**
   - ❌ Message budget tracking
   - ❌ Message passing between agents
   - ❌ Communication protocol

### 📋 Implementation Checklist

- [ ] Create `multi_agent_criticality.py`
- [ ] Extend `CoordinationGame` with communication budget
- [ ] Implement network topology classes
- [ ] Implement message passing protocol
- [ ] Implement failure injection
- [ ] Implement order parameter calculation
- [ ] Create bandwidth variation experiment
- [ ] Create topology variation experiment
- [ ] Create reliability variation experiment
- [ ] Implement critical exponent extraction
- [ ] Integrate with existing phase diagram plotting

### 🎯 Priority Actions

1. **HIGH**: Create multi_agent_criticality.py framework
2. **HIGH**: Implement communication bandwidth control
3. **MEDIUM**: Implement network topology support
4. **MEDIUM**: Implement order parameter calculation
5. **LOW**: Failure injection (can start with perfect agents)

---

## Experiment 4: Compute-Optimal Architecture

### Required Components (from EXPERIMENT_MAPPING.md)

```
├─ 4.1 Agent Size vs Count: (large,1) vs (medium,2) vs (small,5)
├─ 4.2 Depth vs Width: Vary (depth, width) pairs
├─ 4.3 Parallelism: {sequential, parallel, hybrid}
└─ 4.4 Full Optimization: All dimensions
```

### ✅ What Exists

1. **Agent Systems** (`advanced_system2_architectures.py`)
   - ✅ Multiple architectures (verify-refine, debate, memory)
   - ✅ Can use different models

2. **Search Strategies** (`tree_of_thought_enhanced.py`)
   - ✅ Depth control
   - ✅ Width control (beam width)
   - ✅ Parallel search support

3. **Model Variation**
   - ✅ Support for different model sizes
   - ✅ Model cost estimation (implicit)

4. **Metrics Collection**
   - ✅ Token usage tracking (approximate)
   - ✅ Time tracking
   - ✅ Solve rate measurement

### ❌ What's Missing

1. **Compute Budget System**
   - ❌ `ComputeBudget` class
   - ❌ Token cost tracking per agent/model
   - ❌ Budget constraint enforcement
   - ❌ Cost normalization

2. **Architecture Comparison**
   - ❌ `compute_optimal_architecture.py` file
   - ❌ `ArchitectureComparison` class
   - ❌ Size vs count comparison
   - ❌ Depth vs width comparison
   - ❌ Parallelism comparison

3. **Optimization Framework**
   - ❌ Multi-objective optimization (solve_rate, efficiency, latency)
   - ❌ Pareto frontier identification
   - ❌ Optimal configuration search

4. **Efficiency Metrics**
   - ❌ Compute efficiency calculation (solve_rate / cost)
   - ❌ Latency measurement
   - ❌ Resource utilization tracking

### 📋 Implementation Checklist

- [ ] Create `compute_optimal_architecture.py`
- [ ] Create `ComputeBudget` class
- [ ] Implement token cost estimation per model
- [ ] Implement size vs count comparison
- [ ] Implement depth vs width comparison
- [ ] Implement parallelism comparison (seq/parallel/hybrid)
- [ ] Implement efficiency calculation
- [ ] Implement multi-objective optimization
- [ ] Create full optimization surface
- [ ] Generate efficiency curves
- [ ] Create architecture recommendations

### 🎯 Priority Actions

1. **HIGH**: Create ComputeBudget system
2. **HIGH**: Create ArchitectureComparison class
3. **MEDIUM**: Implement individual comparisons (4.1, 4.2, 4.3)
4. **LOW**: Full 5D optimization (can start with 2D slices)

---

## Infrastructure Dependencies

### Shared Infrastructure (✅ Exists)

1. **Power-Law Analysis**
   - ✅ `fit_power_law()` - Used by all experiments
   - ✅ `find_critical_exponent()` - Used by Exps 1, 3
   - ✅ Visualization framework

2. **Benchmarks**
   - ✅ Game of 24
   - ✅ Arithmetic chains
   - ✅ Logic puzzles
   - ✅ Tower of Hanoi
   - ✅ Variable tracking

3. **Criticality Detection**
   - ✅ `find_critical_points()` - Used by Exp 3
   - ✅ `plot_phase_diagrams()` - Used by Exp 3

4. **Search Infrastructure**
   - ✅ Tree of Thought search
   - ✅ Multiple search strategies
   - ✅ Depth/width control

### Missing Shared Infrastructure

1. **Trace Collection**
   - ❌ Standardized trace format
   - ❌ Trace logging in all experiments
   - ❌ Trace storage and retrieval

2. **Result Storage**
   - ⚠️ JSON storage exists but needs standardization
   - ❌ Results directory structure (`results/agentic_scaling/`)
   - ❌ Result aggregation utilities

3. **Compute Tracking**
   - ⚠️ Approximate token tracking exists
   - ❌ Accurate cost calculation
   - ❌ Cost normalization across models

---

## Implementation Priority

### Phase 1: Foundation (Week 1-2) - **CRITICAL**

**Must Have**:
1. ✅ Extend DebateSystem → MultiAgentSystem (N agents)
2. ✅ Create ToolAugmentedAgent class
3. ✅ Create agentic_scaling_experiment.py (basic structure)
4. ✅ Add trace collection to experiments

**Can Defer**:
- Full factorial design
- Advanced network topologies
- Full optimization surface

### Phase 2: Core Experiments (Week 3-4)

**Must Have**:
1. ✅ A-scaling experiment (1.1)
2. ✅ T-scaling experiment (1.2) - mostly exists
3. ✅ K-scaling experiment (1.3)
4. ✅ Basic workflow extraction (2.1)

**Can Defer**:
- Combined scaling (1.4)
- Skill importance ablation (2.4)
- Full criticality suite (3.1-3.5)

### Phase 3: Advanced Experiments (Week 5-7)

**Must Have**:
1. ✅ Communication bandwidth experiments (3.1)
2. ✅ Basic compute-optimal comparisons (4.1, 4.2)

**Can Defer**:
- Network topology experiments (3.2)
- Full optimization (4.4)

---

## Quick Start Recommendations

### To Start Experiment 1.1 (A-Scaling):

1. **Extend DebateSystem** (30 min):
   ```python
   # In advanced_system2_architectures.py
   class MultiAgentSystem:
       def __init__(self, n_agents: int, model: str = "gpt-4o"):
           self.agents = [DebateAgent(f"Agent_{i}", model) 
                         for i in range(n_agents)]
   ```

2. **Create A-scaling runner** (1 hour):
   ```python
   # In agentic_scaling_experiment.py
   def run_a_scaling_experiment():
       A_values = [1, 2, 3, 5, 8]
       # ... implement
   ```

3. **Test with small A** (15 min):
   - Run with A=1, A=2, A=3
   - Verify coordination metrics work

### To Start Experiment 2.1 (Workflow Extraction):

1. **Define trace format** (30 min):
   ```python
   trace = [
       {'step': 1, 'action': 'tool_call', 'tool': 'calculator', ...},
       {'step': 2, 'action': 'reasoning', 'type': 'decompose', ...},
   ]
   ```

2. **Create WorkflowExtractor** (1 hour):
   ```python
   # In workflow_grammar_extractor.py
   class WorkflowExtractor:
       def extract_from_trace(self, trace):
           # Parse trace → production rules
   ```

3. **Test with mock traces** (15 min):
   - Create sample traces
   - Verify extraction works

---

## Files to Create

### High Priority (Week 1-2)

1. **`agentic_scaling_experiment.py`** - Experiment 1 runner
2. **`workflow_grammar_extractor.py`** - Experiment 2 infrastructure
3. **`multi_agent_criticality.py`** - Experiment 3 runner
4. **`compute_optimal_architecture.py`** - Experiment 4 runner

### Medium Priority (Week 3-4)

5. **`tool_augmented_agent.py`** - Tool system for K-scaling
6. **`network_topology.py`** - Network structures for Exp 3
7. **`trace_collector.py`** - Shared trace collection utility

### Low Priority (Week 5+)

8. **`result_aggregator.py`** - Result analysis utilities
9. **`optimization_framework.py`** - Multi-objective optimization

---

## Testing Strategy

### Unit Tests Needed

- [ ] MultiAgentSystem coordination accuracy
- [ ] ToolAugmentedAgent tool usage
- [ ] WorkflowExtractor grammar extraction
- [ ] ComputeBudget cost tracking
- [ ] Order parameter calculations

### Integration Tests Needed

- [ ] A-scaling experiment end-to-end
- [ ] Workflow extraction from real traces
- [ ] Criticality experiment with bandwidth
- [ ] Compute-optimal comparison

### Validation Tests

- [ ] Power-law fits have R² > 0.8
- [ ] Critical points are detected correctly
- [ ] Coordination metrics are meaningful
- [ ] Cost calculations are accurate

---

## Risk Assessment

### High Risk (Address First)

1. **Multi-agent coordination may not show clear scaling**
   - **Mitigation**: Start with simple tasks, use clear metrics
   - **Test**: Run A=1,2,3 first, verify trends

2. **Tool system may be complex to implement**
   - **Mitigation**: Start with 1-2 simple tools, expand later
   - **Test**: Calculator tool first, then add search

3. **Trace collection may be incomplete**
   - **Mitigation**: Add trace logging early, validate format
   - **Test**: Check traces from first experiments

### Medium Risk

4. **Criticality experiments may not show phase transitions**
   - **Mitigation**: Use fine-grained parameter sweeps
   - **Test**: Start with known critical regions

5. **Compute tracking may be inaccurate**
   - **Mitigation**: Use API response metadata, validate estimates
   - **Test**: Compare estimated vs actual costs

---

## Next Steps

1. **Immediate** (Today):
   - [ ] Review this assessment
   - [ ] Prioritize which experiment to start
   - [ ] Create first missing file (likely `agentic_scaling_experiment.py`)

2. **This Week**:
   - [ ] Implement MultiAgentSystem extension
   - [ ] Create basic A-scaling experiment
   - [ ] Test with small A values

3. **Next Week**:
   - [ ] Add ToolAugmentedAgent
   - [ ] Create K-scaling experiment
   - [ ] Start workflow extraction

---

**Status**: Ready to begin implementation  
**Recommended Starting Point**: Experiment 1.1 (A-Scaling) - simplest to implement, builds on existing infrastructure
