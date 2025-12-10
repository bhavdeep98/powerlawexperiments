# Scaling Laws and Critical Phenomena in Agentic Systems

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **A Unified Framework for Model–Data–Interaction Scaling**

This repository implements a comprehensive research program that extends neural scaling laws to **agentic systems**, connecting scaling laws, hierarchical compositional learning, multi-agent criticality, and compute-optimal agent design. The work explicitly **complements** and **extends** the framework in *Scaling Agents* (arXiv:2512.08296).

## 🎯 Research Overview

### Central Question

> **How do scaling laws and critical phenomena manifest in systems of interacting AI agents, and what principles govern the optimal allocation of compute across agent architectures?**

While prior work has established power-law relationships for single models (Kaplan et al., 2020), hierarchical compositional data (Cagnetta et al., 2024), and multi-agent benchmarking (Scaling Agents, 2024), we develop a **unified theoretical framework** that:

1. **Generalizes scaling laws** from single models to agentic systems
2. **Models agent workflows** as hierarchical grammars with power-law skill usage
3. **Analyzes coordination** using criticality theory from statistical physics
4. **Proposes compute-optimal** agent architecture design principles

### Four Core Contributions

#### 1. **Agentic Scaling Laws** 
Extend single-model scaling laws to three system-level scaling variables:
- **A**: Number of agents
- **T**: Interaction depth/episodes
- **K**: Environment/tool richness

**Prediction**: $\mathcal{E}(A, T, K) \sim A^{-\beta_A} T^{-\beta_T} K^{-\beta_K}$

#### 2. **Agent Workflows as Hierarchical Grammars**
Map agent actions, tools, and sub-tasks to production rules in a PCFG. Show that heavy-tailed skill frequency → power-law learning curves.

**Prediction**: Rare skills dominate errors → error ∝ episodes^-γ

#### 3. **Criticality in Multi-Agent Coordination**
Model coordination breakdown/emergence as critical phenomena. Identify phase transitions when varying communication bandwidth, network topology, and agent reliability.

**Prediction**: Above connectivity threshold → reliable coordination; below → fragmentation

#### 4. **Compute-Optimal Agent Architecture**
Analogous to Chinchilla-style optimality, find optimal allocation between:
- Fewer large agents vs many small ones
- Shallow vs deep reasoning
- Parallel vs sequential workflows

**Prediction**: Performance ∝ Compute^α (with optimal architecture)

## 📊 Experimental Program

### Experiment 1: Agentic Scaling Laws

Tests power-law scaling with respect to A, T, and K:

- **A-Scaling**: Vary number of agents {1, 2, 3, 5, 8, 13}
- **T-Scaling**: Vary interaction depth {1, 3, 5, 10, 20, 50}
- **K-Scaling**: Vary tool richness {0, 1, 2, 3, 5}
- **Combined**: Full factorial design (subset)

**Expected Output**: Scaling exponents (β_A, β_T, β_K), critical thresholds (A*, T*, K*)

### Experiment 2: Workflow Grammar Extraction

Extracts agent action sequences as production rules, analyzes skill distributions:

- **Grammar Extraction**: Parse traces → PCFG structure
- **Skill Distribution**: Test Zipf's law (frequency ∝ rank^-α)
- **Learning Curves**: Link rare skills to power-law improvement
- **Skill Importance**: Ablation study to rank skills

**Expected Output**: Grammar structure, Zipf exponent (α), learning exponent (γ)

### Experiment 3: Multi-Agent Criticality

Tests phase transitions in coordination:

- **Communication Bandwidth**: Vary message budget {0, 1, 3, 5, 10, 20, 50}
- **Network Topology**: Test {fully-connected, ring, star, tree, random}
- **Agent Reliability**: Vary failure rate {0%, 10%, 20%, 30%, 50%}
- **Order Parameters**: Measure coordination accuracy, consensus lag, plan coherence

**Expected Output**: Phase diagrams, critical points, critical exponents (β, γ, δ)

### Experiment 4: Compute-Optimal Architecture

Finds optimal agent architecture under compute constraints:

- **Size vs Count**: Compare (large,1) vs (medium,2) vs (small,5)
- **Depth vs Width**: Vary reasoning depth and parallel branches
- **Parallelism**: Compare sequential vs parallel vs hybrid workflows
- **Full Optimization**: 5D optimization surface

**Expected Output**: Optimal configurations, efficiency curves, architecture recommendations

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd powerlawexperiments

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Set Up API Keys

```bash
export OPENAI_API_KEY=your_key_here
```

### Running Experiments

#### Quick Start: Run All Experiments

```bash
# Run comprehensive System 2 experiments
python run_system2_experiments.py

# Run individual experiment components
python system2_criticality_experiment.py  # Experiment 1 & 3
python workflow_grammar_extractor.py      # Experiment 2 (when implemented)
python compute_optimal_architecture.py    # Experiment 4 (when implemented)
```

#### Analyze Results

```bash
# Power law analysis
python system2_power_law_analysis.py results/system2/scaling_results.json

# View results
python view_results.py
```

### Example: Agentic Scaling Experiment

```python
from agentic_scaling_experiment import MultiAgentSystem
from benchmarks import GameOf24Benchmark

# Create multi-agent system
system = MultiAgentSystem(n_agents=5, model="gpt-4o")

# Get benchmark problems
benchmark = GameOf24Benchmark()
problems = benchmark.get_problems(difficulty=3, num_problems=10)

# Run experiment
for problem in problems:
    result = system.solve_with_coordination(problem.problem_text)
    print(f"Coordination accuracy: {result['coordination_accuracy']:.2f}")
```

## 📁 Project Structure

```
powerlawexperiments/
├── README.md                          # This file
├── EXPERIMENTAL_PLAN.md              # Detailed experimental design
├── EXPERIMENT_QUICK_START.md         # Implementation guide
├── RESEARCH_FRAMEWORK_SUMMARY.md     # High-level overview
├── EXPERIMENT_MAPPING.md             # Theory-to-experiment mapping
│
├── # Core Infrastructure
├── system2_criticality_experiment.py # Criticality & scaling experiments
├── system2_power_law_analysis.py     # Power law fitting & analysis
├── tree_of_thought_enhanced.py       # Search strategies
├── advanced_system2_architectures.py  # Multi-agent systems
├── game_theory_mas.py                 # Coordination games
│
├── # Benchmarks
├── benchmarks/
│   ├── base.py                        # Base benchmark class
│   ├── game_of_24.py                  # Game of 24 problems
│   ├── arithmetic_chains.py           # Multi-step arithmetic
│   ├── logic_puzzles.py                # Logic puzzles
│   ├── tower_of_hanoi.py              # Tower of Hanoi
│   └── variable_tracking.py          # State tracking tasks
│
├── # Experiment Runners
├── run_system2_experiments.py         # Main experiment runner
├── run_experiments.py                 # Legacy experiments (Ising, etc.)
│
├── # Analysis & Visualization
├── analyze_results.py                 # Result analysis
├── view_results.py                    # Result viewer
├── generate_plots.py                  # Plot generation
│
└── results/                           # Output directory
    ├── system2/                       # System 2 experiment results
    ├── agentic_scaling/                # Agentic scaling results (new)
    └── ...
```

## 🔬 Theoretical Framework

### Principle 1: Heavy-Tailed Structure Generates Power Laws

Just like grammar rules in hierarchical compositional learning:
- High-frequency "primitive skills" (search, recall, simple retrieval)
- Long tail of rare "expert skills" (planning, debugging, complex tool use)
- Error dominated by rare skills ⇒ power-law improvement with episodes

### Principle 2: Interaction Graphs Induce Critical Dynamics

Inspired by statistical physics (Ising-like behavior):
- Above connectivity threshold → agents coordinate reliably
- Below threshold → behavior fragments into uncoordinated clusters
- Near criticality → perturbations propagate as power-law cascades

### Principle 3: System-Level Scaling Mirrors Model-Level Scaling

Extend Kaplan's formulation:
$$L(N, D, C) \sim N^{-\alpha_N} D^{-\alpha_D} C^{-\alpha_C}$$

to agent systems:
$$\mathcal{E}(A, T, K) \sim A^{-\beta_A} T^{-\beta_T} K^{-\beta_K}$$

Where:
- **A**: Diversity/capacity of distributed reasoning
- **T**: Experience or iteration depth
- **K**: Environment affordances

## 📈 Key Results (Preliminary)

### System 2 Scaling Laws

From existing experiments, we've confirmed:
- **Power-law scaling**: Solve rate ∝ (Model × Depth)^0.205
- **Critical threshold**: ~4.0 compute units for reliable complex reasoning
- **State tracking**: Identified as key bottleneck for long-horizon tasks

![System 2 Power Laws](system2_power_laws.png)

### Multi-Agent Coordination

From game theory experiments:
- **Coordination games**: GPT-3.5 and GPT-4o achieve 100% success in Schelling Point games
- **Emergent cooperation**: Iterated Prisoner's Dilemma shows cooperation emergence
- **Task allocation**: Nash equilibrium enables optimal multi-agent coordination

### Critical Phenomena

From Ising model experiments:
- **Phase transition**: Confirmed at T ≈ 2.25 (Theory: T_c ≈ 2.269)
- **Power-law correlations**: ξ ∝ |τ|^-ν near criticality
- **Implication**: Critical resource levels enable system-wide coherent processing

## 🎓 How This Complements Scaling Agents (2512.08296)

**Scaling Agents provides**:
- Agent benchmarks and evaluation pipelines
- Initial datasets and task formulations
- Architectural motifs (routing, tool use, modular agents)

**We add**:
- **Scaling laws** for agent count, interaction depth, and environment richness
- **Theoretical grounding** for learning curves using hierarchical generative structure
- **Critical phenomena interpretation** for coordination failures
- **Compute-optimal design rules** for choosing agent architectures

**We do NOT modify**:
- Existing benchmarks or evaluation protocols
- Task formulations or architectural patterns
- Core agent implementations

Instead, we provide a **theory + scaling framework** that naturally overlays existing agent benchmarks.

## 📚 References

### Scaling Laws
- Kaplan, J., et al. (2020). *Scaling Laws for Neural Language Models*. arXiv:2001.08361
- Hoffmann, J., et al. (2022). *Training Compute-Optimal Large Language Models*. arXiv:2203.15556
- Cagnetta, F., et al. (2024). *Power-law learning curves from heavy-tailed generative structure*. arXiv:2505.07067

### Agentic Systems
- Scaling Agents (2024). *Scaling Agents: A Framework for Benchmarking Agentic Systems*. arXiv:2512.08296
- Yao, L., et al. (2023). *Tree of Thoughts: Deliberate Problem Solving with Large Language Models*
- Lightman, H., et al. (2023). *Let's Verify Step by Step*

### Critical Phenomena
- Ising, E. (1925). *Beitrag zur Theorie des Ferromagnetismus*
- Metropolis, N., et al. (1953). *Equation of State Calculations by Fast Computing Machines*

### Multi-Agent Systems
- Nash, J. (1950). *Equilibrium Points in N-person Games*
- Axelrod, R. (1984). *The Evolution of Cooperation*

## 🔧 Requirements

- Python 3.8+
- OpenAI API key (for LLM experiments)
- NumPy >= 1.24.0
- Matplotlib >= 3.7.0
- scipy >= 1.10.0
- openai >= 1.0.0

See `requirements.txt` for complete list.

## 📋 Experimental Status

### ✅ Completed
- System 2 criticality framework
- Power law analysis infrastructure
- Multi-agent architectures (debate, verify-refine, memory-augmented)
- State tracking benchmarks
- Tree of Thought search strategies

### 🚧 In Progress
- Agentic scaling experiments (A, T, K)
- Workflow grammar extraction
- Multi-agent criticality experiments
- Compute-optimal architecture search

### 📝 Planned
- Full factorial scaling experiments
- Critical exponent measurements
- Architecture optimization
- Paper preparation

See `EXPERIMENTAL_PLAN.md` for detailed roadmap.

## 🤝 Contributing

This is a research project. Contributions welcome! Please:
1. Review `EXPERIMENTAL_PLAN.md` for research direction
2. Check `EXPERIMENT_QUICK_START.md` for implementation guidelines
3. Open an issue to discuss proposed changes
4. Follow existing code style and documentation patterns

## 📜 License

MIT License - See LICENSE file for details.

## 🙏 Acknowledgments

This work builds on:
- The scaling laws framework from Kaplan et al. (2020)
- Hierarchical compositional learning from Cagnetta et al. (2024)
- The agent benchmarking framework from Scaling Agents (2024)
- Critical phenomena theory from statistical physics

---

**Status**: Active Research Project  
**Last Updated**: December 2024  
**Paper**: In Preparation

For detailed experimental design, see [`EXPERIMENTAL_PLAN.md`](EXPERIMENTAL_PLAN.md).  
For quick implementation guide, see [`EXPERIMENT_QUICK_START.md`](EXPERIMENT_QUICK_START.md).
