# Test Sets Used in Experiments

This document details the test sets and benchmarks used across all experiments.

---

## Overview

The experiments use different test sets depending on the experiment type:

1. **System 2 Reasoning**: Custom benchmarks with synthetic problems
2. **Neural Scaling**: Autoencoder reconstruction tasks (synthetic data)
3. **Ising Model**: Simulated physics (no traditional test set)
4. **Game Theory**: Synthetic game scenarios

---

## 1. System 2 Reasoning Benchmarks

### A. Baseline Evaluation Test Set

**Location**: `run_system2_experiments.py` → `run_baseline_evaluation()`

**Test Set Size**: 5 problems per benchmark (25 total)

**Benchmarks Used**:

1. **Game of 24** (5 problems)
   - Randomly sampled from all difficulty levels
   - Format: "Use numbers X Y Z W and arithmetic operations to obtain 24"
   - Example: "1 2 3 4", "2 2 2 3", etc.

2. **Arithmetic Chains** (5 problems)
   - Randomly sampled from different step counts (5, 10, 15, 20 steps)
   - Format: "Calculate: X + Y - Z * W = ?"
   - Example: "15 + 36 - 46 + 18 + 31 + 23"

3. **Tower of Hanoi** (5 problems)
   - Randomly sampled from 3-7 disk configurations
   - Format: "Solve Tower of Hanoi with N disks"

4. **Variable Tracking** (5 problems)
   - Randomly sampled from different complexity levels
   - Format: Track N variables through M operations
   - Example: Track x1, x2 through 5 operations

5. **Logic Puzzles** (5 problems)
   - Randomly sampled logic problems
   - Format: Various logical reasoning questions

**Note**: These are randomly sampled each run, so the exact problems may vary.

---

### B. Scaling Experiments Test Set

**Location**: `run_system2_experiments.py` → `run_scaling_experiments()`

**Test Set**: **10 hard Game of 24 problems** (difficulty level 3)

**Actual Problems Used** (from results):
```
1. "4 9 10 13"
2. "5 5 5 11"
3. "3 3 7 7"
4. "1 5 5 5"
5. "2 7 7 10"
6. "1 3 4 6"
7. "2 5 5 10"
8. "3 3 3 8"
9. "1 1 5 5"
10. "2 2 2 9"
```

**Configuration**:
- Models: `gpt-4o-mini`, `gpt-4o`
- Search Depths: `[1, 3, 5, 10]`
- Beam Widths: `[1, 3]`
- Branching Factor: `3`
- Samples per config: `1`

**Total Test Runs**: 10 problems × 2 models × 4 depths × 2 beam widths × 2 strategies = **320 test runs**

---

### C. Architecture Comparison Test Set

**Location**: `run_system2_experiments.py` → `run_architecture_comparison()`

**Test Set**: **5 hard Game of 24 problems** (difficulty level 3)

**Architectures Tested**:
1. Verify-Refine
2. Debate
3. Memory-Augmented

---

### D. State Tracking Test Set

**Location**: `state_tracking_benchmarks.py`

**Test Sets**:

1. **Stack Tracking**
   - Step counts: 5, 10, 15, 20
   - Tracks stack operations (push/pop)

2. **Variable Tracking**
   - Step counts: 5, 10, 15, 20
   - Tracks variable values through operations

3. **Counter Tracking**
   - Step counts: 5, 10, 15, 20
   - Tracks counter increments/decrements

**Note**: Results show 0% accuracy across all tasks, suggesting these are very challenging.

---

## 2. Game of 24 Full Benchmark

**Location**: `benchmarks/game_of_24.py`

**Total Problems**: **125 problems** across 5 difficulty levels

### Difficulty Distribution:

- **Difficulty 1 (Easy)**: 25 problems
  - Examples: "1 2 3 4", "2 2 2 3", "3 3 3 3", "4 4 4 4", "5 5 5 1"

- **Difficulty 2 (Medium)**: 25 problems
  - Examples: "1 2 4 6", "3 3 8 8", "10 10 4 4", "2 3 5 7"

- **Difficulty 3 (Hard)**: 25 problems
  - Examples: "4 9 10 13", "5 5 5 11", "3 3 7 7", "1 5 5 5"
  - **This is the set used in scaling experiments**

- **Difficulty 4 (Very Hard)**: 25 problems
  - Examples: "1 1 1 8", "2 2 3 9", "3 3 4 10"

- **Difficulty 5 (Expert)**: 25 problems
  - Examples: "1 1 2 10", "2 2 4 12", "3 3 6 14"

**Note**: Problems are generated deterministically, so the same problems are used across runs.

---

## 3. Arithmetic Chains Benchmark

**Location**: `benchmarks/arithmetic_chains.py`

**Total Problems**: **80 problems**

**Distribution**:
- 5-step chains: 20 problems (difficulty 1)
- 10-step chains: 20 problems (difficulty 2)
- 15-step chains: 20 problems (difficulty 3)
- 20-step chains: 20 problems (difficulty 4)

**Format**: Multi-step arithmetic calculations
- Example: "15 + 36 - 46 + 18 + 31 + 23 = ?"

**Note**: Problems are randomly generated, so they may vary between runs.

---

## 4. Tower of Hanoi Benchmark

**Location**: `benchmarks/tower_of_hanoi.py`

**Total Problems**: **50 problems**

**Distribution**:
- 3 disks: 10 problems
- 4 disks: 10 problems
- 5 disks: 10 problems
- 6 disks: 10 problems
- 7 disks: 10 problems

**Format**: "Solve Tower of Hanoi with N disks, moving from peg A to peg C"

---

## 5. Variable Tracking Benchmark

**Location**: `benchmarks/variable_tracking.py`

**Total Problems**: **100 problems**

**Distribution**:
- 2 variables, 5 operations: 20 problems (difficulty 1)
- 3 variables, 10 operations: 20 problems (difficulty 2)
- 4 variables, 15 operations: 20 problems (difficulty 3)
- 5 variables, 20 operations: 20 problems (difficulty 4)
- 6 variables, 25 operations: 20 problems (difficulty 5)

**Format**: Track N variables through M sequential operations
- Example: Track x1, x2 through operations like "x1 = x1 + x2", "x2++", etc.

---

## 6. Neural Scaling Test Set

**Location**: `autoencoder_scaling.py`

**Test Set**: **Synthetic autoencoder reconstruction task**

**Configuration**:
- Input dimension: 12
- Latent dimensions tested: `[1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16]`
- Training samples: 10,000
- **Test samples: 2,000**
- Epochs: 50
- Batch size: 64
- Trials: 3 (for averaging)

**Task**: Reconstruct input from latent representation

**Note**: This is a synthetic dataset, not a traditional benchmark.

---

## 7. Game Theory Test Sets

**Location**: `game_theory_mas.py`

### A. Adversarial Game (Coder-Critic)
- **Test Set**: Synthetic prompt refinement tasks
- **Format**: Coder generates prompts, Critic evaluates them
- **Size**: Multiple rounds per experiment

### B. Coordination Game
- **Test Set**: Synthetic coordination scenarios
- **Format**: Two agents must coordinate on meeting location
- **Size**: Multiple rounds with different strategies

### C. Iterated Prisoner's Dilemma
- **Test Set**: Tournament format
- **Format**: Multiple strategies compete in repeated games
- **Size**: All strategies play against all others

---

## 8. Ising Model

**Location**: `ising_model.py`

**Test Set**: **N/A** (simulation, not a traditional test set)

**Configuration**:
- Lattice size: 20×20
- Temperature range: 1.5 to 3.5 (21 points)
- Steps per temperature: 5,000
- Warmup steps: 1,000

**Note**: This is a physics simulation, not a machine learning benchmark.

---

## Summary Table

| Experiment | Test Set Type | Size | Source |
|------------|---------------|------|--------|
| **System 2 Baseline** | Mixed benchmarks | 25 problems (5 each) | Synthetic, randomly sampled |
| **System 2 Scaling** | Game of 24 (hard) | 10 problems | `benchmarks/game_of_24.py` (difficulty 3) |
| **System 2 Architecture** | Game of 24 (hard) | 5 problems | `benchmarks/game_of_24.py` (difficulty 3) |
| **System 2 State Tracking** | Stack/Variable/Counter | 3 tasks × 4 step counts | `state_tracking_benchmarks.py` |
| **Neural Scaling** | Autoencoder reconstruction | 2,000 test samples | Synthetic (generated) |
| **Game Theory** | Various game scenarios | Varies by experiment | Synthetic |
| **Ising Model** | N/A (simulation) | 21 temperature points | Physics simulation |

---

## Key Points

1. **No Standard Benchmarks**: Most test sets are synthetic/generated, not from standard ML benchmarks (like GLUE, SuperGLUE, etc.)

2. **Game of 24 is Primary**: The scaling experiments primarily use Game of 24 problems, specifically difficulty level 3 (hard).

3. **Random Sampling**: Baseline evaluation uses random sampling, so exact problems may vary between runs.

4. **Deterministic Generation**: Game of 24 problems are generated deterministically, so the same problems appear across runs.

5. **Small Test Sets**: Most experiments use relatively small test sets (5-10 problems) for cost/time reasons.

6. **No Train/Test Split**: These are zero-shot or few-shot experiments, not traditional train/test splits.

---

## Recommendations

If you want to:
- **Reproduce results**: Use the same difficulty levels and problem counts
- **Extend experiments**: Add more problems from the full benchmark sets
- **Compare to baselines**: Note that these are custom benchmarks, not standard ones
- **Validate findings**: Consider testing on additional problem sets

---

*Last updated: Based on results from `results/system2/` directory*
