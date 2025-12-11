# Experiment 1.2: T-Scaling (Interaction Depth) - Setup

**Experiment**: Agentic Scaling Laws - Variable T (Interaction Depth/Episodes)  
**Status**: Ready to Run  
**Builds on**: Experiment 1.1 (A-Scaling) ✅

---

## Objective

Test the hypothesis that agentic system performance follows a power-law relationship with interaction depth:

**Hypothesis**: $\mathcal{E}(T) \sim T^{-\beta_T}$

Where:
- **T**: Interaction depth/search depth (1, 3, 5, 10, 20, 50)
- **β_T**: Scaling exponent (to be measured)
- **E(T)**: Performance metric (solve rate, search efficiency)

---

## Experimental Design

### Independent Variable
- **T** (Search depth): {1, 3, 5, 10, 20}

### Dependent Variables
1. **Solve Rate**: Fraction of problems solved correctly
2. **Search Efficiency**: Solutions found / nodes explored
3. **Nodes Explored**: Total nodes in search tree
4. **Time to Solution**: Execution time

### Controlled Variables
- **A = 1**: Single agent (fixed)
- **K = 3**: No tools (fixed for now)
- **Model**: gpt-4o-mini (fixed)
- **Beam Width**: 3 (fixed)
- **Branching Factor**: 3 (fixed)

---

## Implementation

Uses existing `System2CriticalityExperiment` framework:
- Tree of Thought search
- Beam search strategy
- Depth variation already supported
- Metrics collection built-in

---

## Time & Cost Estimation

### Per-Run Estimates (approximate)
- **T=1**: ~2-3 seconds, ~500 tokens
- **T=3**: ~5-8 seconds, ~1500 tokens
- **T=5**: ~10-15 seconds, ~3000 tokens
- **T=10**: ~20-30 seconds, ~6000 tokens
- **T=20**: ~40-60 seconds, ~12000 tokens

### Full Run (10 problems, 2 replications)
- **Total runs**: 5 T values × 10 problems × 2 replications = 100 runs
- **Estimated time**: ~30-40 minutes
- **Estimated cost**: ~$0.10 (gpt-4o-mini)

---

## Running the Experiment

### Full Run
```bash
source venv/bin/activate
python agentic_scaling_t_experiment.py \
    --T-values 1 3 5 10 20 \
    --model gpt-4o-mini \
    --num-problems 10 \
    --replications 2
```

### Pilot Run
```bash
python agentic_scaling_t_experiment.py --pilot
```

---

## Expected Output

1. **Power-law fit**: solve_rate = a × T^β_T
2. **Scaling exponent β_T**: Expected range -0.2 to 0.3
3. **R² value**: Should be > 0.8
4. **Critical threshold T***: Point where performance jumps

---

## Success Criteria

- ✅ Power-law fit with R² > 0.8
- ✅ Clear scaling relationship
- ✅ Results saved to JSON
- ✅ Ready for Experiment 1.3

---

**Ready to run Experiment 1.2!**
