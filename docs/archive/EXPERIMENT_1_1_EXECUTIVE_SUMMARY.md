# Experiment 1.1: A-Scaling (Number of Agents) - Executive Summary

**Experiment**: Agentic Scaling Laws - Variable A (Number of Agents)  
**Status**: Ready to Run  
**Priority**: HIGH (Foundation for all other experiments)

---

## Objective

Test the hypothesis that agentic system performance follows a power-law relationship with the number of agents:

**Hypothesis**: $\mathcal{E}(A) \sim A^{-\beta_A}$

Where:
- **A**: Number of agents (1, 2, 3, 5, 8, 13)
- **β_A**: Scaling exponent (to be measured)
- **E(A)**: Performance metric (solve rate, coordination accuracy)

---

## Experimental Design

### Independent Variable
- **A** (Number of agents): {1, 2, 3, 5, 8, 13}

### Dependent Variables (Metrics)
1. **Solve Rate**: Fraction of problems solved correctly (0-1)
2. **Coordination Accuracy**: Agreement rate among agents (0-1)
3. **Consensus Time**: Time to reach consensus
4. **Token Usage**: Total tokens used across all agents

### Controlled Variables
- **Model**: Fixed (gpt-4o-mini for testing, gpt-4o for full run)
- **Task**: Game of 24 (difficulty 3)
- **Problem Set**: Same problems for all A values
- **Replications**: 3 runs per configuration (for statistical significance)

---

## Expected Outcomes

### Primary Result
- **Power-law fit**: solve_rate = a × A^β_A
- **Scaling exponent β_A**: Expected range -0.2 to 0.2
- **R² value**: Should be > 0.8 for valid power law

### Secondary Results
- **Coordination accuracy**: How well agents agree
- **Critical threshold A***: Point where coordination emerges (if exists)
- **Diminishing returns**: Whether more agents help or hurt

### Hypotheses to Test
1. **H1**: More agents → higher solve rate (β_A > 0)
2. **H2**: Coordination accuracy increases with A (up to some point)
3. **H3**: Power-law relationship holds (R² > 0.8)

---

## Setup Requirements

### Infrastructure
- ✅ Multi-agent system (to create)
- ✅ Benchmark problems (Game of 24 exists)
- ✅ Evaluation metrics (exists)
- ✅ Power-law analysis (exists)

### Data Collection
- **Problems**: 10 problems (difficulty 3) per A value
- **Total runs**: 6 A values × 10 problems × 3 replications = 180 runs
- **Output**: JSON file with results and traces

### Analysis
- Power-law fitting using existing `fit_power_law()`
- Critical point detection (if applicable)
- Visualization (log-log plot)

---

## Time & Cost Estimation

### Per-Run Estimates
- **Single agent (A=1)**: ~2-3 seconds, ~500 tokens
- **Two agents (A=2)**: ~4-6 seconds, ~1000 tokens
- **Three agents (A=3)**: ~6-9 seconds, ~1500 tokens
- **Five agents (A=5)**: ~10-15 seconds, ~2500 tokens
- **Eight agents (A=8)**: ~16-24 seconds, ~4000 tokens
- **Thirteen agents (A=13)**: ~26-39 seconds, ~6500 tokens

### Full Experiment (10 problems, 3 replications)

| A | Runs | Time per Run | Total Time | Tokens per Run | Total Tokens |
|---|------|--------------|------------|----------------|--------------|
| 1 | 30 | 2.5s | 75s (1.25 min) | 500 | 15,000 |
| 2 | 30 | 5s | 150s (2.5 min) | 1,000 | 30,000 |
| 3 | 30 | 7.5s | 225s (3.75 min) | 1,500 | 45,000 |
| 5 | 30 | 12.5s | 375s (6.25 min) | 2,500 | 75,000 |
| 8 | 30 | 20s | 600s (10 min) | 4,000 | 120,000 |
| 13 | 30 | 32.5s | 975s (16.25 min) | 6,500 | 195,000 |
| **TOTAL** | **180** | - | **~40 minutes** | - | **~480,000 tokens** |

### Cost Estimation (gpt-4o-mini)
- **Input**: $0.15 per 1M tokens
- **Output**: $0.60 per 1M tokens
- **Estimated cost**: ~$0.30 (assuming 50/50 input/output split)

### Cost Estimation (gpt-4o - full run)
- **Input**: $2.50 per 1M tokens
- **Output**: $10.00 per 1M tokens
- **Estimated cost**: ~$3.00 (assuming 50/50 input/output split)

### Recommended Approach
1. **Pilot Run**: A ∈ {1, 2, 3}, 5 problems, 1 replication
   - Time: ~5 minutes
   - Cost: ~$0.05 (gpt-4o-mini)
   - Purpose: Validate setup, check metrics

2. **Full Run**: A ∈ {1, 2, 3, 5, 8}, 10 problems, 3 replications
   - Time: ~35 minutes
   - Cost: ~$0.25 (gpt-4o-mini) or ~$2.50 (gpt-4o)
   - Purpose: Collect data for analysis

3. **Extended Run** (if needed): Add A=13, more problems
   - Time: +15 minutes
   - Cost: +$0.10 (gpt-4o-mini)

---

## Implementation Steps

### Step 1: Create MultiAgentSystem (30 min)
- Extend existing DebateSystem to support N agents
- Implement coordination measurement
- Implement consensus mechanism

### Step 2: Create Experiment Runner (1 hour)
- Create `agentic_scaling_experiment.py`
- Implement A-scaling experiment function
- Add result collection and saving

### Step 3: Test with Pilot (5 min)
- Run with A ∈ {1, 2, 3}, 3 problems
- Verify metrics work correctly
- Check output format

### Step 4: Run Full Experiment (40 min)
- Execute full experiment
- Monitor progress
- Save results

### Step 5: Analyze Results (15 min)
- Fit power-law
- Generate visualizations
- Extract scaling exponent

**Total Implementation + Execution Time**: ~2.5 hours

---

## Success Criteria

### Must Have
- ✅ Experiment runs without errors
- ✅ All metrics collected correctly
- ✅ Power-law fit with R² > 0.8
- ✅ Results saved to JSON

### Nice to Have
- ✅ Clear coordination threshold detected
- ✅ Diminishing returns identified
- ✅ Traces collected for workflow extraction

---

## Risks & Mitigations

### Risk 1: Coordination accuracy calculation may be inaccurate
- **Mitigation**: Start with simple similarity metric, refine if needed
- **Test**: Check with known similar/different solutions

### Risk 2: Power-law may not hold (low R²)
- **Mitigation**: Ensure A values span sufficient range, use more problems
- **Test**: Run pilot first, check R²

### Risk 3: API costs/time may be higher than estimated
- **Mitigation**: Start with gpt-4o-mini, smaller A values, fewer problems
- **Test**: Monitor first few runs, adjust if needed

### Risk 4: Consensus mechanism may not work well
- **Mitigation**: Start with simple majority vote, improve if needed
- **Test**: Check consensus results manually on sample

---

## Deliverables

1. **Code**: `agentic_scaling_experiment.py`
2. **Results**: `results/agentic_scaling/exp1_1_a_scaling_results.json`
3. **Analysis**: Power-law fit parameters, R² value
4. **Visualization**: Log-log plot (solve_rate vs A)
5. **Traces**: Agent traces for workflow extraction (Experiment 2)

---

## Next Steps After This Experiment

1. **Experiment 1.2 (T-Scaling)**: Vary interaction depth
2. **Experiment 1.3 (K-Scaling)**: Vary tool richness
3. **Experiment 2.1 (Workflow Extraction)**: Use traces from this experiment

---

**Ready to proceed?** Let's create the implementation and run the pilot!
