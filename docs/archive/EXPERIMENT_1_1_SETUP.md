# Experiment 1.1 Setup & Execution Plan

## Time & Cost Summary

### Pilot Run (Recommended First)
- **A values**: [1, 2, 3]
- **Problems**: 3
- **Replications**: 1
- **Total runs**: 9
- **Time**: ~45 seconds (0.8 minutes)
- **Cost**: ~$0.00 (gpt-4o-mini)
- **Purpose**: Validate setup, check metrics work

### Full Run
- **A values**: [1, 2, 3, 5, 8]
- **Problems**: 10
- **Replications**: 3
- **Total runs**: 150
- **Time**: ~24 minutes
- **Cost**: ~$0.09 (gpt-4o-mini) or ~$0.85 (gpt-4o)
- **Purpose**: Collect data for power-law analysis

## Setup Complete ✅

1. ✅ **MultiAgentSystem** class created
2. ✅ **A-scaling experiment runner** implemented
3. ✅ **Power-law analysis** integrated
4. ✅ **Trace collection** for Experiment 2
5. ✅ **Result saving** to JSON

## Running the Experiment

### Option 1: Pilot Run (Recommended)
```bash
python3 agentic_scaling_experiment.py --pilot
```

### Option 2: Custom Run
```bash
python3 agentic_scaling_experiment.py \
    --A-values 1 2 3 5 \
    --model gpt-4o-mini \
    --num-problems 5 \
    --replications 1
```

### Option 3: Full Run
```bash
python3 agentic_scaling_experiment.py \
    --A-values 1 2 3 5 8 \
    --model gpt-4o-mini \
    --num-problems 10 \
    --replications 3
```

## Expected Output

1. **Console Output**:
   - Progress for each A value
   - Solve rates and coordination accuracies
   - Power-law fit results

2. **JSON File**: `results/agentic_scaling/exp1_1_a_scaling_results.json`
   - Complete results
   - Analysis (power-law parameters, R²)
   - Traces for workflow extraction

3. **Metrics**:
   - Solve rate per A value
   - Coordination accuracy per A value
   - Power-law exponent β_A
   - R² value (should be > 0.8)

## Next Steps After Running

1. **Review Results**: Check if power-law fit is valid (R² > 0.8)
2. **Generate Visualizations**: Create log-log plot (solve_rate vs A)
3. **Proceed to Experiment 1.2**: T-Scaling (vary interaction depth)

---

**Ready to run?** Execute the pilot command above!
