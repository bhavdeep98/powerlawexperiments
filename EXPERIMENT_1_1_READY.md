# Experiment 1.1: Ready to Run! 🚀

## Executive Summary

**Experiment**: A-Scaling (Number of Agents)  
**Status**: ✅ Implementation Complete, Ready to Execute  
**First Run**: Pilot experiment (~45 seconds, ~$0.00)

---

## What We've Created

### 1. Implementation Files
- ✅ `agentic_scaling_experiment.py` - Complete experiment runner
- ✅ `EXPERIMENT_1_1_EXECUTIVE_SUMMARY.md` - Detailed summary
- ✅ `EXPERIMENT_1_1_SETUP.md` - Setup instructions
- ✅ `calculate_experiment_time.py` - Time/cost calculator

### 2. Core Components
- ✅ **MultiAgentSystem**: Extends DebateSystem to N agents
- ✅ **Coordination Measurement**: Agreement calculation
- ✅ **Consensus Mechanism**: Majority vote on solutions
- ✅ **Trace Collection**: For workflow extraction (Experiment 2)
- ✅ **Power-Law Analysis**: Integrated from existing code

---

## Time & Cost Estimates

### Pilot Run (Recommended First)
```
A values: [1, 2, 3]
Problems: 3
Replications: 1
Total runs: 9
Time: ~45 seconds
Cost: ~$0.00 (gpt-4o-mini)
```

### Full Run
```
A values: [1, 2, 3, 5, 8]
Problems: 10
Replications: 3
Total runs: 150
Time: ~24 minutes
Cost: ~$0.09 (gpt-4o-mini) or ~$0.85 (gpt-4o)
```

---

## Setup Instructions

### 1. Install Dependencies (if needed)
```bash
pip3 install numpy scipy openai
# Or install all requirements:
pip3 install -r requirements.txt
```

### 2. Set API Key
```bash
export OPENAI_API_KEY=your_key_here
```

### 3. Run Pilot Experiment
```bash
python3 agentic_scaling_experiment.py --pilot
```

### 4. Run Full Experiment (after pilot succeeds)
```bash
python3 agentic_scaling_experiment.py \
    --A-values 1 2 3 5 8 \
    --model gpt-4o-mini \
    --num-problems 10 \
    --replications 3
```

---

## What the Experiment Does

1. **For each A value** (1, 2, 3, 5, 8 agents):
   - Creates N agents
   - Runs them on benchmark problems
   - Measures solve rate
   - Measures coordination accuracy
   - Collects traces

2. **Analyzes Results**:
   - Fits power law: solve_rate = a × A^β_A
   - Calculates R² value
   - Identifies critical threshold (if exists)

3. **Saves Output**:
   - JSON file with all results
   - Traces for workflow extraction
   - Analysis metrics

---

## Expected Results

### Primary Output
- **Power-law fit**: solve_rate ∝ A^β_A
- **Scaling exponent β_A**: Expected range -0.2 to 0.2
- **R² value**: Should be > 0.8 for valid power law

### Secondary Output
- **Coordination accuracy**: How well agents agree
- **Critical threshold A***: Point where coordination emerges
- **Traces**: Agent action sequences for Experiment 2

---

## Success Criteria

✅ Experiment runs without errors  
✅ All metrics collected correctly  
✅ Power-law fit with R² > 0.8  
✅ Results saved to JSON  
✅ Traces collected for workflow extraction

---

## Troubleshooting

### Issue: ModuleNotFoundError (numpy, scipy, etc.)
**Solution**: Install dependencies
```bash
pip3 install numpy scipy openai
```

### Issue: API Key not found
**Solution**: Export API key
```bash
export OPENAI_API_KEY=your_key_here
```

### Issue: Low R² value (< 0.8)
**Possible causes**:
- A values don't span enough range
- Not enough problems
- Solve rates don't vary meaningfully

**Solutions**:
- Use more A values (add 8, 13)
- Increase number of problems
- Check if coordination is actually improving

---

## Next Steps After Running

1. **Review Results**: Check power-law fit quality
2. **Generate Visualizations**: Create log-log plot
3. **Proceed to Experiment 1.2**: T-Scaling (vary interaction depth)
4. **Use Traces for Experiment 2**: Workflow grammar extraction

---

## Files Created

- `agentic_scaling_experiment.py` - Main experiment file
- `EXPERIMENT_1_1_EXECUTIVE_SUMMARY.md` - Detailed summary
- `EXPERIMENT_1_1_SETUP.md` - Setup guide
- `EXPERIMENT_1_1_READY.md` - This file
- `calculate_experiment_time.py` - Time calculator

---

## Quick Start Command

```bash
# Install dependencies (if needed)
pip3 install numpy scipy openai

# Set API key
export OPENAI_API_KEY=your_key_here

# Run pilot
python3 agentic_scaling_experiment.py --pilot
```

---

**Status**: ✅ Ready to Run  
**Estimated Time**: 45 seconds (pilot)  
**Estimated Cost**: ~$0.00 (pilot)

**Let's run it!** 🚀
