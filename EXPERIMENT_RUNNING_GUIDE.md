# System 2 Experiments - Running Guide

## ✅ What's Been Completed

### 1. **Comprehensive Benchmark Suite** (100% Complete)
- **Game of 24**: 125 problems across 5 difficulty levels
- **Arithmetic Chains**: 80 problems (5, 10, 15, 20 steps)
- **Tower of Hanoi**: 50 problems (3-7 disks)
- **Variable Tracking**: 100 problems (systematic complexity scaling)
- **Logic Puzzles**: Logic grid puzzles and reasoning problems

### 2. **Experiment Infrastructure** (100% Complete)
- Enhanced Tree of Thought with 5 search strategies
- System 2 criticality experiment framework
- State tracking benchmarks
- Advanced architectures (Verify-Refine, Debate, Memory-Augmented)
- Power law analysis framework
- Comprehensive experiment runner

### 3. **Ready to Run**
All components are implemented and ready for execution.

---

## 🚀 How to Run Experiments

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

Or if using pip3:
```bash
pip3 install -r requirements.txt
```

### Step 2: Set API Key

```bash
export OPENAI_API_KEY=your_key_here
```

### Step 3: Quick Test (Optional)

Verify everything works:

```bash
python3 quick_test_experiments.py
```

This runs a minimal test to verify:
- Benchmark loading
- Problem evaluation
- Tree of Thought search (minimal)

### Step 4: Run Full Experiments

Run the comprehensive experiment suite:

```bash
python3 run_system2_experiments.py
```

This will:
1. **Baseline Evaluation**: Test GPT-4o on all 5 benchmarks (5 problems each)
2. **Scaling Experiments**: Run Game of 24 with different models, depths, beam widths
3. **Architecture Comparison**: Compare Verify-Refine, Debate, Memory-Augmented
4. **State Tracking**: Measure state tracking fidelity across different step counts

**Expected Time**: 30-60 minutes depending on API speed
**Expected Cost**: ~$5-10 in API credits (using GPT-4o-mini and GPT-4o)

### Step 5: Analyze Results

After experiments complete, analyze results:

```bash
python3 system2_power_law_analysis.py results/system2/scaling_results.json
```

Or load comprehensive results:

```python
import json
with open('results/system2/experiment_summary.json', 'r') as f:
    summary = json.load(f)
```

---

## 📊 Experiment Outputs

All results are saved to `results/system2/`:

- `baseline_results.json`: Baseline performance on all benchmarks
- `scaling_results.json`: Complete scaling experiment results
- `architecture_comparison.json`: Architecture comparison results
- `state_tracking_results.json`: State tracking fidelity measurements
- `experiment_summary.json`: Summary of all experiments

---

## ⚙️ Customizing Experiments

### Run Smaller Experiments

Edit `run_system2_experiments.py` to reduce scope:

```python
# In run_scaling_experiments():
config = ExperimentConfig(
    models=['gpt-4o-mini'],  # Only one model
    search_depths=[1, 3, 5],  # Fewer depths
    beam_widths=[1, 3],      # Fewer beam widths
    ...
)

# In run_baseline_evaluation():
problems = benchmark.get_problems(difficulty=None, num_problems=2)  # Fewer problems
```

### Run Specific Experiments Only

Comment out sections in `main()`:

```python
def main():
    # Only run baseline
    baseline_results = run_baseline_evaluation()
    
    # Skip scaling
    # scaling_results = run_scaling_experiments()
    
    # Skip architecture
    # arch_results = run_architecture_comparison()
```

---

## 🔍 Understanding Results

### Baseline Results

Shows zero-shot performance on each benchmark:
- `accuracy`: Overall solve rate
- `results`: Per-problem results with correctness

### Scaling Results

Shows how performance scales with:
- Model size (GPT-3.5-turbo → GPT-4o)
- Search depth (1 → 20)
- Beam width (1 → 10)

Key metrics:
- `solve_rate`: % of problems solved
- `hallucination_rate`: % of invalid states
- `search_efficiency`: Solutions found / nodes explored
- `critical_points`: Where performance jumps dramatically

### Architecture Comparison

Compares three advanced architectures:
- `verify_refine`: Iterative refinement with verification
- `debate`: Multi-agent debate system
- `memory_augmented`: Memory-augmented reasoning

### State Tracking

Measures state tracking fidelity:
- `accuracy_curve`: Accuracy at each step
- `failure_step`: Where tracking breaks down
- `critical_points`: Critical failure thresholds

---

## 📈 Next Steps After Running

1. **Generate Visualizations**:
   ```python
   from system2_power_law_analysis import plot_power_law_relationships
   import json
   
   with open('results/system2/scaling_results.json', 'r') as f:
       results = json.load(f)
   
   plot_power_law_relationships(results)
   ```

2. **Analyze Critical Points**:
   ```python
   from system2_criticality_experiment import System2CriticalityExperiment
   
   # Load results and find critical points
   experiment = System2CriticalityExperiment()
   # ... analyze critical transitions
   ```

3. **Compare to Ising Model**:
   - Integrate with existing Ising model results
   - Create unified visualization
   - Validate criticality hypothesis

---

## ⚠️ Notes

- **API Costs**: Full experiments use significant API credits. Start with quick test.
- **Time**: Full experiments take 30-60 minutes. Be patient.
- **Rate Limits**: If you hit rate limits, reduce number of problems or add delays.
- **Results**: All results are saved automatically. You can stop and resume.

---

## 🐛 Troubleshooting

### ModuleNotFoundError: No module named 'openai'
```bash
pip install openai
```

### API Key Error
```bash
export OPENAI_API_KEY=your_key_here
```

### Import Errors
Make sure you're in the project directory:
```bash
cd /path/to/powerlawexperiments
```

### Out of Memory
Reduce experiment scope (fewer problems, models, depths)

---

## 📝 Example Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set API key
export OPENAI_API_KEY=sk-...

# 3. Quick test
python3 quick_test_experiments.py

# 4. Run full experiments
python3 run_system2_experiments.py

# 5. Check results
ls -la results/system2/

# 6. Analyze
python3 system2_power_law_analysis.py results/system2/scaling_results.json
```

---

**Ready to run!** All infrastructure is complete. Just install dependencies and execute.
