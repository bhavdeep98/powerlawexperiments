import json
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

# Setup paths
BASE_DIR = Path("results")
PLOTS_DIR = BASE_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

def load_json(path):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return []

def plot_experiment_1_4(data):
    # Exp 1.4: Combined Scaling (T vs K)
    # Expected Structure: List of runs with 'k_value', 't_value', 'success'
    print("\n## Experiment 1.4: Combined Scaling (Search x Tools)")
    print("| T (Depth) | K (Tools) | Success Rate |")
    print("|-----------|-----------|--------------|")
    
    # Process data
    summary = {} # (T, K) -> [results]
    for run in data:
        t = run.get('t_value', 0)
        k = run.get('k_value', 0)
        success = 1 if run.get('success', False) else 0
        key = (t, k)
        if key not in summary: summary[key] = []
        summary[key].append(success)
        
    for (t, k), outcomes in sorted(summary.items()):
        rate = np.mean(outcomes)
        print(f"| {t} | {k} | {rate:.1%} |")
    
    # Plot
    # Group by K
    k_values = sorted(list(set(k for t, k in summary.keys())))
    t_values = sorted(list(set(t for t, k in summary.keys())))
    
    plt.figure(figsize=(8, 5))
    for k in k_values:
        rates = []
        for t in t_values:
            rates.append(np.mean(summary.get((t, k), [0])))
        plt.plot(t_values, rates, marker='o', label=f'K={k} (Tools)')
        
    plt.xlabel('Search Depth (T)')
    plt.ylabel('Success Rate')
    plt.title('Exp 1.4: Synergistic Scaling (Prosthetic Intelligence)')
    plt.legend()
    plt.grid(True)
    plt.savefig(PLOTS_DIR / 'exp1_4_scaling.png')
    plt.close()

def plot_experiment_3_1(data):
    # Exp 3.1: Bandwidth Scaling
    print("\n## Experiment 3.1: Bandwidth Criticality")
    print("| Bandwidth (k) | Consensus Ratio |")
    print("|---------------|-----------------|")
    
    bandwidths = {}
    for run in data:
        bw = run.get('bandwidth', 0)
        ratio = run.get('consensus_ratio', 0)
        if bw not in bandwidths: bandwidths[bw] = []
        bandwidths[bw].append(ratio)
        
    sorted_bws = sorted(bandwidths.keys())
    means = [np.mean(bandwidths[b]) for b in sorted_bws]
    stds = [np.std(bandwidths[b]) for b in sorted_bws]
    
    for b, m in zip(sorted_bws, means):
        print(f"| {b} | {m:.1%} |")
        
    plt.figure(figsize=(8, 5))
    plt.plot(sorted_bws, means, marker='o', color='purple')
    plt.fill_between(sorted_bws, np.array(means)-np.array(stds), np.array(means)+np.array(stds), alpha=0.2, color='purple')
    plt.xlabel('Bandwidth (k)')
    plt.ylabel('Consensus Ratio')
    plt.title('Exp 3.1: Bandwidth Scaling (N=5)')
    plt.grid(True)
    plt.savefig(PLOTS_DIR / 'exp3_1_bandwidth.png')
    plt.close()

def plot_experiment_3_2(data):
    # Exp 3.2: Topology
    print("\n## Experiment 3.2: Network Topology")
    print("| Topology | Consensus Ratio |")
    print("|----------|-----------------|")
    
    topos = {}
    for run in data:
        t = run.get('topology', 'unknown')
        ratio = run.get('consensus_ratio', 0)
        if t not in topos: topos[t] = []
        topos[t].append(ratio)
        
    names = sorted(topos.keys())
    means = [np.mean(topos[n]) for n in names]
    
    for n, m in zip(names, means):
        print(f"| {n} | {m:.1%} |")
        
    plt.figure(figsize=(8, 5))
    plt.bar(names, means, color=['blue', 'green', 'orange', 'red'])
    plt.ylabel('Consensus Ratio')
    plt.title('Exp 3.2: Topology Comparison')
    plt.savefig(PLOTS_DIR / 'exp3_2_topology.png')
    plt.close()

def plot_experiment_3_3(data):
    # Exp 3.3: Reliability
    print("\n## Experiment 3.3: Reliability (Robustness)")
    print("| Reliability | Consensus Ratio |")
    print("|-------------|-----------------|")
    
    rels = {}
    for run in data:
        r = run.get('reliability', 1.0)
        ratio = run.get('consensus_ratio', 0)
        if r not in rels: rels[r] = []
        rels[r].append(ratio)
        
    sorted_rels = sorted(rels.keys(), reverse=True)
    means = [np.mean(rels[r]) for r in sorted_rels]
    
    for r, m in zip(sorted_rels, means):
        print(f"| {r} | {m:.1%} |")
        
    plt.figure(figsize=(8, 5))
    plt.plot(sorted_rels, means, marker='s', color='red')
    plt.gca().invert_xaxis() # 1.0 -> 0.5
    plt.xlabel('Agent Reliability')
    plt.ylabel('Consensus Ratio')
    plt.title('Exp 3.3: Robustness to Failure')
    plt.grid(True)
    plt.savefig(PLOTS_DIR / 'exp3_3_reliability.png')
    plt.close()

def plot_experiment_4(data):
    # Exp 4: Compute Optimal
    print("\n## Experiment 4: Compute Optimal Architecture")
    print("| Architecture | Success Rate | Avg Time (s) | Efficiency (Succ/Cost) |")
    print("|--------------|--------------|--------------|------------------------|")
    
    # Data is list of {problem, ensemble: {}, search: {}, strong: {}}
    if not data: return

    archs = ['ensemble', 'search', 'strong']
    metrics = {a: {'success': [], 'time': [], 'cost': []} for a in archs}
    
    for run in data:
        for a in archs:
            metrics[a]['success'].append(run[a]['success'])
            metrics[a]['time'].append(run[a]['time'])
            metrics[a]['cost'].append(run[a]['cost_units'])
            
    means = {}
    for a in archs:
        means[a] = {
            'success': np.mean(metrics[a]['success']),
            'time': np.mean(metrics[a]['time']),
            'cost': np.mean(metrics[a]['cost'])
        }
        
    # Print Table
    for a in archs:
        m = means[a]
        # Efficiency = Success / Cost (scaled by 100 for readability)
        eff = (m['success'] * 100) / m['cost']
        print(f"| {a.title()} | {m['success']:.1%} | {m['time']:.1f} | {eff:.2f} |")
        
    # Plot Bar Chart
    plt.figure(figsize=(10, 6))
    
    # Multi-bar plot
    x = np.arange(len(archs))
    width = 0.35
    
    fig, ax1 = plt.subplots()
    
    success_vals = [means[a]['success'] for a in archs]
    cost_vals = [means[a]['cost'] for a in archs]
    
    ax1.bar(x - width/2, success_vals, width, label='Success Rate', color='skyblue')
    ax1.set_ylabel('Success Rate', color='blue')
    ax1.set_ylim(0, 1.1)
    
    ax2 = ax1.twinx()
    ax2.plot(x, cost_vals, marker='D', color='red', label='Cost (Units)', linestyle='None')
    ax2.set_ylabel('Compute Cost', color='red')
    
    plt.xticks(x, [a.title() for a in archs])
    plt.title('Exp 4: Success vs Cost')
    
    fig.tight_layout()
    plt.savefig(PLOTS_DIR / 'exp4_compute_optimal.png')
    plt.close()

def main():
    print("# Final Experiment Report")
    
    # Exp 1.4
    d1 = load_json("results/agentic_scaling/exp1_4_combined_scaling_results.json")
    if d1: plot_experiment_1_4(d1)
    
    # Exp 3.1
    d3_1 = load_json("results/criticality/exp3_1_criticality_N5.json")
    if d3_1: plot_experiment_3_1(d3_1)
    
    # Exp 3.2
    d3_2 = load_json("results/criticality/exp3_2_topology_N5.json")
    if d3_2: plot_experiment_3_2(d3_2)
    
    # Exp 3.3
    d3_3 = load_json("results/criticality/exp3_3_reliability_N5.json")
    if d3_3: plot_experiment_3_3(d3_3)
    
    # Exp 4
    d4 = load_json("results/compute_optimal/exp4_results.json")
    if d4: plot_experiment_4(d4)
    
    print(f"\nPlots saved to {PLOTS_DIR.absolute()}")

if __name__ == "__main__":
    main()
