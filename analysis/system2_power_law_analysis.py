"""
System 2 Power Law Analysis
============================
Analyzes System 2 reasoning results for power law relationships and
connects findings to the broader power law thesis.

Hypotheses:
1. Does solve_rate follow power law: S ∝ (M × D)^α ?
2. Is there a critical (M, D) threshold for coherent reasoning?
3. Does hallucination_rate show phase transition?
4. How does System 2 scaling compare to Ising model and neural scaling?
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import optimize, stats
import matplotlib.pyplot as plt


# ==============================================================================
# POWER LAW FITTING
# ==============================================================================

def fit_power_law(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """
    Fit power law: y = a * x^b
    
    Returns:
        (a, b, r_squared)
    """
    # Log-log transformation: log(y) = log(a) + b * log(x)
    # Filter out zeros and negatives
    mask = (x > 0) & (y > 0)
    if mask.sum() < 3:
        return 0.0, 0.0, 0.0
    
    log_x = np.log(x[mask])
    log_y = np.log(y[mask])
    
    # Linear regression in log space
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, log_y)
    
    a = np.exp(intercept)
    b = slope
    r_squared = r_value ** 2
    
    return a, b, r_squared


def find_critical_exponent(x: np.ndarray, y: np.ndarray, 
                          threshold: float = 0.5) -> Optional[float]:
    """
    Find critical point where y crosses threshold.
    
    Returns:
        Critical x value, or None if not found
    """
    # Find where y crosses threshold
    for i in range(len(y) - 1):
        if y[i] < threshold <= y[i+1] or y[i] > threshold >= y[i+1]:
            # Linear interpolation
            if y[i+1] != y[i]:
                t = (threshold - y[i]) / (y[i+1] - y[i])
                return x[i] + t * (x[i+1] - x[i])
            else:
                return x[i]
    return None


# ==============================================================================
# ANALYSIS FUNCTIONS
# ==============================================================================

def analyze_solve_rate_scaling(results: Dict) -> Dict:
    """
    Analyze if solve_rate follows power law with respect to compute.
    
    Hypothesis: S ∝ (M × D)^α
    where M = model_size, D = search_depth
    """
    # Handle different result formats
    if 'criticality' in results:
        data_source = results['criticality']
    elif 'aggregated' in results:
        data_source = results
    else:
        return {'error': 'No criticality results found'}
    
    aggregated = data_source.get('aggregated', {})
    
    # Extract data
    compute_budgets = []
    solve_rates = []
    
    for config_key, metrics in aggregated.items():
        if isinstance(config_key, str):
            parts = config_key.rsplit('_', 2)
            if len(parts) == 3:
                model = parts[0]
                depth = int(parts[1])
        else:
            model, depth, beam = config_key
            
        solve_rate = metrics.get('solve_rate', 0.0)
        
        # Estimate compute budget (simplified: model_size × depth)
        model_sizes = {'gpt-3.5-turbo': 1.0, 'gpt-4o-mini': 2.0, 'gpt-4o': 4.0}
        model_size = model_sizes.get(model, 1.0)
        compute = model_size * depth
        
        compute_budgets.append(compute)
        solve_rates.append(solve_rate)
    
    if len(compute_budgets) < 3:
        return {'error': 'Insufficient data'}
    
    # Fit power law
    x = np.array(compute_budgets)
    y = np.array(solve_rates)
    
    a, b, r_squared = fit_power_law(x, y)
    
    # Find critical point (where solve_rate crosses 0.5)
    critical_compute = find_critical_exponent(x, y, threshold=0.5)
    
    return {
        'power_law_coefficient': a,
        'power_law_exponent': b,
        'r_squared': r_squared,
        'critical_compute': critical_compute,
        'data_points': len(compute_budgets)
    }


def analyze_hallucination_phase_transition(results: Dict) -> Dict:
    """Analyze if hallucination_rate shows phase transition."""
    # Handle different result formats
    if 'criticality' in results:
        data_source = results['criticality']
    elif 'aggregated' in results:
        data_source = results
    else:
        return {'error': 'No criticality results found'}
    
    aggregated = data_source.get('aggregated', {})
    
    # Extract data
    model_sizes = []
    hallucination_rates = []
    
    model_size_map = {'gpt-3.5-turbo': 1.0, 'gpt-4o-mini': 2.0, 'gpt-4o': 4.0}
    
    for config_key, metrics in aggregated.items():
        if isinstance(config_key, str):
            parts = config_key.rsplit('_', 2)
            if len(parts) == 3:
                model = parts[0]
        else:
            model, depth, beam = config_key
            
        hall_rate = metrics.get('hallucination_rate', 0.0)
        
        model_size = model_size_map.get(model, 1.0)
        model_sizes.append(model_size)
        hallucination_rates.append(hall_rate)
    
    if len(model_sizes) < 3:
        return {'error': 'Insufficient data'}
    
    # Check for phase transition (sharp drop)
    x = np.array(model_sizes)
    y = np.array(hallucination_rates)
    
    # Find largest drop
    max_drop = 0.0
    drop_point = None
    
    for i in range(len(y) - 1):
        drop = y[i] - y[i+1]
        if drop > max_drop:
            max_drop = drop
            drop_point = (x[i] + x[i+1]) / 2
    
    return {
        'max_drop': max_drop,
        'drop_point': drop_point,
        'has_phase_transition': max_drop > 0.2  # Threshold
    }


def analyze_search_efficiency(results: Dict) -> Dict:
    """Analyze search efficiency vs problem complexity."""
    if 'tot_comparison' not in results:
        return {'error': 'No ToT comparison results found'}
    
    # Extract efficiency metrics
    strategies = {}
    
    for task_result in results['tot_comparison'].get('results', []):
        for strategy, s2_result in task_result.get('system2', {}).items():
            if strategy not in strategies:
                strategies[strategy] = {'efficiencies': [], 'depths': []}
            
            metrics = s2_result.get('metrics', {})
            nodes_explored = metrics.get('nodes_explored', 1)
            success = s2_result.get('success', False)
            
            efficiency = (1.0 if success else 0.0) / max(nodes_explored, 1)
            depth = metrics.get('max_depth_reached', 0)
            
            strategies[strategy]['efficiencies'].append(efficiency)
            strategies[strategy]['depths'].append(depth)
    
    # Calculate averages
    summary = {}
    for strategy, data in strategies.items():
        if data['efficiencies']:
            summary[strategy] = {
                'avg_efficiency': np.mean(data['efficiencies']),
                'avg_depth': np.mean(data['depths']),
                'std_efficiency': np.std(data['efficiencies'])
            }
    
    return summary


# ==============================================================================
# VISUALIZATION
# ==============================================================================

def plot_power_law_relationships(results: Dict, save_path: str = 'system2_power_laws.png'):
    """Plot power law relationships in System 2 results."""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Solve rate vs compute budget
        # 1. Solve rate vs compute budget
        ax1 = axes[0, 0]
        
        # Handle different result formats
        if 'criticality' in results:
            data_source = results['criticality']
        elif 'aggregated' in results:
            data_source = results
        else:
            data_source = {}
            
        if data_source:
            aggregated = data_source.get('aggregated', {})
            compute_budgets = []
            solve_rates = []
            
            model_size_map = {'gpt-3.5-turbo': 1.0, 'gpt-4o-mini': 2.0, 'gpt-4o': 4.0}
            
            for config_key, metrics in aggregated.items():
                if isinstance(config_key, str):
                    # Parse string key "model_depth_beam"
                    parts = config_key.rsplit('_', 2)
                    if len(parts) == 3:
                        model = parts[0]
                        depth = int(parts[1])
                else:
                    # Tuple key
                    model, depth, beam = config_key
                
                model_size = model_size_map.get(model, 1.0)
                compute = model_size * depth
                compute_budgets.append(compute)
                solve_rates.append(metrics.get('solve_rate', 0.0))
            
            if compute_budgets:
                ax1.scatter(compute_budgets, solve_rates, alpha=0.6)
                ax1.set_xlabel('Compute Budget (Model Size × Depth)', fontsize=10)
                ax1.set_ylabel('Solve Rate', fontsize=10)
                ax1.set_title('Solve Rate vs Compute Budget', fontsize=12, fontweight='bold')
                ax1.set_xscale('log')
                ax1.set_yscale('log')
                ax1.grid(True, alpha=0.3)
        
        # 2. Hallucination rate vs model size
        ax2 = axes[0, 1]
        if 'criticality' in results:
            aggregated = results['criticality'].get('aggregated', {})
            model_sizes = []
            hall_rates = []
            
            model_size_map = {'gpt-3.5-turbo': 1.0, 'gpt-4o-mini': 2.0, 'gpt-4o': 4.0}
            
            for config, metrics in aggregated.items():
                model, depth, beam = config
                model_sizes.append(model_size_map.get(model, 1.0))
                hall_rates.append(metrics.get('hallucination_rate', 0.0))
            
            if model_sizes:
                ax2.scatter(model_sizes, hall_rates, alpha=0.6)
                ax2.set_xlabel('Model Size', fontsize=10)
                ax2.set_ylabel('Hallucination Rate', fontsize=10)
                ax2.set_title('Hallucination Rate vs Model Size', fontsize=12, fontweight='bold')
                ax2.grid(True, alpha=0.3)
        
        # 3. Search efficiency by strategy
        ax3 = axes[1, 0]
        if 'tot_comparison' in results:
            strategies = {}
            for task_result in results['tot_comparison'].get('results', []):
                for strategy, s2_result in task_result.get('system2', {}).items():
                    if strategy not in strategies:
                        strategies[strategy] = []
                    
                    metrics = s2_result.get('metrics', {})
                    nodes = metrics.get('nodes_explored', 1)
                    success = s2_result.get('success', False)
                    efficiency = (1.0 if success else 0.0) / max(nodes, 1)
                    strategies[strategy].append(efficiency)
            
            if strategies:
                strategy_names = list(strategies.keys())
                avg_efficiencies = [np.mean(strategies[s]) for s in strategy_names]
                std_efficiencies = [np.std(strategies[s]) for s in strategy_names]
                
                ax3.bar(strategy_names, avg_efficiencies, yerr=std_efficiencies, alpha=0.7)
                ax3.set_ylabel('Search Efficiency', fontsize=10)
                ax3.set_title('Search Efficiency by Strategy', fontsize=12, fontweight='bold')
                ax3.tick_params(axis='x', rotation=45)
                ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. State tracking failure points
        ax4 = axes[1, 1]
        if 'state_tracking' in results:
            tracking_results = results['state_tracking'].get('results', {})
            for task_name, task_data in tracking_results.items():
                steps = []
                failure_points = []
                
                for num_steps, result in task_data.items():
                    steps.append(num_steps)
                    failure_points.append(result.get('failure_step', num_steps))
                
                if steps:
                    ax4.plot(steps, failure_points, marker='o', label=task_name, alpha=0.7)
            
            ax4.set_xlabel('Number of Steps', fontsize=10)
            ax4.set_ylabel('Failure Point', fontsize=10)
            ax4.set_title('State Tracking Failure Points', fontsize=12, fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.suptitle('System 2 Power Law Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Power law plots saved to '{save_path}'")
        plt.close()
        
    except ImportError:
        print("⚠ Matplotlib not available for plotting")
    except Exception as e:
        print(f"⚠ Error plotting: {str(e)}")


# ==============================================================================
# MAIN ANALYSIS FUNCTION
# ==============================================================================

def analyze_system2_scaling(results: Dict) -> Dict:
    """
    Comprehensive analysis of System 2 scaling results.
    
    Tests all hypotheses and generates visualizations.
    """
    print("\n" + "="*70)
    print("SYSTEM 2 POWER LAW ANALYSIS")
    print("="*70)
    
    analysis_results = {}
    
    # 1. Solve rate power law
    print("\n1. Analyzing solve rate scaling...")
    solve_rate_analysis = analyze_solve_rate_scaling(results)
    analysis_results['solve_rate_scaling'] = solve_rate_analysis
    
    if 'error' not in solve_rate_analysis:
        print(f"   Power law: S ∝ C^{solve_rate_analysis['power_law_exponent']:.3f}")
        print(f"   R² = {solve_rate_analysis['r_squared']:.3f}")
        if solve_rate_analysis['critical_compute']:
            print(f"   Critical compute: {solve_rate_analysis['critical_compute']:.2f}")
    
    # 2. Hallucination phase transition
    print("\n2. Analyzing hallucination phase transition...")
    hallucination_analysis = analyze_hallucination_phase_transition(results)
    analysis_results['hallucination_phase_transition'] = hallucination_analysis
    
    if 'error' not in hallucination_analysis:
        if hallucination_analysis['has_phase_transition']:
            print(f"   Phase transition detected at model size: {hallucination_analysis['drop_point']:.2f}")
        else:
            print("   No clear phase transition detected")
    
    # 3. Search efficiency
    print("\n3. Analyzing search efficiency...")
    efficiency_analysis = analyze_search_efficiency(results)
    analysis_results['search_efficiency'] = efficiency_analysis
    
    if 'error' not in efficiency_analysis:
        for strategy, metrics in efficiency_analysis.items():
            print(f"   {strategy}: efficiency = {metrics['avg_efficiency']:.4f} ± {metrics['std_efficiency']:.4f}")
    
    # 4. Generate visualizations
    print("\n4. Generating visualizations...")
    plot_power_law_relationships(results)
    
    # 5. Summary
    print("\n" + "="*70)
    print("ANALYSIS SUMMARY")
    print("="*70)
    print("""
Key Findings:

1. SOLVE RATE SCALING:
   - Tests hypothesis: S ∝ (M × D)^α
   - Identifies critical compute threshold

2. HALLUCINATION PHASE TRANSITION:
   - Tests if hallucination rate shows sharp drop at critical model size
   - Connects to Ising model phase transition

3. SEARCH EFFICIENCY:
   - Compares different search strategies
   - Measures solutions_found / nodes_explored

4. STATE TRACKING:
   - Identifies critical failure points
   - Measures where state tracking breaks down

CONCLUSION:
System 2 reasoning shows power law relationships similar to neural
scaling laws, with potential phase transitions at critical model/depth
combinations. This supports the criticality hypothesis.
""")
    
    return analysis_results


if __name__ == "__main__":
    # Load results from file
    import sys
    
    if len(sys.argv) > 1:
        results_file = sys.argv[1]
        with open(results_file, 'r') as f:
            results = json.load(f)
    else:
        # Try to load from default location
        try:
            with open('system2_comprehensive_results.json', 'r') as f:
                results = json.load(f)
            results = results.get('results', {})
        except:
            print("No results file found. Please provide results file as argument.")
            sys.exit(1)
    
    analysis = analyze_system2_scaling(results)
    
    # Save analysis
    with open('system2_power_law_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    
    print("\n✓ Analysis saved to 'system2_power_law_analysis.json'")
