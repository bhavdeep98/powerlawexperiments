#!/usr/bin/env python3
"""
Quick Results Viewer
====================
Displays key findings from all experiments in a readable format.
"""

import json
import os
from pathlib import Path

def print_section(title):
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def load_json(filepath):
    """Load JSON file, return None if not found."""
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return None

def view_ising_results():
    """Display Ising model results."""
    data = load_json('results/ising_model_results.json')
    if not data:
        print("⚠ Ising model results not found")
        return
    
    print_section("ISING MODEL - Phase Transition")
    print(f"Theoretical Critical Temperature: T_c = {data['theoretical_critical_temp']:.4f}")
    print(f"Observed Transition: T ≈ {data['observed_transition_temp']:.4f}")
    print(f"Transition Error: {data['statistics']['transition_temp_error']:.4f}")
    print(f"Magnetization Range: {data['statistics']['magnetization_range']:.4f}")
    print(f"Low-T Magnetization: {data['statistics']['low_T_magnetization']:.4f}")
    print(f"High-T Magnetization: {data['statistics']['high_T_magnetization']:.4f}")

def view_neural_scaling():
    """Display neural scaling results."""
    data = load_json('results/neural_scaling_results.json')
    if not data:
        print("⚠ Neural scaling results not found")
        return
    
    print_section("NEURAL SCALING LAWS")
    scaling = data['scaling_law']
    print(f"Power Law Formula: {scaling['formula']}")
    print(f"Scaling Exponent (α): {scaling['exponent']:.4f}")
    print(f"R-squared: {scaling['r_squared']:.4f}")
    
    if data['emergence']['emergence_latent_dim']:
        print(f"Emergence Point: L = {data['emergence']['emergence_latent_dim']}")
    
    stats = data['statistics']
    print(f"Loss Range: {stats['min_loss']:.4f} - {stats['max_loss']:.4f}")
    print(f"Accuracy Range: {stats['min_accuracy']:.4f} - {stats['max_accuracy']:.4f}")
    print(f"Accuracy Improvement: {stats['accuracy_improvement']:.4f}")

def view_game_theory():
    """Display game theory results."""
    data = load_json('results/game_theory_results.json')
    if not data:
        print("⚠ Game theory results not found")
        return
    
    print_section("GAME THEORY & MULTI-AGENT SYSTEMS")
    
    if 'adversarial_game' in data.get('sub_experiments', {}):
        adv = data['sub_experiments']['adversarial_game']
        print("\nAdversarial Game (Coder-Critic):")
        print(f"  Adaptive Accuracy: {adv['adaptive']['accuracy']:.2%}")
        print(f"  Static Baseline: {adv['static_baseline']['accuracy']:.2%}")
        print(f"  Improvement: {adv['improvement']['accuracy_delta']:.2%}")
    
    if 'coordination_game' in data.get('sub_experiments', {}):
        coord = data['sub_experiments']['coordination_game']
        print("\nCoordination Game:")
        for method, stats in coord['results'].items():
            print(f"  {method:20s}: {stats['mean']:.2%} ± {stats['std']:.2%}")
    
    if 'ipd_tournament' in data.get('sub_experiments', {}):
        ipd = data['sub_experiments']['ipd_tournament']
        print("\nIPD Tournament Rankings:")
        sorted_strategies = sorted(ipd['results'].items(),
                                 key=lambda x: x[1]['avg_score'], reverse=True)
        for i, (strat, result) in enumerate(sorted_strategies, 1):
            print(f"  {i}. {strat:20s}: {result['avg_score']:.1f} points")

def view_system2_baseline():
    """Display System 2 baseline results."""
    data = load_json('results/system2/baseline_results.json')
    if not data:
        print("⚠ System 2 baseline results not found")
        return
    
    print_section("SYSTEM 2 - Baseline Performance (GPT-4o Zero-shot)")
    
    for benchmark, results in data.items():
        accuracy = results.get('accuracy', 0)
        num_problems = results.get('num_problems', 0)
        print(f"\n{benchmark.replace('_', ' ').title()}:")
        print(f"  Accuracy: {accuracy:.1%} ({int(accuracy * num_problems)}/{num_problems} problems)")

def view_system2_scaling():
    """Display System 2 scaling results summary."""
    data = load_json('results/system2/scaling_results.json')
    if not data:
        print("⚠ System 2 scaling results not found")
        return
    
    print_section("SYSTEM 2 - Scaling Results")
    
    aggregated = data.get('aggregated', {})
    print(f"Total configurations tested: {len(aggregated)}")
    
    # Group by model
    by_model = {}
    for config_key, metrics in aggregated.items():
        if isinstance(config_key, str):
            model = config_key.split('_')[0]
        else:
            model = config_key[0]
        
        if model not in by_model:
            by_model[model] = []
        by_model[model].append((config_key, metrics))
    
    print("\nPerformance by Model:")
    for model, configs in sorted(by_model.items()):
        solve_rates = [m['solve_rate'] for _, m in configs]
        hall_rates = [m['hallucination_rate'] for _, m in configs]
        print(f"\n  {model}:")
        print(f"    Avg Solve Rate: {sum(solve_rates)/len(solve_rates):.1%}")
        print(f"    Avg Hallucination Rate: {sum(hall_rates)/len(hall_rates):.1%}")
        print(f"    Best Solve Rate: {max(solve_rates):.1%}")
    
    # Show critical points
    if 'critical_points' in data and data['critical_points']:
        print("\nCritical Points (where performance jumps):")
        for cp in data['critical_points'][:3]:  # Show first 3
            config = cp['config']
            print(f"  {config[0]} (depth={config[1]}, beam={config[2]}): "
                  f"solve_rate={cp['solve_rate']:.1%}")

def view_system2_power_law():
    """Display System 2 power law analysis."""
    data = load_json('system2_power_law_analysis.json')
    if not data:
        print("⚠ System 2 power law analysis not found")
        return
    
    print_section("SYSTEM 2 - Power Law Analysis")
    
    if 'solve_rate_scaling' in data:
        srs = data['solve_rate_scaling']
        if 'error' not in srs:
            print(f"Power Law: S ∝ C^{srs['power_law_exponent']:.3f}")
            print(f"Coefficient: {srs['power_law_coefficient']:.3f}")
            print(f"R² = {srs['r_squared']:.3f}")
            if srs.get('critical_compute'):
                print(f"Critical Compute Threshold: {srs['critical_compute']:.2f}")
    
    if 'hallucination_phase_transition' in data:
        hpt = data['hallucination_phase_transition']
        if 'error' not in hpt:
            print(f"\nHallucination Phase Transition:")
            print(f"  Max Drop: {hpt['max_drop']:.3f}")
            print(f"  Drop Point: {hpt['drop_point']}")
            print(f"  Has Phase Transition: {hpt['has_phase_transition']}")

def main():
    """Main entry point."""
    print("\n" + "="*70)
    print("  POWER LAW EXPERIMENTS - RESULTS SUMMARY")
    print("="*70)
    
    # Check if results directory exists
    if not os.path.exists('results'):
        print("\n❌ Results directory not found!")
        print("   Run experiments first using: python3 run_experiments.py")
        return
    
    # Display all results
    view_ising_results()
    view_neural_scaling()
    view_game_theory()
    view_system2_baseline()
    view_system2_scaling()
    view_system2_power_law()
    
    print("\n" + "="*70)
    print("  SUMMARY COMPLETE")
    print("="*70)
    print("\nFor detailed analysis, see: RESULTS_SUMMARY.md")
    print("For visualizations, run: python3 analyze_results.py results --plot")

if __name__ == "__main__":
    main()
