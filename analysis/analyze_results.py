#!/usr/bin/env python3
"""
Analyze Recorded Experiment Results
====================================

Helper script to load and analyze results from run_experiments.py

Usage:
    python analyze_results.py [results_dir]
"""

import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def load_json(filepath: str):
    """Load JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def analyze_ising_results(results_dir: str = "results"):
    """Analyze Ising Model results."""
    filepath = os.path.join(results_dir, 'ising_model_results.json')
    if not os.path.exists(filepath):
        print(f"⚠ Ising Model results not found: {filepath}")
        return None
    
    data = load_json(filepath)
    
    print("\n" + "="*70)
    print("ISING MODEL ANALYSIS")
    print("="*70)
    print(f"Critical Temperature: T_c = {data['theoretical_critical_temp']:.4f}")
    print(f"Observed Transition: T ≈ {data['observed_transition_temp']:.4f}")
    print(f"Transition Error: {data['statistics']['transition_temp_error']:.4f}")
    print(f"Magnetization Range: {data['statistics']['magnetization_range']:.4f}")
    
    return data


def analyze_scaling_results(results_dir: str = "results"):
    """Analyze Neural Scaling Laws results."""
    filepath = os.path.join(results_dir, 'neural_scaling_results.json')
    if not os.path.exists(filepath):
        print(f"⚠ Scaling results not found: {filepath}")
        return None
    
    data = load_json(filepath)
    
    print("\n" + "="*70)
    print("NEURAL SCALING LAWS ANALYSIS")
    print("="*70)
    print(f"Power Law: {data['scaling_law']['formula']}")
    print(f"Exponent (α): {data['scaling_law']['exponent']:.4f}")
    print(f"R-squared: {data['scaling_law']['r_squared']:.4f}")
    if data['emergence']['emergence_latent_dim']:
        print(f"Emergence Point: L = {data['emergence']['emergence_latent_dim']}")
    print(f"Accuracy Improvement: {data['statistics']['accuracy_improvement']:.4f}")
    
    return data


def analyze_gametheory_results(results_dir: str = "results"):
    """Analyze Game Theory results."""
    filepath = os.path.join(results_dir, 'game_theory_results.json')
    if not os.path.exists(filepath):
        print(f"⚠ Game Theory results not found: {filepath}")
        return None
    
    data = load_json(filepath)
    
    print("\n" + "="*70)
    print("GAME THEORY ANALYSIS")
    print("="*70)
    
    # Adversarial
    if 'adversarial_game' in data['sub_experiments']:
        adv = data['sub_experiments']['adversarial_game']
        print("\nAdversarial Game:")
        print(f"  Adaptive Accuracy: {adv['adaptive']['accuracy']:.4f}")
        print(f"  Static Baseline: {adv['static_baseline']['accuracy']:.4f}")
        print(f"  Improvement: {adv['improvement']['accuracy_delta']:.4f}")
    
    # Coordination
    if 'coordination_game' in data['sub_experiments']:
        coord = data['sub_experiments']['coordination_game']
        print("\nCoordination Game:")
        for method, stats in coord['results'].items():
            print(f"  {method:20s}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    # IPD
    if 'ipd_tournament' in data['sub_experiments']:
        ipd = data['sub_experiments']['ipd_tournament']
        print("\nIPD Tournament Rankings:")
        sorted_strategies = sorted(ipd['results'].items(),
                                 key=lambda x: x[1]['avg_score'], reverse=True)
        for i, (strat, result) in enumerate(sorted_strategies, 1):
            print(f"  {i}. {strat:20s}: {result['avg_score']:.1f}")
    
    return data


def plot_comparison(results_dir: str = "results", save: bool = True):
    """Create comparison plots from recorded data."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Ising Model
    ising_file = os.path.join(results_dir, 'ising_model_results.json')
    if os.path.exists(ising_file):
        data = load_json(ising_file)
        temps = np.array(data['temperature_data']['temperatures'])
        mags = np.array(data['temperature_data']['magnetizations'])
        
        ax = axes[0, 0]
        ax.plot(temps, mags, 'o-', color='#2E86AB', linewidth=2)
        ax.axvline(x=data['theoretical_critical_temp'], color='r', 
                  linestyle='--', label='T_c')
        ax.set_xlabel('Temperature')
        ax.set_ylabel('Magnetization')
        ax.set_title('Ising Model Phase Transition')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 2: Scaling Laws
    scaling_file = os.path.join(results_dir, 'neural_scaling_results.json')
    if os.path.exists(scaling_file):
        data = load_json(scaling_file)
        latent_dims = np.array(data['data']['latent_dims'])
        losses = np.array(data['data']['reconstruction_loss'])
        accs = np.array(data['data']['task_accuracy'])
        
        ax = axes[0, 1]
        ax2 = ax.twinx()
        line1 = ax.plot(latent_dims, losses, 'o-', color='#2E86AB', 
                       label='Loss', linewidth=2)
        line2 = ax2.plot(latent_dims, accs, 's-', color='#E94F37',
                        label='Accuracy', linewidth=2)
        ax.set_xlabel('Latent Dimension')
        ax.set_ylabel('Loss', color='#2E86AB')
        ax2.set_ylabel('Accuracy', color='#E94F37')
        ax.set_title('Neural Scaling Laws')
        ax.tick_params(axis='y', labelcolor='#2E86AB')
        ax2.tick_params(axis='y', labelcolor='#E94F37')
        ax.grid(True, alpha=0.3)
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='center right')
    
    # Plot 3: Coordination Game
    gt_file = os.path.join(results_dir, 'game_theory_results.json')
    if os.path.exists(gt_file):
        data = load_json(gt_file)
        if 'coordination_game' in data['sub_experiments']:
            coord = data['sub_experiments']['coordination_game']
            methods = list(coord['results'].keys())
            means = [coord['results'][m]['mean'] for m in methods]
            stds = [coord['results'][m]['std'] for m in methods]
            
            ax = axes[1, 0]
            ax.bar(methods, means, yerr=stds, capsize=5, alpha=0.7,
                  color=['#E94F37', '#2E86AB', '#27AE60'])
            ax.set_ylabel('Success Rate')
            ax.set_title('Coordination Game Comparison')
            ax.grid(True, alpha=0.3, axis='y')
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')
    
    # Plot 4: IPD Tournament
    if os.path.exists(gt_file):
        data = load_json(gt_file)
        if 'ipd_tournament' in data['sub_experiments']:
            ipd = data['sub_experiments']['ipd_tournament']
            strategies = list(ipd['results'].keys())
            scores = [ipd['results'][s]['avg_score'] for s in strategies]
            
            # Sort
            sorted_idx = np.argsort(scores)[::-1]
            strategies = [strategies[i] for i in sorted_idx]
            scores = [scores[i] for i in sorted_idx]
            
            ax = axes[1, 1]
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(strategies)))
            ax.barh(strategies, scores, color=colors, alpha=0.7)
            ax.set_xlabel('Average Score')
            ax.set_title('IPD Tournament Results')
            ax.grid(True, alpha=0.3, axis='x')
    
    plt.suptitle('Experiment Results Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save:
        output_path = os.path.join(results_dir, 'results_comparison.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Comparison plot saved to {output_path}")
    
    plt.show()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Analyze recorded experiment results'
    )
    parser.add_argument('results_dir', type=str, nargs='?', default='results',
                      help='Results directory (default: results)')
    parser.add_argument('--plot', action='store_true',
                      help='Generate comparison plots')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_dir):
        print(f"❌ Results directory not found: {args.results_dir}")
        print("   Run 'python run_experiments.py' first to generate results.")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("ANALYZING EXPERIMENT RESULTS")
    print("="*70)
    print(f"Results directory: {args.results_dir}\n")
    
    # Analyze each experiment
    ising_data = analyze_ising_results(args.results_dir)
    scaling_data = analyze_scaling_results(args.results_dir)
    gt_data = analyze_gametheory_results(args.results_dir)
    
    # Generate plots if requested
    if args.plot:
        plot_comparison(args.results_dir, save=True)
    
    # Summary
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    
    if ising_data:
        print("✓ Ising Model results analyzed")
    if scaling_data:
        print("✓ Neural Scaling results analyzed")
    if gt_data:
        print("✓ Game Theory results analyzed")
    
    print(f"\nUse --plot to generate comparison visualizations.")


if __name__ == "__main__":
    main()

