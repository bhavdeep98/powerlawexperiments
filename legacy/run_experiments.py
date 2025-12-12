#!/usr/bin/env python3
"""
Run All Experiments and Record Results
======================================

This script runs all three experiments and saves comprehensive results
to JSON and CSV files for later analysis and reproducibility.

Usage:
    python run_experiments.py [--quick] [--output-dir RESULTS]
"""

import argparse
import json
import csv
import os
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import time
from typing import Dict, Any


def ensure_dir(path: str):
    """Ensure directory exists."""
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(data: Dict[str, Any], filepath: str):
    """Save data to JSON file with proper numpy serialization."""
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj
    
    serializable_data = convert_numpy(data)
    with open(filepath, 'w') as f:
        json.dump(serializable_data, f, indent=2)


def run_ising_experiment(quick: bool = False, output_dir: str = "results") -> Dict[str, Any]:
    """Run Ising Model experiment and record results."""
    print("\n" + "="*70)
    print("EXPERIMENT 1: 2D ISING MODEL")
    print("="*70)
    
    from ising_model import run_temperature_sweep, T_CRITICAL, run_simulation
    
    # Configuration
    if quick:
        config = {
            'lattice_size': 15,
            'n_temps': 11,
            'n_steps': 2000,
            'n_warmup': 500,
            'T_min': 1.5,
            'T_max': 3.5
        }
    else:
        config = {
            'lattice_size': 20,
            'n_temps': 21,
            'n_steps': 5000,
            'n_warmup': 1000,
            'T_min': 1.5,
            'T_max': 3.5
        }
    
    start_time = time.time()
    
    # Run experiment
    temperatures, magnetizations, energies = run_temperature_sweep(
        N=config['lattice_size'],
        T_min=config['T_min'],
        T_max=config['T_max'],
        n_temps=config['n_temps'],
        n_steps=config['n_steps'],
        n_warmup=config['n_warmup'],
        save_plot=True,
        show_plot=False  # Don't display, just save
    )
    
    execution_time = time.time() - start_time
    
    # Find transition point
    transition_idx = np.argmax(np.abs(np.diff(magnetizations)))
    T_observed = (temperatures[transition_idx] + temperatures[transition_idx + 1]) / 2
    
    # Calculate statistics
    low_T_mag = magnetizations[0]
    high_T_mag = magnetizations[-1]
    max_mag = np.max(magnetizations)
    min_mag = np.min(magnetizations)
    
    # Results dictionary
    results = {
        'experiment': 'ising_model',
        'timestamp': datetime.now().isoformat(),
        'execution_time_seconds': execution_time,
        'configuration': config,
        'theoretical_critical_temp': float(T_CRITICAL),
        'observed_transition_temp': float(T_observed),
        'temperature_data': {
            'temperatures': temperatures.tolist(),
            'magnetizations': magnetizations.tolist(),
            'energies': energies.tolist()
        },
        'statistics': {
            'low_T_magnetization': float(low_T_mag),
            'high_T_magnetization': float(high_T_mag),
            'max_magnetization': float(max_mag),
            'min_magnetization': float(min_mag),
            'magnetization_range': float(max_mag - min_mag),
            'transition_temp_error': float(abs(T_observed - T_CRITICAL))
        }
    }
    
    # Save results
    ensure_dir(output_dir)
    json_path = os.path.join(output_dir, 'ising_model_results.json')
    save_json(results, json_path)
    
    # Save CSV for easy plotting
    csv_path = os.path.join(output_dir, 'ising_model_data.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['temperature', 'magnetization', 'energy'])
        for T, M, E in zip(temperatures, magnetizations, energies):
            writer.writerow([float(T), float(M), float(E)])
    
    print(f"✓ Results saved to {json_path}")
    print(f"✓ Data saved to {csv_path}")
    print(f"  Execution time: {execution_time:.2f} seconds")
    print(f"  Observed transition: T ≈ {T_observed:.4f} (theoretical: {T_CRITICAL:.4f})")
    
    return results


def run_autoencoder_experiment(quick: bool = False, output_dir: str = "results") -> Dict[str, Any]:
    """Run Autoencoder scaling experiment and record results."""
    print("\n" + "="*70)
    print("EXPERIMENT 2: NEURAL SCALING LAWS")
    print("="*70)
    
    from autoencoder_scaling import run_scaling_experiment, fit_power_law
    
    # Configuration
    if quick:
        config = {
            'input_dim': 12,
            'latent_dims': [2, 4, 6, 8, 10, 12],
            'num_train': 5000,
            'num_test': 1000,
            'epochs': 30,
            'batch_size': 64,
            'n_trials': 2
        }
    else:
        config = {
            'input_dim': 12,
            'latent_dims': [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16],
            'num_train': 10000,
            'num_test': 2000,
            'epochs': 50,
            'batch_size': 64,
            'n_trials': 3
        }
    
    start_time = time.time()
    
    # Run experiment
    results_dict = run_scaling_experiment(
        input_dim=config['input_dim'],
        latent_dims=config['latent_dims'],
        num_train=config['num_train'],
        num_test=config['num_test'],
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        n_trials=config['n_trials'],
        verbose=True
    )
    
    execution_time = time.time() - start_time
    
    # Fit power law
    latent_dims = np.array(results_dict['latent_dims'])
    recon_loss = np.array(results_dict['reconstruction_loss'])
    alpha, a, r_squared = fit_power_law(latent_dims, recon_loss)
    
    # Find emergence point (largest accuracy jump)
    task_acc = np.array(results_dict['task_accuracy'])
    if len(task_acc) > 1:
        acc_diff = np.diff(task_acc)
        emergence_idx = np.argmax(acc_diff) + 1
        emergence_latent_dim = latent_dims[emergence_idx] if emergence_idx < len(latent_dims) else None
    else:
        emergence_idx = None
        emergence_latent_dim = None
    
    # Results dictionary
    results = {
        'experiment': 'neural_scaling_laws',
        'timestamp': datetime.now().isoformat(),
        'execution_time_seconds': execution_time,
        'configuration': config,
        'scaling_law': {
            'exponent': float(alpha),
            'coefficient': float(a),
            'r_squared': float(r_squared),
            'formula': f'Loss = {a:.4f} * L^(-{alpha:.4f})'
        },
        'emergence': {
            'emergence_latent_dim': float(emergence_latent_dim) if emergence_latent_dim else None,
            'emergence_index': int(emergence_idx) if emergence_idx else None
        },
        'data': {
            'latent_dims': results_dict['latent_dims'],
            'reconstruction_loss': results_dict['reconstruction_loss'],
            'reconstruction_loss_std': results_dict['reconstruction_loss_std'],
            'task_accuracy': results_dict['task_accuracy'],
            'task_accuracy_std': results_dict['task_accuracy_std'],
            'task_loss': results_dict['task_loss'],
            'task_loss_std': results_dict['task_loss_std']
        },
        'statistics': {
            'min_loss': float(np.min(recon_loss)),
            'max_loss': float(np.max(recon_loss)),
            'min_accuracy': float(np.min(task_acc)),
            'max_accuracy': float(np.max(task_acc)),
            'accuracy_improvement': float(np.max(task_acc) - np.min(task_acc))
        }
    }
    
    # Save results
    ensure_dir(output_dir)
    json_path = os.path.join(output_dir, 'neural_scaling_results.json')
    save_json(results, json_path)
    
    # Save CSV
    csv_path = os.path.join(output_dir, 'neural_scaling_data.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['latent_dim', 'reconstruction_loss', 'reconstruction_loss_std',
                         'task_accuracy', 'task_accuracy_std', 'task_loss', 'task_loss_std'])
        for i, L in enumerate(results_dict['latent_dims']):
            writer.writerow([
                L,
                results_dict['reconstruction_loss'][i],
                results_dict['reconstruction_loss_std'][i],
                results_dict['task_accuracy'][i],
                results_dict['task_accuracy_std'][i],
                results_dict['task_loss'][i],
                results_dict['task_loss_std'][i]
            ])
    
    print(f"✓ Results saved to {json_path}")
    print(f"✓ Data saved to {csv_path}")
    print(f"  Execution time: {execution_time:.2f} seconds")
    print(f"  Power law exponent: α = {alpha:.4f} (R² = {r_squared:.4f})")
    if emergence_latent_dim:
        print(f"  Emergence point: L = {emergence_latent_dim}")
    
    return results


def run_gametheory_experiment(quick: bool = False, output_dir: str = "results") -> Dict[str, Any]:
    """Run Game Theory experiments and record results."""
    print("\n" + "="*70)
    print("EXPERIMENT 3: GAME THEORY & MULTI-AGENT SYSTEMS")
    print("="*70)
    
    from game_theory_mas import (
        run_adversarial_game, AgentConfig,
        CoordinationGame, run_ipd_tournament, Strategy
    )
    
    start_time = time.time()
    results = {
        'experiment': 'game_theory_mas',
        'timestamp': datetime.now().isoformat(),
        'sub_experiments': {}
    }
    
    # ========================================
    # Sub-experiment 1: Adversarial Game
    # ========================================
    print("\n  Running Adversarial Game...")
    exp1_start = time.time()
    
    coder_config = AgentConfig("Coder", base_competence=0.6, learning_rate=0.005)
    critic_config = AgentConfig("Critic", base_competence=0.6, learning_rate=0.005)
    
    adaptive_results = run_adversarial_game(
        n_rounds=2000,
        coder_config=coder_config,
        critic_config=critic_config,
        adaptive=True
    )
    
    # Static baseline
    coder_static = AgentConfig("Coder", base_competence=0.6, learning_rate=0)
    critic_static = AgentConfig("Critic", base_competence=0.6, learning_rate=0)
    
    static_results = run_adversarial_game(
        n_rounds=2000,
        coder_config=coder_static,
        critic_config=critic_static,
        adaptive=False
    )
    
    results['sub_experiments']['adversarial_game'] = {
        'execution_time_seconds': time.time() - exp1_start,
        'adaptive': {
            'initial_competence': 0.6,
            'final_coder_competence': float(adaptive_results['final_coder_competence']),
            'final_critic_competence': float(adaptive_results['final_critic_competence']),
            'accuracy': float(adaptive_results['accuracy']),
            'precision': float(adaptive_results['precision']),
            'recall': float(adaptive_results['recall']),
            'true_positives': int(adaptive_results['true_positives']),
            'true_negatives': int(adaptive_results['true_negatives']),
            'false_positives': int(adaptive_results['false_positives']),
            'false_negatives': int(adaptive_results['false_negatives']),
            'coder_history': adaptive_results['coder_history'],
            'critic_history': adaptive_results['critic_history']
        },
        'static_baseline': {
            'accuracy': float(static_results['accuracy']),
            'precision': float(static_results['precision']),
            'recall': float(static_results['recall'])
        },
        'improvement': {
            'accuracy_delta': float(adaptive_results['accuracy'] - static_results['accuracy']),
            'precision_delta': float(adaptive_results['precision'] - static_results['precision']),
            'recall_delta': float(adaptive_results['recall'] - static_results['recall'])
        }
    }
    
    # ========================================
    # Sub-experiment 2: Coordination Game
    # ========================================
    print("  Running Coordination Game...")
    exp2_start = time.time()
    
    n_trials = 50 if not quick else 20
    coordination_results = {
        'Random': {'scores': []},
        'Greedy': {'scores': []},
        'Nash Equilibrium': {'scores': []}
    }
    
    for _ in range(n_trials):
        game = CoordinationGame(n_agents=5, n_tasks=10)
        _, random_success = game.random_allocation()
        coordination_results['Random']['scores'].append(random_success)
        
        game = CoordinationGame(n_agents=5, n_tasks=10)
        _, greedy_success = game.greedy_allocation()
        coordination_results['Greedy']['scores'].append(greedy_success)
        
        game = CoordinationGame(n_agents=5, n_tasks=10)
        _, nash_success = game.nash_equilibrium_allocation()
        coordination_results['Nash Equilibrium']['scores'].append(nash_success)
    
    coordination_stats = {}
    for method in coordination_results:
        scores = coordination_results[method]['scores']
        coordination_stats[method] = {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'min': float(np.min(scores)),
            'max': float(np.max(scores)),
            'scores': scores
        }
    
    results['sub_experiments']['coordination_game'] = {
        'execution_time_seconds': time.time() - exp2_start,
        'n_trials': n_trials,
        'configuration': {'n_agents': 5, 'n_tasks': 10},
        'results': coordination_stats
    }
    
    # ========================================
    # Sub-experiment 3: IPD Tournament
    # ========================================
    print("  Running IPD Tournament...")
    exp3_start = time.time()
    
    strategies = [
        Strategy.ALWAYS_COOPERATE,
        Strategy.ALWAYS_DEFECT,
        Strategy.TIT_FOR_TAT,
        Strategy.RANDOM,
        Strategy.PAVLOV
    ]
    
    ipd_results = run_ipd_tournament(strategies, n_rounds=100, n_matches=20)
    
    # Convert to serializable format
    ipd_serializable = {}
    for strat, data in ipd_results.items():
        ipd_serializable[strat] = {
            'total_score': int(data['total_score']),
            'matches': int(data['matches']),
            'avg_score': float(data['avg_score'])
        }
    
    results['sub_experiments']['ipd_tournament'] = {
        'execution_time_seconds': time.time() - exp3_start,
        'configuration': {'n_rounds': 100, 'n_matches': 20},
        'results': ipd_serializable
    }
    
    execution_time = time.time() - start_time
    results['execution_time_seconds'] = execution_time
    
    # Save results
    ensure_dir(output_dir)
    json_path = os.path.join(output_dir, 'game_theory_results.json')
    save_json(results, json_path)
    
    # Save CSV for coordination game
    csv_path = os.path.join(output_dir, 'coordination_game_data.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['method', 'trial', 'success_rate'])
        for method in coordination_results:
            for i, score in enumerate(coordination_results[method]['scores']):
                writer.writerow([method, i+1, score])
    
    # Save CSV for IPD tournament
    csv_path_ipd = os.path.join(output_dir, 'ipd_tournament_data.csv')
    with open(csv_path_ipd, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['strategy', 'total_score', 'matches', 'avg_score'])
        for strat, data in ipd_results.items():
            writer.writerow([strat, data['total_score'], data['matches'], data['avg_score']])
    
    print(f"✓ Results saved to {json_path}")
    print(f"✓ Coordination data saved to {csv_path}")
    print(f"✓ IPD tournament data saved to {csv_path_ipd}")
    print(f"  Execution time: {execution_time:.2f} seconds")
    
    return results


def generate_summary_report(all_results: Dict[str, Any], output_dir: str = "results"):
    """Generate a human-readable summary report."""
    report_path = os.path.join(output_dir, 'experiment_summary.txt')
    
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("POWER LAW EXPERIMENTS - SUMMARY REPORT\n")
        f.write("="*70 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Experiment 1: Ising Model
        if 'ising' in all_results:
            ising = all_results['ising']
            f.write("EXPERIMENT 1: 2D ISING MODEL\n")
            f.write("-"*70 + "\n")
            f.write(f"Execution Time: {ising['execution_time_seconds']:.2f} seconds\n")
            f.write(f"Theoretical Critical Temperature: T_c = {ising['theoretical_critical_temp']:.4f}\n")
            f.write(f"Observed Transition Temperature: T ≈ {ising['observed_transition_temp']:.4f}\n")
            f.write(f"Transition Error: {ising['statistics']['transition_temp_error']:.4f}\n")
            f.write(f"Low-T Magnetization: {ising['statistics']['low_T_magnetization']:.4f}\n")
            f.write(f"High-T Magnetization: {ising['statistics']['high_T_magnetization']:.4f}\n")
            f.write(f"Magnetization Range: {ising['statistics']['magnetization_range']:.4f}\n\n")
        
        # Experiment 2: Neural Scaling
        if 'autoencoder' in all_results:
            scaling = all_results['autoencoder']
            f.write("EXPERIMENT 2: NEURAL SCALING LAWS\n")
            f.write("-"*70 + "\n")
            f.write(f"Execution Time: {scaling['execution_time_seconds']:.2f} seconds\n")
            f.write(f"Power Law Formula: {scaling['scaling_law']['formula']}\n")
            f.write(f"Scaling Exponent (α): {scaling['scaling_law']['exponent']:.4f}\n")
            f.write(f"R-squared: {scaling['scaling_law']['r_squared']:.4f}\n")
            if scaling['emergence']['emergence_latent_dim']:
                f.write(f"Emergence Point: L = {scaling['emergence']['emergence_latent_dim']}\n")
            f.write(f"Loss Range: {scaling['statistics']['min_loss']:.4f} - {scaling['statistics']['max_loss']:.4f}\n")
            f.write(f"Accuracy Range: {scaling['statistics']['min_accuracy']:.4f} - {scaling['statistics']['max_accuracy']:.4f}\n")
            f.write(f"Accuracy Improvement: {scaling['statistics']['accuracy_improvement']:.4f}\n\n")
        
        # Experiment 3: Game Theory
        if 'gametheory' in all_results:
            gt = all_results['gametheory']
            f.write("EXPERIMENT 3: GAME THEORY & MULTI-AGENT SYSTEMS\n")
            f.write("-"*70 + "\n")
            f.write(f"Execution Time: {gt['execution_time_seconds']:.2f} seconds\n\n")
            
            # Adversarial
            if 'adversarial_game' in gt['sub_experiments']:
                adv = gt['sub_experiments']['adversarial_game']
                f.write("Adversarial Game:\n")
                f.write(f"  Adaptive Accuracy: {adv['adaptive']['accuracy']:.4f}\n")
                f.write(f"  Static Baseline: {adv['static_baseline']['accuracy']:.4f}\n")
                f.write(f"  Improvement: {adv['improvement']['accuracy_delta']:.4f}\n")
                f.write(f"  Final Coder Competence: {adv['adaptive']['final_coder_competence']:.4f}\n")
                f.write(f"  Final Critic Competence: {adv['adaptive']['final_critic_competence']:.4f}\n\n")
            
            # Coordination
            if 'coordination_game' in gt['sub_experiments']:
                coord = gt['sub_experiments']['coordination_game']
                f.write("Coordination Game:\n")
                for method, stats in coord['results'].items():
                    f.write(f"  {method:20s}: {stats['mean']:.4f} ± {stats['std']:.4f}\n")
                f.write("\n")
            
            # IPD
            if 'ipd_tournament' in gt['sub_experiments']:
                ipd = gt['sub_experiments']['ipd_tournament']
                f.write("IPD Tournament Rankings:\n")
                sorted_strategies = sorted(ipd['results'].items(), 
                                         key=lambda x: x[1]['avg_score'], reverse=True)
                for i, (strat, data) in enumerate(sorted_strategies, 1):
                    f.write(f"  {i}. {strat:20s}: {data['avg_score']:.1f} points\n")
                f.write("\n")
        
        # Total time
        total_time = sum([
            all_results.get('ising', {}).get('execution_time_seconds', 0),
            all_results.get('autoencoder', {}).get('execution_time_seconds', 0),
            all_results.get('gametheory', {}).get('execution_time_seconds', 0)
        ])
        f.write("="*70 + "\n")
        f.write(f"TOTAL EXECUTION TIME: {total_time:.2f} seconds ({total_time/60:.2f} minutes)\n")
        f.write("="*70 + "\n")
    
    print(f"✓ Summary report saved to {report_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run all experiments and record results',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--quick', action='store_true',
                       help='Run quick versions with fewer iterations')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for results (default: results)')
    parser.add_argument('--skip-ising', action='store_true',
                       help='Skip Ising Model experiment')
    parser.add_argument('--skip-autoencoder', action='store_true',
                       help='Skip Autoencoder experiment')
    parser.add_argument('--skip-gametheory', action='store_true',
                       help='Skip Game Theory experiment')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("POWER LAW EXPERIMENTS - RESULTS RECORDING")
    print("="*70)
    print(f"Output directory: {args.output_dir}")
    print(f"Quick mode: {args.quick}")
    print("="*70)
    
    all_results = {}
    
    try:
        # Experiment 1: Ising Model
        if not args.skip_ising:
            all_results['ising'] = run_ising_experiment(
                quick=args.quick,
                output_dir=args.output_dir
            )
        
        # Experiment 2: Autoencoder
        if not args.skip_autoencoder:
            all_results['autoencoder'] = run_autoencoder_experiment(
                quick=args.quick,
                output_dir=args.output_dir
            )
        
        # Experiment 3: Game Theory
        if not args.skip_gametheory:
            all_results['gametheory'] = run_gametheory_experiment(
                quick=args.quick,
                output_dir=args.output_dir
            )
        
        # Generate summary report
        generate_summary_report(all_results, output_dir=args.output_dir)
        
        # Save combined results
        combined_path = os.path.join(args.output_dir, 'all_results.json')
        save_json(all_results, combined_path)
        print(f"\n✓ Combined results saved to {combined_path}")
        
        print("\n" + "="*70)
        print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY")
        print("="*70)
        print(f"\nResults saved in: {args.output_dir}/")
        print("  • ising_model_results.json")
        print("  • ising_model_data.csv")
        print("  • neural_scaling_results.json")
        print("  • neural_scaling_data.csv")
        print("  • game_theory_results.json")
        print("  • coordination_game_data.csv")
        print("  • ipd_tournament_data.csv")
        print("  • experiment_summary.txt")
        print("  • all_results.json")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Experiments interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

