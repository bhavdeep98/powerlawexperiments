#!/usr/bin/env python3
"""
Power Law Experiments: Deep Learning Criticality Survey
========================================================

Main runner script for the three scientific experiments exploring:
1. Physics Proof: 2D Ising Model Phase Transition
2. LLM Proof: Neural Scaling Laws & Emergent Capabilities  
3. MAS Proof: Game Theory Coordination

This code accompanies the survey:
"A Survey on Criticality and Strategic Interaction in Deep Learning:
 From Power Law Scaling to Multi-Agent Games"

Usage:
    python main.py              # Run all experiments
    python main.py --ising      # Run only Ising Model
    python main.py --autoencoder # Run only Autoencoder scaling
    python main.py --gametheory # Run only Game Theory
    python main.py --quick      # Run quick versions (fewer iterations)

Author: Generated for Power Law Experiments Survey
"""

import argparse
import sys
import time
from datetime import timedelta


def print_banner():
    """Print the experiment banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║     ██████╗  ██████╗ ██╗    ██╗███████╗██████╗     ██╗      █████╗ ██╗    ██╗║
║     ██╔══██╗██╔═══██╗██║    ██║██╔════╝██╔══██╗    ██║     ██╔══██╗██║    ██║║
║     ██████╔╝██║   ██║██║ █╗ ██║█████╗  ██████╔╝    ██║     ███████║██║ █╗ ██║║
║     ██╔═══╝ ██║   ██║██║███╗██║██╔══╝  ██╔══██╗    ██║     ██╔══██║██║███╗██║║
║     ██║     ╚██████╔╝╚███╔███╔╝███████╗██║  ██║    ███████╗██║  ██║╚███╔███╔╝║
║     ╚═╝      ╚═════╝  ╚══╝╚══╝ ╚══════╝╚═╝  ╚═╝    ╚══════╝╚═╝  ╚═╝ ╚══╝╚══╝ ║
║                                                                              ║
║            EXPERIMENTS IN CRITICALITY & STRATEGIC INTERACTION                ║
║                                                                              ║
║     Bridging Statistical Physics, Deep Learning, and Game Theory             ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def run_ising_experiment(quick: bool = False):
    """
    Run the 2D Ising Model phase transition experiment.
    
    This demonstrates:
    - Critical phenomena at the Curie point (T_c ≈ 2.269)
    - Power law scaling near the phase transition
    - Divergence of correlation length at criticality
    """
    print("\n" + "="*78)
    print("EXPERIMENT 1: 2D ISING MODEL - PHYSICS PROOF OF CRITICALITY")
    print("="*78)
    print("""
    The Ising Model demonstrates how local spin interactions lead to
    global magnetization (emergent order). At the Curie Point (T_c ≈ 2.269),
    properties like correlation length diverge according to a power law.
    
    In AI, this implies that at a critical resource level, the information
    correlation length spans the entire network, enabling system-wide,
    coherent processing.
    """)
    
    from ising_model import run_temperature_sweep, visualize_spin_configurations, T_CRITICAL
    
    # Adjust parameters for quick mode
    if quick:
        n_temps = 11
        n_steps = 2000
        n_warmup = 500
        lattice_size = 15
    else:
        n_temps = 21
        n_steps = 5000
        n_warmup = 1000
        lattice_size = 20
    
    # Run phase transition experiment
    temperatures, magnetizations, energies = run_temperature_sweep(
        N=lattice_size,
        T_min=1.5,
        T_max=3.5,
        n_temps=n_temps,
        n_steps=n_steps,
        n_warmup=n_warmup,
        save_plot=True,
        show_plot=True
    )
    
    # Visualize spin configurations
    if not quick:
        print("\nGenerating spin configuration visualizations...")
        visualize_spin_configurations(
            N=50,
            temperatures=[1.5, T_CRITICAL, 3.5],
            n_equilibrate=1500,
            save_plot=True,
            show_plot=True
        )
    
    return temperatures, magnetizations, energies


def run_autoencoder_experiment(quick: bool = False):
    """
    Run the Neural Scaling Laws experiment with autoencoder.
    
    This demonstrates:
    - Smooth power law decay of reconstruction loss with capacity
    - Potential emergent capabilities in secondary task accuracy
    - The relationship between smooth scaling and sharp emergence
    """
    print("\n" + "="*78)
    print("EXPERIMENT 2: NEURAL SCALING LAWS - LLM PROOF OF EMERGENCE")
    print("="*78)
    print("""
    This experiment shows how:
    - Loss decreases smoothly as a power law of resources (L ∝ P^(-α))
    - Task accuracy may show sharp transitions (emergent capabilities)
    - The "mirage" vs "real emergence" debate in modern AI
    
    We use an autoencoder with variable bottleneck size as a minimal
    model for scaling laws in deep learning.
    """)
    
    from autoencoder_scaling import run_scaling_experiment, visualize_scaling_results
    
    # Adjust parameters for quick mode
    if quick:
        latent_dims = [2, 4, 6, 8, 10, 12]
        num_train = 5000
        num_test = 1000
        epochs = 30
        n_trials = 2
    else:
        latent_dims = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16]
        num_train = 10000
        num_test = 2000
        epochs = 50
        n_trials = 3
    
    results = run_scaling_experiment(
        input_dim=12,
        latent_dims=latent_dims,
        num_train=num_train,
        num_test=num_test,
        epochs=epochs,
        n_trials=n_trials,
        verbose=True
    )
    
    alpha, r_squared = visualize_scaling_results(
        results,
        save_plot=True,
        show_plot=True
    )
    
    return results


def run_gametheory_experiment(quick: bool = False):
    """
    Run the Game Theory Multi-Agent Systems experiment.
    
    This demonstrates:
    - Adversarial dynamics (GAN-inspired) improve agent competence
    - Coordination games benefit from Nash equilibrium strategies
    - Cooperation emerges in iterated games (Prisoner's Dilemma)
    """
    print("\n" + "="*78)
    print("EXPERIMENT 3: GAME THEORY - MAS PROOF OF COORDINATION")
    print("="*78)
    print("""
    Game Theory provides the structure to manage emergent dynamics in
    Multi-Agent Systems:
    
    1. ADVERSARIAL GAMES: GAN-like minimax optimization forces improvement
    2. COORDINATION GAMES: Nash equilibrium enables task allocation
    3. ITERATED GAMES: Reciprocity leads to emergent cooperation
    
    Strategic interaction transforms raw capability into robust,
    emergent task success.
    """)
    
    from game_theory_mas import run_all_experiments
    
    results = run_all_experiments(
        save_plots=True,
        show_plots=True
    )
    
    return results


def main():
    """Main entry point for running experiments."""
    parser = argparse.ArgumentParser(
        description='Power Law Experiments: Deep Learning Criticality Survey',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py              # Run all experiments
    python main.py --ising      # Run only Ising Model
    python main.py --autoencoder # Run only Autoencoder scaling
    python main.py --gametheory # Run only Game Theory
    python main.py --quick      # Run quick versions (fewer iterations)
    python main.py --quick --ising  # Quick Ising Model only
        """
    )
    
    parser.add_argument('--ising', action='store_true',
                       help='Run Ising Model experiment only')
    parser.add_argument('--autoencoder', action='store_true',
                       help='Run Autoencoder scaling experiment only')
    parser.add_argument('--gametheory', action='store_true',
                       help='Run Game Theory experiment only')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick versions with fewer iterations')
    parser.add_argument('--no-display', action='store_true',
                       help='Save plots but do not display them')
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Determine which experiments to run
    run_all = not (args.ising or args.autoencoder or args.gametheory)
    
    start_time = time.time()
    results = {}
    
    try:
        # Experiment 1: Ising Model
        if run_all or args.ising:
            exp_start = time.time()
            results['ising'] = run_ising_experiment(quick=args.quick)
            print(f"\n⏱ Ising Model completed in {timedelta(seconds=int(time.time()-exp_start))}")
        
        # Experiment 2: Autoencoder Scaling
        if run_all or args.autoencoder:
            exp_start = time.time()
            results['autoencoder'] = run_autoencoder_experiment(quick=args.quick)
            print(f"\n⏱ Autoencoder experiment completed in {timedelta(seconds=int(time.time()-exp_start))}")
        
        # Experiment 3: Game Theory
        if run_all or args.gametheory:
            exp_start = time.time()
            results['gametheory'] = run_gametheory_experiment(quick=args.quick)
            print(f"\n⏱ Game Theory experiment completed in {timedelta(seconds=int(time.time()-exp_start))}")
        
        # Final summary
        total_time = time.time() - start_time
        print("\n" + "="*78)
        print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY")
        print("="*78)
        print(f"\n⏱ Total execution time: {timedelta(seconds=int(total_time))}")
        print("\nGenerated plots:")
        
        if run_all or args.ising:
            print("  • ising_model_phase_transition.png")
            if not args.quick:
                print("  • ising_spin_configurations.png")
        
        if run_all or args.autoencoder:
            print("  • neural_scaling_laws.png")
        
        if run_all or args.gametheory:
            print("  • game_theory_adversarial.png")
            print("  • game_theory_coordination.png")
            print("  • game_theory_ipd.png")
        
        print("\n" + "="*78)
        print("CONCLUSION: The Criticality Hypothesis")
        print("="*78)
        print("""
The three experiments provide complementary evidence for the survey's
central thesis:

1. PHYSICS (Ising Model): Demonstrates that at critical points,
   small changes yield massive, power-law-governed qualitative shifts.
   The correlation length diverges, enabling system-wide coherence.

2. DEEP LEARNING (Scaling Laws): Shows that neural network loss follows
   smooth power laws, while task accuracy can exhibit sharp transitions.
   This supports the "criticality" view of emergent abilities.

3. GAME THEORY (MAS): Proves that strategic interaction (adversarial
   or cooperative) acts as a mechanism to exploit underlying capabilities,
   transforming individual competence into emergent coordination.

UNIFIED FRAMEWORK:
Modern AI systems are engineered to operate near critical points,
leveraging power law scaling for maximum sensitivity and coordination
synergy through game-theoretic structures.
        """)
        
    except KeyboardInterrupt:
        print("\n\n⚠ Experiments interrupted by user.")
        sys.exit(1)
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("Please install required packages: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        raise
    
    return results


if __name__ == "__main__":
    main()

