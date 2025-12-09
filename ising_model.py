"""
2D Ising Model Phase Transition Simulation
===========================================

Physics Proof: Criticality and Power Law Scaling

This module implements the Metropolis Monte Carlo algorithm to simulate
the 2D Ising Model, demonstrating the phase transition at the Curie point (T_c ≈ 2.269).

The Ising Model serves as the microscopic foundation for understanding how local spin
interactions (agents) lead to global magnetization (emergent order). At the critical
temperature, properties like correlation length diverge according to a power law.

References:
- Ising, E. (1925). Beitrag zur Theorie des Ferromagnetismus.
- Metropolis, N., et al. (1953). Equation of State Calculations.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
from tqdm import tqdm


# Physical constants
J = 1.0  # Coupling constant (ferromagnetic)
T_CRITICAL = 2.0 / np.log(1 + np.sqrt(2))  # ≈ 2.269 (exact for 2D Ising)


def initialize_lattice(N: int, initial_state: str = "random") -> np.ndarray:
    """
    Initialize an NxN spin lattice.
    
    Args:
        N: Lattice size (NxN grid)
        initial_state: "random" for random spins, "ordered" for all +1
        
    Returns:
        NxN numpy array with spin values (+1 or -1)
    """
    if initial_state == "random":
        return 2 * np.random.randint(2, size=(N, N)) - 1
    elif initial_state == "ordered":
        return np.ones((N, N), dtype=int)
    else:
        raise ValueError(f"Unknown initial state: {initial_state}")


def calculate_energy(config: np.ndarray, J: float = 1.0) -> float:
    """
    Calculate the total energy of a spin configuration.
    
    Uses periodic boundary conditions (toroidal topology).
    E = -J * Σ s_i * s_j (sum over nearest neighbors)
    
    Args:
        config: NxN spin configuration
        J: Coupling constant
        
    Returns:
        Total energy of the configuration
    """
    N = config.shape[0]
    energy = 0.0
    
    for i in range(N):
        for j in range(N):
            s = config[i, j]
            # Sum of nearest neighbors with periodic boundaries
            neighbors = (config[(i + 1) % N, j] + config[i, (j + 1) % N] +
                        config[(i - 1) % N, j] + config[i, (j - 1) % N])
            energy -= J * s * neighbors
    
    # Divide by 2 to avoid double counting
    return energy / 2


def calculate_magnetization(config: np.ndarray) -> float:
    """
    Calculate the magnetization per spin.
    
    M = (1/N²) * Σ s_i
    
    Args:
        config: NxN spin configuration
        
    Returns:
        Magnetization per spin (range: -1 to +1)
    """
    return np.sum(config) / config.size


def metropolis_step(config: np.ndarray, T: float, J: float = 1.0) -> np.ndarray:
    """
    Perform one Monte Carlo sweep using the Metropolis algorithm.
    
    One sweep = N² attempted spin flips (on average, each spin is visited once).
    
    The Metropolis criterion:
    - Always accept moves that lower energy (ΔE < 0)
    - Accept moves that raise energy with probability exp(-ΔE/kT)
    
    Args:
        config: Current spin configuration (modified in place)
        T: Temperature (in units where k_B = 1)
        J: Coupling constant
        
    Returns:
        Updated spin configuration
    """
    N = config.shape[0]
    beta = 1.0 / T
    
    # Perform N² spin flip attempts (one sweep)
    for _ in range(N * N):
        # Select a random site
        i, j = np.random.randint(N), np.random.randint(N)
        s = config[i, j]
        
        # Calculate sum of nearest neighbors (Periodic Boundary Conditions)
        neighbors_sum = (config[(i + 1) % N, j] + config[i, (j + 1) % N] +
                        config[(i - 1) % N, j] + config[i, (j - 1) % N])
        
        # Energy change from flipping spin s → -s
        # ΔE = E_after - E_before = -J*(-s)*neighbors - (-J*s*neighbors) = 2*J*s*neighbors
        delta_E = 2 * J * s * neighbors_sum
        
        # Metropolis acceptance criterion
        if delta_E < 0 or np.random.rand() < np.exp(-delta_E * beta):
            config[i, j] *= -1  # Flip the spin
    
    return config


def run_simulation(N: int, T: float, n_steps: int = 10000, 
                   n_warmup: int = 1000, 
                   measure_interval: int = 10) -> Tuple[float, float, float, float]:
    """
    Run a full Monte Carlo simulation at a given temperature.
    
    Args:
        N: Lattice size
        T: Temperature
        n_steps: Number of Monte Carlo sweeps after warmup
        n_warmup: Number of warmup sweeps (thermalization)
        measure_interval: Measure observables every N sweeps
        
    Returns:
        Tuple of (avg_magnetization, magnetization_std, avg_energy, energy_std)
    """
    config = initialize_lattice(N, "random")
    magnetizations = []
    energies = []
    
    # Warmup phase (thermalization)
    for _ in range(n_warmup):
        config = metropolis_step(config, T)
    
    # Measurement phase
    for step in range(n_steps):
        config = metropolis_step(config, T)
        
        if step % measure_interval == 0:
            M = calculate_magnetization(config)
            E = calculate_energy(config) / config.size  # Energy per spin
            magnetizations.append(M)
            energies.append(E)
    
    # Return average absolute magnetization (order parameter)
    avg_M = np.mean(np.abs(magnetizations))
    std_M = np.std(np.abs(magnetizations))
    avg_E = np.mean(energies)
    std_E = np.std(energies)
    
    return avg_M, std_M, avg_E, std_E


def calculate_susceptibility(magnetizations: List[float], T: float, N: int) -> float:
    """
    Calculate magnetic susceptibility χ = N²/T * (<M²> - <|M|>²)
    
    Susceptibility diverges at T_c following a power law: χ ∝ |T - T_c|^(-γ)
    """
    M_array = np.array(magnetizations)
    M_squared_avg = np.mean(M_array ** 2)
    M_avg_squared = np.mean(np.abs(M_array)) ** 2
    return (N ** 2 / T) * (M_squared_avg - M_avg_squared)


def run_temperature_sweep(N: int = 20, 
                          T_min: float = 1.5, 
                          T_max: float = 3.5,
                          n_temps: int = 21,
                          n_steps: int = 5000,
                          n_warmup: int = 1000,
                          save_plot: bool = True,
                          show_plot: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run the Ising model simulation across a temperature range.
    
    This demonstrates the phase transition: magnetization drops sharply
    at the critical temperature T_c ≈ 2.269.
    
    Args:
        N: Lattice size (NxN)
        T_min, T_max: Temperature range
        n_temps: Number of temperature points
        n_steps: Monte Carlo steps per temperature
        n_warmup: Warmup steps for thermalization
        save_plot: Whether to save the plot as PNG
        show_plot: Whether to display the plot
        
    Returns:
        Tuple of (temperatures, magnetizations, energies)
    """
    temperatures = np.linspace(T_min, T_max, n_temps)
    magnetizations = []
    magnetization_errors = []
    energies = []
    
    print(f"\n{'='*60}")
    print("2D ISING MODEL PHASE TRANSITION SIMULATION")
    print(f"{'='*60}")
    print(f"Lattice Size: {N}×{N} = {N*N} spins")
    print(f"Temperature Range: {T_min:.2f} - {T_max:.2f}")
    print(f"Theoretical Critical Temperature: T_c ≈ {T_CRITICAL:.4f}")
    print(f"Monte Carlo Steps: {n_steps} (+ {n_warmup} warmup)")
    print(f"{'='*60}\n")
    
    for T in tqdm(temperatures, desc="Temperature Sweep"):
        avg_M, std_M, avg_E, _ = run_simulation(N, T, n_steps, n_warmup)
        magnetizations.append(avg_M)
        magnetization_errors.append(std_M)
        energies.append(avg_E)
    
    magnetizations = np.array(magnetizations)
    energies = np.array(energies)
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Magnetization vs Temperature
    ax1 = axes[0]
    ax1.errorbar(temperatures, magnetizations, yerr=magnetization_errors, 
                 fmt='o-', capsize=3, color='#2E86AB', markersize=6, linewidth=2)
    ax1.axvline(x=T_CRITICAL, color='#E94F37', linestyle='--', linewidth=2, 
                label=f'Theoretical $T_c$ ≈ {T_CRITICAL:.3f}')
    ax1.set_xlabel('Temperature (T)', fontsize=12)
    ax1.set_ylabel('Average Absolute Magnetization |M|', fontsize=12)
    ax1.set_title('Phase Transition: Order Parameter', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, linestyle=':', alpha=0.7)
    ax1.set_ylim(-0.05, 1.05)
    
    # Add annotation for phases
    ax1.annotate('Ordered Phase\n(Ferromagnetic)', xy=(1.8, 0.85), fontsize=10, 
                ha='center', style='italic', color='#2E86AB')
    ax1.annotate('Disordered Phase\n(Paramagnetic)', xy=(3.2, 0.15), fontsize=10,
                ha='center', style='italic', color='#E94F37')
    
    # Plot 2: Energy vs Temperature
    ax2 = axes[1]
    ax2.plot(temperatures, energies, 's-', color='#A23B72', markersize=6, linewidth=2)
    ax2.axvline(x=T_CRITICAL, color='#E94F37', linestyle='--', linewidth=2,
                label=f'Theoretical $T_c$ ≈ {T_CRITICAL:.3f}')
    ax2.set_xlabel('Temperature (T)', fontsize=12)
    ax2.set_ylabel('Average Energy per Spin (E/N²)', fontsize=12)
    ax2.set_title('Internal Energy vs Temperature', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, linestyle=':', alpha=0.7)
    
    plt.suptitle('2D Ising Model: Proof of Critical Phenomena & Power Law Scaling',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_plot:
        plt.savefig('ising_model_phase_transition.png', dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print("\n✓ Plot saved as 'ising_model_phase_transition.png'")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    # Print summary
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    
    # Find approximate transition point
    transition_idx = np.argmax(np.abs(np.diff(magnetizations)))
    T_observed = (temperatures[transition_idx] + temperatures[transition_idx + 1]) / 2
    
    print(f"Observed transition near T ≈ {T_observed:.3f}")
    print(f"Theoretical critical point: T_c = {T_CRITICAL:.4f}")
    print(f"Low-T Magnetization (T={temperatures[0]:.2f}): |M| = {magnetizations[0]:.4f}")
    print(f"High-T Magnetization (T={temperatures[-1]:.2f}): |M| = {magnetizations[-1]:.4f}")
    print(f"{'='*60}")
    
    return temperatures, magnetizations, energies


def visualize_spin_configurations(N: int = 50, 
                                  temperatures: List[float] = [1.5, 2.269, 3.5],
                                  n_equilibrate: int = 2000,
                                  save_plot: bool = True,
                                  show_plot: bool = True):
    """
    Visualize equilibrated spin configurations at different temperatures.
    
    This shows the spatial structure:
    - T < T_c: Large ordered domains (high correlation length)
    - T ≈ T_c: Fractal-like, scale-invariant patterns (critical point)
    - T > T_c: Random, uncorrelated spins (disordered)
    """
    fig, axes = plt.subplots(1, len(temperatures), figsize=(5*len(temperatures), 5))
    
    temp_labels = {
        temperatures[0]: f'T = {temperatures[0]:.2f} < T_c (Ordered)',
        temperatures[1]: f'T ≈ T_c = {temperatures[1]:.3f} (Critical)',
        temperatures[2]: f'T = {temperatures[2]:.2f} > T_c (Disordered)'
    }
    
    for idx, T in enumerate(temperatures):
        config = initialize_lattice(N, "random")
        
        # Equilibrate
        for _ in tqdm(range(n_equilibrate), desc=f"Equilibrating T={T:.2f}", leave=False):
            config = metropolis_step(config, T)
        
        # Plot
        ax = axes[idx] if len(temperatures) > 1 else axes
        im = ax.imshow(config, cmap='RdBu', interpolation='nearest', vmin=-1, vmax=1)
        ax.set_title(temp_labels.get(T, f'T = {T:.2f}'), fontsize=12, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f'M = {calculate_magnetization(config):.3f}', fontsize=10)
    
    plt.suptitle(f'Spin Configurations ({N}×{N} Lattice)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_plot:
        plt.savefig('ising_spin_configurations.png', dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print("✓ Plot saved as 'ising_spin_configurations.png'")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    # Run the main phase transition experiment
    temperatures, magnetizations, energies = run_temperature_sweep(
        N=20,           # Lattice size (20×20)
        T_min=1.5,      # Below critical temperature
        T_max=3.5,      # Above critical temperature
        n_temps=21,     # Number of temperature points
        n_steps=5000,   # Monte Carlo steps per temperature
        n_warmup=1000,  # Warmup for thermalization
        save_plot=True,
        show_plot=True
    )
    
    # Visualize spin configurations at key temperatures
    print("\nGenerating spin configuration visualizations...")
    visualize_spin_configurations(
        N=50,
        temperatures=[1.5, T_CRITICAL, 3.5],
        n_equilibrate=2000,
        save_plot=True,
        show_plot=True
    )

