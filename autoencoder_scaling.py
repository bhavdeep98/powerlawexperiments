"""
Neural Scaling Laws & Emergent Capabilities Experiment
=======================================================

LLM Proof: Power Law Scaling and Emergence in Deep Learning

This module demonstrates the relationship between:
1. Neural Scaling Laws: Loss decreases smoothly as a power law of resources
2. Emergent Capabilities: Task accuracy can show sharp transitions

The experiment uses an autoencoder with variable bottleneck size to show
how reconstruction loss follows smooth scaling while a secondary task
(parity detection) exhibits emergent behavior at a critical capacity.

References:
- Kaplan et al. (2020). Scaling Laws for Neural Language Models.
- Wei et al. (2022). Emergent Abilities of Large Language Models.
- Schaeffer et al. (2023). Are Emergent Abilities of LLMs a Mirage?
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import warnings

# Suppress TensorFlow warnings for cleaner output
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


def create_parity_data(num_samples: int, input_dim: int = 10, 
                       noise_level: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create synthetic data for the scaling experiment.
    
    Task 1 (Autoencoder): Reconstruct the input binary vector
    Task 2 (Emergence): Predict parity (even/odd number of 1s)
    
    Args:
        num_samples: Number of data samples
        input_dim: Dimension of input vectors
        noise_level: Optional noise to add to inputs (0-1)
        
    Returns:
        X: Input data (binary vectors)
        Y_reconstruction: Target for autoencoder (same as X)
        Y_parity: Target for parity task (0 or 1)
    """
    X = np.random.randint(0, 2, size=(num_samples, input_dim))
    
    # Add optional noise
    if noise_level > 0:
        noise_mask = np.random.rand(num_samples, input_dim) < noise_level
        X_noisy = X.copy().astype(float)
        X_noisy[noise_mask] = 1 - X_noisy[noise_mask]  # Flip bits
        X = X_noisy
    
    # Parity: 1 if even number of 1s, 0 if odd
    Y_parity = (np.sum(X > 0.5, axis=1) % 2 == 0).astype(float)
    
    return X.astype(float), X.astype(float), Y_parity


def create_xor_pattern_data(num_samples: int, input_dim: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create more complex XOR-pattern data requiring deeper representations.
    
    The emergent task requires detecting specific bit patterns that
    only become learnable with sufficient latent capacity.
    """
    X = np.random.randint(0, 2, size=(num_samples, input_dim))
    
    # Complex pattern: XOR of first half with second half
    half = input_dim // 2
    pattern = np.logical_xor(X[:, :half], X[:, half:2*half])
    Y_pattern = (np.sum(pattern, axis=1) > half // 2).astype(float)
    
    return X.astype(float), X.astype(float), Y_pattern


def build_scaling_autoencoder(input_dim: int, latent_dim: int, 
                               hidden_dim: int = 32,
                               use_batch_norm: bool = True) -> Model:
    """
    Build an autoencoder with multi-task output for scaling experiments.
    
    Architecture:
        Input → Encoder → Latent Bottleneck → Decoder → Reconstruction
                              ↓
                        Classifier → Task Prediction
    
    Args:
        input_dim: Input dimension
        latent_dim: Bottleneck dimension (the "resource" we scale)
        hidden_dim: Hidden layer dimension
        use_batch_norm: Whether to use batch normalization
        
    Returns:
        Compiled Keras model with two outputs
    """
    # Input layer
    input_layer = Input(shape=(input_dim,), name='input')
    
    # Encoder
    x = Dense(hidden_dim, activation='relu', name='encoder_1')(input_layer)
    if use_batch_norm:
        x = BatchNormalization()(x)
    x = Dense(hidden_dim // 2, activation='relu', name='encoder_2')(x)
    
    # Latent bottleneck (the critical resource)
    latent = Dense(latent_dim, activation='relu', name='latent_bottleneck')(x)
    
    # Decoder
    x = Dense(hidden_dim // 2, activation='relu', name='decoder_1')(latent)
    if use_batch_norm:
        x = BatchNormalization()(x)
    x = Dense(hidden_dim, activation='relu', name='decoder_2')(x)
    
    # Output 1: Reconstruction (smooth scaling)
    reconstruction = Dense(input_dim, activation='sigmoid', name='reconstruction')(x)
    
    # Output 2: Classification task (potential emergence)
    # This head operates on the latent representation
    task_hidden = Dense(latent_dim * 2, activation='relu', name='task_hidden')(latent)
    classification = Dense(1, activation='sigmoid', name='classification')(task_hidden)
    
    # Combined model
    model = Model(inputs=input_layer, outputs=[reconstruction, classification])
    
    # Compile with weighted losses
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss={
            'reconstruction': 'mse',
            'classification': 'binary_crossentropy'
        },
        loss_weights={
            'reconstruction': 1.0,
            'classification': 0.5  # Secondary task
        },
        metrics={
            'reconstruction': 'mae',
            'classification': 'accuracy'
        }
    )
    
    return model


def run_scaling_experiment(input_dim: int = 12,
                           latent_dims: List[int] = None,
                           num_train: int = 10000,
                           num_test: int = 2000,
                           epochs: int = 50,
                           batch_size: int = 64,
                           n_trials: int = 3,
                           verbose: bool = True) -> Dict:
    """
    Run the neural scaling law experiment.
    
    This sweeps across different latent dimensions (the "compute resource")
    and measures both reconstruction loss (smooth scaling) and task accuracy
    (potential emergence).
    
    Args:
        input_dim: Dimension of input vectors
        latent_dims: List of latent dimensions to test
        num_train: Number of training samples
        num_test: Number of test samples
        epochs: Training epochs per model
        batch_size: Training batch size
        n_trials: Number of trials per latent dim (for error bars)
        verbose: Print progress
        
    Returns:
        Dictionary with results
    """
    if latent_dims is None:
        latent_dims = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16]
    
    # Generate data
    X_train, Y_recon_train, Y_task_train = create_parity_data(num_train, input_dim)
    X_test, Y_recon_test, Y_task_test = create_parity_data(num_test, input_dim)
    
    results = {
        'latent_dims': latent_dims,
        'reconstruction_loss': [],
        'reconstruction_loss_std': [],
        'task_accuracy': [],
        'task_accuracy_std': [],
        'task_loss': [],
        'task_loss_std': []
    }
    
    print(f"\n{'='*70}")
    print("NEURAL SCALING LAWS & EMERGENT CAPABILITIES EXPERIMENT")
    print(f"{'='*70}")
    print(f"Input Dimension: {input_dim}")
    print(f"Latent Dimensions: {latent_dims}")
    print(f"Training Samples: {num_train}")
    print(f"Trials per Latent Dim: {n_trials}")
    print(f"{'='*70}\n")
    
    for L in latent_dims:
        trial_recon_loss = []
        trial_task_acc = []
        trial_task_loss = []
        
        for trial in range(n_trials):
            # Build and train model
            tf.keras.backend.clear_session()
            model = build_scaling_autoencoder(input_dim, L)
            
            # Early stopping for efficiency
            early_stop = EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=0
            )
            
            # Train
            model.fit(
                X_train,
                {'reconstruction': Y_recon_train, 'classification': Y_task_train},
                validation_split=0.1,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stop],
                verbose=0
            )
            
            # Evaluate
            eval_results = model.evaluate(
                X_test,
                {'reconstruction': Y_recon_test, 'classification': Y_task_test},
                verbose=0
            )
            
            # Results: [total_loss, recon_loss, class_loss, recon_mae, class_accuracy]
            trial_recon_loss.append(eval_results[1])
            trial_task_loss.append(eval_results[2])
            trial_task_acc.append(eval_results[-1])
        
        # Aggregate results
        results['reconstruction_loss'].append(np.mean(trial_recon_loss))
        results['reconstruction_loss_std'].append(np.std(trial_recon_loss))
        results['task_accuracy'].append(np.mean(trial_task_acc))
        results['task_accuracy_std'].append(np.std(trial_task_acc))
        results['task_loss'].append(np.mean(trial_task_loss))
        results['task_loss_std'].append(np.std(trial_task_loss))
        
        if verbose:
            print(f"Latent Dim L={L:2d}: "
                  f"Recon Loss = {np.mean(trial_recon_loss):.4f} ± {np.std(trial_recon_loss):.4f}, "
                  f"Task Acc = {np.mean(trial_task_acc):.4f} ± {np.std(trial_task_acc):.4f}")
    
    return results


def fit_power_law(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """
    Fit a power law y = a * x^(-α) using log-log linear regression.
    
    Returns:
        Tuple of (alpha, a, r_squared)
    """
    # Filter positive values for log
    mask = (x > 0) & (y > 0)
    log_x = np.log(x[mask])
    log_y = np.log(y[mask])
    
    # Linear regression in log space
    coeffs = np.polyfit(log_x, log_y, 1)
    alpha = -coeffs[0]  # Power law exponent
    a = np.exp(coeffs[1])  # Coefficient
    
    # R-squared
    y_pred = coeffs[0] * log_x + coeffs[1]
    ss_res = np.sum((log_y - y_pred) ** 2)
    ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return alpha, a, r_squared


def visualize_scaling_results(results: Dict, 
                               save_plot: bool = True,
                               show_plot: bool = True):
    """
    Visualize the scaling experiment results.
    
    Creates a multi-panel figure showing:
    1. Reconstruction loss vs latent dim (power law)
    2. Task accuracy vs latent dim (emergence)
    3. Log-log plot for power law fitting
    """
    latent_dims = np.array(results['latent_dims'])
    recon_loss = np.array(results['reconstruction_loss'])
    recon_std = np.array(results['reconstruction_loss_std'])
    task_acc = np.array(results['task_accuracy'])
    task_std = np.array(results['task_accuracy_std'])
    
    # Fit power law to reconstruction loss
    alpha, a, r_squared = fit_power_law(latent_dims, recon_loss)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Color palette
    BLUE = '#2E86AB'
    RED = '#E94F37'
    PURPLE = '#8E44AD'
    GREEN = '#27AE60'
    
    # Plot 1: Reconstruction Loss (Linear Scale)
    ax1 = axes[0, 0]
    ax1.errorbar(latent_dims, recon_loss, yerr=recon_std,
                 fmt='o-', capsize=4, color=BLUE, markersize=8, linewidth=2,
                 label='Reconstruction Loss')
    ax1.set_xlabel('Latent Dimension (L)', fontsize=12)
    ax1.set_ylabel('MSE Loss', fontsize=12)
    ax1.set_title('Neural Scaling Law: Smooth Loss Decay', fontsize=14, fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.7)
    ax1.legend(fontsize=10)
    
    # Plot 2: Task Accuracy (Emergence)
    ax2 = axes[0, 1]
    ax2.errorbar(latent_dims, task_acc, yerr=task_std,
                 fmt='s-', capsize=4, color=RED, markersize=8, linewidth=2,
                 label='Parity Task Accuracy')
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random Baseline')
    ax2.set_xlabel('Latent Dimension (L)', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Emergent Capability: Task Accuracy Jump', fontsize=14, fontweight='bold')
    ax2.set_ylim(0.4, 1.05)
    ax2.grid(True, linestyle=':', alpha=0.7)
    ax2.legend(fontsize=10)
    
    # Annotate emergence region
    emergence_idx = np.argmax(np.diff(task_acc)) + 1
    if emergence_idx < len(latent_dims):
        ax2.axvline(x=latent_dims[emergence_idx], color=PURPLE, linestyle='--', 
                    alpha=0.7, label='Emergence Threshold')
        ax2.annotate('Emergence\nRegion', 
                     xy=(latent_dims[emergence_idx], task_acc[emergence_idx]),
                     xytext=(latent_dims[emergence_idx] + 2, task_acc[emergence_idx] - 0.15),
                     fontsize=10, color=PURPLE,
                     arrowprops=dict(arrowstyle='->', color=PURPLE))
    
    # Plot 3: Log-Log Plot (Power Law)
    ax3 = axes[1, 0]
    ax3.errorbar(latent_dims, recon_loss, yerr=recon_std,
                 fmt='o', capsize=4, color=BLUE, markersize=8,
                 label='Data')
    
    # Plot fitted power law
    x_fit = np.linspace(latent_dims.min(), latent_dims.max(), 100)
    y_fit = a * x_fit ** (-alpha)
    ax3.plot(x_fit, y_fit, '--', color=GREEN, linewidth=2,
             label=f'Power Law: L$^{{-{alpha:.2f}}}$ (R²={r_squared:.3f})')
    
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.set_xlabel('Latent Dimension (L) [log scale]', fontsize=12)
    ax3.set_ylabel('MSE Loss [log scale]', fontsize=12)
    ax3.set_title(f'Power Law Fit: α = {alpha:.3f}', fontsize=14, fontweight='bold')
    ax3.grid(True, linestyle=':', alpha=0.7, which='both')
    ax3.legend(fontsize=10)
    
    # Plot 4: Combined View
    ax4 = axes[1, 1]
    
    # Normalize for comparison
    recon_normalized = (recon_loss - recon_loss.min()) / (recon_loss.max() - recon_loss.min())
    
    ax4.plot(latent_dims, 1 - recon_normalized, 'o-', color=BLUE, 
             markersize=8, linewidth=2, label='1 - Normalized Loss')
    ax4.plot(latent_dims, task_acc, 's-', color=RED,
             markersize=8, linewidth=2, label='Task Accuracy')
    ax4.set_xlabel('Latent Dimension (L)', fontsize=12)
    ax4.set_ylabel('Performance', fontsize=12)
    ax4.set_title('Comparison: Smooth Scaling vs Sharp Emergence', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, linestyle=':', alpha=0.7)
    
    # Add text annotation
    ax4.text(0.05, 0.95, 
             'Loss: Smooth power law decay\nAccuracy: Potential sharp transition',
             transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Neural Scaling Laws & Emergent Capabilities\n'
                 '(Autoencoder with Parity Detection Task)',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_plot:
        plt.savefig('neural_scaling_laws.png', dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print("\n✓ Plot saved as 'neural_scaling_laws.png'")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    # Print summary
    print(f"\n{'='*70}")
    print("SCALING LAW ANALYSIS")
    print(f"{'='*70}")
    print(f"Power Law Exponent (α): {alpha:.4f}")
    print(f"Power Law Coefficient (a): {a:.4f}")
    print(f"R-squared (goodness of fit): {r_squared:.4f}")
    print(f"\nScaling Law: Loss ∝ L^(-{alpha:.2f})")
    print(f"{'='*70}")
    
    return alpha, r_squared


if __name__ == "__main__":
    # Run the scaling experiment
    print("Starting Neural Scaling Laws Experiment...")
    print("This may take a few minutes depending on your hardware.\n")
    
    results = run_scaling_experiment(
        input_dim=12,                                    # Input dimension
        latent_dims=[1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16],  # Resource scaling
        num_train=10000,                                 # Training samples
        num_test=2000,                                   # Test samples
        epochs=50,                                       # Max epochs
        batch_size=64,                                   # Batch size
        n_trials=3,                                      # Trials for averaging
        verbose=True
    )
    
    # Visualize results
    alpha, r_squared = visualize_scaling_results(
        results,
        save_plot=True,
        show_plot=True
    )

