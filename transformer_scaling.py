"""
Transformer Scaling Experiment
==============================

Demonstrates Neural Scaling Laws using a realistic Transformer architecture
(NanoGPT style) on a character-level language modeling task.

This provides stronger evidence than the Autoencoder experiment.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, losses, callbacks
import matplotlib.pyplot as plt
import time
import os
import json

# ==============================================================================
# 1. TRANSFORMER COMPONENTS
# ==============================================================================

class CausalSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)
        self.dropout_layer = layers.Dropout(dropout)

    def attention(self, query, key, value, mask=None):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        
        if mask is not None:
             # Mask is 0 for valid, 1 for masked (following keras convention usually involves adding -1e9)
             # Here we assume mask is additive (0 for keep, -1e9 for mask)
             scaled_score += mask
             
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.head_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, mask=None):
        batch_size = tf.shape(inputs)[0]
        
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)
        
        # Causal mask
        seq_len = tf.shape(inputs)[1]
        causal_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        causal_mask = causal_mask * -1e9
        
            
        attention, self.attention_weights = self.attention(query, key, value, causal_mask)
        
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))
        
        output = self.combine_heads(concat_attention)
        return self.dropout_layer(output)

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.att = CausalSelfAttention(embed_dim, num_heads, dropout)
        self.ffn = models.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

def create_transformer_model(vocab_size, max_len, embed_dim, num_heads, ff_dim, num_layers, dropout=0.1):
    inputs = layers.Input(shape=(max_len,))
    
    # Token Embedding
    token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)(inputs)
    
    # Positional Embedding
    positions = tf.range(start=0, limit=max_len, delta=1)
    pos_emb = layers.Embedding(input_dim=max_len, output_dim=embed_dim)(positions)
    
    x = token_emb + pos_emb
    
    # Transformer Blocks
    for _ in range(num_layers):
        x = TransformerBlock(embed_dim, num_heads, ff_dim, dropout)(x)
        
    # Output Head
    outputs = layers.Dense(vocab_size, activation="softmax")(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

# ==============================================================================
# 2. DATA GENERATION (Synthetic Text)
# ==============================================================================

def generate_synthetic_text(num_samples=1000, seq_len=32, vocab_size=50):
    """
    Generate synthetic data that has structure (so learning is possible).
    Pattern: Arithmetic-like sequences or repeating motifs.
    """
    X = []
    y = [] # Target is next token prediction
    
    # We'll use a simple repeating pattern task: A B C A B C ...
    # And some predictable probabilistic transitions for difficulty measure
    
    for _ in range(num_samples + 100): # +100 to ensure we have enough valid
        # Pattern 1: Counting
        start = np.random.randint(0, vocab_size - seq_len - 1)
        seq = np.arange(start, start + seq_len + 1) % vocab_size
        
        if len(seq) == seq_len + 1:
            X.append(seq[:-1])
            y.append(seq[1:]) # Next token prediction
            
        if len(X) >= num_samples:
            break
            
    return np.array(X), np.array(y)

# ==============================================================================
# 3. EXPERIMENT RUNNER
# ==============================================================================

def run_transformer_scaling(quick=False):
    print("\n" + "="*70)
    print("TRANSFORMER SCALING EXPERIMENT")
    print("="*70)
    
    # Parameters
    VOCAB_SIZE = 64
    SEQ_LEN = 32
    
    if quick:
        embed_dims = [16, 32, 64]
        epochs = 10
        num_samples = 1000
    else:
        embed_dims = [16, 32, 64, 128, 256]
        epochs = 20
        num_samples = 5000
        
    results = {
        'embed_dims': embed_dims,
        'final_loss': [],
        'param_counts': []
    }
    
    # Data
    X, y = generate_synthetic_text(num_samples, SEQ_LEN, VOCAB_SIZE)
    
    # One-hot encode targets for Sparse Categorical Crossentropy
    # Actually Keras handles integer targets fine with 'sparse_categorical_crossentropy'
    
    for dim in embed_dims:
        print(f"\nTraining Transformer with d_model={dim}...")
        
        # Scale heads and ff_dim relative to d_model
        num_heads = 4 if dim >= 32 else 2 # Ensure divisibility
        if dim == 16: num_heads = 2
        
        model = create_transformer_model(
            vocab_size=VOCAB_SIZE,
            max_len=SEQ_LEN,
            embed_dim=dim,
            num_heads=num_heads,
            ff_dim=dim*4,
            num_layers=2,
            dropout=0.0
        )
        
        model.compile(
            optimizer="adam", 
            loss="sparse_categorical_crossentropy", 
            metrics=["accuracy"]
        )
        
        # Calculate params
        params = model.count_params()
        results['param_counts'].append(params)
        print(f"  Parameters: {params:,}")
        
        # Train
        hist = model.fit(
            X, y,
            batch_size=64,
            epochs=epochs,
            validation_split=0.2,
            verbose=0,
            callbacks=[callbacks.EarlyStopping(patience=3, restore_best_weights=True)]
        )
        
        final_loss = hist.history['val_loss'][-1]
        print(f"  Final Val Loss: {final_loss:.4f}")
        results['final_loss'].append(final_loss)
        
    return results

def visualize_transformer_scaling(results):
    params = np.array(results['param_counts'])
    losses = np.array(results['final_loss'])
    
    # Fit power law: Loss = a * N^(-alpha)
    log_params = np.log(params)
    log_losses = np.log(losses)
    
    coeffs = np.polyfit(log_params, log_losses, 1)
    alpha = -coeffs[0]
    
    plt.figure(figsize=(10, 6))
    
    # Data points
    plt.scatter(params, losses, color='#E94F37', s=100, label='Transformer Runs')
    
    # Fit line
    x_range = np.linspace(min(params), max(params), 100)
    y_fit = np.exp(coeffs[1]) * x_range**(-alpha)
    plt.plot(x_range, y_fit, '--', color='#2E86AB', linewidth=2, 
             label=f'Scaling Law ($\\alpha={alpha:.2f}$)')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Parameters (N)', fontsize=12)
    plt.ylabel('Test Loss (L)', fontsize=12)
    plt.title('Transformer Neural Scaling Law', fontsize=14, fontweight='bold')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    
    plt.savefig('transformer_scaling_results.png', dpi=150, bbox_inches='tight')
    print("\n✓ Plot saved as 'transformer_scaling_results.png'")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()
    
    results = run_transformer_scaling(quick=args.quick)
    
    with open('transformer_scaling_results.json', 'w') as f:
        json.dump(results, f, indent=2)
        
    visualize_transformer_scaling(results)
