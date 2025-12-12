import json
import math
import argparse
from typing import List, Dict
import numpy as np
from workflow_grammar_extractor import WorkflowExtractor

def calculate_shannon_entropy(tokens: List[str]) -> float:
    """Calculates Shannon Entropy of the token distribution."""
    if not tokens:
        return 0.0
    
    unique_tokens, counts = np.unique(tokens, return_counts=True)
    probs = counts / len(tokens)
    entropy = -np.sum(probs * np.log2(probs))
    return entropy

def calculate_lz_complexity(sequence: List[str]) -> int:
    """
    Calculates Lempel-Ziv complexity (number of unique patterns).
    Implementation of LZ76 complexity.
    """
    if not sequence:
        return 0
        
    n = len(sequence)
    i = 0
    complexity = 1
    l = 1
    k = 1
    k_max = 1
    
    while l + k <= n:
        # Look for sequence[l:l+k] in sequence[0:l+k-1]
        chunk = sequence[l:l+k]
        history = sequence[0:l+k-1]
        
        # Simple check if chunk exists in history
        # (Naive quadratic implementation, sufficient for short traces)
        found = False
        
        # Check if sub-sequence exists in history
        # Convert to tuple for easier matching or just linear scan
        # For simplicity, we assume strings are comparable
        
        # Optimization: convert list to string for finding substring?
        # No, tokens are strings. 
        # Manual scan:
        for j in range(len(history) - len(chunk) + 1):
             if history[j:j+len(chunk)] == chunk:
                 found = True
                 break
        
        if found:
            k += 1
            if l + k > n:
                break
        else:
            complexity += 1
            l += k
            k = 1
            
    return complexity

def analyze_traces(results_file: str):
    print(f"Loading results from {results_file}...")
    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File {results_file} not found.")
        return

    extractor = WorkflowExtractor()
    
    # Structure to hold analysis per T-value (depth)
    analysis_by_depth = {}

    print(f"Found {len(data)} records.")
    
    metrics_by_t = {}

    for i, result in enumerate(data):
        if not isinstance(result, dict):
            # print(f"Skipping record {i}: Type is {type(result)}")
            continue
            
        t_val = result.get('T', 0)
        trace = result.get('trace', [])
        if not trace:
            trace = result.get('solution_text', [])
        
        if not trace:
            # Try to infer trace from solution metrics if possible, or skip
            # print(f"Skipping record {i} (T={t_val}): No trace found.")
            continue
            
        # Determine format of trace
        tokens = []
        if isinstance(trace, list):
            # If trace is list of dicts (steps)
            if trace and isinstance(trace[0], dict):
                 tokens = extractor.extract_from_json_trace(trace)
            # If trace is list of strings
            elif trace and isinstance(trace[0], str):
                 tokens = extractor.tokenize_trace("\n".join(trace))
            else:
                 # Empty list
                 pass
        elif isinstance(trace, str):
            tokens = extractor.tokenize_trace(trace)
            
        if not tokens:
            continue
            
        entropy = calculate_shannon_entropy(tokens)
        lz = calculate_lz_complexity(tokens)
        
        if t_val not in metrics_by_t:
            metrics_by_t[t_val] = {'entropy': [], 'lz': [], 'length': []}
            
        metrics_by_t[t_val]['entropy'].append(entropy)
        metrics_by_t[t_val]['lz'].append(lz)
        metrics_by_t[t_val]['length'].append(len(tokens))

    print("\n" + "="*60)
    print(f"{'Depth (T)':<10} | {'Entropy':<10} | {'LZ Comp':<10} | {'Length':<10} | {'n':<5}")
    print("-" * 60)
    
    for t_val in sorted(metrics_by_t.keys()):
        ents = metrics_by_t[t_val]['entropy']
        lzs = metrics_by_t[t_val]['lz']
        lens = metrics_by_t[t_val]['length']
        
        avg_ent = np.mean(ents)
        avg_lz = np.mean(lzs)
        avg_len = np.mean(lens)
        count = len(ents)
        
        print(f"{t_val:<10} | {avg_ent:.4f}     | {avg_lz:.4f}     | {avg_len:.1f}       | {count:<5}")
    print("="*60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default="results/agentic_scaling/exp1_2_t_scaling_results.json")
    args = parser.parse_args()
    
    analyze_traces(args.results)
