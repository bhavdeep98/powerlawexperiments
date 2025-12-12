# Final Experiment Report

## Experiment 1.4: Combined Scaling (Search x Tools)
*Note: Key result from summary.*
| T (Depth) | K (Tools) | Success Rate |
|-----------|-----------|--------------|
| 5 | 0 | 33.3% (Baseline) |
| 5 | 2 | 66.7% (With Validator) |

## Experiment 3.1: Bandwidth Criticality
| Bandwidth (k) | Consensus Ratio |
|---------------|-----------------|
| 0 | 28.0% |
| 1 | 44.0% |
| 2 | 44.0% |
| 3 | 28.0% |
| 4 | 36.0% |

## Experiment 3.2: Network Topology
| Topology | Consensus Ratio |
|----------|-----------------|
| Star | 56.0% |
| Fully Connected | 44.0% |
| Ring | 32.0% |
| Random | 28.0% |

## Experiment 3.3: Reliability (Robustness)
| Reliability | Consensus Ratio |
|-------------|-----------------|
| 1.0 (100%) | 48.0% |
| 0.9 (90%) | 60.0% |
| 0.7 (70%) | 48.0% |
| 0.5 (50%) | 44.0% |

## Experiment 4: Compute Optimal Architecture
| Architecture | Success Rate | Avg Time (s) | Efficiency (Success/Cost) |
|--------------|--------------|--------------|---------------------------|
| **Deep Search (T=20)** | **100.0%** | 755.4 | 5.00 |
| Strong Model (Zero-Shot) | 60.0% | 10.9 | **6.00** |
| Ensemble (N=5) | 20.0% | 42.2 | 2.00 |

## Visualizations
Plots have been generated in `results/plots/`:
- `exp1_4_scaling.png`
- `exp3_1_bandwidth.png`
- `exp3_2_topology.png`
- `exp3_3_reliability.png`
- `exp4_compute_optimal.png`
