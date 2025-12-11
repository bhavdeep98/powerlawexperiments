#!/usr/bin/env python3
"""Calculate time and cost estimates for Experiment 1.1"""

# Time estimates per run (seconds)
time_per_run = {
    1: 2.5,
    2: 5.0,
    3: 7.5,
    5: 12.5,
    8: 20.0,
    13: 32.5
}

# Token estimates per run
tokens_per_run = {
    1: 500,
    2: 1000,
    3: 1500,
    5: 2500,
    8: 4000,
    13: 6500
}

# Pilot run
A_pilot = [1, 2, 3]
problems_pilot = 3
reps_pilot = 1

total_runs_pilot = len(A_pilot) * problems_pilot * reps_pilot
total_time_pilot = sum(time_per_run[A] * problems_pilot * reps_pilot for A in A_pilot)
total_tokens_pilot = sum(tokens_per_run[A] * problems_pilot * reps_pilot for A in A_pilot)

print('='*70)
print('TIME ESTIMATION FOR EXPERIMENT 1.1: A-SCALING')
print('='*70)
print(f'\nPILOT RUN:')
print(f'  A values: {A_pilot}')
print(f'  Problems: {problems_pilot}')
print(f'  Replications: {reps_pilot}')
print(f'  Total runs: {total_runs_pilot}')
print(f'  Estimated time: {total_time_pilot:.1f} seconds ({total_time_pilot/60:.1f} minutes)')
print(f'  Estimated tokens: ~{total_tokens_pilot:,}')
print(f'  Estimated cost (gpt-4o-mini): ~${total_tokens_pilot/1e6 * 0.3:.2f}')

# Full run
A_full = [1, 2, 3, 5, 8]
problems_full = 10
reps_full = 3

total_runs_full = len(A_full) * problems_full * reps_full
total_time_full = sum(time_per_run[A] * problems_full * reps_full for A in A_full)
total_tokens_full = sum(tokens_per_run[A] * problems_full * reps_full for A in A_full)

print(f'\nFULL RUN:')
print(f'  A values: {A_full}')
print(f'  Problems: {problems_full}')
print(f'  Replications: {reps_full}')
print(f'  Total runs: {total_runs_full}')
print(f'  Estimated time: {total_time_full:.1f} seconds ({total_time_full/60:.1f} minutes)')
print(f'  Estimated tokens: ~{total_tokens_full:,}')
print(f'  Estimated cost (gpt-4o-mini): ~${total_tokens_full/1e6 * 0.3:.2f}')
print(f'  Estimated cost (gpt-4o): ~${total_tokens_full/1e6 * 3.0:.2f}')

print(f'\n{"="*70}')
print('RECOMMENDATION: Start with PILOT run to validate setup')
print('='*70)
