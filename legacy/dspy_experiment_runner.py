"""
DSPy Experiment Runner
======================
Runs the DSPy optimization experiment for System 2 reasoning.
Compares:
1. Zero-shot GPT-4o
2. Uncompiled DSPy ChainOfThought
3. Compiled DSPy ChainOfThought (BootstrapFewShot)
"""

import os
import dspy
from dspy.teleprompt import BootstrapFewShot
from typing import List, Dict

# Configuration
API_KEY = os.environ.get("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OPENAI_API_KEY not found")

# Configure DSPy
gpt4 = dspy.LM('openai/gpt-4o', api_key=API_KEY)
dspy.configure(lm=gpt4)


class GameOf24Signature(dspy.Signature):
    """Signature for Game of 24."""
    numbers = dspy.InputField(desc="The 4 numbers to use (e.g., '4 9 10 13')")
    reasoning = dspy.OutputField(desc="Step-by-step thinking to find the solution")
    equation = dspy.OutputField(desc="The final equation (e.g., '(13-9)*(10-4)=24')")


class GameOf24Solver(dspy.Module):
    """Basic Game of 24 solver."""
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought(GameOf24Signature)
    
    def forward(self, numbers):
        return self.prog(numbers=numbers)


def validate_24(example, pred, trace=None):
    """Checks if the equation equals 24 and uses correct numbers."""
    try:
        eqn = pred.equation.split("=")[0]
        # Safety: simple eval for arithmetic
        result = eval(eqn, {"__builtins__": None}, {})
        if abs(result - 24.0) > 1e-5:
            return False
        return True
    except:
        return False


def run_dspy_comparison():
    print(f"\n{'='*70}")
    print("DSPy OPTIMIZATION EXPERIMENT")
    print(f"{'='*70}")

    # 1. Define Training Data (Hard examples)
    trainset = [
        dspy.Example(numbers="5 5 5 1", equation="(5 - 1 / 5) * 5 = 24").with_inputs('numbers'),
        dspy.Example(numbers="3 3 8 8", equation="8 / (3 - 8 / 3) = 24").with_inputs('numbers'),
        dspy.Example(numbers="10 10 4 4", equation="(10 * 10 - 4) / 4 = 24").with_inputs('numbers'),
        dspy.Example(numbers="2 2 2 3", equation="(2 + 2) * (2 * 3) = 24").with_inputs('numbers'),
    ]

    # 2. Define Test Data (System 2 Benchmark)
    # These are the ones ToT struggled with or solved slowly
    test_problems = [
        "4 9 10 13",
        "1 2 4 6",
        "5 5 5 11",
        "3 3 7 7",
        "1 5 5 5"
    ]
    testset = [dspy.Example(numbers=p).with_inputs('numbers') for p in test_problems]

    results = {
        'uncompiled': [],
        'compiled': []
    }

    # 3. Test Uncompiled
    print("\nTesting Uncompiled (Zero-Shot CoT)...")
    solver = GameOf24Solver()
    correct = 0
    for ex in testset:
        pred = solver(numbers=ex.numbers)
        is_correct = validate_24(ex, pred)
        results['uncompiled'].append({
            'input': ex.numbers,
            'prediction': pred.equation,
            'correct': is_correct
        })
        if is_correct: correct += 1
        print(f"  {ex.numbers}: {is_correct} ({pred.equation})")
    print(f"  Accuracy: {correct/len(testset):.2f}")

    # 4. Compile with BootstrapFewShot
    print("\nCompiling with BootstrapFewShot...")
    teleprompter = BootstrapFewShot(metric=validate_24, max_bootstrapped_demos=3)
    compiled_solver = teleprompter.compile(solver, trainset=trainset)

    # 5. Test Compiled
    print("\nTesting Compiled (Optimized CoT)...")
    correct = 0
    for ex in testset:
        pred = compiled_solver(numbers=ex.numbers)
        is_correct = validate_24(ex, pred)
        results['compiled'].append({
            'input': ex.numbers,
            'prediction': pred.equation,
            'correct': is_correct
        })
        if is_correct: correct += 1
        print(f"  {ex.numbers}: {is_correct} ({pred.equation})")
    print(f"  Accuracy: {correct/len(testset):.2f}")

    return results

if __name__ == "__main__":
    run_dspy_comparison()
