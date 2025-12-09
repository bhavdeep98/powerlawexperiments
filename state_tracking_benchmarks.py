"""
State Tracking Benchmarks
==========================
Experiments specifically targeting state tracking fidelity in System 2 reasoning.

Tasks:
- Stack tracking: Track a stack of blocks through operations
- Variable tracking: Track variables through assignment operations
- Graph state: Track edge additions/removals in a graph
- Counter tracking: Maintain multiple counters through increments/decrements
- Game state: Track chess/checkers board through moves

Key Metric: Find the critical point where state tracking breaks down.
"""

import os
import json
import time
import random
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from openai import OpenAI

# Configuration
API_KEY = os.environ.get("OPENAI_API_KEY")
if not API_KEY:
    print("⚠ OPENAI_API_KEY not found. Please export it.")
    exit(1)

client = OpenAI(api_key=API_KEY)
DEFAULT_MODEL = "gpt-4o"


# ==============================================================================
# STATE TRACKING TASKS
# ==============================================================================

class StackTrackingTask:
    """Track a stack of blocks through operations."""
    
    def __init__(self, num_blocks: int = 5):
        self.num_blocks = num_blocks
        self.blocks = list(range(1, num_blocks + 1))
        self.stack = []
        self.operations = []
        self.ground_truth_states = []
    
    def push(self, block: int):
        """Push a block onto the stack."""
        if block in self.blocks and block not in self.stack:
            self.stack.append(block)
            self.operations.append(f"PUSH {block}")
            self.ground_truth_states.append(self.stack.copy())
    
    def pop(self):
        """Pop a block from the stack."""
        if self.stack:
            popped = self.stack.pop()
            self.operations.append(f"POP")
            self.ground_truth_states.append(self.stack.copy())
            return popped
        return None
    
    def generate_sequence(self, num_operations: int) -> Tuple[List[str], List[List[int]]]:
        """Generate a random sequence of operations."""
        self.stack = []
        self.operations = []
        self.ground_truth_states = []
        
        available_blocks = self.blocks.copy()
        
        for _ in range(num_operations):
            if self.stack and random.random() < 0.5:
                # Pop
                self.pop()
            elif available_blocks:
                # Push
                block = random.choice(available_blocks)
                available_blocks.remove(block)
                self.push(block)
        
        return self.operations, self.ground_truth_states
    
    def format_for_llm(self, operations: List[str], step: int) -> str:
        """Format operations up to step for LLM."""
        history = "\n".join([f"{i+1}. {op}" for i, op in enumerate(operations[:step+1])])
        return f"""Track the state of a stack through these operations:

{history}

What is the current state of the stack? Return as a list of numbers, e.g., [1, 3, 2]"""


class VariableTrackingTask:
    """Track variables through assignment operations."""
    
    def __init__(self, num_variables: int = 5):
        self.num_variables = num_variables
        self.variables = {f"x{i}": 0 for i in range(1, num_variables + 1)}
        self.operations = []
        self.ground_truth_states = []
    
    def assign(self, var: str, value: int):
        """Assign a value to a variable."""
        if var in self.variables:
            self.variables[var] = value
            self.operations.append(f"{var} = {value}")
            self.ground_truth_states.append(self.variables.copy())
    
    def increment(self, var: str, amount: int = 1):
        """Increment a variable."""
        if var in self.variables:
            self.variables[var] += amount
            self.operations.append(f"{var} += {amount}")
            self.ground_truth_states.append(self.variables.copy())
    
    def generate_sequence(self, num_operations: int) -> Tuple[List[str], List[Dict[str, int]]]:
        """Generate a random sequence of operations."""
        self.variables = {f"x{i}": 0 for i in range(1, self.num_variables + 1)}
        self.operations = []
        self.ground_truth_states = []
        
        var_names = list(self.variables.keys())
        
        for _ in range(num_operations):
            var = random.choice(var_names)
            if random.random() < 0.7:
                # Assignment
                value = random.randint(1, 100)
                self.assign(var, value)
            else:
                # Increment
                amount = random.randint(1, 10)
                self.increment(var, amount)
        
        return self.operations, self.ground_truth_states
    
    def format_for_llm(self, operations: List[str], step: int) -> str:
        """Format operations up to step for LLM."""
        history = "\n".join([f"{i+1}. {op}" for i, op in enumerate(operations[:step+1])])
        return f"""Track the values of variables through these operations:

{history}

What are the current values of all variables? Return as a dictionary, e.g., {{"x1": 5, "x2": 10, ...}}"""


class CounterTrackingTask:
    """Track multiple counters through increments/decrements."""
    
    def __init__(self, num_counters: int = 3):
        self.num_counters = num_counters
        self.counters = {f"counter_{i}": 0 for i in range(1, num_counters + 1)}
        self.operations = []
        self.ground_truth_states = []
    
    def increment(self, counter: str):
        """Increment a counter."""
        if counter in self.counters:
            self.counters[counter] += 1
            self.operations.append(f"INCREMENT {counter}")
            self.ground_truth_states.append(self.counters.copy())
    
    def decrement(self, counter: str):
        """Decrement a counter."""
        if counter in self.counters and self.counters[counter] > 0:
            self.counters[counter] -= 1
            self.operations.append(f"DECREMENT {counter}")
            self.ground_truth_states.append(self.counters.copy())
    
    def generate_sequence(self, num_operations: int) -> Tuple[List[str], List[Dict[str, int]]]:
        """Generate a random sequence of operations."""
        self.counters = {f"counter_{i}": 0 for i in range(1, self.num_counters + 1)}
        self.operations = []
        self.ground_truth_states = []
        
        counter_names = list(self.counters.keys())
        
        for _ in range(num_operations):
            counter = random.choice(counter_names)
            if random.random() < 0.6:
                self.increment(counter)
            else:
                self.decrement(counter)
        
        return self.operations, self.ground_truth_states
    
    def format_for_llm(self, operations: List[str], step: int) -> str:
        """Format operations up to step for LLM."""
        history = "\n".join([f"{i+1}. {op}" for i, op in enumerate(operations[:step+1])])
        return f"""Track the values of counters through these operations:

{history}

What are the current values of all counters? Return as a dictionary, e.g., {{"counter_1": 3, "counter_2": 1, ...}}"""


# ==============================================================================
# STATE TRACKING EVALUATOR
# ==============================================================================

class StateTrackingEvaluator:
    """Evaluates state tracking fidelity."""
    
    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
    
    def predict_state(self, prompt: str) -> Any:
        """Use LLM to predict state."""
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            result = response.choices[0].message.content.strip()
            
            # Try to parse as list or dict
            if result.startswith('['):
                import ast
                return ast.literal_eval(result)
            elif result.startswith('{'):
                import ast
                return ast.literal_eval(result)
            else:
                return result
        except Exception as e:
            return None
    
    def compare_states(self, predicted: Any, ground_truth: Any) -> float:
        """Compare predicted state with ground truth. Returns accuracy (0-1)."""
        try:
            if isinstance(ground_truth, list):
                if isinstance(predicted, list):
                    # Compare lists (order matters for stack)
                    if len(predicted) != len(ground_truth):
                        return 0.0
                    return 1.0 if predicted == ground_truth else 0.0
                else:
                    return 0.0
            
            elif isinstance(ground_truth, dict):
                if isinstance(predicted, dict):
                    # Compare dictionaries
                    if set(predicted.keys()) != set(ground_truth.keys()):
                        return 0.0
                    correct = sum(1 for k in ground_truth 
                                if k in predicted and predicted[k] == ground_truth[k])
                    return correct / len(ground_truth) if ground_truth else 1.0
                else:
                    return 0.0
            
            else:
                return 1.0 if predicted == ground_truth else 0.0
        except:
            return 0.0
    
    def measure_state_tracking_fidelity(self,
                                       task,
                                       num_steps: int,
                                       num_trials: int = 5) -> Dict:
        """Measure state tracking fidelity across steps."""
        all_accuracies = []
        failure_points = []
        
        for trial in range(num_trials):
            # Generate sequence
            if hasattr(task, 'generate_sequence'):
                operations, ground_truth_states = task.generate_sequence(num_steps)
            else:
                continue
            
            step_accuracies = []
            
            for step in range(min(num_steps, len(operations))):
                # Format prompt
                prompt = task.format_for_llm(operations, step)
                
                # Predict state
                predicted_state = self.predict_state(prompt)
                
                # Compare with ground truth
                if step < len(ground_truth_states):
                    accuracy = self.compare_states(predicted_state, 
                                                   ground_truth_states[step])
                    step_accuracies.append(accuracy)
                else:
                    step_accuracies.append(0.0)
            
            all_accuracies.append(step_accuracies)
            
            # Find failure point (first major deviation)
            for i, acc in enumerate(step_accuracies):
                if acc < 0.5:  # Threshold for "failure"
                    failure_points.append(i)
                    break
            else:
                failure_points.append(num_steps)  # No failure
        
        # Aggregate results
        avg_accuracies = [np.mean([acc[i] for acc in all_accuracies 
                                  if i < len(acc)]) 
                         for i in range(num_steps)]
        
        return {
            'accuracy_curve': avg_accuracies,
            'failure_step': np.mean(failure_points) if failure_points else num_steps,
            'failure_std': np.std(failure_points) if failure_points else 0.0,
            'final_accuracy': avg_accuracies[-1] if avg_accuracies else 0.0
        }


# ==============================================================================
# EXPERIMENT RUNNER
# ==============================================================================

def run_state_tracking_experiments(model: str = DEFAULT_MODEL,
                                   num_steps_list: List[int] = None) -> Dict:
    """Run comprehensive state tracking experiments."""
    if num_steps_list is None:
        num_steps_list = [5, 10, 15, 20, 25, 30]
    
    evaluator = StateTrackingEvaluator(model=model)
    results = {}
    
    print(f"\n{'='*70}")
    print("STATE TRACKING FIDELITY EXPERIMENTS")
    print(f"{'='*70}")
    print(f"Model: {model}")
    print(f"Step counts: {num_steps_list}")
    print(f"{'='*70}\n")
    
    # 1. Stack Tracking
    print("Task 1: Stack Tracking...")
    stack_task = StackTrackingTask(num_blocks=5)
    stack_results = {}
    for num_steps in num_steps_list:
        print(f"  Testing {num_steps} steps...")
        result = evaluator.measure_state_tracking_fidelity(stack_task, num_steps)
        stack_results[num_steps] = result
        print(f"    Failure point: {result['failure_step']:.1f} ± {result['failure_std']:.1f}")
    results['stack_tracking'] = stack_results
    
    # 2. Variable Tracking
    print("\nTask 2: Variable Tracking...")
    var_task = VariableTrackingTask(num_variables=5)
    var_results = {}
    for num_steps in num_steps_list:
        print(f"  Testing {num_steps} steps...")
        result = evaluator.measure_state_tracking_fidelity(var_task, num_steps)
        var_results[num_steps] = result
        print(f"    Failure point: {result['failure_step']:.1f} ± {result['failure_std']:.1f}")
    results['variable_tracking'] = var_results
    
    # 3. Counter Tracking
    print("\nTask 3: Counter Tracking...")
    counter_task = CounterTrackingTask(num_counters=3)
    counter_results = {}
    for num_steps in num_steps_list:
        print(f"  Testing {num_steps} steps...")
        result = evaluator.measure_state_tracking_fidelity(counter_task, num_steps)
        counter_results[num_steps] = result
        print(f"    Failure point: {result['failure_step']:.1f} ± {result['failure_std']:.1f}")
    results['counter_tracking'] = counter_results
    
    return results


def find_critical_tracking_point(results: Dict) -> Dict:
    """Find the critical point where state tracking breaks down."""
    critical_points = {}
    
    for task_name, task_results in results.items():
        failure_points = []
        for num_steps, result in task_results.items():
            failure_points.append((num_steps, result['failure_step']))
        
        # Find where failure point becomes significantly less than num_steps
        for num_steps, failure_step in failure_points:
            if failure_step < num_steps * 0.8:  # Threshold
                critical_points[task_name] = {
                    'critical_steps': num_steps,
                    'failure_step': failure_step,
                    'gap': num_steps - failure_step
                }
                break
    
    return critical_points


if __name__ == "__main__":
    import numpy as np
    
    results = run_state_tracking_experiments()
    
    # Find critical points
    critical_points = find_critical_tracking_point(results)
    
    # Save results
    output = {
        'results': results,
        'critical_points': critical_points
    }
    
    with open('state_tracking_results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n✓ Results saved to 'state_tracking_results.json'")
    
    if critical_points:
        print(f"\n{'='*70}")
        print("CRITICAL TRACKING POINTS")
        print(f"{'='*70}")
        for task, cp in critical_points.items():
            print(f"\n{task}:")
            print(f"  Critical at {cp['critical_steps']} steps")
            print(f"  Failure at {cp['failure_step']:.1f} steps")
            print(f"  Gap: {cp['gap']:.1f} steps")
