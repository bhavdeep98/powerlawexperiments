"""
Variable Tracking Benchmark
============================
Track N variables through M operations - systematic complexity scaling.
"""

import random
from typing import List, Dict
from .base import Benchmark, Problem


class VariableTrackingBenchmark(Benchmark):
    """Variable tracking problems with systematic complexity scaling."""
    
    def __init__(self):
        super().__init__("Variable Tracking")
        self._generate_problems()
    
    def _generate_variable_problem(self, num_vars: int, num_ops: int, difficulty: int) -> tuple:
        """Generate a variable tracking problem."""
        var_names = [f"x{i+1}" for i in range(num_vars)]
        initial_values = {var: random.randint(0, 10) for var in var_names}
        
        operations = []
        current_values = initial_values.copy()
        ground_truth_path = [current_values.copy()]
        
        for i in range(num_ops):
            var = random.choice(var_names)
            op_type = random.choice(['assign', 'increment', 'decrement', 'add', 'multiply'])
            
            if op_type == 'assign':
                value = random.randint(0, 20)
                current_values[var] = value
                operations.append(f"{var} = {value}")
            elif op_type == 'increment':
                current_values[var] += 1
                operations.append(f"{var}++")
            elif op_type == 'decrement':
                current_values[var] = max(0, current_values[var] - 1)
                operations.append(f"{var}--")
            elif op_type == 'add':
                other_var = random.choice([v for v in var_names if v != var])
                current_values[var] += current_values[other_var]
                operations.append(f"{var} = {var} + {other_var}")
            else:  # multiply
                factor = random.randint(2, 5)
                current_values[var] *= factor
                operations.append(f"{var} = {var} * {factor}")
            
            ground_truth_path.append(current_values.copy())
        
        problem_text = (
            f"Track {num_vars} variables through {num_ops} operations.\n"
            f"Initial values: {initial_values}\n"
            f"Operations:\n" + "\n".join([f"{i+1}. {op}" for i, op in enumerate(operations)]) +
            f"\nWhat are the final values of all variables?"
        )
        
        return problem_text, current_values, ground_truth_path
    
    def _generate_problems(self):
        """Generate 100 synthetic problems."""
        problems = []
        
        # Different configurations
        configs = [
            (2, 5, 1),   # 2 vars, 5 ops, difficulty 1
            (3, 10, 2),  # 3 vars, 10 ops, difficulty 2
            (4, 15, 3),  # 4 vars, 15 ops, difficulty 3
            (5, 20, 4),  # 5 vars, 20 ops, difficulty 4
            (6, 25, 5),  # 6 vars, 25 ops, difficulty 5
        ]
        
        problem_id = 0
        for num_vars, num_ops, difficulty in configs:
            for _ in range(20):  # 20 problems per configuration
                problem_id += 1
                problem_text, final_values, ground_truth_path = \
                    self._generate_variable_problem(num_vars, num_ops, difficulty)
                
                problems.append(Problem(
                    problem_id=f"var_track_{problem_id}",
                    problem_text=problem_text,
                    difficulty=difficulty,
                    ground_truth_solution=final_values,
                    ground_truth_path=ground_truth_path,
                    metadata={
                        'num_vars': num_vars,
                        'num_ops': num_ops
                    }
                ))
        
        self.problems = problems
    
    def load_problems(self, difficulty: int = None, num_problems: int = None) -> List[Problem]:
        """Load problems from the benchmark."""
        return self.get_problems(difficulty, num_problems)
    
    def evaluate_solution(self, problem: Problem, solution: str) -> tuple:
        """Evaluate if solution matches final variable values."""
        try:
            import ast
            import re
            
            # Try to parse as dictionary
            if '{' in solution:
                # Extract dictionary
                dict_match = re.search(r'\{[^}]+\}', solution)
                if dict_match:
                    predicted = ast.literal_eval(dict_match.group())
                else:
                    return False, 0.0
            else:
                # Try to extract values
                predicted = {}
                var_names = [f"x{i+1}" for i in range(problem.metadata['num_vars'])]
                for var in var_names:
                    match = re.search(f'{var}[\\s:=]+(\\d+)', solution)
                    if match:
                        predicted[var] = int(match.group(1))
            
            ground_truth = problem.ground_truth_solution
            
            # Compare dictionaries
            if set(predicted.keys()) != set(ground_truth.keys()):
                return False, 0.0
            
            correct = sum(1 for k in ground_truth 
                         if k in predicted and predicted[k] == ground_truth[k])
            accuracy = correct / len(ground_truth) if ground_truth else 1.0
            
            return accuracy == 1.0, accuracy
            
        except:
            return False, 0.0
