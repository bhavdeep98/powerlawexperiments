"""
Multi-Step Arithmetic Chain Benchmark
======================================
Chain calculation problems with progressive complexity (5, 10, 15, 20 steps).
"""

import random
from typing import List
from .base import Benchmark, Problem


class ArithmeticChainBenchmark(Benchmark):
    """Multi-step arithmetic chain problems."""
    
    def __init__(self):
        super().__init__("Arithmetic Chains")
        self._generate_problems()
    
    def _generate_chain(self, num_steps: int, difficulty: int) -> tuple:
        """Generate a chain calculation problem."""
        operations = ['+', '-', '*', '/']
        numbers = []
        
        # Start with a number
        start = random.randint(1, 20)
        numbers.append(start)
        current = start
        
        chain = [str(start)]
        ground_truth = [start]
        
        for i in range(num_steps):
            # Choose operation based on difficulty
            if difficulty <= 2:
                op = random.choice(['+', '-'])
            elif difficulty == 3:
                op = random.choice(['+', '-', '*'])
            else:
                op = random.choice(operations)
            
            # Prevent invalid operations
            if op == '-' and current < 1:
                op = '+'
            if op == '/' and abs(current) < 1:
                op = '+'
            
            # Choose next number
            if op == '+':
                next_num = random.randint(1, 50)
            elif op == '-':
                # Ensure we have a valid range for randint
                limit = max(1, int(current))
                next_num = random.randint(1, limit)
            elif op == '*':
                next_num = random.randint(2, 10)
            else:  # division
                next_num = random.choice([2, 3, 4, 5])
            
            # Apply operation
            if op == '+':
                current = current + next_num
            elif op == '-':
                current = current - next_num
            elif op == '*':
                current = current * next_num
            else:
                current = current / next_num
            
            chain.append(f" {op} {next_num}")
            numbers.append(next_num)
            ground_truth.append(current)
        
        problem_text = f"Calculate: {''.join(chain)} = ?"
        answer = current
        
        return problem_text, answer, ground_truth
    
    def _generate_problems(self):
        """Generate problems across different step counts and difficulties."""
        problems = []
        
        step_counts = [5, 10, 15, 20]
        difficulties = [1, 2, 3, 4, 5]
        
        problem_id = 0
        for steps in step_counts:
            for difficulty in difficulties:
                for _ in range(4):  # 4 problems per combination
                    problem_id += 1
                    problem_text, answer, ground_truth = self._generate_chain(steps, difficulty)
                    
                    problems.append(Problem(
                        problem_id=f"arithmetic_{problem_id}",
                        problem_text=problem_text,
                        difficulty=difficulty,
                        ground_truth_solution=answer,
                        ground_truth_path=ground_truth,
                        metadata={'steps': steps, 'answer': answer}
                    ))
        
        self.problems = problems
    
    def load_problems(self, difficulty: int = None, num_problems: int = None) -> List[Problem]:
        """Load problems from the benchmark."""
        return self.get_problems(difficulty, num_problems)
    
    def evaluate_solution(self, problem: Problem, solution: str) -> tuple:
        """Evaluate if solution matches ground truth."""
        try:
            # Try to extract number from solution
            import re
            numbers = re.findall(r'-?\d+\.?\d*', solution)
            if numbers:
                predicted = float(numbers[-1])  # Take last number
            else:
                predicted = float(solution)
            
            ground_truth = problem.ground_truth_solution
            
            if abs(predicted - ground_truth) < 1e-5:
                return True, 1.0
            
            # Partial credit
            distance = abs(predicted - ground_truth)
            partial = max(0.0, 1.0 - distance / max(abs(ground_truth), 1))
            return False, partial
            
        except:
            return False, 0.0
