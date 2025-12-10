"""
Tower of Hanoi Benchmark
========================
3-7 disk problems with state tracking critical.
"""

from typing import List, Tuple
from .base import Benchmark, Problem


class TowerOfHanoiBenchmark(Benchmark):
    """Tower of Hanoi problems requiring state tracking."""
    
    def __init__(self):
        super().__init__("Tower of Hanoi")
        self._generate_problems()
    
    def _generate_hanoi_problem(self, num_disks: int) -> tuple:
        """Generate a Tower of Hanoi problem."""
        # Initial state: all disks on rod A
        initial_state = {
            'A': list(range(num_disks, 0, -1)),  # [3, 2, 1] for 3 disks
            'B': [],
            'C': []
        }
        
        # Calculate minimum moves (2^n - 1)
        min_moves = 2 ** num_disks - 1
        
        problem_text = (
            f"Tower of Hanoi with {num_disks} disks. "
            f"Initial state: All disks on rod A (largest on bottom). "
            f"Goal: Move all disks to rod C. "
            f"What is the minimum number of moves required? "
            f"Also describe the first 5 moves."
        )
        
        # Ground truth: first 5 moves for 3-disk case
        if num_disks == 3:
            ground_truth_path = [
                "Move disk 1 from A to C",
                "Move disk 2 from A to B",
                "Move disk 1 from C to B",
                "Move disk 3 from A to C",
                "Move disk 1 from B to A"
            ]
        else:
            ground_truth_path = [f"Move sequence for {num_disks} disks"]
        
        return problem_text, min_moves, ground_truth_path, initial_state
    
    def _generate_problems(self):
        """Generate problems for 3-7 disks."""
        problems = []
        
        disk_counts = [3, 4, 5, 6, 7]
        difficulties = {3: 1, 4: 2, 5: 3, 6: 4, 7: 5}
        
        for num_disks in disk_counts:
            for i in range(10):  # 10 problems per disk count
                problem_text, answer, ground_truth_path, initial_state = \
                    self._generate_hanoi_problem(num_disks)
                
                problems.append(Problem(
                    problem_id=f"hanoi_{num_disks}_{i+1}",
                    problem_text=problem_text,
                    difficulty=difficulties[num_disks],
                    ground_truth_solution=answer,
                    ground_truth_path=ground_truth_path,
                    metadata={
                        'num_disks': num_disks,
                        'initial_state': initial_state,
                        'min_moves': answer
                    }
                ))
        
        self.problems = problems
    
    def load_problems(self, difficulty: int = None, num_problems: int = None) -> List[Problem]:
        """Load problems from the benchmark."""
        return self.get_problems(difficulty, num_problems)
    
    def evaluate_solution(self, problem: Problem, solution: str) -> tuple:
        """Evaluate if solution matches minimum moves."""
        try:
            import re
            # Extract number from solution
            numbers = re.findall(r'\d+', solution)
            if numbers:
                predicted = int(numbers[0])
            else:
                return False, 0.0
            
            ground_truth = problem.ground_truth_solution
            
            if predicted == ground_truth:
                return True, 1.0
            
            # Partial credit based on how close
            distance = abs(predicted - ground_truth)
            partial = max(0.0, 1.0 - distance / max(ground_truth, 1))
            return False, partial
            
        except:
            return False, 0.0
