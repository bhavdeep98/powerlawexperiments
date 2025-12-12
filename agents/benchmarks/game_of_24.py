"""
Game of 24 Benchmark
====================
Expanded dataset with 100+ problems across 5 difficulty levels.
"""

import random
from typing import List
from .base import Benchmark, Problem


class GameOf24Benchmark(Benchmark):
    """Game of 24 benchmark with expanded problem set."""
    
    def __init__(self):
        super().__init__("Game of 24")
        self._generate_problems()
    
    def _generate_problems(self):
        """Generate comprehensive problem set."""
        # Easy problems (difficulty 1)
        easy = [
            "1 2 3 4", "2 2 2 3", "3 3 3 3", "4 4 4 4", "5 5 5 1",
            "1 1 2 12", "2 2 3 6", "3 3 4 4", "4 4 5 5", "1 2 3 6",
            "2 3 4 5", "1 3 4 8", "2 4 6 6", "3 5 5 7", "1 4 5 6",
            "2 2 4 6", "3 3 6 6", "4 4 8 8", "1 5 6 8", "2 3 5 10",
            "1 1 3 9", "2 2 5 10", "3 3 7 7", "4 4 9 9", "1 6 7 8"
        ]
        
        # Medium problems (difficulty 2)
        medium = [
            "1 2 4 6", "3 3 8 8", "10 10 4 4", "2 3 5 7", "4 5 6 7",
            "1 3 5 9", "2 4 6 8", "3 5 7 9", "1 2 5 10", "4 6 8 10",
            "1 4 6 9", "2 5 7 11", "3 6 8 12", "1 3 6 10", "2 4 7 9",
            "4 5 7 8", "1 5 7 11", "2 6 8 10", "3 4 7 10", "1 2 7 12",
            "5 6 7 8", "1 4 8 11", "2 3 8 11", "3 7 9 11", "1 6 9 12"
        ]
        
        # Hard problems (difficulty 3)
        hard = [
            "4 9 10 13", "5 5 5 11", "3 3 7 7", "1 5 5 5", "2 7 7 10",
            "1 3 4 6", "2 5 5 10", "3 3 3 8", "1 1 5 5", "2 2 2 9",
            "4 6 9 13", "1 7 8 9", "2 4 9 11", "3 5 8 12", "1 2 8 13",
            "5 6 9 14", "1 4 9 12", "2 6 10 14", "3 7 11 15", "1 3 7 13",
            "4 8 12 16", "2 5 9 13", "1 6 10 14", "3 8 11 15", "2 7 11 16"
        ]
        
        # Very Hard problems (difficulty 4)
        very_hard = [
            "1 1 1 8", "2 2 3 9", "3 3 4 10", "4 4 5 11", "1 2 6 12",
            "2 3 7 13", "3 4 8 14", "4 5 9 15", "1 3 8 14", "2 4 9 15",
            "3 5 10 16", "1 4 10 16", "2 5 11 17", "3 6 12 18", "1 5 12 18",
            "2 6 13 19", "3 7 14 20", "1 6 14 20", "2 7 15 21", "3 8 16 22",
            "1 7 16 22", "2 8 17 23", "3 9 18 24", "1 8 18 24", "2 9 19 25"
        ]
        
        # Expert problems (difficulty 5)
        expert = [
            "1 1 2 10", "2 2 4 12", "3 3 6 14", "4 4 8 16", "1 2 7 15",
            "2 3 8 16", "3 4 9 17", "4 5 10 18", "1 3 9 17", "2 4 10 18",
            "3 5 11 19", "1 4 11 19", "2 5 12 20", "3 6 13 21", "1 5 13 21",
            "2 6 14 22", "3 7 15 23", "1 6 15 23", "2 7 16 24", "3 8 17 25",
            "1 7 17 25", "2 8 18 26", "3 9 19 27", "1 8 19 27", "2 9 20 28"
        ]
        
        # Combine all problems
        all_problems = []
        for difficulty, problem_list in [
            (1, easy), (2, medium), (3, hard), (4, very_hard), (5, expert)
        ]:
            for i, problem_text in enumerate(problem_list):
                all_problems.append(Problem(
                    problem_id=f"game24_{difficulty}_{i+1}",
                    problem_text=f"Use numbers {problem_text} and arithmetic operations (+ - * /) to obtain 24.",
                    difficulty=difficulty,
                    metadata={'numbers': problem_text}
                ))
        
        self.problems = all_problems
    
    def load_problems(self, difficulty: int = None, num_problems: int = None) -> List[Problem]:
        """Load problems from the benchmark."""
        return self.get_problems(difficulty, num_problems)
    
    def evaluate_solution(self, problem: Problem, solution: str) -> tuple:
        """
        Evaluate if solution equals 24.
        
        Returns:
            (is_correct, partial_credit)
        """
        try:
            # Extract equation from solution
            if "=" in solution:
                lhs = solution.split("=")[0].strip()
            else:
                lhs = solution.strip()
            
            # Evaluate
            result = eval(lhs)
            
            # Check if equals 24
            if abs(result - 24.0) < 1e-5:
                return True, 1.0
            
            # Partial credit based on how close
            distance = abs(result - 24.0)
            partial = max(0.0, 1.0 - distance / 24.0)
            return False, partial
            
        except:
            return False, 0.0
    
    def get_ground_truth_path(self, problem: Problem) -> List[str]:
        """Get ground truth solution path (simplified)."""
        # In practice, would have pre-computed solutions
        # For now, return empty list
        return []
