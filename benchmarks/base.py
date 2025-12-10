"""
Base classes for benchmarks.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass


@dataclass
class Problem:
    """Represents a single problem instance."""
    problem_id: str
    problem_text: str
    difficulty: int  # 1-5 scale
    ground_truth_solution: Any = None
    ground_truth_path: List[Any] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class Benchmark(ABC):
    """Abstract base class for benchmarks."""
    
    def __init__(self, name: str):
        self.name = name
        self.problems: List[Problem] = []
    
    @abstractmethod
    def load_problems(self, difficulty: int = None, num_problems: int = None) -> List[Problem]:
        """Load problems from the benchmark."""
        pass
    
    @abstractmethod
    def evaluate_solution(self, problem: Problem, solution: Any) -> Tuple[bool, float]:
        """
        Evaluate a solution.
        
        Returns:
            (is_correct, partial_credit) where partial_credit is 0.0-1.0
        """
        pass
    
    def get_ground_truth_path(self, problem: Problem) -> List[Any]:
        """Get the correct reasoning path for a problem."""
        return problem.ground_truth_path or []
    
    def get_problems(self, difficulty: int = None, num_problems: int = None) -> List[Problem]:
        """Get problems, loading if necessary."""
        if not self.problems:
            self.problems = self.load_problems(difficulty, num_problems)
        
        # Filter by difficulty if specified
        if difficulty is not None:
            filtered = [p for p in self.problems if p.difficulty == difficulty]
        else:
            filtered = self.problems
        
        # Limit number if specified
        if num_problems is not None:
            filtered = filtered[:num_problems]
        
        return filtered
