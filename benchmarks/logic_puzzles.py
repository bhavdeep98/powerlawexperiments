"""
Logic Puzzle Benchmark
======================
Logic grid puzzles including Zebra puzzles and Einstein's riddle variants.
"""

from typing import List
from .base import Benchmark, Problem


class LogicPuzzleBenchmark(Benchmark):
    """Logic puzzle problems."""
    
    def __init__(self):
        super().__init__("Logic Puzzles")
        self._generate_problems()
    
    def _generate_problems(self):
        """Generate logic puzzle problems."""
        problems = []
        
        # Simple logic problems
        simple = [
            {
                'text': "If all roses are flowers, and some flowers are red, can we conclude all roses are red?",
                'answer': 'no',
                'difficulty': 1
            },
            {
                'text': "Alice is taller than Bob. Bob is taller than Charlie. Is Alice taller than Charlie?",
                'answer': 'yes',
                'difficulty': 1
            },
            {
                'text': "All birds can fly. Penguins are birds. Can penguins fly?",
                'answer': 'no',
                'difficulty': 1
            },
        ]
        
        # Transitivity problems
        transitivity = [
            {
                'text': "A > B, B > C, C > D. Is A > D?",
                'answer': 'yes',
                'difficulty': 2
            },
            {
                'text': "X is faster than Y. Y is faster than Z. Is X faster than Z?",
                'answer': 'yes',
                'difficulty': 2
            },
        ]
        
        # Zebra puzzle variants
        zebra = [
            {
                'text': """Five houses in a row. Each house has a different color, owner, pet, drink, and car.
House 1: Red, owns a dog, drinks coffee, drives a sedan
House 2: Blue, owns a cat, drinks tea, drives a truck
House 3: Green, owns a bird, drinks juice, drives a coupe
House 4: Yellow, owns a fish, drinks water, drives an SUV
House 5: White, owns a hamster, drinks milk, drives a convertible
Who owns the fish?""",
                'answer': 'House 4',
                'difficulty': 3
            },
        ]
        
        # Einstein's riddle variants
        einstein = [
            {
                'text': """Five people live in five houses. Each has a different nationality, drink, smoke, and pet.
The Englishman lives in the red house.
The Spaniard owns a dog.
Coffee is drunk in the green house.
The Ukrainian drinks tea.
The green house is immediately to the right of the ivory house.
The Old Gold smoker owns snails.
Kools are smoked in the yellow house.
Milk is drunk in the middle house.
The Norwegian lives in the first house.
Who drinks water?""",
                'answer': 'The Norwegian',
                'difficulty': 4
            },
        ]
        
        all_puzzles = simple + transitivity + zebra + einstein
        
        for i, puzzle in enumerate(all_puzzles):
            problems.append(Problem(
                problem_id=f"logic_{i+1}",
                problem_text=puzzle['text'],
                difficulty=puzzle['difficulty'],
                ground_truth_solution=puzzle['answer'],
                metadata={'type': 'logic'}
            ))
        
        self.problems = problems
    
    def load_problems(self, difficulty: int = None, num_problems: int = None) -> List[Problem]:
        """Load problems from the benchmark."""
        return self.get_problems(difficulty, num_problems)
    
    def evaluate_solution(self, problem: Problem, solution: str) -> tuple:
        """Evaluate if solution matches ground truth."""
        ground_truth = problem.ground_truth_solution.lower()
        solution_lower = solution.lower()
        
        # Check if answer appears in solution
        if ground_truth in solution_lower or solution_lower in ground_truth:
            return True, 1.0
        
        # Partial credit for similar answers
        if any(word in solution_lower for word in ground_truth.split()):
            return False, 0.5
        
        return False, 0.0
