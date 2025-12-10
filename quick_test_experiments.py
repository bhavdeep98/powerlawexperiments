"""
Quick Test of System 2 Experiments
===================================
Runs a minimal test to verify all components work.
"""

import os
from benchmarks import GameOf24Benchmark
from tree_of_thought_enhanced import EnhancedGameOf24, SearchStrategy

# Configuration
API_KEY = os.environ.get("OPENAI_API_KEY")
if not API_KEY:
    print("⚠ OPENAI_API_KEY not found. Please export it.")
    exit(1)

def quick_test():
    """Run a quick test of the system."""
    print("="*70)
    print("QUICK TEST: System 2 Experiments")
    print("="*70)
    
    # Test 1: Benchmark loading
    print("\n1. Testing benchmark loading...")
    benchmark = GameOf24Benchmark()
    problems = benchmark.get_problems(difficulty=1, num_problems=2)
    print(f"   ✓ Loaded {len(problems)} problems")
    
    # Test 2: Problem evaluation
    print("\n2. Testing problem evaluation...")
    problem = problems[0]
    test_solution = "(1 + 2) * (3 + 4) = 21"  # Wrong answer
    is_correct, credit = benchmark.evaluate_solution(problem, test_solution)
    print(f"   ✓ Evaluation works: correct={is_correct}, credit={credit:.2f}")
    
    # Test 3: Tree of Thought (minimal)
    print("\n3. Testing Tree of Thought (minimal search)...")
    if problems:
        numbers = problems[0].metadata.get('numbers', '1 2 3 4')
        solver = EnhancedGameOf24(numbers, model="gpt-4o-mini")  # Use cheaper model for test
        
        print(f"   Testing with numbers: {numbers}")
        result = solver.solve_tot(
            strategy=SearchStrategy.BFS,
            branching_factor=2,
            max_depth=2,  # Very shallow for quick test
            beam_width=2
        )
        
        print(f"   ✓ ToT search completed")
        print(f"   Success: {result['success']}")
        print(f"   Nodes explored: {result['metrics']['nodes_explored']}")
    
    print("\n" + "="*70)
    print("✓ All quick tests passed!")
    print("="*70)
    print("\nTo run full experiments:")
    print("  python3 run_system2_experiments.py")

if __name__ == "__main__":
    quick_test()
