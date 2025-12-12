"""
Tree of Thought (ToT) Experiment
================================
Simulates "System 2" reasoning using Breadth-First Search (BFS) over LLM-generated thoughts.

Task: Game of 24
- Input: 4 numbers (e.g., "4 9 10 13")
- Goal: Use +, -, *, / to reach 24.

Hypothesis:
- System 1 (Zero-Shot): Fails on complex combinations.
- System 2 (ToT Search): Succeeds by exploring the thought space.
"""

import os
import itertools
from openai import OpenAI
import time

# Configuration
API_KEY = os.environ.get("OPENAI_API_KEY")
MODEL = "gpt-4o"

if not API_KEY:
    print("⚠ OPENAI_API_KEY not found. Please export it.")
    exit(1)

client = OpenAI(api_key=API_KEY)

# ==============================================================================
# PROMPTS
# ==============================================================================

PROPOSE_PROMPT = """
Input: {input}
Possible next steps:
"""

VALUE_PROMPT = """
Evaluate if correct 24 can be reached from: {input}
Output "sure", "likely", or "impossible".
"""

# ==============================================================================
# LOGIC
# ==============================================================================

class GameOf24:
    def __init__(self, numbers: str):
        self.numbers = numbers

    def solve_zero_shot(self):
        """System 1: Direct Answer"""
        prompt = f"Use numbers {self.numbers} and basic arithmetic operations (+ - * /) to obtain 24. Return just the equation."
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content

    def solve_tot(self, branching_factor=3, depth=3):
        """System 2: Tree Search"""
        # State format: "Current Value: X | History: (...) | Remaining: [...]"
        initial_state = f"Current Numbers: [{self.numbers}] | History: Start"
        current_thoughts = [initial_state]
        
        for step in range(depth):
            next_thoughts = []
            print(f"  System 2 Depth {step+1}: Exploring {len(current_thoughts)} branches...")
            
            for state in current_thoughts:
                # 1. Generate (Propose) with explicit state tracking
                prompt = (
                    f"State: {state}\n"
                    f"Objective: Reach 24.\n"
                    f"Propose {branching_factor} valid next steps. "
                    f"Format: 'Operation: <op> | Remaining: <list>'\n"
                    f"Example: 'Operation: 10 + 4 = 14 | Remaining: [14, 9, 13]'"
                )
                
                try:
                    response = client.chat.completions.create(
                        model=MODEL,
                        messages=[{"role": "user", "content": prompt}],
                         temperature=0.7
                    )
                    candidates = response.choices[0].message.content.split('\n')
                    
                    for cand in candidates:
                        # Debug print
                        # print(f"DEBUG CAND: {cand}") 
                        if "|" in cand and "Remaining" in cand:
                            # Normalize spacing
                            next_thoughts.append(state + " -> " + cand.strip())
                            print(f"    -> Candidate: {cand.strip()}") # Show progress
                except:
                    continue

            # Prune to keep search manageable
            current_thoughts = next_thoughts[:5] 
            
            # Check for solution
            for t in current_thoughts:
                # Check if "24" appears in the Remaining list
                if "Remaining" in t:
                    try:
                        rem_str = t.split("Remaining:")[-1].strip(" []")
                        rem_vals = [float(x.strip()) for x in rem_str.split(",") if x.strip()]
                        
                        # If 24 is reached (checking loosely for float precision)
                        for val in rem_vals:
                            if abs(val - 24.0) < 1e-5:
                                return f"FOUND SOLUTION PATH: {t}"
                    except:
                        pass
        
        return "No solution found in search budget."

def basic_verify(equation):
    try:
        # Very unsafe eval, but fine for controlled experiment inputs
        # Clean string to just the math part if possible
        # This is a heuristic check
        if "=" in equation:
            lhs = equation.split("=")[0]
            return abs(eval(lhs) - 24) < 1e-5
        return abs(eval(equation) - 24) < 1e-5
    except:
        return False

# ==============================================================================
# EXPERIMENT RUNNER
# ==============================================================================

def run_experiment():
    # Hard cases from ToT paper
    tasks = ["4 9 10 13", "1 2 4 6", "5 5 5 11"] # 4,9,10,13 is notoriously hard
    
    print(f"Running Tree of Thoughts on {MODEL}...")
    
    for task in tasks:
        print(f"\nTask: {task}")
        
        # System 1
        print("Running System 1 (Zero-Shot)...")
        start = time.time()
        s1_ans = GameOf24(task).solve_zero_shot()
        s1_time = time.time() - start
        print(f"  Result: {s1_ans} ({s1_time:.2f}s)")
        
        # System 2
        print("Running System 2 (Tree of Thought)...")
        start = time.time()
        s2_ans = GameOf24(task).solve_tot()
        s2_time = time.time() - start
        print(f"  Result: {s2_ans} ({s2_time:.2f}s)\n")
        
        # Note: Verification is left manual/heuristic for now as parsing "thoughts" is non-trivial

if __name__ == "__main__":
    run_experiment()
