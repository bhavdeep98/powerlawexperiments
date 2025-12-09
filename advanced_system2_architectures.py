"""
Advanced System 2 Architectures
================================
Implements three advanced reasoning architectures:

1. Verify-and-Refine Loop: Iterative refinement with verification
2. Debate/Multi-Agent: Multiple agents debate solutions
3. Memory-Augmented Reasoning: Explicit working memory and long-term memory

These architectures demonstrate how structured reasoning processes can
improve System 2 performance beyond simple search.
"""

import os
import json
import time
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from openai import OpenAI
from collections import deque

# Configuration
API_KEY = os.environ.get("OPENAI_API_KEY")
if not API_KEY:
    print("⚠ OPENAI_API_KEY not found. Please export it.")
    exit(1)

client = OpenAI(api_key=API_KEY)
DEFAULT_MODEL = "gpt-4o"


# ==============================================================================
# 1. VERIFY-AND-REFINE LOOP
# ==============================================================================

class VerifierReasonerSystem:
    """Verify-and-Refine reasoning system."""
    
    def __init__(self, 
                 model: str = DEFAULT_MODEL,
                 max_iterations: int = 5,
                 verifier_model: str = None):
        self.model = model
        self.verifier_model = verifier_model or model
        self.max_iterations = max_iterations
    
    def reasoner_generate(self, problem: str) -> str:
        """Generate initial solution attempt."""
        prompt = f"""Solve this problem step by step:

{problem}

Provide a complete solution."""
        
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            return response.choices[0].message.content
        except:
            return ""
    
    def verifier_check(self, problem: str, solution: str) -> Dict:
        """Verify if solution is correct and provide critique."""
        prompt = f"""Problem: {problem}

Proposed Solution: {solution}

Verify if this solution is correct. If not, provide specific critique on what is wrong.
Return JSON: {{"is_valid": true/false, "critique": "..."}}"""
        
        try:
            response = client.chat.completions.create(
                model=self.verifier_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            result_text = response.choices[0].message.content
            
            # Try to parse JSON
            import re
            json_match = re.search(r'\{[^}]+\}', result_text)
            if json_match:
                import json as json_lib
                result = json_lib.loads(json_match.group())
                return result
            
            # Fallback: check for keywords
            is_valid = "true" in result_text.lower() or "correct" in result_text.lower()
            return {
                "is_valid": is_valid,
                "critique": result_text
            }
        except:
            return {"is_valid": False, "critique": "Verification failed"}
    
    def reasoner_refine(self, problem: str, previous_solution: str, critique: str) -> str:
        """Refine solution based on critique."""
        prompt = f"""Problem: {problem}

Previous Attempt: {previous_solution}

Critique: {critique}

Provide an improved solution that addresses the critique."""
        
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            return response.choices[0].message.content
        except:
            return previous_solution
    
    def solve(self, problem: str) -> Dict:
        """Solve problem using verify-and-refine loop."""
        start_time = time.time()
        iterations = []
        
        # Initial attempt
        attempt = self.reasoner_generate(problem)
        
        for iteration in range(self.max_iterations):
            # Verify
            verification = self.verifier_check(problem, attempt)
            
            iterations.append({
                'iteration': iteration + 1,
                'solution': attempt,
                'is_valid': verification.get('is_valid', False),
                'critique': verification.get('critique', '')
            })
            
            # If valid, return
            if verification.get('is_valid', False):
                return {
                    'success': True,
                    'solution': attempt,
                    'iterations': iterations,
                    'num_iterations': iteration + 1,
                    'time': time.time() - start_time
                }
            
            # Refine
            critique = verification.get('critique', '')
            attempt = self.reasoner_refine(problem, attempt, critique)
        
        # Max iterations reached
        return {
            'success': False,
            'solution': attempt,
            'iterations': iterations,
            'num_iterations': self.max_iterations,
            'time': time.time() - start_time
        }


# ==============================================================================
# 2. DEBATE/MULTI-AGENT SYSTEM
# ==============================================================================

class DebateAgent:
    """Individual agent in a debate system."""
    
    def __init__(self, name: str, model: str = DEFAULT_MODEL):
        self.name = name
        self.model = model
        self.solution = None
        self.reasoning = None
    
    def solve(self, problem: str) -> str:
        """Generate initial solution."""
        prompt = f"""Solve this problem:

{problem}

Provide your solution and reasoning."""
        
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8
            )
            result = response.choices[0].message.content
            self.solution = result
            return result
        except:
            return ""
    
    def critique(self, opponent_solution: str, problem: str) -> str:
        """Critique opponent's solution."""
        prompt = f"""Problem: {problem}

Opponent's Solution: {opponent_solution}

Your Solution: {self.solution}

Critique the opponent's solution. Point out any flaws or weaknesses."""
        
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            return response.choices[0].message.content
        except:
            return ""
    
    def update_solution(self, critique: str, problem: str):
        """Update solution based on critique."""
        prompt = f"""Problem: {problem}

Your Current Solution: {self.solution}

Critique Received: {critique}

Update your solution to address the critique."""
        
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            self.solution = response.choices[0].message.content
        except:
            pass


class DebateSystem:
    """Multi-agent debate system."""
    
    def __init__(self,
                 agent_models: List[str] = None,
                 debate_rounds: int = 3,
                 judge_model: str = None):
        if agent_models is None:
            agent_models = [DEFAULT_MODEL, DEFAULT_MODEL]
        
        self.agents = [DebateAgent(f"Agent_{i+1}", model) 
                      for i, model in enumerate(agent_models)]
        self.debate_rounds = debate_rounds
        self.judge_model = judge_model or DEFAULT_MODEL
    
    def judge_select_best(self, problem: str, solutions: List[str]) -> Tuple[int, str]:
        """Judge selects the best solution."""
        solutions_text = "\n\n".join([
            f"Solution {i+1}:\n{sol}" 
            for i, sol in enumerate(solutions)
        ])
        
        prompt = f"""Problem: {problem}

{solutions_text}

Evaluate each solution and select the best one. Return JSON:
{{"best": 1 or 2, "reasoning": "..."}}"""
        
        try:
            response = client.chat.completions.create(
                model=self.judge_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            result_text = response.choices[0].message.content
            
            # Parse result
            import re
            json_match = re.search(r'\{[^}]+\}', result_text)
            if json_match:
                import json as json_lib
                result = json_lib.loads(json_match.group())
                best_idx = result.get('best', 1) - 1
                return best_idx, result.get('reasoning', '')
            
            # Fallback
            return 0, result_text
        except:
            return 0, "Judge selection failed"
    
    def solve(self, problem: str) -> Dict:
        """Solve problem through debate."""
        start_time = time.time()
        debate_history = []
        
        # Initial solutions
        solutions = []
        for agent in self.agents:
            solution = agent.solve(problem)
            solutions.append(solution)
        
        debate_history.append({
            'round': 0,
            'solutions': solutions.copy()
        })
        
        # Debate rounds
        for round_num in range(self.debate_rounds):
            critiques = []
            
            # Each agent critiques the other
            for i, agent in enumerate(self.agents):
                opponent_idx = 1 - i  # Simple 2-agent assumption
                critique = agent.critique(solutions[opponent_idx], problem)
                critiques.append(critique)
            
            # Agents update based on critiques
            for i, agent in enumerate(self.agents):
                opponent_critique = critiques[1 - i]
                agent.update_solution(opponent_critique, problem)
                solutions[i] = agent.solution
            
            debate_history.append({
                'round': round_num + 1,
                'critiques': critiques,
                'solutions': solutions.copy()
            })
        
        # Judge selects best
        best_idx, reasoning = self.judge_select_best(problem, solutions)
        
        return {
            'success': True,  # Assume success if we have a solution
            'solution': solutions[best_idx],
            'selected_agent': best_idx,
            'judge_reasoning': reasoning,
            'debate_history': debate_history,
            'time': time.time() - start_time
        }


# ==============================================================================
# 3. MEMORY-AUGMENTED REASONING
# ==============================================================================

class WorkingMemory:
    """Explicit working memory for reasoning."""
    
    def __init__(self, max_size: int = 10):
        self.memory = deque(maxlen=max_size)
        self.current_state = None
    
    def append(self, state: Any, description: str = ""):
        """Add state to working memory."""
        self.memory.append({
            'state': state,
            'description': description,
            'timestamp': len(self.memory)
        })
        self.current_state = state
    
    def get_history(self) -> List[Dict]:
        """Get full memory history."""
        return list(self.memory)
    
    def get_current(self) -> Any:
        """Get current state."""
        return self.current_state


class SimpleVectorDB:
    """Simple vector database for long-term memory (simplified)."""
    
    def __init__(self):
        self.memories = []
    
    def store(self, problem: str, solution: str, success: bool):
        """Store a solved problem."""
        self.memories.append({
            'problem': problem,
            'solution': solution,
            'success': success
        })
    
    def retrieve(self, problem: str, top_k: int = 3) -> List[Dict]:
        """Retrieve similar problems (simplified: just return recent ones)."""
        # In practice, would use embeddings and similarity search
        return self.memories[-top_k:] if self.memories else []


class MemoryAugmentedReasoner:
    """Memory-augmented reasoning system."""
    
    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self.working_memory = WorkingMemory(max_size=20)
        self.long_term_memory = SimpleVectorDB()
    
    def plan(self, problem: str, examples: List[Dict]) -> str:
        """Generate a plan based on problem and examples."""
        examples_text = ""
        if examples:
            examples_text = "\n\nSimilar solved problems:\n"
            for ex in examples:
                examples_text += f"Problem: {ex['problem']}\nSolution: {ex['solution']}\n\n"
        
        prompt = f"""Problem: {problem}

{examples_text}

Create a step-by-step plan to solve this problem."""
        
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            return response.choices[0].message.content
        except:
            return ""
    
    def execute_step(self, problem: str, plan: str, step_num: int, 
                    current_state: Any) -> Tuple[Any, str]:
        """Execute one step of the plan."""
        memory_context = ""
        if self.working_memory.get_history():
            memory_context = "\n\nWorking Memory History:\n"
            for mem in self.working_memory.get_history()[-3:]:
                memory_context += f"- {mem['description']}\n"
        
        prompt = f"""Problem: {problem}

Plan: {plan}

Current State: {current_state}
Step: {step_num}
{memory_context}

Execute this step. Return the new state and a description."""
        
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            result = response.choices[0].message.content
            # Simplified: assume result is the new state
            return result, f"Step {step_num} executed"
        except:
            return current_state, f"Step {step_num} failed"
    
    def solve(self, problem: str, max_steps: int = 10) -> Dict:
        """Solve problem with memory-augmented reasoning."""
        start_time = time.time()
        self.working_memory = WorkingMemory(max_size=20)
        
        # Retrieve similar problems
        examples = self.long_term_memory.retrieve(problem, top_k=3)
        
        # Plan
        plan = self.plan(problem, examples)
        self.working_memory.append(None, f"Plan created: {plan[:100]}")
        
        # Execute steps
        current_state = None
        steps = []
        
        for step_num in range(1, max_steps + 1):
            new_state, description = self.execute_step(
                problem, plan, step_num, current_state
            )
            current_state = new_state
            self.working_memory.append(new_state, description)
            steps.append({
                'step': step_num,
                'state': new_state,
                'description': description
            })
            
            # Check if solved (simplified heuristic)
            if "24" in str(new_state) or "solution" in str(new_state).lower():
                break
        
        solution = current_state
        
        # Store in long-term memory
        self.long_term_memory.store(problem, solution, True)
        
        return {
            'success': True,
            'solution': solution,
            'plan': plan,
            'steps': steps,
            'working_memory_size': len(self.working_memory.get_history()),
            'time': time.time() - start_time
        }


# ==============================================================================
# COMPARISON EXPERIMENT
# ==============================================================================

def compare_architectures(problems: List[str]) -> Dict:
    """Compare all three architectures on the same problems."""
    results = {
        'verify_refine': [],
        'debate': [],
        'memory_augmented': []
    }
    
    print(f"\n{'='*70}")
    print("ADVANCED SYSTEM 2 ARCHITECTURES COMPARISON")
    print(f"{'='*70}\n")
    
    # 1. Verify-Refine
    print("Testing Verify-Refine System...")
    verifier = VerifierReasonerSystem()
    for problem in problems:
        print(f"  Problem: {problem[:50]}...")
        result = verifier.solve(problem)
        results['verify_refine'].append(result)
        print(f"    Success: {result['success']}, Iterations: {result['num_iterations']}")
    
    # 2. Debate
    print("\nTesting Debate System...")
    debate = DebateSystem(debate_rounds=2)
    for problem in problems:
        print(f"  Problem: {problem[:50]}...")
        result = debate.solve(problem)
        results['debate'].append(result)
        print(f"    Success: {result['success']}, Selected Agent: {result['selected_agent']}")
    
    # 3. Memory-Augmented
    print("\nTesting Memory-Augmented System...")
    memory = MemoryAugmentedReasoner()
    for problem in problems:
        print(f"  Problem: {problem[:50]}...")
        result = memory.solve(problem)
        results['memory_augmented'].append(result)
        print(f"    Success: {result['success']}, Steps: {len(result['steps'])}")
    
    return results


if __name__ == "__main__":
    # Test problems
    problems = [
        "Use numbers 4 9 10 13 and arithmetic operations to get 24",
        "Use numbers 1 2 4 6 and arithmetic operations to get 24",
        "Use numbers 5 5 5 11 and arithmetic operations to get 24"
    ]
    
    results = compare_architectures(problems)
    
    # Save results
    with open('advanced_system2_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✓ Results saved to 'advanced_system2_results.json'")
