"""
LLM Game Theory Experiment
==========================

Uses OpenAI's API to simulate Game Theory scenarios across different models
and complexity levels.

Models: gpt-3.5-turbo, gpt-4o
Complexity: Level 1 (Basic), Level 2 (Noisy/Adversarial)

Requires: OPENAI_API_KEY environment variable.
"""

import os
import time
import json
import random
from openai import OpenAI
from typing import List, Dict

# Configuration
API_KEY = os.environ.get("OPENAI_API_KEY")
MOCK_MODE = not API_KEY

if not MOCK_MODE:
    client = OpenAI(api_key=API_KEY)
else:
    client = None

class LLMAgent:
    def __init__(self, name, model_name="gpt-4o"):
        self.name = name
        self.model_name = model_name
        self.history = []

    def generate(self, prompt, temperature=0.7):
        if MOCK_MODE:
            return self._mock_response(prompt)
        
        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": f"You are {self.name}, a helpful AI."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"API Error: {e}")
            return "Error generating response."

    def _mock_response(self, prompt):
        # Simulation logic for Mock Mode to demonstrate hypothesis
        is_gpt4 = "gpt-4" in self.model_name
        is_complex = "NOISE" in prompt or "CONSTRAINT" in prompt
        
        # Base probabilities
        success_prob = 0.8 if is_gpt4 else 0.5
        if is_complex:
            success_prob -= 0.2
        
        # 1. Coordination Game Response
        if "coordinate" in prompt.lower() or "where" in prompt.lower():
            locations = ["Grand Central Terminal", "Times Square", "Central Park"]
            # GPT-4 acts as a Schelling point optimizer (picks the most obvious)
            if random.random() < success_prob:
                return "Grand Central Terminal"
            else:
                return random.choice(locations)
                
        # 2. Adversarial Critic Response
        elif "critic" in prompt.lower():
            quality = 8 if is_gpt4 else 5
            return f"Score: {quality}/10. Feedback: Detailed critique."
            
        # 3. Adversarial Generator Response
        else:
            return "Here is the optimized python code..."

# ==============================================================================
# EXPERIMENT 1: COORDINATION (SCHELLING POINT)
# ==============================================================================

def run_coordination_game(model_name, complexity=1, n_rounds=10):
    print(f"  Running Coordination (Model: {model_name}, Level: {complexity})...")
    
    agent_a = LLMAgent("Agent A", model_name)
    agent_b = LLMAgent("Agent B", model_name)
    
    base_scenario = """
    You are lost in New York City with no way to communicate. 
    You need to meet your friend. You pick one famous landmark to go to.
    Where do you go? Answer with just the location name.
    """
    
    if complexity == 2:
        base_scenario += "\nCONSTRAINT: You cannot choose Times Square or Empire State Building. There is a thunderstorm."
        
    matches = 0
    for i in range(n_rounds):
        scenario = f"{base_scenario}\n(Round {i+1})"
        
        # Add random noise for complexity level 2
        if complexity == 2:
            scenario += f"\nNOISE: {random.randint(100,999)}"
            
        loc_a = agent_a.generate(scenario, temperature=0.5).strip().lower()
        loc_b = agent_b.generate(scenario, temperature=0.5).strip().lower()
        
        match = (loc_a in loc_b) or (loc_b in loc_a)
        if match: matches += 1
        
    success_rate = matches / n_rounds
    print(f"    Success Rate: {success_rate:.2%}")
    return success_rate

# ==============================================================================
# MAIN COMPARISON LOOP
# ==============================================================================

def run_comparison():
    models = ["gpt-3.5-turbo", "gpt-4o"]
    complexities = [1, 2]
    
    results = {
        "models": models,
        "complexities": complexities,
        "coordination_scores": {}  # keys: (model, complexity)
    }
    
    print("\n" + "="*60)
    print("MODEL COMPARISON EXPERIMENT")
    print("="*60)
    
    for model in models:
        for level in complexities:
            score = run_coordination_game(model, level)
            results["coordination_scores"][f"{model}_L{level}"] = score
            
    return results

if __name__ == "__main__":
    if MOCK_MODE:
        print("⚠ OPENAI_API_KEY not found. Running in MOCK MODE (Simulated Results).")
    
    data = run_comparison()
    
    with open('model_comparison_results.json', 'w') as f:
        json.dump(data, f, indent=2)
        
    print("\n✓ Comparison results saved.")
