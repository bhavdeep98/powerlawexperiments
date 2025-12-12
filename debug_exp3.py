
import random
from agents.multi_agent_system import MultiAgentSystem

def debug_run():
    mas = MultiAgentSystem(num_agents=3)
    problem = "Use numbers [3, 3, 8, 8] to make 24."
    print(f"Running debug debate on: {problem}")
    
    # Run for k=1 (Ring)
    result = mas.run_debate(problem, rounds=2, bandwidth=1)
    
    print("\n--- DEBUG OUTPUT ---")
    print(f"Consensus Ratio: {result['consensus_ratio']}")
    print("Individual Answers (Parsed):", result['individual_answers'])
    
    print("\n--- RAW LAST MESSAGES ---")
    for i, agent in enumerate(mas.agents):
        if agent.history:
            last_msg = agent.history[-1]['content']
            print(f"\nAgent {i} Last Message:\n[START]\n{last_msg}\n[END]")
        else:
            print(f"\nAgent {i}: No history.")

if __name__ == "__main__":
    debug_run()
