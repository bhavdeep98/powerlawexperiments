"""
Multi-Agent System for Experiment 3: Criticality in Coordination
================================================================
Implements a flexible multi-agent debate environment to test:
1. Impact of Communication Bandwidth (Topology)
2. Phase Transitions in Consensus
"""

import time
import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from openai import OpenAI

DEFAULT_MODEL = "gpt-4o-mini"

@dataclass
class Message:
    sender_id: int
    content: str
    round_id: int
    timestamp: float

class DebateAgent:
    """
    An individual agent participating in the debate.
    Maintains its own history and "belief" (current best solution).
    """
    def __init__(self, agent_id: int, model: str = DEFAULT_MODEL, systemic_role: str = ""):
        self.agent_id = agent_id
        self.model = model
        self.client = OpenAI()
        self.history: List[Dict[str, str]] = []
        self.system_prompt = f"""You are Agent {agent_id}. You are collaborating with others to solve a problem.
{systemic_role}
Your goal is to find the correct answer AND reach consensus with others.

CRITICAL: You MUST use the following format for every response:

THOUGHT: (Private reasoning about what others said and your current state)
MESSAGE: (Public message to others. Propose checking a calculation, or suggest a solution)
FINAL_ANSWER: (The mathematical solution, e.g., "3 * 8 = 24". This MUST be the last line of your response.)

If you agree with a solution, you MUST repeat it in the FINAL_ANSWER field.
"""

    def perceive(self, messages: List[Message]):
        """Review messages from others."""
        if not messages:
            return
            
        context = "Messages from others:\n"
        for msg in messages:
            context += f"Agent {msg.sender_id}: {msg.content}\n"
        
        self.history.append({"role": "user", "content": context})

    def speak(self, problem_statement: str, current_round: int) -> Message:
        """Generate a response."""
        # Ensure system prompt is first
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Add problem statement
        messages.append({"role": "user", "content": f"Problem: {problem_statement}"})
        
        # Add history
        # (Filtering relevant history to avoid context overflow could be done here)
        messages.extend(self.history)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7
            )
            content = response.choices[0].message.content
            
            # Save own output to history
            self.history.append({"role": "assistant", "content": content})
            
            # Extract public message part for others
            public_content = content
            if "MESSAGE:" in content:
                parts = content.split("MESSAGE:")
                if len(parts) > 1:
                    public_content = parts[1].split("FINAL_ANSWER:")[0].strip()
            
            # Clean generic "THOUGHT" parts from public message if format wasn't strictly followed
            # (Simple heuristic)
            
            return Message(
                sender_id=self.agent_id,
                content=public_content,
                round_id=current_round,
                timestamp=time.time()
            )
        except Exception as e:
            return Message(self.agent_id, f"Error: {str(e)}", current_round, time.time())

    def get_final_answer(self) -> Optional[str]:
        """Extract final answer from the last assistant message."""
        # Search backwards for the last assistant message
        for msg in reversed(self.history):
            if msg['role'] == 'assistant':
                content = msg['content']
                if "FINAL_ANSWER:" in content:
                    raw = content.split("FINAL_ANSWER:")[-1].strip()
                    # Take only the first line if multiple, to avoid capturing trailing chatter
                    first_line = raw.split('\n')[0].strip()
                    clean = first_line.replace("ANSWER:", "").strip().rstrip('.')
                    clean = " ".join(clean.split())
                    return clean
        return None

class MultiAgentSystem:
    """
    Orchestrates the interaction between agents.
    Manages topology (who sees whom).
    """
    def __init__(self, num_agents: int, model: str = DEFAULT_MODEL):
        self.agents = [DebateAgent(i, model) for i in range(num_agents)]
        self.num_agents = num_agents
        
    def run_debate(self, problem: str, rounds: int = 3, bandwidth: int = 2, 
                  topology: str = "random", reliability: float = 1.0) -> Dict:
        """
        Run a debate.
        
        Args:
            problem: The problem text.
            rounds: Number of communication rounds.
            bandwidth: Used if topology is 'random'. Number of random peers to see.
            topology: 'random', 'fully_connected', 'ring', 'star'.
            reliability: Probability (0.0-1.0) that an agent speaks successfully per round.
        """
        all_messages: List[Message] = []
        
        for r in range(rounds):
            round_messages = []
            
            # 1. Speak phase
            for agent in self.agents:
                # Reliability check: Agent might fail to speak
                if random.random() > reliability:
                    # Agent fails to speak (Silence)
                    # We can either add a "Silence" message or nothing. 
                    # Nothing simulates packet loss / crash better.
                    continue

                msg = agent.speak(problem, r)
                round_messages.append(msg)
            
            all_messages.extend(round_messages)
            
            # 2. Listen phase (Topology enforcement)
            for i, agent in enumerate(self.agents):
                potential_peers = [m for m in round_messages if m.sender_id != i]
                
                visible_messages = []
                
                if topology == "fully_connected":
                    visible_messages = potential_peers
                    
                elif topology == "ring":
                    # Ring: i-1 and i+1
                    left = (i - 1) % self.num_agents
                    right = (i + 1) % self.num_agents
                    visible_messages = [m for m in potential_peers if m.sender_id in (left, right)]
                    
                elif topology == "star":
                    # Star: Agent 0 is Hub.
                    # Hub sees all. Leaves see Hub.
                    if i == 0: # Hub
                        visible_messages = potential_peers # Sees all
                    else: # Leaf
                        visible_messages = [m for m in potential_peers if m.sender_id == 0]
                        
                else: # "random"
                    if bandwidth >= len(potential_peers):
                        visible_messages = potential_peers
                    else:
                        visible_messages = random.sample(potential_peers, bandwidth)
                
                agent.perceive(visible_messages)
        
        # 3. Aggregation
        raw_answers = [a.get_final_answer() for a in self.agents]
        # Filter None
        valid_answers = [a for a in raw_answers if a]
        
        consensus_answer = None
        consensus_count = 0
        consensus_ratio = 0.0
        
        if valid_answers:
            from collections import Counter
            c = Counter(valid_answers)
            most_common = c.most_common(1)[0]
            consensus_answer = most_common[0]
            consensus_count = most_common[1]
            consensus_ratio = consensus_count / self.num_agents
            
        return {
            "problem": problem,
            "rounds": rounds,
            "bandwidth": bandwidth,
            "topology": topology,
            "reliability": reliability,
            "consensus_answer": consensus_answer,
            "consensus_ratio": consensus_ratio,
            "individual_answers": raw_answers,
            "transcript_len": len(all_messages)
        }
