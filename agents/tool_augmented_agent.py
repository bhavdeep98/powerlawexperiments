import re
import math
from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass
from openai import OpenAI

@dataclass
class Tool:
    name: str
    description: str
    func: Callable[[str], str]

class ToolAugmentedAgent:
    """
    Agent capable of using tools in a ReAct loop.
    Supports varying K (number/richness of tools).
    """
    
    def __init__(self, model: str = "gpt-4o-mini", tools: List[Tool] = None, max_steps: int = 10):
        self.model = model
        self.tools = tools or []
        self.max_steps = max_steps
        self.client = OpenAI() # Assumes env var is set
        self.tool_map = {t.name: t for t in self.tools}

    def _get_system_prompt(self) -> str:
        tool_desc = "\n".join([f"- {t.name}: {t.description}" for t in self.tools])
        
        base_prompt = """You are a helpful AI assistant solving a problem.
You have access to the following tools:

{tool_desc}

Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!
"""
        return base_prompt.format(
            tool_desc=tool_desc if self.tools else "No tools available. Rely on your internal knowledge.",
            tool_names=", ".join(self.tool_map.keys()) if self.tools else "None"
        )

    def solve(self, question: str) -> Dict[str, Any]:
        """
        Runs the ReAct loop to solve the question.
        """
        system_prompt = self._get_system_prompt()
        history = f"Question: {question}\n"
        
        trace = []
        
        for step in range(self.max_steps):
            # 1. Generate Thought and Action
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": history}
                    ],
                    stop=["Observation:"], # Stop before generating observation hallucination
                    temperature=0.0
                )
                output = response.choices[0].message.content
            except Exception as e:
                return {"success": False, "error": str(e), "trace": trace, "final_answer": None}
            
            history += output + "\n"
            trace.append(output)
            print(f"    [Step {step+1}] {output[:50]}..." if len(output) > 50 else f"    [Step {step+1}] {output}")
            
            # 2. Parse Action
            if "Final Answer:" in output:
                final_answer = output.split("Final Answer:")[-1].strip()
                return {"success": True, "final_answer": final_answer, "trace": trace, "steps": step+1}
            
            # Regex to find Action and Action Input
            # pattern = r"Action:\s*(.+?)\nAction Input:\s*(.+)"
            # Let's try a robust line-based parse
            lines = output.split('\n')
            action = None
            action_input = None
            
            for line in lines:
                if line.startswith("Action:"):
                    action = line.replace("Action:", "").strip()
                if line.startswith("Action Input:"):
                    action_input = line.replace("Action Input:", "").strip()
            
            if action and action in self.tool_map and action_input is not None:
                # 3. Execute Tool
                tool = self.tool_map[action]
                try:
                    observation = tool.func(action_input)
                except Exception as e:
                    observation = f"Error executing tool: {str(e)}"
                
                obs_str = f"Observation: {observation}\n"
                history += obs_str
                trace.append(obs_str)
            elif action and action not in self.tool_map:
                 obs_str = f"Observation: Error: Action '{action}' not found. Available tools: {list(self.tool_map.keys())}\n"
                 history += obs_str
                 trace.append(obs_str)
            else:
                # No action found, or malformed.
                # If no tools are available (K=0), the model should give Final Answer.
                # If it didn't, we might be stuck.
                if not self.tools:
                    # Look for answer in text without "Final Answer" prefix?
                    pass
                
                # If loop continues without action, it might be just thinking. 
                # Provide a nudge or just continue?
                # For now, if no action triggers, we append a newline to prompt continuation?
                # Actually, if it stops at 'Observation:', but we didn't add one, the next turn might be weird.
                # Let's try forcing an observation of "Please continue."
                pass
                
        return {"success": False, "error": "Max steps reached", "trace": trace, "final_answer": None}

# ==============================================================================
# DEFAULT TOOLS
# ==============================================================================

def python_calculator(expression: str) -> str:
    """Evaluates a Python math expression."""
    try:
        # WHITELIST: only allow digits, +, -, *, /, (, ), ., whitespace
        if not re.match(r'^[\d\+\-\*\/\(\)\.\s]+$', expression):
             return "Error: Invalid characters in expression."
        return str(eval(expression))
    except Exception as e:
        return f"Error: {str(e)}"

def check_24(numbers: str) -> str:
    """Checks if the result achieves 24."""
    try:
        # Input might be "4 * 6" or just "24"
        val = eval(numbers)
        if math.isclose(val, 24.0, rel_tol=1e-5):
            return "YES. The result is 24."
        else:
            return f"NO. The result is {val}, not 24."
    except:
        return "Error checking 24."

BASIC_TOOLS = [
    Tool("Calculator", "Useful for calculating math expressions. Input should be a valid python expression.", python_calculator)
]

VALIDATOR_TOOLS = [
    Tool("Calculator", "Useful for calculating math expressions.", python_calculator),
    Tool("Validator", "Checks if a value equals 24.", check_24)
]
