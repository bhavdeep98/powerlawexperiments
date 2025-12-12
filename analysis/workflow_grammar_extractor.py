import re
import json
from typing import List, Dict, Any

class WorkflowExtractor:
    """
    Extracts abstract production rules (grammar) from agent execution traces.
    """
    
    def __init__(self):
        self.action_types = {
            'thought': r"Reasoning:.*",
            'call': r"Tool Call:.*",
            'observation': r"Observation:.*",
            'error': r"Error:.*"
        }

    def tokenize_trace(self, trace_text: str) -> List[str]:
        """
        Converts a raw text trace into a sequence of abstract tokens.
        E.g., "Reasoning: ... Tool Call: calculator ..." -> ["REASON", "TOOL_CALC", ...]
        """
        tokens = []
        
        # Split by typical separators or just scan line by line
        # Check if it's a single-line trace (contains "->")
        if "\n" not in trace_text and ("->" in trace_text or "|" in trace_text):
             # Normalize separators
             trace_text = trace_text.replace("|", "\n").replace("->", "\n")
             
        lines = trace_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Simple heuristic mapping
            if "Reasoning:" in line or "Thinking:" in line:
                tokens.append("THOUGHT")
            elif "Tool Call:" in line or "Action:" in line:
                # Extract specific tool if possible
                if "calculate" in line.lower():
                    tokens.append("TOOL_CALC")
                elif "search" in line.lower():
                    tokens.append("TOOL_SEARCH")
                else:
                    tokens.append("TOOL_GENERIC")
            elif "Observation:" in line or "Result:" in line:
                tokens.append("OBSERVATION")
            elif "Error:" in line or "Invalid" in line:
                tokens.append("ERROR")
            elif "Final Answer:" in line or "Solution:" in line:
                tokens.append("SOLUTION")
            elif "Operation:" in line:
                tokens.append("ACTION_OP")
            elif "Start" in line:
                tokens.append("START")
                
        return tokens

    def extract_production_rules(self, tokens: List[str]) -> List[str]:
        """
        Converts a token sequence into 2-gram production rules.
        E.g., ["THOUGHT", "TOOL_CALC"] -> "THOUGHT -> TOOL_CALC"
        """
        rules = []
        for i in range(len(tokens) - 1):
            rule = f"{tokens[i]} -> {tokens[i+1]}"
            rules.append(rule)
        return rules

    def extract_from_json_trace(self, trace_data: List[Dict]) -> List[str]:
        """
        Parses a structured JSON trace (if available) into tokens.
        """
        tokens = []
        for step in trace_data:
            # Handle standard step dictionary structure
            if 'thought' in step and step['thought']:
                tokens.append("THOUGHT")
            if 'action' in step and step['action']:
                tokens.append("ACTION")
            if 'observation' in step:
                tokens.append("OBSERVATION")
            
            # Heuristic for Tree of Thought nodes
            if 'node_type' in step:
                 tokens.append(f"NODE_{step['node_type'].upper()}")
                 
        return tokens
