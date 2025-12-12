# Consensus Mechanism Explained

## Overview

The consensus mechanism is used in the A-scaling experiment to combine solutions from multiple agents into a single final answer. When N agents each generate a solution, we need to decide which one to use (or combine them).

---

## The Problem

**Input**: N agents each generate a solution (long text explanations)  
**Output**: A single equation that evaluates to 24 (for Game of 24)

**Challenge**: Agents generate verbose explanations like:
```
"To solve the problem using the numbers 4, 9, 10, and 13...
One approach is: (10 - 4) * (13 - 9) = 24"
```

We need to extract just the equation: `(10 - 4) * (13 - 9)`

---

## Current Implementation

### Step 1: Get Solutions from All Agents

```python
solutions = []
for agent in self.agents:
    solution = agent.solve(problem)  # Each agent generates text
    solutions.append(solution)
```

**Example with 3 agents:**
- Agent 1: "To solve... (10 - 4) * (13 - 9) = 24"
- Agent 2: "Using numbers 4, 9, 10, 13... (13 - 9) * (10 - 4) = 24"  
- Agent 3: "The solution is: 10 - 4 = 6, then 13 - 9 = 4, so 6 * 4 = 24"

### Step 2: Extract Equations from Each Solution

The `_reach_consensus()` method tries **three patterns** to find equations:

#### Pattern 1: Parenthesized Expressions
```python
# Looks for: (a) op (b), (a) op n, n op (a)
paren_patterns = [
    r'\([^)]+\)\s*[*+\-/]\s*\([^)]+\)',  # (10-4) * (13-9)
    r'\([^)]+\)\s*[*+\-/]\s*\d+',        # (10-4) * 2
    r'\d+\s*[*+\-/]\s*\([^)]+\)',        # 2 * (10-4)
]
```

**What it does:**
- Searches for expressions with parentheses
- Evaluates each match to see if it equals 24
- If yes, adds it to the equations list

**Example:**
- Input: "The solution is (10 - 4) * (13 - 9) = 24"
- Match: `(10 - 4) * (13 - 9)`
- Evaluates: `(10 - 4) * (13 - 9)` = 6 * 4 = 24 ✓
- Adds: `"(10 - 4) * (13 - 9)"` to equations list

#### Pattern 2: "= 24" Pattern
```python
eq_match = re.search(r'([\d+\-*/().\s]+)\s*=\s*24\b', sol, re.IGNORECASE)
```

**What it does:**
- Finds text like "X = 24" or "X equals 24"
- Extracts the left side (X)
- Cleans it (removes LaTeX, markdown)
- Verifies it evaluates to 24

**Example:**
- Input: "The answer is (10 - 4) * (13 - 9) = 24"
- Match: `(10 - 4) * (13 - 9) = 24`
- Extracts: `(10 - 4) * (13 - 9)`
- Evaluates: 24 ✓
- Adds to equations list

#### Pattern 3: Lines Containing "24"
```python
# Looks for lines with "24" and extracts expressions
if '24' in line:
    expr_match = re.search(r'([\(]?[\d+\-*/().\s]+[\)]?)\s*[=:]\s*24', line)
```

**What it does:**
- Finds lines containing "24"
- Extracts expressions before "= 24" or ": 24"
- Verifies they evaluate to 24

**Example:**
- Input line: "So the solution is: (10-4)*(13-9) = 24"
- Match: `(10-4)*(13-9) = 24`
- Extracts: `(10-4)*(13-9)`
- Evaluates: 24 ✓

### Step 3: Majority Vote

```python
if equations:
    most_common = Counter(equations).most_common(1)[0][0]
    return most_common
```

**What it does:**
- Counts how many times each equation appears
- Returns the most common one
- If agents agree on the same equation, that's the consensus

**Example:**
- Agent 1: `"(10 - 4) * (13 - 9)"`
- Agent 2: `"(13 - 9) * (10 - 4)"` (same, different order)
- Agent 3: `"(10 - 4) * (13 - 9)"`
- Result: `"(10 - 4) * (13 - 9)"` (appears twice)

### Step 4: Fallback

If no equations are found:
```python
return solutions[0]  # Return full text of first solution
```

The evaluator will try to extract an equation from the full text.

---

## Why It's Failing

### Problem 1: Extracting Numbers Instead of Equations

**Current behavior:**
- Consensus solution: `"6"` (just a number)
- Should be: `"(10 - 4) * (13 - 9)"` (an equation)

**Why this happens:**
1. Agents generate long explanations with multiple numbers
2. The regex patterns might match intermediate steps (like "10 - 4 = 6")
3. The consensus mechanism extracts "6" instead of the full equation
4. The evaluator tries to evaluate "6" → gets 6, not 24 → marked incorrect

### Problem 2: Pattern Matching Issues

**Example agent response:**
```
"To solve the problem:
1. First: 10 - 4 = 6
2. Then: 13 - 9 = 4  
3. Finally: 6 * 4 = 24"
```

**What happens:**
- Pattern 1 might match: `10 - 4` (evaluates to 6, not 24) ❌
- Pattern 2 might match: `6 * 4 = 24` but extract `6 * 4` (correct!) ✓
- But if it extracts just `6` from step 1, it fails ❌

### Problem 3: LaTeX/Markdown Formatting

**Agent response:**
```
"\[ (10 - 4) \times (13 - 9) = 24 \]"
```

**What happens:**
- Contains LaTeX: `\[`, `\]`, `\times`
- Regex might not match correctly
- Cleaning might remove important parts

---

## How to Fix It

### Solution 1: Prioritize Equations That Equal 24

Only extract equations that **actually evaluate to 24**:

```python
# Always verify before adding
result = eval(clean_expr)
if abs(result - 24.0) < 1e-5:  # Must equal 24
    equations.append(clean_expr)
```

### Solution 2: Look for Final Answers

Prioritize expressions that appear **after** "= 24" or near the end:

```python
# Look for patterns like "X = 24" where X evaluates to 24
# Prefer these over intermediate steps
```

### Solution 3: Better Pattern Matching

Improve regex to handle:
- LaTeX formatting: `\[`, `\]`, `\times`
- Markdown: Code blocks, math blocks
- Different formats: `(a-b)*(c-d)`, `(a-b) * (c-d)`, `(a-b)*(c-d)`

### Solution 4: Use LLM to Extract Equation

Instead of regex, use the LLM itself to extract the equation:

```python
prompt = f"""Extract the final equation from this solution that equals 24:
{solution}

Return only the mathematical expression, e.g., (10-4)*(13-9)"""
```

---

## Current Status

✅ **Working:**
- Extracts some equations
- Verifies they equal 24
- Uses majority vote

❌ **Not Working:**
- Sometimes extracts numbers instead of equations
- Misses equations in complex formatting
- Doesn't handle all LaTeX/markdown formats

---

## Example Flow

### Input (3 agents):

**Agent 1:**
```
"To solve using 4, 9, 10, 13:
(10 - 4) * (13 - 9) = 24"
```

**Agent 2:**
```
"Solution: (13 - 9) * (10 - 4) = 24"
```

**Agent 3:**
```
"Using the numbers:
Step 1: 10 - 4 = 6
Step 2: 13 - 9 = 4
Step 3: 6 * 4 = 24"
```

### Processing:

1. **Pattern 1** finds: `(10 - 4) * (13 - 9)` from Agent 1 ✓
2. **Pattern 2** finds: `(13 - 9) * (10 - 4)` from Agent 2 ✓
3. **Pattern 3** finds: `6 * 4` from Agent 3 ✓

### Equations List:
- `"(10 - 4) * (13 - 9)"` (from Agent 1)
- `"(13 - 9) * (10 - 4)"` (from Agent 2)
- `"6 * 4"` (from Agent 3)

### Majority Vote:
- `"(10 - 4) * (13 - 9)"` appears 1 time
- `"(13 - 9) * (10 - 4)"` appears 1 time  
- `"6 * 4"` appears 1 time
- **Result**: First one found (or could normalize to handle equivalent forms)

### Output:
`"(10 - 4) * (13 - 9)"` → Evaluates to 24 ✓

---

## Recommendations

1. **Normalize equivalent equations**: `(a-b)*(c-d)` = `(c-d)*(a-b)`
2. **Prioritize final answers**: Look for expressions near "= 24"
3. **Better cleaning**: Handle LaTeX, markdown, formatting
4. **Fallback to LLM**: If regex fails, ask LLM to extract equation
5. **Validation**: Always verify equation equals 24 before using

---

**Status**: Mechanism works but needs refinement to handle edge cases better.
