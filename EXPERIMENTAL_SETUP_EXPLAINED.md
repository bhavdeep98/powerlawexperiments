# Experimental Setup Explained

## The Big Picture: What Are We Testing?

### Research Question

**"How does performance scale when we add more agents to solve a problem?"**

We're testing if there's a **power-law relationship** between:
- **Number of agents (A)** → **Performance (solve rate)**

**Hypothesis**: Performance = a × A^β_A

Where β_A is the scaling exponent we want to measure.

---

## Why Multiple Agents?

### The Scaling Law Question

In traditional neural scaling laws (Kaplan et al., 2020), we ask:
- "How does performance scale with model size (parameters)?"
- Answer: Performance ∝ Parameters^α

**We're extending this to agentic systems:**
- "How does performance scale with number of agents?"
- Answer: Performance ∝ Agents^β_A (we're testing this)

### Why This Matters

1. **Practical**: If you have a compute budget, should you use:
   - 1 large agent? 
   - 5 medium agents?
   - 10 small agents?

2. **Theoretical**: Do agentic systems follow the same scaling laws as single models?

3. **Coordination**: As you add agents, do they coordinate better or worse?

---

## The Experimental Setup

### What We Do

1. **Create N agents** (where N = 1, 2, 3, 5, 8, 13)
2. **Give them the same problem** (e.g., "Use 4, 9, 10, 13 to get 24")
3. **Each agent solves independently** (no communication)
4. **Measure two things**:
   - **Coordination**: Do they agree? (coordination accuracy)
   - **Performance**: Is the final answer correct? (solve rate)

### The Flow

```
Problem: "Use 4, 9, 10, 13 to get 24"
    ↓
[Agent 1] → Solution 1: "(10-4)*(13-9) = 24"
[Agent 2] → Solution 2: "(13-9)*(10-4) = 24"  
[Agent 3] → Solution 3: "6 * 4 = 24"
    ↓
Consensus Mechanism → Picks best/common solution
    ↓
Evaluate: Is it correct? → Solve Rate
Measure: Do agents agree? → Coordination Accuracy
```

---

## Why Do We Need a Consensus Mechanism?

### The Problem

When you have **multiple agents**, you get **multiple solutions**:
- Agent 1: "(10-4)*(13-9) = 24"
- Agent 2: "(13-9)*(10-4) = 24" (same, different order)
- Agent 3: "6 * 4 = 24" (equivalent, simplified)

**Question**: Which one do we use to measure "solve rate"?

### Why Not Just Pick One?

**Option 1: Use first agent's solution**
- ❌ Biased toward Agent 1
- ❌ Doesn't measure coordination
- ❌ Doesn't test if multiple agents help

**Option 2: Use best agent's solution**
- ❌ How do we know which is "best"?
- ❌ Requires evaluating all solutions (expensive)
- ❌ Doesn't measure coordination

**Option 3: Use consensus (what we do)**
- ✅ Tests if agents can coordinate
- ✅ Measures agreement (coordination accuracy)
- ✅ Simulates real multi-agent systems (they need to agree)
- ✅ Tests if "wisdom of crowds" helps

---

## What the Consensus Mechanism Does

### Step 1: Extract Equations

Each agent produces a long explanation:
```
Agent 1: "To solve using 4, 9, 10, 13:
          First: 10 - 4 = 6
          Then: 13 - 9 = 4
          Finally: 6 * 4 = 24"
```

We extract the **final equation**: `6 * 4` or `(10-4)*(13-9)`

### Step 2: Find Agreement

- Agent 1: `(10-4)*(13-9)`
- Agent 2: `(13-9)*(10-4)` (same, different order)
- Agent 3: `6 * 4` (equivalent)

**Coordination Accuracy**: How many pairs agree?
- Agents 1 & 2: Agree (same equation) ✓
- Agents 1 & 3: Agree (equivalent) ✓
- Agents 2 & 3: Agree (equivalent) ✓
- **Result**: 100% coordination

### Step 3: Pick Consensus Solution

**Majority Vote**: Use the most common equation
- `(10-4)*(13-9)` appears 2 times
- `6 * 4` appears 1 time
- **Consensus**: `(10-4)*(13-9)`

### Step 4: Evaluate

- **Consensus solution**: `(10-4)*(13-9)`
- **Evaluates to**: 24 ✓
- **Solve rate**: 1.0 (correct!)

---

## Why This Design?

### 1. Tests Coordination

**Research Question**: "Do agents coordinate better with more agents?"

**How we measure**:
- Coordination accuracy = agreement rate
- If agents agree → high coordination
- If agents disagree → low coordination

**What we test**:
- Does coordination improve with more agents?
- Is there a critical number where coordination emerges?

### 2. Tests Performance

**Research Question**: "Does more agents → better performance?"

**How we measure**:
- Solve rate = fraction of correct solutions
- Consensus solution is evaluated

**What we test**:
- Does solve rate scale with number of agents?
- Is there diminishing returns?

### 3. Tests Power-Law Scaling

**Research Question**: "Is there a power-law relationship?"

**How we measure**:
- Fit: solve_rate = a × A^β_A
- Extract exponent β_A

**What we test**:
- Does performance follow power-law scaling?
- What is the scaling exponent?

---

## The Experimental Variables

### Independent Variable (What We Change)
- **A**: Number of agents {1, 2, 3, 5, 8, 13}

### Dependent Variables (What We Measure)
1. **Solve Rate**: Fraction correct (0-1)
2. **Coordination Accuracy**: Agreement rate (0-1)
3. **Consensus Time**: How long to reach consensus

### Controlled Variables (What We Keep Fixed)
- **Model**: Same LLM for all agents (gpt-4o-mini)
- **Problem**: Same problems for all A values
- **Task**: Game of 24 (arithmetic puzzle)

---

## Why This Specific Setup?

### Why Game of 24?
- ✅ Clear evaluation (equals 24 or not)
- ✅ Multiple solution paths (tests coordination)
- ✅ Not too easy (needs reasoning)
- ✅ Not too hard (agents can solve it)

### Why Independent Agents?
- ✅ Tests if multiple agents help (no communication)
- ✅ Simpler to implement
- ✅ Clearer to analyze
- ✅ Tests "wisdom of crowds" effect

### Why Consensus Instead of Best?
- ✅ Tests coordination (key research question)
- ✅ More realistic (agents need to agree)
- ✅ Measures agreement (coordination accuracy)
- ✅ Tests if agreement correlates with performance

---

## Alternative Designs (Why We Didn't Use Them)

### Alternative 1: Agents Debate/Communicate
**Why not?**
- More complex to implement
- Harder to isolate A-scaling effect
- Communication adds another variable
- We test this in Experiment 3 (criticality)

### Alternative 2: Use Best Agent's Solution
**Why not?**
- Doesn't test coordination
- Requires evaluating all solutions (expensive)
- Doesn't measure agreement
- Not realistic (how do you know which is best?)

### Alternative 3: Average/Combine Solutions
**Why not?**
- Hard to combine equations
- Doesn't make sense for discrete problems
- Doesn't test coordination
- Consensus is more natural

---

## What We're Really Testing

### Primary Hypothesis
**H1**: Performance scales as a power law with number of agents
- solve_rate = a × A^β_A
- We measure β_A

### Secondary Hypotheses
**H2**: Coordination improves with more agents (up to a point)
- Coordination accuracy increases with A
- Then plateaus or decreases (diminishing returns)

**H3**: Coordination correlates with performance
- High coordination → high solve rate
- Low coordination → low solve rate

---

## The Consensus Mechanism's Role

### It's Not Just a Convenience

The consensus mechanism is **central to the experiment** because:

1. **It measures coordination**: How well do agents agree?
2. **It tests coordination-performance link**: Does agreement help?
3. **It simulates real systems**: Multi-agent systems need consensus
4. **It enables scaling analysis**: We can measure how coordination scales

### Without Consensus, We Can't Answer:
- ❌ Do agents coordinate better with more agents?
- ❌ Does coordination help performance?
- ❌ Is there a critical number where coordination emerges?

### With Consensus, We Can Answer:
- ✅ Coordination accuracy as a function of A
- ✅ Solve rate as a function of A
- ✅ Relationship between coordination and performance
- ✅ Power-law scaling of both metrics

---

## Summary

### The Setup
1. **Create N agents** (vary N = 1, 2, 3, 5, 8, 13)
2. **Each solves independently** (no communication)
3. **Extract solutions** (equations from each agent)
4. **Reach consensus** (pick most common/agreed solution)
5. **Measure**: Solve rate, coordination accuracy

### Why Consensus?
- **Tests coordination** (key research question)
- **Measures agreement** (coordination accuracy)
- **Enables scaling analysis** (how does coordination scale?)
- **Simulates real systems** (agents need to agree)

### What We Learn
- Does performance scale with number of agents? (power-law?)
- Does coordination improve with more agents?
- Is there a critical number where coordination emerges?
- What is the scaling exponent β_A?

---

**The consensus mechanism is not just a technical detail—it's essential to testing our research hypotheses about coordination and scaling in multi-agent systems.**
