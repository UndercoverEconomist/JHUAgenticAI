This repo contains code for Parts 0, 1, and 2 (data loading/validation, Simple Reflex, and Simple Goal-Based Agents).

# How to run
- Ensure Python 3.8+ is installed.
- Install dependencies:
  pip install -r requirements.txt
- Run Part 0 (load & validate):
  python3 src/part0_load_validate.py
- Run Part 1 (Simple Reflex Agent + demos):
  python3 src/part1_reflex_agent.py
- Run Part 2 (Simple Goal-Based Agent + demos):
  python3 src/part2_goal_based_agent.py

# What the scripts do
- part0_load_validate.py: Loads data from data/meals.xlsx (sheets: Meals, IngredientCosts), validates that every ingredient listed in Meals exists in IngredientCosts, and prints a short summary.
- part1_reflex_agent.py: Implements the Simple Reflex Agent that maps day → meal (based on Meals.DefaultDay) and reports ingredients + estimated cost. Runs two demos (Monday & Thursday).
- part2_goal_based_agent.py: Implements the Simple Goal-Based Agent: given a pantry list, computes MissingIngredients and AdditionalCost for every meal and selects the meal minimizing AdditionalCost (deterministic tie-break by total ingredient cost then name). Runs two demos:
  - Case A: pantry covering at least one meal (AdditionalCost = 0 expected).
  - Case B: empty pantry (shows MissingIngredients and AdditionalCost).

# Running demos & interacting with nodes
- Run scripts (each prints summary / demos to the console):
  
  ```
  # Part 0: load & validate
  python3 src/part0_load_validate.py

  # Part 1: Simple Reflex Agent + demos (Monday & Thursday)
  python3 src/part1_reflex_agent.py

  # Part 2: Simple Goal-Based Agent + demos (Case A & Case B)
  python3 src/part2_goal_based_agent.py
  ```

# Part 1 Reflection

Demo runs (console output):

```
--- Demo for day: Monday ---
Chosen meal: <unknown>
Ingredients: ['chicken', 'tortilla', 'onion', 'salsa', 'lime']
TotalEstimatedCost: 6.2
-------------------------

--- Demo for day: Thursday ---
Chosen meal: <unknown>
Ingredients: ['tuna', 'lettuce', 'tomato', 'cucumber', 'dressing', 'bread']
TotalEstimatedCost: 5.1
-------------------------
```

Limitations:

The Simple Reflex Agent implements a direct rule lookup from day → DefaultDay and reports estimated ingredient cost. This makes it predictable but narrowly scoped: it does not consider pantry state, user budgets, portion sizes, or substitutions, so recommendations may be infeasible or more expensive than necessary. The mapping from day to meal is brittle — any change in the spreadsheet or unexpected day formats can break selection. Costs are simple sums of unit costs (assumes one unit per ingredient) and do not handle missing or negative unit costs robustly. The agent also lacks any learning, preference handling, or multi-criteria optimization (e.g., nutrition, time-to-cook). These limitations motivate the Goal-Based agent in Part 2 and future nodes that incorporate pantry, budget, and fallback logic.

# Part 2 Reflection

Demo runs (console output):

```
--- Goal-Based Demo: Case A (pantry covers first meal -> AdditionalCost=0 expected) ---
Pantry: ['chicken', 'tortilla', 'onion', 'salsa', 'lime']
Selected meal: <unknown>
MissingIngredients: []
AdditionalCost: 0
TotalIngredientCost: 6.2
-------------------------------

--- Goal-Based Demo: Case B (empty pantry -> show MissingIngredients and AdditionalCost) ---
Pantry: []
Selected meal: <unknown>
MissingIngredients: ['eggs', 'cheese', 'spinach', 'bread', 'butter']
AdditionalCost: 2.65
TotalIngredientCost: 2.65
-------------------------------

Goal Agent graph: Input(pantry) -> ComputeCosts(tool) -> ArgminSelector -> Output(selection)
```

Assumptions & edge cases (≈120 words)

The Goal-Based agent assumes ingredients are normalized (trimmed, lowercase) and that each listed ingredient requires one unit. Ingredient unit costs are read from the IngredientCosts sheet; missing unit-cost entries are treated as zero in the current implementation. Costs can therefore understate true additional cost if IngredientCosts is incomplete. Zero or negative costs in the spreadsheet are accepted numerically, which may produce misleading selections; validation or clamping would be advisable. When multiple meals tie on AdditionalCost, deterministic tie-breaking uses total ingredient cost then meal name. An empty pantry returns full missing-ingredient lists and larger AdditionalCost values. Unknown ingredient names (typos, synonyms) will be treated as missing — consider data cleaning, synonyms, or explicit error reporting in future iterations.

# Part 3: Report, Diagrams, and Repro

The README file is here and already includes instructions.

Artifacts produced by running the demos
- figures/reflex_agent.png — minimal diagram for the Simple Reflex Agent.
- figures/goal_agent.png — minimal diagram for the Goal-Based Agent.
- runs/reflex_monday.txt, runs/reflex_thursday.txt — console logs for the two reflex demos.
- runs/goal_caseA.txt, runs/goal_caseB.txt — console logs for the two goal-based demos.

These files are created under the repo folder ./figures and ./runs. The diagram images are simple matplotlib visuals representing node flow; they are suitable for inclusion in the final report.
