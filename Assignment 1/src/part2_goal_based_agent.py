import pandas as pd
from pathlib import Path
import os
import matplotlib.pyplot as plt
from datetime import datetime

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "meals.xlsx"

# Node: load data
def load_data_node(path: Path = DATA_PATH):
    if not path.exists():
        raise FileNotFoundError(f"Excel file not found: {path}")
    meals = pd.read_excel(path, sheet_name="Meals")
    costs = pd.read_excel(path, sheet_name="IngredientCosts")
    return meals, costs

# Node: normalize ingredient
def normalize_ingredient_node(name: str) -> str:
    return name.strip().lower()

# Node: extract ingredients from a cell
def extract_ingredients_node(cell) -> list:
    if pd.isna(cell):
        return []
    return [normalize_ingredient_node(p) for p in str(cell).split(";") if p.strip()]

# Node: build cost lookup
def build_cost_lookup_node(costs_df: pd.DataFrame) -> dict:
    lookup = {}
    for _, r in costs_df.iterrows():
        key = normalize_ingredient_node(r["Ingredient"])
        try:
            lookup[key] = float(r["UnitCost"])
        except Exception:
            lookup[key] = 0.0
    return lookup

# Node: compute costs for every meal given pantry (single-purpose)
def compute_costs_node(meals_df: pd.DataFrame, cost_lookup: dict, pantry: list):
    pantry_set = {normalize_ingredient_node(p) for p in pantry}
    results = []
    for _, r in meals_df.iterrows():
        meal_name = r.get("MealName", "<unknown>")
        ings = extract_ingredients_node(r.get("Ingredients", ""))
        missing = [i for i in ings if i not in pantry_set]
        additional = sum(cost_lookup.get(i, 0.0) for i in missing)
        total_ing_cost = sum(cost_lookup.get(i, 0.0) for i in ings)
        results.append({
            "MealName": meal_name,
            "Ingredients": ings,
            "MissingIngredients": missing,
            "AdditionalCost": additional,
            "TotalIngredientCost": total_ing_cost,
            "Row": r
        })
    return results

# Node: deterministic argmin selector
def argmin_selector_node(costs_list: list):
    if not costs_list:
        return None
    return sorted(
        costs_list,
        key=lambda x: (x["AdditionalCost"], x["TotalIngredientCost"], x["MealName"])
    )[0]

# High-level wrapper for LangChain Tool: takes pantry list and returns dict with best selection
def goal_compute_tool(pantry: list):
    meals_df, costs_df = load_data_node()
    lookup = build_cost_lookup_node(costs_df)
    costs = compute_costs_node(meals_df, lookup, pantry)
    best = argmin_selector_node(costs)
    return {"costs": costs, "best": best}

# Backward-compatible class API
class GoalBasedAgent:
    """
    Given a pantry P (list of ingredient strings), select meal minimizing:
      AdditionalCost(m) = sum(UnitCost(i) for i in Ingredients(m) \\ P)
    Tie-break: (1) smallest TotalIngredientCost, (2) MealName alphabetical.
    """
    def __init__(self, meals_df: pd.DataFrame, costs_df: pd.DataFrame):
        self.meals = meals_df
        self.cost_lookup = build_cost_lookup_node(costs_df)

    def compute_costs(self, pantry: list):
        return compute_costs_node(self.meals, self.cost_lookup, pantry)

    def select_min_additional(self, costs_list: list):
        return argmin_selector_node(costs_list)

# New node: ensure artifact dirs (reuse pattern)
def _ensure_artifact_dirs():
    base = Path(__file__).resolve().parents[1]
    figs = base / "figures"
    runs = base / "runs"
    figs.mkdir(parents=True, exist_ok=True)
    runs.mkdir(parents=True, exist_ok=True)
    return figs, runs

# New node: save a minimal diagram for the goal agent
def save_goal_diagram(path=None):
    figs, _ = _ensure_artifact_dirs()
    if path is None:
        path = figs / "goal_agent.png"
    fig, ax = plt.subplots(figsize=(6,3))
    ax.axis("off")
    ax.text(0.1, 0.7, "Input: pantry", bbox=dict(boxstyle="round", fc="lightblue"))
    ax.text(0.45, 0.7, "ComputeCosts (tool)", bbox=dict(boxstyle="round", fc="lightgreen"))
    ax.text(0.75, 0.4, "ArgminSelector", bbox=dict(boxstyle="round", fc="wheat"))
    ax.annotate("", xy=(0.4,0.7), xytext=(0.2,0.7), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(0.7,0.5), xytext=(0.5,0.6), arrowprops=dict(arrowstyle="->"))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path)

# New node: write run log
def write_run_log(filename: str, text: str):
    _, runs = _ensure_artifact_dirs()
    safe = "".join(c for c in filename if c.isalnum() or c in ("_", ".", "-"))
    out = runs / safe
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)

# Update demo_goal to also write run logs
def demo_goal(agent: GoalBasedAgent, pantry: list, label: str):
    lines = []
    lines.append(f"--- Goal-Based Demo: {label} ---")
    lines.append(f"Pantry: {pantry}")
    costs = agent.compute_costs(pantry)
    best = agent.select_min_additional(costs)
    if best is None:
        lines.append("No meals available.")
    else:
        lines.append(f"Selected meal: {best['MealName']}")
        lines.append(f"MissingIngredients: {best['MissingIngredients']}")
        lines.append(f"AdditionalCost: {best['AdditionalCost']}")
        lines.append(f"TotalIngredientCost: {best['TotalIngredientCost']}")
    lines.append("-------------------------------\n")
    out_text = "\n".join(lines)
    print(out_text)
    # filename from label
    fname = "goal_caseA.txt" if "Case A" in label else "goal_caseB.txt"
    write_run_log(fname, out_text)

def print_graph_description():
    print("Goal Agent graph: Input(pantry) -> ComputeCosts(tool) -> ArgminSelector -> Output(selection)\n")

def main():
    meals_df, costs_df = load_data_node(DATA_PATH)
    goal = GoalBasedAgent(meals_df, costs_df)

    # Case A: pantry fully covers at least one meal (use first meal's ingredients if available)
    if len(meals_df) > 0:
        first_row = meals_df.iloc[0]
        pantry_a = extract_ingredients_node(first_row.get("Ingredients", ""))
    else:
        pantry_a = []
    # Case B: empty pantry (missing items)
    pantry_b = []

    demo_goal(goal, pantry_a, "Case A (pantry covers first meal -> AdditionalCost=0 expected)")
    demo_goal(goal, pantry_b, "Case B (empty pantry -> show MissingIngredients and AdditionalCost)")
    p = save_goal_diagram()
    print(f"Goal diagram saved to: {p}")

if __name__ == "__main__":
    main()
