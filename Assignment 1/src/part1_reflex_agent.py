import pandas as pd
from pathlib import Path
import os
import matplotlib.pyplot as plt
from datetime import datetime

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "meals.xlsx"

# Node: load data
def load_data_node(path: Path = DATA_PATH):
    """Node: load Meals and IngredientCosts sheets and return (meals_df, costs_df)."""
    if not path.exists():
        raise FileNotFoundError(f"Excel file not found: {path}")
    meals = pd.read_excel(path, sheet_name="Meals")
    costs = pd.read_excel(path, sheet_name="IngredientCosts")
    return meals, costs

# Node: normalize single ingredient
def normalize_ingredient_node(name: str) -> str:
    return name.strip().lower()

# Node: extract ingredients from a cell
def extract_ingredients_node(cell) -> list:
    if pd.isna(cell):
        return []
    return [normalize_ingredient_node(p) for p in str(cell).split(";") if p.strip()]

# Node: build cost lookup from costs_df
def build_cost_lookup_node(costs_df: pd.DataFrame) -> dict:
    lookup = {}
    for _, r in costs_df.iterrows():
        key = normalize_ingredient_node(r["Ingredient"])
        try:
            lookup[key] = float(r["UnitCost"])
        except Exception:
            lookup[key] = 0.0
    return lookup

# Node: day selector (accepts full day or 3-letter abbrev)
def day_selector_node(day: str) -> str:
    if not day:
        return ""
    return day.strip().lower()[:3]

# Node: meal lookup by day token
def meal_lookup_node(meals_df: pd.DataFrame, day_token: str):
    if not day_token:
        return None
    for _, r in meals_df.iterrows():
        dd = str(r.get("DefaultDay", "")).strip().lower()
        if dd.startswith(day_token):
            return r
    return None

# Node: cost report tool (given meal_row and cost_lookup)
def cost_report_node(meal_row, cost_lookup: dict):
    if meal_row is None:
        return None
    name = meal_row.get("MealName", "<unknown>")
    ings = extract_ingredients_node(meal_row.get("Ingredients", ""))
    total = 0.0
    missing_costs = []
    for ing in ings:
        if ing in cost_lookup:
            total += cost_lookup[ing]
        else:
            missing_costs.append(ing)
    return {"MealName": name, "Ingredients": ings, "TotalEstimatedCost": total, "MissingCosts": missing_costs}

# New node: ensure directories
def _ensure_artifact_dirs():
    base = Path(__file__).resolve().parents[1]
    figs = base / "figures"
    runs = base / "runs"
    figs.mkdir(parents=True, exist_ok=True)
    runs.mkdir(parents=True, exist_ok=True)
    return figs, runs

# New node: save a minimal diagram for the reflex agent
def save_reflex_diagram(path=None):
    figs, _ = _ensure_artifact_dirs()
    if path is None:
        path = figs / "reflex_agent.png"
    fig, ax = plt.subplots(figsize=(6,3))
    ax.axis("off")
    ax.text(0.1, 0.7, "Input: day", bbox=dict(boxstyle="round", fc="lightblue"))
    ax.text(0.5, 0.7, "DaySelector", bbox=dict(boxstyle="round", fc="lightgreen"))
    ax.text(0.5, 0.4, "MealLookup (tool)", bbox=dict(boxstyle="round", fc="wheat"))
    ax.text(0.85, 0.4, "Output: report", bbox=dict(boxstyle="round", fc="lightgrey"))
    ax.annotate("", xy=(0.38,0.7), xytext=(0.18,0.7), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(0.65,0.65), xytext=(0.55,0.65), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(0.8,0.45), xytext=(0.65,0.45), arrowprops=dict(arrowstyle="->"))
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

# High-level wrapper (composes nodes) — useful for LangChain Tool
def reflex_day_lookup_tool(day: str):
    meals_df, costs_df = load_data_node()
    lookup = build_cost_lookup_node(costs_df)
    tok = day_selector_node(day)
    meal_row = meal_lookup_node(meals_df, tok)
    report = cost_report_node(meal_row, lookup)
    return report

# Backward-compatible class API (keeps existing demos)
class SimpleReflexAgent:
    """
    Simple wrapper that uses the single-purpose nodes defined above.
    """
    def __init__(self, meals_df: pd.DataFrame, costs_df: pd.DataFrame):
        self.meals = meals_df
        # use the node that builds the lookup
        self.cost_lookup = build_cost_lookup_node(costs_df)

    def choose_meal_for_day(self, day: str):
        tok3 = day_selector_node(day)
        return meal_lookup_node(self.meals, tok3)

    def cost_report(self, meal_row):
        return cost_report_node(meal_row, self.cost_lookup)

def demo(agent: SimpleReflexAgent, day: str):
    lines = []
    lines.append(f"--- Demo for day: {day} ---")
    meal = agent.choose_meal_for_day(day)
    if meal is None:
        lines.append(f"No meal found for day '{day}'")
    else:
        rpt = agent.cost_report(meal)
        lines.append(f"Chosen meal: {rpt['MealName']}")
        lines.append(f"Ingredients: {rpt['Ingredients']}")
        lines.append(f"TotalEstimatedCost: {rpt['TotalEstimatedCost']}")
        if rpt["MissingCosts"]:
            lines.append(f"Missing unit costs: {rpt['MissingCosts']}")
    lines.append("-------------------------\n")
    out_text = "\n".join(lines)
    print(out_text)
    # save run log
    fname = f"reflex_{day.lower()}.txt"
    write_run_log(fname, out_text)

# ------------------ New: Goal-Based Agent ------------------
class GoalBasedAgent:
    """
    Goal: Given pantry P (list of ingredient strings), select meal minimizing:
      AdditionalCost(m) = sum(UnitCost(i) for i in Ingredients(m) \ P)
    Tie-break: smallest TotalIngredientCost, then MealName alphabetical.
    """
    def __init__(self, meals_df: pd.DataFrame, costs_df: pd.DataFrame):
        self.meals = meals_df
        self.cost_lookup = {}
        for _, r in costs_df.iterrows():
            key = normalize_ingredient(r["Ingredient"])
            try:
                self.cost_lookup[key] = float(r["UnitCost"])
            except Exception:
                self.cost_lookup[key] = 0.0

    def compute_costs(self, pantry: list):
        pantry_set = {normalize_ingredient(p) for p in pantry}
        results = []
        for _, r in self.meals.iterrows():
            meal_name = r.get("MealName", "<unknown>")
            ings = extract_ingredients(r.get("Ingredients", ""))
            missing = [i for i in ings if i not in pantry_set]
            additional = sum(self.cost_lookup.get(i, 0.0) for i in missing)
            total_ing_cost = sum(self.cost_lookup.get(i, 0.0) for i in ings)
            results.append({
                "MealName": meal_name,
                "Ingredients": ings,
                "MissingIngredients": missing,
                "AdditionalCost": additional,
                "TotalIngredientCost": total_ing_cost,
                "Row": r
            })
        return results

    def select_min_additional(self, costs_list: list):
        if not costs_list:
            return None
        # find minimum AdditionalCost
        costs_list_sorted = sorted(
            costs_list,
            key=lambda x: (x["AdditionalCost"], x["TotalIngredientCost"], x["MealName"])
        )
        return costs_list_sorted[0]

def demo_goal(agent: GoalBasedAgent, pantry: list, label: str):
    print(f"--- Goal-Based Demo: {label} ---")
    print(f"Pantry: {pantry}")
    costs = agent.compute_costs(pantry)
    best = agent.select_min_additional(costs)
    if best is None:
        print("No meals available.")
    else:
        print(f"Selected meal: {best['MealName']}")
        print(f"MissingIngredients: {best['MissingIngredients']}")
        print(f"AdditionalCost: {best['AdditionalCost']}")
        print(f"TotalIngredientCost: {best['TotalIngredientCost']}")
    print("-------------------------------\n")

# Minimal textual graphs
def print_graph_descriptions():
    print("Reflex Agent graph: Input(day) -> DaySelector -> MealLookup(tool) -> Output(report)")
    print("Goal Agent graph: Input(pantry) -> ComputeCosts(tool) -> ArgminSelector -> Output(selection)\n")

# ------------------ Integrate demos into main ------------------
def main():
    # Only run reflex demos here (GoalBasedAgent is implemented in part2_goal_based_agent.py)
    meals_df, costs_df = load_data_node(DATA_PATH)
    agent = SimpleReflexAgent(meals_df, costs_df)
    demo(agent, "Monday")
    demo(agent, "Thursday")
    # save diagram
    p = save_reflex_diagram()
    print(f"Reflex diagram saved to: {p}")

if __name__ == "__main__":
    main()