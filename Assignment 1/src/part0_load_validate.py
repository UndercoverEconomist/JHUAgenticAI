import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "meals.xlsx"

def normalize_ingredient(name: str) -> str:
    return name.strip().lower()

def extract_ingredients(cell) -> list:
    if pd.isna(cell):
        return []
    return [normalize_ingredient(p) for p in str(cell).split(";") if p.strip()]

def load_data(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Excel file not found: {path}")
    meals = pd.read_excel(path, sheet_name="Meals")
    costs = pd.read_excel(path, sheet_name="IngredientCosts")
    return meals, costs

def validate_ingredients(meals_df: pd.DataFrame, costs_df: pd.DataFrame):
    meal_ings = set()
    for _, r in meals_df.iterrows():
        meal_ings.update(extract_ingredients(r.get("Ingredients", "")))
    cost_ings = {normalize_ingredient(i) for i in costs_df["Ingredient"].astype(str).tolist()}
    missing = sorted(meal_ings - cost_ings)
    if missing:
        raise AssertionError(f"Missing ingredients in IngredientCosts: {missing}")
    return meal_ings, cost_ings

def print_summary(meals_df: pd.DataFrame, costs_df: pd.DataFrame, unique_ings: set):
    print("=== Part 0: Data Summary ===")
    print(f"Number of meals: {len(meals_df)}")
    print(f"Number of unique ingredients (in Meals): {len(unique_ings)}")
    print("\nExample Meals row:")
    print(meals_df.iloc[0].to_dict() if len(meals_df) else "(Meals sheet empty)")
    print("\nExample IngredientCosts row:")
    print(costs_df.iloc[0].to_dict() if len(costs_df) else "(IngredientCosts sheet empty)")
    print("============================\n")

def main():
    meals_df, costs_df = load_data(DATA_PATH)
    meal_ings, cost_ings = validate_ingredients(meals_df, costs_df)
    print_summary(meals_df, costs_df, meal_ings)

if __name__ == "__main__":
    main()
