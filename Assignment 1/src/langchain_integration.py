import sys
from pathlib import Path
import json

# Ensure src is importable
SRC_DIR = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(SRC_DIR))

# Explicitly import langchain (user requested direct import)
try:
    import langchain  # noqa: F401
    from langchain.tools import Tool
except Exception as e:
    Tool = None  # we'll handle absence below

# Import agent node wrappers
try:
    import part1_reflex_agent as reflex_mod
except Exception:
    reflex_mod = None

try:
    import part2_goal_based_agent as goal_mod
except Exception:
    goal_mod = None

def _reflex_tool_impl(day: str):
    """Wrapper composing reflex nodes and returning JSON string."""
    if reflex_mod is None:
        return json.dumps({"error": "reflex module not available"})
    try:
        rpt = reflex_mod.reflex_day_lookup_tool(day)
        return json.dumps(rpt, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})

def _goal_tool_impl(pantry):
    """Wrapper composing goal nodes and returning JSON string."""
    if goal_mod is None:
        return json.dumps({"error": "goal module not available"})
    try:
        result = goal_mod.goal_compute_tool(pantry)
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})

def create_tools_if_langchain_available():
    """Return Tool objects if langchain is installed, else instruct install."""
    if Tool is None:
        return {"available": False, "message": "langchain not installed. pip install langchain", "tools": []}
    # Create Tool wrappers using the wrapper functions
    day_tool = Tool.from_function(_reflex_tool_impl, name="day_lookup", description="Lookup meal by day and return cost report JSON")
    compute_tool = Tool.from_function(_goal_tool_impl, name="compute_costs", description="Compute missing ingredients and additional cost for pantry")
    return {"available": True, "message": "Tools created", "tools": [day_tool, compute_tool]}

def demo_direct_calls():
    """Demonstrate calling the wrapper functions directly (no LangChain required)."""
    print("Demo: direct tool-like calls (no langchain required).")
    print("Day lookup (Monday):")
    print(_reflex_tool_impl("Monday"))
    print("\nCompute costs with pantry ['eggs', 'rice']:")
    print(_goal_tool_impl(["eggs", "rice"]))
    print("\nTo get langchain Tool objects, call create_tools_if_langchain_available().")

def main():
    info = create_tools_if_langchain_available()
    if not info["available"]:
        print("LangChain not detected.")
        print(info["message"])
        print("You can still use the demo functions without langchain:")
        demo_direct_calls()
        return
    print("LangChain detected — created tools:")
    for t in info["tools"]:
        print(f"- {t.name}: {t.description}")
    demo_direct_calls()

if __name__ == "__main__":
    main()
