import re
from typing import Optional

try:
    import sympy as sp
    HAS_SYMPY = True
except Exception:
    HAS_SYMPY = False


def extract_last_number(text: str) -> Optional[str]:
    matches = re.findall(r"-?\d+(?:\.\d+)?", text)
    if not matches:
        return None
    return matches[-1]


def log_turn(state, speaker: str, content: str) -> None:
    if "dialogue" not in state or state["dialogue"] is None:
        state["dialogue"] = []
    state["dialogue"].append({"speaker": speaker, "content": content})


# ANSI color helpers for terminal visualization
CSI = "\033["
RESET = CSI + "0m"
BLUE = CSI + "34m"
GREEN = CSI + "32m"


def color_prompt_blue(text: str) -> str:
    """Return a blue-colored version of `text` for terminal display.

    Note: This only affects terminal output (display). Prompts sent to the
    LLM should remain plain/uncolored; this helper is intended for printing
    the prompt for human inspection.
    """
    return f"{BLUE}{text}{RESET}"


def color_llm_green(text: str) -> str:
    """Return a green-colored version of LLM output for terminal display."""
    return f"{GREEN}{text}{RESET}"


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from text."""
    return re.sub(r"\x1b\[[0-9;]*m", "", text)
