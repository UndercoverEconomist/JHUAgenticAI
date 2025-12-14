"""
Multi-Agent Math Solver using LangGraph

Agents:
- OrchestratorAgent
- GeneratorAgent
- ValidatorAgent (hybrid: symbolic + LLM critique)
- CriticAgent
- RefinerAgent
- EvaluatorAgent

Usage example at bottom of file.
"""

from __future__ import annotations

from typing import TypedDict, Optional, Dict, Any, List, Callable
import re

# Shared utilities (centralized to avoid circular imports)
from utils import log_turn, extract_last_number, HAS_SYMPY

from langgraph.graph import StateGraph, END
import os
import subprocess
import shlex
from dotenv import load_dotenv

# Load environment variables from .env (if present)
load_dotenv()


# ============================================================
# 1. GLOBAL STATE DEFINITION
# ============================================================

class MathState(TypedDict, total=False):
    # Core problem
    question: str
    solution_key: Optional[str]  # Optional ground-truth numeric answer (e.g. "27")

    # Agent outputs
    initial_answer: str          # Generator's first solution
    validator_report: Dict[str, Any]  # Validator's structured feedback
    critic_report: str           # Critic's textual correction instructions
    refined_answer: str          # Refiner's improved solution
    evaluation: str              # Evaluator's rubric-based evaluation
    final_answer: str            # Orchestrator's published final answer

    # Metrics
    automatic_metrics: Dict[str, Any]

    # Dialogue log for transparency
    dialogue: List[Dict[str, str]]  # [{ "speaker": "Generator", "content": "..." }, ...]


# ============================================================
# 2. UTILITIES
# ============================================================

def extract_last_number(text: str) -> Optional[str]:
    """
    Extract the last integer or decimal number from text.
    Very simple, but works fine for many math Q&A formats.
    """
    matches = re.findall(r"-?\d+(?:\.\d+)?", text)
    if not matches:
        return None
    return matches[-1]


def log_turn(state: MathState, speaker: str, content: str) -> None:
    """Append a dialogue turn to the state's dialogue log."""
    if "dialogue" not in state or state["dialogue"] is None:
        state["dialogue"] = []
    state["dialogue"].append({"speaker": speaker, "content": content})


# ============================================================
# 3. DEFAULT MODEL FUNCTION (REPLACE WITH REAL LLM)
# ============================================================


# Qwen 32B model integration (pseudo-code, replace with actual API)
def qwen_32b_model(prompt: str, temperature: float = 0.0) -> str:
    """Run a local Qwen model via the Ollama Python client (preferred) or CLI.

    Behavior:
    - Reads env var `OLLAMA_QWEN_MODEL` for the model name (default: `qwen2:32b`).
    - Calls `ollama run <model> --no-stream --prompt '<prompt>' --temperature <t>`.
    - Returns the stdout on success, or a helpful error string on failure.

    This keeps the integration optional: if Ollama or the model isn't available,
    the function returns an informative placeholder string instead of raising.
    """
    model_name = os.getenv("OLLAMA_QWEN_MODEL", "qwen2:32b")

    use_python_client = os.getenv("USE_OLLAMA_PYTHON_CLIENT", "false").lower() in ("1", "true", "yes")

    # 1) Try Python client if requested
    if use_python_client:
        try:
            import ollama
            # Try common client APIs dynamically
            try:
                # preferred API: ollama.chat or ollama.generate
                if hasattr(ollama, "chat"):
                    resp = ollama.chat(model=model_name, prompt=prompt, temperature=float(temperature))
                    return str(resp)
                if hasattr(ollama, "generate"):
                    resp = ollama.generate(model=model_name, prompt=prompt, temperature=float(temperature))
                    return str(resp)
                # fallback: look for a client class
                if hasattr(ollama, "Ollama"):
                    client = ollama.Ollama()
                    if hasattr(client, "chat"):
                        resp = client.chat(model=model_name, prompt=prompt, temperature=float(temperature))
                        return str(resp)
                    if hasattr(client, "generate"):
                        resp = client.generate(model=model_name, prompt=prompt, temperature=float(temperature))
                        return str(resp)
            except Exception as e:
                # If python client exists but call failed, return informative message
                return f"[ollama-py-error] {e}"
        except Exception:
            # Python client not available or import failed; fall back to CLI below
            pass

    # 2) Fallback to CLI invocation (previous behavior)
    # Some ollama versions don't support `--no-stream`; call with the more
    # broadly-supported flags and include temperature if available.
    # Ollama CLI expects: `ollama run MODEL [PROMPT]` (prompt is positional)
    cmd = ["ollama", "run", model_name, prompt]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if proc.returncode == 0:
            return proc.stdout.strip()
        err = proc.stderr.strip() or proc.stdout.strip()
        return f"[ollama-error] {err}"
    except FileNotFoundError:
        return "[ollama-missing] ollama CLI not found in PATH"
    except subprocess.TimeoutExpired:
        return "[ollama-error] ollama run timed out"
    except Exception as e:
        return f"[ollama-error] {e}"

MODEL = qwen_32b_model


# ============================================================


# Import agents from agents folder
from agents.orchestrator_agent import OrchestratorAgent
from agents.generator_agent import GeneratorAgent
from agents.validator_agent import ValidatorAgent
from agents.critic_agent import CriticAgent
from agents.refiner_agent import RefinerAgent
from agents.evaluator_agent import EvaluatorAgent


# ============================================================


# ============================================================
# 6. BUILD LANGGRAPH WORKFLOW
# ============================================================




# Create global agent instances, passing model and temperature
orchestrator = OrchestratorAgent(name="Orchestrator", model=MODEL, temperature=0.0)
generator = GeneratorAgent(model=MODEL, temperature=0.2)
validator = ValidatorAgent(model=MODEL, temperature=0.0)
critic = CriticAgent(model=MODEL, temperature=0.1)
refiner = RefinerAgent(model=MODEL, temperature=0.3)
evaluator = EvaluatorAgent(model=MODEL, temperature=0.0)


def orchestrator_node(state: MathState) -> MathState:
    """Wrapper to use OrchestratorAgent inside LangGraph."""
    return orchestrator.act(state)


def orchestrator_router(state: MathState) -> str:
    """Router for conditional edges out of the orchestrator node."""
    return orchestrator.decide_next(state)


def generator_node(state: MathState) -> MathState:
    return generator.act(state)


def validator_node(state: MathState) -> MathState:
    return validator.act(state)


def critic_node(state: MathState) -> MathState:
    return critic.act(state)


def refiner_node(state: MathState) -> MathState:
    return refiner.act(state)


def evaluator_node(state: MathState) -> MathState:
    return evaluator.act(state)


def build_app():
    workflow = StateGraph(MathState)

    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("generator", generator_node)
    workflow.add_node("validator", validator_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("refiner", refiner_node)
    workflow.add_node("evaluator", evaluator_node)

    workflow.set_entry_point("orchestrator")

    workflow.add_conditional_edges(
        "orchestrator",
        orchestrator_router,
        {
            "generator": "generator",
            "validator": "validator",
            "critic": "critic",
            "refiner": "refiner",
            "evaluator": "evaluator",
            "end": END,
        },
    )

    # Linear pipeline between worker agents
    workflow.add_edge("generator", "validator")
    workflow.add_edge("validator", "critic")
    workflow.add_edge("critic", "refiner")
    workflow.add_edge("refiner", "evaluator")
    workflow.add_edge("evaluator", "orchestrator")  # go back to orchestrator to finish

    return workflow.compile()


app = build_app()


# ============================================================
# 7. PUBLIC RUN FUNCTION
# ============================================================

def solve_math_problem(
    question: str,
    solution_key: Optional[str] = None,
) -> MathState:
    """
    Run the full 5-agent pipeline on an arbitrary math problem.

    Arguments:
        question: the math problem as a string.
        solution_key: optional numeric ground-truth answer as string (e.g., "27").

    Returns:
        final_state: MathState containing all intermediate artifacts.
    """
    initial_state: MathState = {
        "question": question,
        "solution_key": solution_key,
        "dialogue": [],
    }
    final_state = app.invoke(initial_state)
    return final_state


# ============================================================
# 8. EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    # Simple example
    q = (
        "A rectangle has length 8 cm and width 5 cm. "
        "If the length is increased by 50% and the width is decreased by 20%, "
        "what is the new area of the rectangle in square centimeters?"
    )
    # Let's quickly compute the true answer for demo:
    # original: 8 * 5 = 40 (not used)
    # new length = 8 * 1.5 = 12
    # new width  = 5 * 0.8 = 4
    # new area   = 12 * 4 = 48
    final = solve_math_problem(q, solution_key="48")

    print("\n================= FINAL ANSWER =================\n")
    print(final.get("final_answer", ""))

    print("\n================= AUTOMATIC METRICS ============\n")
    print(final.get("automatic_metrics", {}))

    print("\n================= EVALUATION ===================\n")
    print(final.get("evaluation", ""))

    print("\n================= DIALOGUE LOG =================\n")
    for turn in final.get("dialogue", []):
        print(f"[{turn['speaker']}] {turn['content']}\n")
