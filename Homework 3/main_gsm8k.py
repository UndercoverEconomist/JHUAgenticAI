"""
Multi-Agent Math Solver (GSM8K runner)

This file is a thin copy of `main.py` that adds a GSM8K evaluation loop
so you can run the multi-agent pipeline over the GSM8K dataset and
produce a results file + summary.

Usage:
    python3 main_gsm8k.py --split test --max-examples 200

If you want a quick smoke-test, run with `--max-examples 2` (default used
by the quick-run below).
"""

from __future__ import annotations

from typing import TypedDict, Optional, Dict, Any, List, Callable
import re
import os
import subprocess
import shlex
from dotenv import load_dotenv

# Shared utilities (centralized to avoid circular imports)
from utils import log_turn, extract_last_number, HAS_SYMPY, strip_ansi

from langgraph.graph import StateGraph, END

# Load environment variables from .env (if present)
load_dotenv()

# Allow overriding model and client choice via early CLI args so agents are built
# with the requested model. We use parse_known_args to avoid interfering with
# the later run-specific args.
import argparse as _argparse
_early_parser = _argparse.ArgumentParser(add_help=False)
_early_parser.add_argument("--model", default=os.getenv("OLLAMA_QWEN_MODEL", "qwen2:32b"), help="Model name to use for ollama (overrides .env)")
_early_parser.add_argument("--use-ollama-py", action="store_true", help="Use Ollama python client when available")
_early_args, _remaining = _early_parser.parse_known_args()
if _early_args.model:
    os.environ["OLLAMA_QWEN_MODEL"] = _early_args.model
if _early_args.use_ollama_py:
    os.environ["USE_OLLAMA_PYTHON_CLIENT"] = "true"


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
    - Calls `ollama run <model> <prompt>`.
    - Returns the stdout on success, or a helpful error string on failure.
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
# GSM8K EVALUATION LOOP
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


import argparse
import json
import csv
from tqdm import tqdm


def normalize_number(s: str):
    if s is None:
        return None
    s = str(s).strip()
    s = s.replace(",", "")
    s = s.replace("$", "")
    s = s.rstrip(".\n ")
    try:
        if "." in s:
            return float(s)
        return int(s)
    except Exception:
        last = extract_last_number(s)
        if last is None:
            return None
        try:
            if "." in last:
                return float(last)
            return int(last)
        except Exception:
            return None


def numeric_eq(a, b):
    if a is None or b is None:
        return False
    try:
        af = float(a)
        bf = float(b)
        return abs(af - bf) < 1e-6
    except Exception:
        return str(a).strip() == str(b).strip()


def run_gsm8k(split: str = "test", max_examples: int = None, resume: bool = False, resume_run_dir: Optional[str] = None):
    try:
        from datasets import load_dataset
    except Exception as e:
        print("Please install the `datasets` package (pip install datasets) to run GSM8K evaluation.")
        raise

    # Load dataset
    dataset = None
    for config in ["main", None]:
        try:
            ds = load_dataset("gsm8k", config, split=split)
            dataset = ds
            break
        except Exception:
            continue
    if dataset is None:
        try:
            dataset = load_dataset("gsm8k", split=split)
        except Exception as e:
            print("Failed to load GSM8K via `datasets`. Error:", e)
            return

    n = len(dataset)
    print(f"Loaded GSM8K split={split} with {n} examples")
    if max_examples is not None:
        dataset = dataset.select(range(min(max_examples, n)))

    os.makedirs("output", exist_ok=True)
    # Determine run directory: either resume an existing run or create a new timestamped run dir
    from datetime import datetime

    if resume_run_dir:
        if not os.path.isdir(resume_run_dir):
            print(f"Resume run dir not found: {resume_run_dir}")
            return
        run_dir = resume_run_dir
    elif resume:
        # pick the latest run directory under output/
        if not os.path.isdir("output"):
            print("No output directory found to resume from.")
            return
        cand = [os.path.join("output", d) for d in os.listdir("output") if os.path.isdir(os.path.join("output", d))]
        if not cand:
            print("No previous runs found in output/ to resume.")
            return
        # choose the most recently modified directory
        run_dir = max(cand, key=os.path.getmtime)
        print(f"Resuming from latest run dir: {run_dir}")
    else:
        ts_iso = datetime.utcnow().isoformat()  # e.g. 2025-11-18T12:34:56.789012
        run_dir = os.path.join("output", ts_iso)
        os.makedirs(run_dir, exist_ok=True)

    results_path = os.path.join(run_dir, "gsm8k_results.jsonl")
    summary_path = os.path.join(run_dir, "gsm8k_summary.csv")

    # If resuming and results file exists, count already-processed examples and load cumulative correctness
    already_processed = 0
    correct = 0
    failures = []
    if (resume or resume_run_dir) and os.path.exists(results_path):
        try:
            with open(results_path, "r", encoding="utf-8") as rf:
                for line in rf:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        j = json.loads(line)
                        already_processed += 1
                        if j.get("correct"):
                            correct += 1
                    except Exception:
                        # ignore malformed lines
                        continue
        except Exception as e:
            print(f"Failed to read existing results for resume: {e}")

    # Open results file for append if resuming, else write new
    results_f = open(results_path, "a" if ((resume or resume_run_dir) and already_processed > 0) else "w", encoding="utf-8")

    total = already_processed

    # If we've already processed the entire (possibly truncated) dataset, nothing to do
    if already_processed >= len(dataset):
        print(f"Run already has {already_processed} entries which meets/exceeds the target {len(dataset)}. Nothing to do.")
        results_f.close()
        return

    for idx, ex in enumerate(tqdm(dataset, desc="GSM8K")):
        # skip already-processed examples when resuming
        if idx < already_processed:
            continue
        total += 1
        question = ex.get("question") or ex.get("Problem") or ex.get("question_text")
        gold_raw = ex.get("answer") or ex.get("correct_answer") or ex.get("solution") or ""
        gold_norm = normalize_number(gold_raw)

        print("\n================== Example #{} ==================\n".format(total))
        print(question)

        # Run pipeline
        final_state = solve_math_problem(question, solution_key=None)

        # Try to extract predicted answer from the compiled state
        pred_candidates = []
        if final_state.get("final_answer"):
            pred_candidates.append(final_state.get("final_answer"))
        if final_state.get("refined_answer"):
            pred_candidates.append(final_state.get("refined_answer"))
        if final_state.get("initial_answer"):
            pred_candidates.append(final_state.get("initial_answer"))

        pred = None
        for c in pred_candidates:
            if not c:
                continue
            c_plain = strip_ansi(c)
            n = normalize_number(c_plain)
            if n is not None:
                pred = n
                break

        correct_flag = numeric_eq(pred, gold_norm)
        if correct_flag:
            correct += 1
        else:
            failures.append({
                "question": question,
                "gold": gold_raw,
                "pred": pred,
                "state": {k: strip_ansi(str(v))[:1000] for k, v in final_state.items() if k in ("final_answer","refined_answer","initial_answer")}
            })

        out = {"question": question, "gold": gold_raw, "gold_norm": gold_norm, "pred": pred, "correct": correct_flag}
        results_f.write(json.dumps(out, ensure_ascii=False) + "\n")

    results_f.close()

    # Write summary CSV
    with open(summary_path, "w", newline="", encoding="utf-8") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["split", "total", "correct", "accuracy"])
        writer.writerow([split, total, correct, float(correct) / float(total) if total else 0.0])

    print("\nGSM8K run complete")
    print(f"Total: {total}, Correct: {correct}, Accuracy: {float(correct)/float(total) if total else 0.0:.4f}")
    print(f"Results written to: {results_path}")
    print(f"Summary written to: {summary_path}")

    if failures:
        print(f"Failures saved: {len(failures)} (first 3 shown)")
        for f in failures[:3]:
            print(json.dumps(f, ensure_ascii=False, indent=2))

    # Also write a JSON summary into the run directory with per-example results and accuracy
    try:
        from datetime import datetime
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        model_name = os.getenv("OLLAMA_QWEN_MODEL", "<unknown>")
        out_path = os.path.join(run_dir, f"gsm8k_summary_{ts}.json")
        # collect entries from results_path
        entries = []
        try:
            with open(results_path, "r", encoding="utf-8") as rf:
                for line in rf:
                    try:
                        entries.append(json.loads(line))
                    except Exception:
                        pass
        except Exception:
            entries = []

        summary_obj = {
            "timestamp": ts,
            "model": model_name,
            "split": split,
            "total": total,
            "correct": correct,
            "accuracy": float(correct) / float(total) if total else None,
            "entries": entries,
            "run_dir": run_dir,
        }
        with open(out_path, "w", encoding="utf-8") as of:
            json.dump(summary_obj, of, ensure_ascii=False, indent=2)
        print(f"Run directory: {run_dir}")
        print(f"Summary JSON written to: {out_path}")
    except Exception as e:
        print(f"Failed to write timestamped JSON summary: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", help="datasets split to evaluate (test|validation|train)")
    parser.add_argument("--max-examples", type=int, default=2, help="Limit number of examples for quick runs")
    parser.add_argument("--resume", action="store_true", help="Resume from the latest run in output/")
    parser.add_argument("--run-dir", default=None, help="Path to a specific run directory to resume (overrides --resume)")
    args = parser.parse_args()
    run_gsm8k(split=args.split, max_examples=args.max_examples, resume=args.resume, resume_run_dir=args.run_dir)
