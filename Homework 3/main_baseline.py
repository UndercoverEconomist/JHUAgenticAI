"""
Baseline GSM8K runner that prompts the LLM directly (no agents).

Usage:
  python3 main_baseline.py --model qwen2.5:32b --max-examples 200

This script mirrors `main_gsm8k.py`'s evaluation logic but sends a single
prompt to the model per example and extracts the final numeric answer.
"""

from __future__ import annotations
import os
import argparse
import json
import csv
from dotenv import load_dotenv
load_dotenv()

# Model runner (same fallback behavior as main files)
import subprocess


def qwen_model_run(prompt: str, model_name: str, use_python_client: bool = False, temperature: float = 0.0) -> str:
    # try python client
    if use_python_client:
        try:
            import ollama
            if hasattr(ollama, "chat"):
                return str(ollama.chat(model=model_name, prompt=prompt, temperature=float(temperature)))
            if hasattr(ollama, "generate"):
                return str(ollama.generate(model=model_name, prompt=prompt, temperature=float(temperature)))
            if hasattr(ollama, "Ollama"):
                client = ollama.Ollama()
                if hasattr(client, "chat"):
                    return str(client.chat(model=model_name, prompt=prompt, temperature=float(temperature)))
                if hasattr(client, "generate"):
                    return str(client.generate(model=model_name, prompt=prompt, temperature=float(temperature)))
        except Exception:
            pass
    # CLI fallback
    cmd = ["ollama", "run", model_name, prompt]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if proc.returncode == 0:
            return proc.stdout.strip()
        return proc.stderr.strip() or proc.stdout.strip()
    except Exception as e:
        return f"[error] {e}"


# helpers copied/adapted from main_gsm8k
import re
from utils import strip_ansi, extract_last_number


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


# Runner
from datasets import load_dataset
from tqdm import tqdm


def run_baseline(model_name: str, use_python_client: bool, split: str = "test", max_examples: int = None, temperature: float = 0.0):
    # load GSM8K
    try:
        ds = load_dataset("gsm8k", split=split)
    except Exception:
        ds = load_dataset("gsm8k", "main", split=split)

    n = len(ds)
    print(f"Loaded GSM8K split={split} with {n} examples")
    if max_examples is not None:
        ds = ds.select(range(min(max_examples, n)))

    os.makedirs("output", exist_ok=True)
    results_path = os.path.join("output", "gsm8k_baseline_results.jsonl")
    summary_path = os.path.join("output", "gsm8k_baseline_summary.csv")

    results_f = open(results_path, "w", encoding="utf-8")

    total = 0
    correct = 0
    failures = []

    for ex in tqdm(ds, desc="GSM8K"):
        total += 1
        question = ex.get("question")
        gold_raw = ex.get("answer")
        gold_norm = normalize_number(gold_raw)

        # build a simple prompt
        prompt = (
            "You are a careful math solver. Answer the problem and put the final answer on its own line like:\nFinal Answer: <answer>\n\n"
            f"Problem:\n{question}\n"
        )
        out = qwen_model_run(prompt, model_name=model_name, use_python_client=use_python_client, temperature=temperature)

        # clean and extract
        out_plain = strip_ansi(out)
        pred = normalize_number(out_plain)
        correct_flag = numeric_eq(pred, gold_norm)
        if correct_flag:
            correct += 1
        else:
            failures.append({"question": question, "gold": gold_raw, "pred": pred, "raw_output": out_plain[:1000]})

        results_f.write(json.dumps({"question": question, "gold": gold_raw, "pred": pred, "correct": correct_flag}, ensure_ascii=False) + "\n")

    results_f.close()
    with open(summary_path, "w", newline="", encoding="utf-8") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["split", "total", "correct", "accuracy"])
        writer.writerow([split, total, correct, float(correct) / float(total) if total else 0.0])

    print("\nBaseline run complete")
    print(f"Total: {total}, Correct: {correct}, Accuracy: {float(correct)/float(total) if total else 0.0:.4f}")
    print(f"Results: {results_path}")
    print(f"Summary: {summary_path}")

    if failures:
        print(f"Failures: {len(failures)} (showing first 3)")
        for f in failures[:3]:
            print(json.dumps(f, ensure_ascii=False, indent=2))

    # write timestamped JSON summary
    try:
        from datetime import datetime
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        model_used = model_name
        out_path = os.path.join("output", f"gsm8k_baseline_{ts}.json")
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
            "model": model_used,
            "split": split,
            "total": total,
            "correct": correct,
            "accuracy": float(correct) / float(total) if total else None,
            "entries": entries,
        }
        with open(out_path, "w", encoding="utf-8") as of:
            json.dump(summary_obj, of, ensure_ascii=False, indent=2)
        print(f"Timestamped baseline JSON written to: {out_path}")
    except Exception as e:
        print(f"Failed to write baseline timestamped JSON: {e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.getenv("OLLAMA_QWEN_MODEL", "qwen2:32b"), help="Model name to use")
    ap.add_argument("--use-ollama-py", action="store_true", help="Use Ollama python client when available")
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-examples", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    run_baseline(model_name=args.model, use_python_client=args.use_ollama_py, split=args.split, max_examples=args.max_examples, temperature=args.temperature)
