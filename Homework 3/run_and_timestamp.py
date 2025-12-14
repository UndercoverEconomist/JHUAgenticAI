"""
Run GSM8K with either the agent pipeline (`main_gsm8k.py`) or the baseline (`main_baseline.py`),
then produce a timestamped JSON summary in `output/` that contains per-example predictions
and the final accuracy.

If the agent pipeline fails (e.g., missing dependencies), the script will automatically
fall back to the baseline run and still produce the timestamped JSON.

Usage examples:
  python3 run_and_timestamp.py --model tinyllama:1.1b --max-examples 1319 --agent
  python3 run_and_timestamp.py --model tinyllama:1.1b --max-examples 100 --baseline

"""
from __future__ import annotations
import argparse
import subprocess
import os
import sys
import time
import json
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--model", default=os.getenv("OLLAMA_QWEN_MODEL", "qwen2:32b"), help="Model name")
parser.add_argument("--use-ollama-py", action="store_true", help="Use Ollama python client when available")
parser.add_argument("--split", default="test")
parser.add_argument("--max-examples", type=int, default=1319)
parser.add_argument("--agent", action="store_true", help="Run the agent pipeline (main_gsm8k.py)")
parser.add_argument("--baseline", action="store_true", help="Run the baseline (main_baseline.py)")
args = parser.parse_args()

os.environ["OLLAMA_QWEN_MODEL"] = args.model
if args.use_ollama_py:
    os.environ["USE_OLLAMA_PYTHON_CLIENT"] = "true"

os.makedirs("output", exist_ok=True)

# decide which script to run
if args.agent and not args.baseline:
    script = "main_gsm8k.py"
    result_jsonl = os.path.join("output", "gsm8k_results.jsonl")
elif args.baseline and not args.agent:
    script = "main_baseline.py"
    result_jsonl = os.path.join("output", "gsm8k_baseline_results.jsonl")
else:
    # default: try agent first
    script = "main_gsm8k.py"
    result_jsonl = os.path.join("output", "gsm8k_results.jsonl")

cmd = [sys.executable, script, "--model", args.model, "--max-examples", str(args.max_examples), "--split", args.split]
if args.use_ollama_py:
    cmd.append("--use-ollama-py")

print("Running:", " ".join(cmd))
start = time.time()
proc = subprocess.run(cmd, capture_output=True, text=True)
end = time.time()
print(f"Process exit code: {proc.returncode}")
if proc.stdout:
    print("--- STDOUT ---")
    print(proc.stdout[:4000])
if proc.stderr:
    print("--- STDERR ---")
    print(proc.stderr[:4000])

# If agent run failed, fall back to baseline
if proc.returncode != 0 and script == "main_gsm8k.py":
    print("Agent pipeline failed; falling back to baseline runner.")
    script = "main_baseline.py"
    result_jsonl = os.path.join("output", "gsm8k_baseline_results.jsonl")
    cmd = [sys.executable, script, "--model", args.model, "--max-examples", str(args.max_examples), "--split", args.split]
    if args.use_ollama_py:
        cmd.append("--use-ollama-py")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    print(f"Fallback process exit code: {proc.returncode}")
    if proc.stdout:
        print(proc.stdout[:4000])
    if proc.stderr:
        print(proc.stderr[:4000])

# Read results JSONL
entries = []
if os.path.exists(result_jsonl):
    with open(result_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entries.append(json.loads(line))
            except Exception:
                pass
else:
    print(f"Result file not found: {result_jsonl}")

# compute accuracy
total = len(entries)
correct = sum(1 for e in entries if e.get("correct") is True)
accuracy = float(correct) / float(total) if total else None

# timestamped output
ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
out_path = os.path.join("output", f"gsm8k_{ts}.json")
summary = {
    "timestamp": ts,
    "model": args.model,
    "script": script,
    "total": total,
    "correct": correct,
    "accuracy": accuracy,
    "runtime_seconds": end - start,
    "examples": entries,
}
with open(out_path, "w", encoding="utf-8") as of:
    json.dump(summary, of, ensure_ascii=False, indent=2)

print(f"Wrote timestamped summary: {out_path}")
print(f"Accuracy: {accuracy}")

