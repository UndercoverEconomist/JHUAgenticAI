from agents.base_agent import BaseAgent
from utils import log_turn

GENERATOR_SYSTEM_PROMPT = """
You are a careful, rigorous mathematical problem solver.

You:
- read the question slowly
- identify what's being asked
- solve step by step
- keep reasoning explicit
- aim for correct, not clever.
""".strip()

from typing import Any, Optional
import re
from agents.tools import calculate_expression


class GeneratorAgent(BaseAgent):
    def __init__(self, model=None, temperature=0.2):
        super().__init__(
            name="Generator",
            model=model,
            system_prompt=GENERATOR_SYSTEM_PROMPT,
            temperature=temperature,
        )

    def act(self, state: dict) -> dict:
        question = state["question"]

        # 1) First model pass: generate a full solution / reasoning
        prompt_parts = [
            "Solve the following math problem carefully.",
            f"Problem:\n{question}",
            "Show detailed reasoning.",
            "At the end, clearly mark the final answer on its own line, e.g:",
            "Final Answer: <answer here>"
        ]
        prompt = "\n\n".join(prompt_parts)

        first_out = self.call(prompt)
        log_turn(state, self.name + "-pass1", first_out)

        # 2) Tool invocation: try to extract an expression from the model output
        tool_result: Optional[str] = None

        # If model provides a "Final Answer: ..." line, try to evaluate that
        final_line = re.search(r"Final Answer:\s*(.+)$", first_out, re.IGNORECASE | re.MULTILINE)
        if final_line:
            candidate = final_line.group(1).strip()
            if re.search(r"[0-9]", candidate):
                tool_result = calculate_expression(candidate)

        # Otherwise try to find a simple arithmetic expression inside the reasoning
        if tool_result is None:
            expr_match = re.search(r"(-?\d+(?:\.\d+)?(?:\s*[+\-*/%^]\s*-?\d+(?:\.\d+)?)+)", first_out)
            if expr_match:
                expr = expr_match.group(1)
                tool_result = calculate_expression(expr)

        if tool_result is None:
            tool_result = "[no-tool-result]"

        state["tool_result"] = tool_result

        # 3) Second model pass: provide the tool output and ask model to finalize/confirm
        prompt2_parts = [
            "Refine the solution given the computed tool output below.",
            "Original model reasoning:",
            first_out,
            f"Tool computation result:\n{tool_result}",
            "Please produce a final, self-contained solution and clearly mark the final answer line."
        ]
        prompt2 = "\n\n".join(prompt2_parts)

        final_out = self.call(prompt2)
        state["initial_answer"] = final_out
        log_turn(state, self.name + "-final", final_out)
        return state
