from agents.base_agent import BaseAgent
from utils import log_turn
from agents.tools import calculate_expression
import re

REFINER_SYSTEM_PROMPT = """
You are a Math Refiner.

You:
- rewrite solutions clearly
- apply corrections faithfully
- fix errors in reasoning and calculation
- keep the solution readable and well structured
- ensure the final answer is explicit and correct if possible.
""".strip()


class RefinerAgent(BaseAgent):
    def __init__(self, model=None, temperature=0.3):
        super().__init__(
            name="Refiner",
            model=model,
            system_prompt=REFINER_SYSTEM_PROMPT,
            temperature=temperature,
        )

    def act(self, state):
        original = state.get("initial_answer", "")
        corrections = state.get("critic_report", "")

        prompt = f"""You will rewrite the solution to a math problem.

Original solution:
{original}

Corrections and instructions from the Critic:
{corrections}

Write a new solution that:
- fixes all identified issues
- is logically sound and clearly explained
- shows each major step
- ends with a clearly marked final answer line, e.g.:

Final Answer: <answer here>
"""

        # First model pass to produce a refined draft
        draft = self.call(prompt)
        log_turn(state, self.name + "-draft", draft)

        # Try to extract a final answer or arithmetic expression from draft
        tool_result = None
        final_match = re.search(r"Final Answer:\s*(.+)$", draft, re.IGNORECASE | re.MULTILINE)
        if final_match:
            candidate = final_match.group(1).strip()
            if re.search(r"[0-9]", candidate):
                tool_result = calculate_expression(candidate)

        if tool_result is None:
            expr_match = re.search(r"(-?\d+(?:\.\d+)?(?:\s*[+\-*/%^]\s*-?\d+(?:\.\d+)?)+)", draft)
            if expr_match:
                tool_result = calculate_expression(expr_match.group(1))

        if tool_result is None:
            tool_result = "[no-tool-result]"

        state["refiner_tool_result"] = tool_result

        # Second pass: ask model to incorporate/confirm computed result and finalize
        prompt2 = (
            "Please produce a final, corrected solution incorporating the computed result below.\n\n"
            f"Draft:\n{draft}\n\nComputed result:\n{tool_result}\n\n"
            "Make any necessary corrections and clearly mark the final answer line."
        )
        final_refined = self.call(prompt2)
        state["refined_answer"] = final_refined
        log_turn(state, self.name + "-final", final_refined)
        return state
