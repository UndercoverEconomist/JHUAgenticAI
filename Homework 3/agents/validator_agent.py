from agents.base_agent import BaseAgent
from utils import log_turn, extract_last_number, HAS_SYMPY

VALIDATOR_SYSTEM_PROMPT = """
You are a strict mathematical validator and critic.

You:
- check arithmetic and algebra step by step
- verify logical validity of transformations
- check consistency with the problem statement
- identify missing steps or unjustified leaps
- are skeptical by default.
""".strip()

class ValidatorAgent(BaseAgent):
    def __init__(self, model=None, temperature=0.0):
        super().__init__(
            name="Validator",
            model=model,
            system_prompt=VALIDATOR_SYSTEM_PROMPT,
            temperature=temperature,
        )

    def act(self, state):
        reasoning = state["initial_answer"]
        predicted = extract_last_number(reasoning)
        gold = state.get("solution_key")
        symbolic_ok = None
        if HAS_SYMPY and gold is not None and predicted is not None:
            try:
                symbolic_ok = float(predicted) == float(gold)
            except Exception:
                symbolic_ok = None
        critique_prompt = f"""You are given a solution to a math problem.\n\nSolution:\n{reasoning}\n\nYour tasks:\n1. Check each step for:\n   - arithmetic errors\n   - algebraic errors\n   - logical inconsistencies\n   - misinterpretations of the question\n2. Identify any missing steps or unjustified assertions.\n3. Decide how confident you are that the final answer is correct.\n\nRespond in a JSON-like format (you don't have to be perfectly valid JSON):\n\nerrors: [\n  \"Description of issue 1...\",\n  \"Description of issue 2...\"\n]\nstrengths: [\n  \"Good property 1...\",\n  \"Good property 2...\"\n]\noverall_quality: \"high\" | \"medium\" | \"low\"\nrevision_instructions: \"Detailed, actionable guidance on how to improve the solution.\"\nconfidence: <0-100 integer, your confidence that the final answer is correct>\n"""
        llm_critique = self.call(critique_prompt)
        validator_report = {
            "predicted_answer": predicted,
            "gold_answer": gold,
            "symbolic_check": symbolic_ok,
            "llm_critique": llm_critique,
        }
        state["validator_report"] = validator_report
        log_turn(state, self.name, str(validator_report))
        return state
