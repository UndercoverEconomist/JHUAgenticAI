from agents.base_agent import BaseAgent
from utils import log_turn, extract_last_number

EVALUATOR_SYSTEM_PROMPT = """
You are a math solution evaluator and teacher.

You:
- compare two solutions
- judge correctness, clarity, rigor, completeness
- evaluate whether the refined solution is an improvement
- give numeric scores and short explanations.
""".strip()

class EvaluatorAgent(BaseAgent):
    def __init__(self, model=None, temperature=0.0):
        super().__init__(
            name="Evaluator",
            model=model,
            system_prompt=EVALUATOR_SYSTEM_PROMPT,
            temperature=temperature,
        )

    def act(self, state):
        original = state["initial_answer"]
        refined = state["refined_answer"]
        gold = state.get("solution_key")
        auto = {}
        base_num = extract_last_number(original) if original else None
        ref_num = extract_last_number(refined) if refined else None
        auto["baseline_extracted_answer"] = base_num
        auto["refined_extracted_answer"] = ref_num
        auto["solution_key"] = gold
        if gold is not None:
            try:
                base_ok = (base_num is not None) and (float(base_num) == float(gold))
            except Exception:
                base_ok = None
            try:
                ref_ok = (ref_num is not None) and (float(ref_num) == float(gold))
            except Exception:
                ref_ok = None
        else:
            base_ok = None
            ref_ok = None
        auto["baseline_correct"] = base_ok
        auto["refined_correct"] = ref_ok
        auto["improved"] = (
            base_ok is False and ref_ok is True
        ) if (base_ok is not None and ref_ok is not None) else None
        state["automatic_metrics"] = auto
        prompt = f"""Evaluate the improvement from the original to the refined math solution.\n\nProblem:\n{state['question']}\n\nOriginal solution:\n{original}\n\nRefined solution:\n{refined}\n\nOptional gold numeric answer (may be None): {gold}\n\nScore the REFINED solution relative to the original on:\n- Correctness (0-10)\n- Rigor / soundness of reasoning (0-10)\n- Clarity (0-10)\n- Completeness (0-10)\n- Improvement over original (0-10)\n\nThen give:\n- Total score (0-50)\n- 2-4 sentence summary of what improved, what is still weak.\n\nFormat:\n\nScores:\n- Correctness: <0-10> - <short comment>\n- Rigor: <0-10> - <short comment>\n- Clarity: <0-10> - <short comment>\n- Completeness: <0-10> - <short comment>\n- Improvement: <0-10> - <short comment>\n\nTotal Score: <0-50>\n\nSummary:\n<2-4 sentences>\n"""
        evaluation = self.call(prompt)
        state["evaluation"] = evaluation
        log_turn(state, self.name, evaluation)
        return state
