from agents.base_agent import BaseAgent
from utils import log_turn

CRITIC_SYSTEM_PROMPT = """
You are a Math Critic.

You:
- read validator feedback
- turn it into concise, actionable corrections
- specify which steps must be changed
- identify missing justifications
- are precise and constructive.
""".strip()

class CriticAgent(BaseAgent):
    def __init__(self, model=None, temperature=0.1):
        super().__init__(
            name="Critic",
            model=model,
            system_prompt=CRITIC_SYSTEM_PROMPT,
            temperature=temperature,
        )

    def act(self, state):
        report = state["validator_report"]
        llm_feedback = report.get("llm_critique", "")
        prompt = f"""You are given validator feedback on a math solution.\n\nValidator feedback:\n{llm_feedback}\n\nWrite a clear set of corrections in bullet points, aimed at the solver.\nFocus on what to change, refine, or justify, step by step.\n"""
        critique = self.call(prompt)
        state["critic_report"] = critique
        log_turn(state, self.name, critique)
        return state
