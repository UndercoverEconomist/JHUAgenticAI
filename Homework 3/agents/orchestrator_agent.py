from agents.base_agent import BaseAgent
from utils import log_turn

class OrchestratorAgent(BaseAgent):
    def decide_next(self, state) -> str:
        if "initial_answer" not in state or not state["initial_answer"]:
            return "generator"
        if "validator_report" not in state or not state["validator_report"]:
            return "validator"
        if "critic_report" not in state or not state["critic_report"]:
            return "critic"
        if "refined_answer" not in state or not state["refined_answer"]:
            return "refiner"
        if "evaluation" not in state or not state["evaluation"]:
            return "evaluator"
        return "end"

    def act(self, state):
        next_step = self.decide_next(state)
        if next_step == "generator":
            log_turn(state, "Orchestrator", "Starting solution with Generator.")
        elif next_step == "validator":
            log_turn(state, "Orchestrator", "Sending solution to Validator.")
        elif next_step == "critic":
            log_turn(state, "Orchestrator", "Forwarding Validator report to Critic.")
        elif next_step == "refiner":
            log_turn(state, "Orchestrator", "Sending corrections to Refiner.")
        elif next_step == "evaluator":
            log_turn(state, "Orchestrator", "Sending original vs refined to Evaluator.")
        elif next_step == "end":
            final_ans = state.get("refined_answer") or state.get("initial_answer", "")
            state["final_answer"] = final_ans
            log_turn(state, "Orchestrator", "Final answer ready.")
        return state
