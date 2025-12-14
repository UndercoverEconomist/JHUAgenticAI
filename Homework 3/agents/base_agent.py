from typing import TypedDict, Optional, Dict, Any, List, Callable

default_model = None  # Will be set in main.py

from utils import color_prompt_blue, color_llm_green


class BaseAgent:
    def __init__(
        self,
        name: str,
        model: Callable[[str, float], str] = None,
        system_prompt: str = "",
        temperature: float = 0.0,
    ):
        self.name = name
        self.model = model if model is not None else default_model
        self.system_prompt = system_prompt.strip()
        self.temperature = temperature
        self.memory: List[Dict[str, str]] = []  # agent-private memory

    def _build_prompt(self, message: str) -> str:
        if self.system_prompt:
            return self.system_prompt + "\n\n" + message
        return message

    def call(self, message: str) -> str:
        prompt = self._build_prompt(message)
        # Print the prompt for human inspection (blue) but send plain prompt to model
        try:
            print(color_prompt_blue(f"\n[{self.name} PROMPT]\n{prompt}\n"))
        except Exception:
            # Fallback: printing without color if something goes wrong
            print(f"\n[{self.name} PROMPT]\n{prompt}\n")

        output = self.model(prompt, temperature=self.temperature)
        if output is None:
            output = ""
        output = str(output)

        # Print the LLM output in green for terminal visualization
        try:
            print(color_llm_green(f"[{self.name} RESPONSE]\n{output}\n"))
        except Exception:
            print(f"[{self.name} RESPONSE]\n{output}\n")

        # Keep memory/plain records uncolored
        self.memory.append({"role": "input", "content": message})
        self.memory.append({"role": "output", "content": output})
        return output

    def act(self, state):
        raise NotImplementedError
