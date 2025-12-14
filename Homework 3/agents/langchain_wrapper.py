"""
LangChain wrapper for the local Ollama/Qwen model.
Provides:
- `OllamaLLM`: a small LangChain-compatible LLM that delegates to `main.qwen_32b_model`.
- `make_ollama_tool`: convenience to create a LangChain `Tool` that calls the model.

This keeps dependencies optional: LangChain integration will work if `langchain` is installed.
"""
from typing import Optional, Dict, Any

try:
    from langchain.llms.base import LLM
    from langchain.tools import Tool
except Exception:
    # LangChain not installed; provide lightweight fallbacks
    LLM = object
    Tool = None

from main import qwen_32b_model


class OllamaLLM(LLM):
    """LangChain LLM wrapper that calls the qwen_32b_model function in `main`.

    Usage:
        llm = OllamaLLM(model_name='qwen2:32b', temperature=0.2)
        out = llm.predict('Hello')  # uses LLM API
    """

    def __init__(self, model_name: Optional[str] = None, temperature: float = 0.0):
        self.model_name = model_name
        self.temperature = temperature

    @property
    def _llm_type(self) -> str:
        return "ollama-qwen"

    def _call(self, prompt: str, stop: Optional[list] = None) -> str:  # type: ignore[override]
        # Delegate to main.qwen_32b_model which handles python client or CLI
        return qwen_32b_model(prompt, temperature=self.temperature)

    def _identifying_params(self) -> Dict[str, Any]:
        return {"model": self.model_name, "temperature": self.temperature}


def make_ollama_tool(name: str = "ollama_qwen", description: str = "Run Qwen via Ollama"):
    """Return a LangChain Tool that calls the qwen model.

    The returned tool will have a `.run(input_str)` method for synchronous calls.
    """
    def _run(input_str: str) -> str:
        return qwen_32b_model(input_str)

    if Tool is None:
        # LangChain not installed: return a simple callable
        return _run
    return Tool(name=name, func=_run, description=description)
