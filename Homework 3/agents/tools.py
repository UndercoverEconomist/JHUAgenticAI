"""
Small tools module for agents.
Currently provides `calculate_expression(expr: str) -> str` which safely evaluates
simple arithmetic expressions and returns the result as a string.

This is intentionally minimal and can be replaced by a LangChain tool or
external calculator later.
"""
import ast
import operator as op
from typing import Any

# Supported operators
_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
}


def _eval(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Num):  # type: ignore[attr-defined]
        return node.n
    if isinstance(node, ast.Constant):  # Python 3.8+
        return node.value
    if isinstance(node, ast.BinOp):
        left = _eval(node.left)
        right = _eval(node.right)
        op_type = type(node.op)
        if op_type in _OPERATORS:
            return _OPERATORS[op_type](left, right)
        raise ValueError(f"Unsupported binary operator: {op_type}")
    if isinstance(node, ast.UnaryOp):
        operand = _eval(node.operand)
        op_type = type(node.op)
        if op_type in _OPERATORS:
            return _OPERATORS[op_type](operand)
        raise ValueError(f"Unsupported unary operator: {op_type}")
    raise ValueError(f"Unsupported expression: {type(node)}")


def calculate_expression(expr: str) -> str:
    """Safely evaluate a simple arithmetic expression and return result as string.

    If the expression cannot be parsed or contains disallowed nodes, an error
    string is returned instead of raising.
    """
    try:
        # Parse expression only (no statements)
        parsed = ast.parse(expr, mode="eval")
        result = _eval(parsed)
        return str(result)
    except Exception as e:
        return f"[tool-error] {e}"
