from __future__ import annotations

import ast
import operator
from typing import NoReturn

from .base import Tool

_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _eval_node(node) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("only numeric constants are allowed")
    if isinstance(node, ast.BinOp):
        return _SAFE_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        return _SAFE_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("unsupported expression")


def safe_eval(expression: str) -> float:
    """Evaluate a restricted arithmetic expression using ast (no eval())."""
    tree = ast.parse(expression, mode="eval").body
    return _eval_node(tree)


class CalculatorTool(Tool):
    name = "calculator"
    description = "Evaluate a basic arithmetic expression, e.g. '12*8'."
    parameters = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Arithmetic expression using + - * / % **",
            }
        },
        "required": ["expression"],
    }

    def run(self, expression: str) -> str:
        return str(safe_eval(expression))
