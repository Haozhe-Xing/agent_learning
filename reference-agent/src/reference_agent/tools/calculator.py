from __future__ import annotations

import ast
import operator
import math
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
    ast.UAdd: operator.pos,
}


def _bounded(value):
    if type(value) not in (int, float) or abs(value) > 1e100 or not math.isfinite(value):
        raise ValueError("numeric budget exceeded")
    return value


def _eval_node(node) -> float:
    if isinstance(node, ast.Constant):
        if type(node.value) in (int, float):
            return _bounded(node.value)
        raise ValueError("only numeric constants are allowed")
    if isinstance(node, ast.BinOp):
        if type(node.op) not in _SAFE_OPS:
            raise ValueError("unsupported operator")
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("exponent budget exceeded")
        return _bounded(_SAFE_OPS[type(node.op)](left, right))
    if isinstance(node, ast.UnaryOp):
        if type(node.op) not in (ast.USub, ast.UAdd):
            raise ValueError("unsupported unary operator")
        return _bounded(_SAFE_OPS[type(node.op)](_eval_node(node.operand)))
    raise ValueError("unsupported expression")


def safe_eval(expression: str) -> float:
    """Evaluate a restricted arithmetic expression using ast (no eval())."""
    if not isinstance(expression, str) or len(expression) > 256:
        raise ValueError("expression length budget exceeded")
    tree = ast.parse(expression, mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 64:
        raise ValueError("expression complexity budget exceeded")
    return _eval_node(tree.body)


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
