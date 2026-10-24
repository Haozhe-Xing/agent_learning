"""Small, bounded arithmetic interpreter for untrusted calculator inputs."""
import ast
import math
import operator

BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
          ast.Div: operator.truediv, ast.Mod: operator.mod, ast.Pow: operator.pow}
UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
FUNCTIONS = {name: getattr(math, name) for name in ('sqrt', 'sin', 'cos', 'tan', 'log', 'log10', 'exp', 'ceil', 'floor')}
FUNCTIONS.update(abs=abs, round=round)
CONSTANTS = {'pi': math.pi, 'e': math.e}


def safe_math(expression: str) -> float:
    if not isinstance(expression, str) or len(expression) > 256:
        raise ValueError('expression must be a string of at most 256 characters')
    tree = ast.parse(expression, mode='eval')
    if sum(1 for _ in ast.walk(tree)) > 64:
        raise ValueError('expression is too complex')

    def bounded(value):
        if type(value) not in (int, float) or not math.isfinite(value) or abs(value) > 1e100:
            raise ValueError('result is outside the numeric budget')
        return value

    def visit(node):
        if isinstance(node, ast.Constant):
            return bounded(node.value)
        if isinstance(node, ast.Name) and node.id in CONSTANTS:
            return CONSTANTS[node.id]
        if isinstance(node, ast.UnaryOp) and type(node.op) in UNARY:
            return bounded(UNARY[type(node.op)](visit(node.operand)))
        if isinstance(node, ast.BinOp) and type(node.op) in BINARY:
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError('exponent exceeds budget')
            return bounded(BINARY[type(node.op)](left, right))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FUNCTIONS:
            if node.keywords or not 1 <= len(node.args) <= 2:
                raise ValueError('unsupported function arguments')
            args = [visit(arg) for arg in node.args]
            if node.func.id == 'round' and len(args) == 2 and (type(args[1]) is not int or abs(args[1]) > 100):
                raise ValueError('round precision exceeds budget')
            return bounded(FUNCTIONS[node.func.id](*args))
        raise ValueError('unsupported expression')

    return visit(tree.body)
