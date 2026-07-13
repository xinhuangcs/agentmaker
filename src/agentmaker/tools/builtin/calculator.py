"""agentmaker.tools.builtin.calculator: calculator tool.

Parses the expression into an abstract syntax tree with ast, then evaluates only
whitelisted operators and functions, eliminating at the root the risk of eval
executing arbitrary code.
"""

import ast
import math
import operator
from typing import List

from ...prompts import DEFAULT_PROMPTS
from ..base import Tool, ToolParameter
from ..response import ToolResponse

# Resource limits: prevent a model-generated expression from exhausting memory/CPU
# (e.g. astronomically large powers like 9**9**9).
_MAX_EXPR_LEN = 1000        # Maximum number of characters in the expression.
_MAX_AST_NODES = 200        # AST node cap (guards against deep nesting / overly long expressions causing recursion and slowdowns).
_MAX_RESULT_DIGITS = 1000   # Cap on the number of decimal digits in a power-operation result.


class _CalcError(Exception):
    """Internal calculator evaluation error: carries only a prompt key plus render variables, which run() turns into readable text in the current language.

    _eval / _safe_pow are module-level pure functions that do not hold prompts, so they do
    not assemble user-facing text here; the wording choice is left to run() (which holds
    prompts), so the calculator's internal errors can also be fully localized (i18n closure).
    """

    def __init__(self, key: str, **kw):
        self.key = key
        self.kw = kw
        super().__init__(key)


def _safe_pow(base, exp):
    """Power operation with a scale gate: estimate the digit count of base ** exp first and reject if it exceeds _MAX_RESULT_DIGITS, avoiding astronomically large computations.

    Only int ** positive int is estimated (Python big integers are unbounded and are the real
    exhaustion point); floats / negative or zero exponents are left to the underlying operator,
    and any overflow raises OverflowError, caught by run() as a readable error. When the exponent
    is enormous (even base=2 would exceed the digit cap), reject outright first, which also avoids
    an int->float overflow in the `exp x log10` step (otherwise it would raise an uninformative
    OverflowError).
    """
    if isinstance(base, int) and isinstance(exp, int) and exp > 0 and base not in (0, 1, -1):
        # Result digit count = floor(exp x log10(|base|)) + 1; reject if over the cap (floor+1 is exact, avoiding off-by-one).
        if exp > 4 * _MAX_RESULT_DIGITS or math.floor(exp * math.log10(abs(base))) + 1 > _MAX_RESULT_DIGITS:
            raise _CalcError("tool.msg.calc.too_large")
    return operator.pow(base, exp)


# Whitelist: binary operators.
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: _safe_pow,
}
# Whitelist: unary operators (sign).
_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
# Whitelist: functions and constants.
_FUNCS = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "log": math.log,
    "sin": math.sin,
    "cos": math.cos,
}
_CONSTS = {
    "pi": math.pi,
    "e": math.e,
}


def _eval(node):
    """Recursively evaluate an ast node, accepting only whitelisted operators / functions / constants and erroring otherwise."""
    if isinstance(node, ast.Constant):           # Numeric literal.
        if isinstance(node.value, (int, float)):
            return node.value
        raise _CalcError("tool.msg.calc.bad_constant", value=repr(node.value))
    if isinstance(node, ast.BinOp):              # Binary operation a + b.
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise _CalcError("tool.msg.calc.bad_operator")
        return op(_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp):            # Unary operation -a.
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise _CalcError("tool.msg.calc.bad_unary")
        return op(_eval(node.operand))
    if isinstance(node, ast.Call):               # Function call sqrt(2) / round(x, 2).
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS:
            raise _CalcError("tool.msg.calc.bad_function")
        if node.keywords:                        # Keyword arguments (e.g. round(x, ndigits=2)) would be silently ignored since only node.args is taken below; reject explicitly rather than return a wrong result.
            raise _CalcError("tool.msg.calc.no_kwargs")
        return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
    if isinstance(node, ast.Name):               # Constant name pi / e.
        if node.id in _CONSTS:
            return _CONSTS[node.id]
        raise _CalcError("tool.msg.calc.bad_name", name=node.id)
    raise _CalcError("tool.msg.calc.unparseable")


class CalculatorTool(Tool):
    """Evaluate math expressions. Supports + - * / // % **, sign, plus sqrt/abs/round/log/sin/cos and pi/e."""

    def __init__(self, *, prompts=None):
        self.prompts = prompts or DEFAULT_PROMPTS        # Tool description / parameter text come from the registry (fully localizable).
        super().__init__("calculator", self.prompts.text("tool.desc.calculator"))

    def get_parameters(self) -> List[ToolParameter]:
        return [ToolParameter("expression", "string", self.prompts.text("tool.param.calculator.expression"))]

    def run(self, parameters: dict) -> ToolResponse:
        """Parse and evaluate expression; on success data carries the numeric value; return an error when empty / too long / too complex / invalid."""
        expression = (parameters.get("expression") or "").strip()
        if not expression:
            return ToolResponse.error(self.prompts.text("tool.msg.calc.empty"))
        if len(expression) > _MAX_EXPR_LEN:
            return ToolResponse.error(self.prompts.render("tool.msg.calc.too_long", max=_MAX_EXPR_LEN))
        try:
            tree = ast.parse(expression, mode="eval")
            if sum(1 for _ in ast.walk(tree)) > _MAX_AST_NODES:
                return ToolResponse.error(self.prompts.render("tool.msg.calc.too_complex", max=_MAX_AST_NODES))
            value = _eval(tree.body)
            text = str(value)          # Kept inside try: a huge integer whose str() raises ValueError is also caught as a readable error rather than propagating raw.
        except ZeroDivisionError:
            return ToolResponse.error(self.prompts.text("tool.msg.calc.div_zero"))
        except _CalcError as e:        # Internal evaluation error: carries a prompt key; render its detail in the current language, then wrap it in eval_failed.
            detail = self.prompts.render(e.key, **e.kw)
            return ToolResponse.error(self.prompts.render("tool.msg.calc.eval_failed", err=detail))
        except Exception as e:
            return ToolResponse.error(self.prompts.render("tool.msg.calc.eval_failed", err=e))
        return ToolResponse.ok(text, data=value)


