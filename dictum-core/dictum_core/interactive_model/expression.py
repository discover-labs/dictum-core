from functools import wraps
from typing import Optional

from lark import Token, Tree

from dictum_core.interactive_model.resolve import resolve_interactive_expression
from dictum_core.model.expr import parse_expr
from dictum_core.model.expr.introspection import get_expr_kind
from dictum_core.schema.model.types import resolve_type


def handle_literals(fn):
    @wraps(fn)
    def wrapped(self, other):
        if isinstance(other, AbstractInteractiveExpression):
            return fn(self, other)

        token_kind = None
        if isinstance(other, int):
            token_kind = "INTEGER"
        if isinstance(other, str):
            token_kind = "STRING"
        if isinstance(other, float):
            token_kind = "FLOAT"
        if isinstance(other, bool):
            token_kind = str(other).upper()

        if token_kind is None:
            raise ValueError(f"Unsupported expression type: {type(other)}")

        return fn(self, InteractiveExpression(children=[Token(token_kind, str(other))]))

    return wrapped


def build_op_method(op: str, right: bool = False):
    @handle_literals
    def method(self, other: "InteractiveExpression") -> "InteractiveExpression":
        _op = op_map.get(op, op)
        if right:
            return InteractiveExpression([Tree(_op, [other, self])])
        return InteractiveExpression([Tree(_op, [self, other])])

    return method


op_list = [
    "add",
    "sub",
    "mul",
    "truediv",
    "floordiv",
    "eq",
    "ne",
    "gt",
    "ge",
    "lt",
    "le",
]
op_map = {"truediv": "div", "floordiv": "fdiv"}


class InteractiveExpressionMeta(type):
    def __new__(cls, name, bases, attrs):
        for op in op_list:
            attrs[f"__{op}__"] = build_op_method(op)
            attrs[f"__r{op}__"] = build_op_method(op, right=True)
        return super().__new__(cls, name, bases, attrs)


class InteractiveExpressionFunction:
    def __init__(self, parent: "InteractiveExpression", call: str):
        self.parent = parent
        self.call = call

    def __call__(self, *args):
        return InteractiveExpression([Tree("call", [self.call, self.parent])])


class AbstractInteractiveExpression(Tree, metaclass=InteractiveExpressionMeta):
    def __init__(self, data: str, children: Optional[list] = None) -> None:
        if children is None:
            children = []
        self.type_ = None
        # None is expression name
        # will be added in __set_name__
        # has to be the first child to be available in the resolver
        super().__init__(data, children=[None, *children])

    @classmethod
    def get_kind(cls, expression: str) -> str:
        parsed = parse_expr(expression)
        kind = get_expr_kind(parsed)
        if kind in {"column", "scalar"}:
            return "dimension"
        if kind == "aggregate" and len(list(parsed.find_data("column"))) > 0:
            return "measure"
        return "metric"

    def type(self, type_: str):
        self.type_ = resolve_type(type_)
        return self

    def __getattr__(self, attr: str):
        return InteractiveExpressionFunction(self, attr)

    def __get__(self, obj, objtype=None):
        return self

    def __ror__(self, table):
        """An operator for defining a table for table calculations.

        invoices = Table()
        invoice_items = Table()
        invoice_items.InvoiceId >> invoices.InvoiceId

        unique_customers = invoice_items | invoices.CustomerId.countd()
        """
        for ref in self.find_data("interactive_column"):
            name, *tables, col = ref.children
            ref.children = [name, table, *tables, col]
        return self

    def __set_name__(self, owner, name: str):
        """When the interactive model is built, we need to create all the calculations
        in the actual model.
        - Figure out calculation type (measure/metric/dimension)
        - Create calculation on the actual model
        """
        self.children[0] = name  # set name for the resolvers of dependant expressions

    def get_str_expr(self) -> str:
        return resolve_interactive_expression(
            AbstractInteractiveExpression(self.data, self.children[1:])
        )


class InteractiveExpression(AbstractInteractiveExpression):
    def __init__(self, children: Optional[list] = None):
        super().__init__("expr", children)


class InteractiveColumn(AbstractInteractiveExpression):
    def __init__(self, table, column: str):
        super().__init__("interactive_column", [table, column])

    def __rshift__(self, other: "InteractiveColumn"):
        self.children[1]._InteractiveTable__foreign_keys.append((self, other))
