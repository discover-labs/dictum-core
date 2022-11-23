from copy import deepcopy
from typing import List

from lark import Tree


def merged_expr(name: str):
    return Tree("expr", [Tree("column", [None, name])])


def prefixed_expr(expr: Tree, prefix: List[str]) -> Tree:
    """Prefix the expression with the given join path."""
    result = deepcopy(expr)
    for ref in result.find_data("column"):
        # skip first child: host table's name
        _, *path, field = ref.children
        ref.children = [*prefix, *path, field]
    return result
