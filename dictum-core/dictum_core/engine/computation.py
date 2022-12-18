from dataclasses import dataclass
from typing import List

from lark import Tree


@dataclass
class Column:
    """Represents a column selected from a relational calculation result.

    Arguments:
        name — column name in the resulting query
        expr — Lark expression for the column
        type — column data type
        display_info — info for displaying the column in the formatted table
            or an Altair chart
    """

    name: str
    expr: Tree

    @property
    def join_paths(self) -> List[str]:
        result = []
        for ref in self.expr.find_data("column"):
            path = ref.children[1:-1]
            if path:
                result.append(path)
        return result


@dataclass
class LiteralOrderItem:
    name: str
    ascending: bool
