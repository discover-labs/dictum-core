from lark import Lark, Token, Transformer, Tree

from dictum_core import grammars

grammars = grammars.__file__

ql = Lark.open("ql.lark", rel_to=grammars, start="query")


def filter_tree(name: str, dimension: Tree, value):
    return Tree("filter", [dimension, Tree("call", [name, value])])


def append_scalar(name: str):
    def method(self, children: list):
        term, *args = children
        term.children = [*term.children, Tree("scalar_transform", [name, *args])]
        return term

    return method


class Preprocessor(Transformer):
    transform_aliases = {
        "not": "invert",
    }

    def identifier(self, children: list):
        return children[0]

    def IDENTIFIER(self, token: Token):
        return token.value

    def QUOTED_IDENTIFIER(self, token: Token):
        return token.value[1:-1]  # unquote

    def STRING(self, value: str):
        return value[1:-1]  # unquote

    def INTEGER(self, value: str):
        return int(value)

    def UINTEGER(self, value: str):
        return int(value)

    def FLOAT(self, value: str):
        return float(value)

    def scalar_transform(self, children: list):
        id, *args = children
        id = self.transform_aliases.get(id, id)
        return Tree("scalar_transform", [id, *args])

    gt = append_scalar("gt")
    ge = append_scalar("ge")
    lt = append_scalar("lt")
    le = append_scalar("le")
    eq = append_scalar("eq")
    ne = append_scalar("ne")
    isnull = append_scalar("isnull")
    isnotnull = append_scalar("isnotnull")
    isin = append_scalar("isin")


pre = Preprocessor()


def parse_ql(query: str):
    return pre.transform(ql.parse(query))


dimension_expr = Lark.open("ql.lark", rel_to=grammars, start="dimension_expr")


def parse_dimension_expr(expr: str):
    """
    A separate function to parse string transform definitions during interactive use
    """
    return pre.transform(dimension_expr.parse(expr))


dimension_request = Lark.open("ql.lark", rel_to=grammars, start="dimension_request")


def parse_dimension_request(expr: str):
    return pre.transform(dimension_request.parse(expr))


metric_expr = Lark.open("ql.lark", rel_to=grammars, start="metric_expr")


def parse_metric_expr(expr: str):
    return pre.transform(metric_expr.parse(expr))


metric_request = Lark.open("ql.lark", rel_to=grammars, start="metric_request")


def parse_metric_request(expr: str):
    return pre.transform(metric_request.parse(expr))
