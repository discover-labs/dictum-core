from functools import wraps

from lark import Lark, Token, Transformer, Tree
from lark.exceptions import UnexpectedInput

from dictum_core import grammars
from dictum_core.exceptions import ShorthandSyntaxError

grammars = grammars.__file__

related_parser = Lark.open("magics.lark", rel_to=grammars, start="related")
metric_parser = Lark.open(
    "magics.lark", rel_to=grammars, start="metric", propagate_positions=True
)
dimension_parser = Lark.open(
    "magics.lark", rel_to=grammars, start="dimension", propagate_positions=True
)
table_parser = Lark.open(
    "magics.lark", rel_to=grammars, start="table_full", propagate_positions=True
)
format_parser = Lark.open("magics.lark", rel_to=grammars, start="format")


class Preprocessor(Transformer):
    def IDENTIFIER(self, token: Token):
        return token.value

    def identifier(self, children: list):
        return children[0]

    def key_value(self, children: list):
        return tuple(children)

    def key_values(self, children: list):
        return dict(children)

    def ql__IDENTIFIER(self, token: Token):
        return token.value.strip('"')

    def ql__QUOTED_IDENTIFIER(self, token: Token):
        return token.value.strip('"')

    def table_metric(self, children: list):
        return children[0]

    def table_dimension(self, children: list):
        return children[0]

    def table_related(self, children: list):
        return children[0]

    def filter(self, children: list):
        """Inline expr node directly into filter"""
        return Tree("filter", children[0].children, meta=children[0].meta)


preprocessor = Preprocessor()


def catch_syntax_errors(fn: callable):
    @wraps(fn)
    def wrapped(definition: str):
        try:
            return fn(definition)
        except UnexpectedInput as e:
            raise ShorthandSyntaxError(e, definition)

    return wrapped


@catch_syntax_errors
def parse_shorthand_table(definition: str) -> Tree:
    return preprocessor.transform(table_parser.parse(definition))


@catch_syntax_errors
def parse_shorthand_metric(definition: str) -> Tree:
    return preprocessor.transform(metric_parser.parse(definition))


@catch_syntax_errors
def parse_shorthand_dimension(definition: str) -> Tree:
    return preprocessor.transform(dimension_parser.parse(definition))


@catch_syntax_errors
def parse_shorthand_related(definition: str) -> Tree:
    return preprocessor.transform(related_parser.parse(definition))


@catch_syntax_errors
def parse_shorthand_format(definition: str) -> Tree:
    return preprocessor.transform(format_parser.parse(definition))
