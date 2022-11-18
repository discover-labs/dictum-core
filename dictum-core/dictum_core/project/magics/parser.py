from lark import Lark, Token, Transformer, Tree

from dictum_core import grammars

grammars = grammars.__file__

related_parser = Lark.open("magics.lark", rel_to=grammars, start="related_standalone")
calculation_parser = Lark.open(
    "magics.lark", rel_to=grammars, start="calculation", propagate_positions=True
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


preprocessor = Preprocessor()


def parse_shorthand_table(definition: str) -> Tree:
    return preprocessor.transform(table_parser.parse(definition))


def parse_shorthand_calculation(definition: str) -> Tree:
    return preprocessor.transform(calculation_parser.parse(definition))


def parse_shorthand_related(definition: str) -> Tree:
    return preprocessor.transform(related_parser.parse(definition))


def parse_shorthand_format(definition: str) -> Tree:
    return preprocessor.transform(format_parser.parse(definition))
