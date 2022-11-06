from lark import Transformer


class ExpressionResolver(Transformer):
    """Resolves instances of delayed calculations (table columns) with proper table
    and column names
    """

    def interactive_column(self, children: list):
        *_, column = children
        return column

    def expr(self, children: list):
        name, expression = children
        if name is not None:
            from dictum_core.interactive_model.expression import InteractiveExpression

            kind = InteractiveExpression.get_kind(expression)
            if kind == "dimension":
                return f":{name}"
            return f"${name}"
        return expression

    def add(self, children):
        a, b = children
        return f"({a} + {b})"

    def sub(self, children):
        a, b = children
        return f"({a} - {b})"

    def mul(self, children):
        a, b = children
        return f"({a} * {b})"

    def div(self, children):
        a, b = children
        return f"({a} / {b})"

    def fdiv(self, children):
        a, b = children
        return f"({a} // {b})"

    def call(self, children):
        fn, *args = children
        _args = ", ".join(args)
        return f"{fn}({_args})"


resolver = ExpressionResolver()


def resolve_interactive_expression(interactive_expression) -> str:
    return resolver.transform(interactive_expression)
