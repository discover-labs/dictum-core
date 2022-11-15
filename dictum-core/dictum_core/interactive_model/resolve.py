from lark import Transformer


class ExpressionResolver(Transformer):
    """Resolves instances of delayed calculations (table columns) with proper table
    and column names
    """

    def interactive_column(self, children: list):
        _, *tables, column = children
        if len(tables) == 1:
            return column

        source, target = tables
        source_id = source._InteractiveTable__id
        target_id = target._InteractiveTable__id
        path = source._InteractiveTable__table.allowed_join_paths.get(
            target._InteractiveTable__table
        )
        if path is None:
            raise ValueError(
                f"There is no join path from table {source_id} to {target_id}"
            )
        return ".".join([*path, column])

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
