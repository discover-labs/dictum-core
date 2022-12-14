import pytest
from lark import Token, Tree

from dictum_core.engine import Column, DisplayInfo
from dictum_core.format import Format
from dictum_core.model.scalar import LiteralTransform, ScalarTransform, transforms
from dictum_core.model.types import Type


@pytest.fixture(scope="function")
def col():
    return Column(
        name="test",
        type="int",
        expr=Tree("expr", [Token("INTEGER", "1")]),
    )


def test_transform():
    transform = ScalarTransform()
    assert transform.get_return_type("test") == "test"
    assert transform.get_format("test") == "test"
    with pytest.raises(NotImplementedError):
        transform.transform_expr(None)


def test_literal_transform(col: Column):
    class LiteralTest(LiteralTransform):
        expr = "@ > value"
        args = ["value"]
        return_type = "bool"

    transform = LiteralTest(0)
    assert transform._expr.children[0] == Tree(
        "gt", [Token("ARG", "@"), Tree("column", ["value"])]
    )
    assert transform.transform_expr(Token("TEST", None)) == Tree(
        "gt", [Token("TEST", None), Token("INTEGER", "0")]
    )
    assert transform(col) == Column(
        name="test",
        type="bool",
        expr=Tree(
            "expr",
            [
                Tree(
                    "gt",
                    [
                        Token("INTEGER", "1"),
                        Token("INTEGER", "0"),
                    ],
                )
            ],
        ),
    )


def test_booleans(col: Column):
    for key in ["eq", "ne", "gt", "ge", "lt", "le"]:
        transform = transforms[key](0)
        result = transform(col)
        assert result.type.name == "bool"
        assert result.expr.children[0].data == key
        assert result.expr.children[0].children == [
            Token("INTEGER", "1"),
            Token("INTEGER", "0"),
        ]


def test_nulls(col: Column):
    result = transforms["isnull"]()(col)
    assert result.type.name == "bool"
    assert result.expr == Tree("expr", [Tree("isnull", [Token("INTEGER", "1")])])

    result = transforms["isnotnull"]()(col)
    assert result.type.name == "bool"
    assert result.expr == Tree(
        "expr", [Tree("NOT", [Tree("isnull", [Token("INTEGER", "1")])])]
    )


def test_inrange(col: Column):
    result = transforms["inrange"](-1, 1)(col)
    assert result.type.name == "bool"
    assert result.expr == Tree(
        "expr",
        [
            Tree(
                "AND",
                [
                    Tree("ge", [Token("INTEGER", "1"), Token("INTEGER", -1)]),
                    Tree("le", [Token("INTEGER", "1"), Token("INTEGER", "1")]),
                ],
            )
        ],
    )


def test_in(col: Column):
    result = transforms["isin"](0, 1, 2)(col)
    assert result.type.name == "bool"
    assert result.expr.children[0] == Tree(
        "IN",
        [
            col.expr.children[0],
            Token("INTEGER", "0"),
            Token("INTEGER", "1"),
            Token("INTEGER", "2"),
        ],
    )


def test_datepart(col: Column):
    result = transforms["datepart"]("month")(col)
    assert result.type.name == "int"
    assert result.expr.children[0] == Tree(
        "call", ["datepart", "month", col.expr.children[0]]
    )

    result = transforms["month"]()(col)
    assert result.type.name == "int"
    assert result.expr.children[0] == Tree(
        "call", ["datepart", "month", col.expr.children[0]]
    )


def test_datetrunc(col: Column):
    col.display_info = DisplayInfo(
        display_name="xxx",
        column_name="xxx",
        format=Format(locale="en", type=Type(name="datetime")),
        kind="dimension",
        altair_time_unit="month",
    )
    datetrunc = transforms["datetrunc"]("month")
    result = datetrunc(col)
    assert result.type.name == "datetime"
    assert result.type.grain == "month"
    assert result.expr.children[0] == Tree(
        "call", ["datetrunc", "month", col.expr.children[0]]
    )
