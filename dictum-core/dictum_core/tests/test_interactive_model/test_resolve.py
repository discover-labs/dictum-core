import pytest
from dictum_core.interactive_model.resolve import resolve_interactive_expression
from dictum_core.interactive_model.table import InteractiveTable
from dictum_core.model.expr.parser import parse_expr
from lark import Tree


@pytest.fixture(scope="module")
def tbl():
    return InteractiveTable(id="tbl")


def resolve_parse(expression):
    return parse_expr(resolve_interactive_expression(expression)).children[0]


def test_resolve_column(tbl):
    assert resolve_parse(tbl.col) == Tree("column", ["col"])


def test_resolve_add(tbl):
    assert resolve_parse(tbl.col + tbl.col2) == Tree(
        "add", [Tree("column", ["col"]), Tree("column", ["col2"])]
    )


def test_resolve_sub(tbl):
    assert resolve_parse(tbl.col - tbl.col2) == Tree(
        "sub", [Tree("column", ["col"]), Tree("column", ["col2"])]
    )


def test_resolve_mul(tbl):
    assert resolve_parse(tbl.col * tbl.col2) == Tree(
        "mul", [Tree("column", ["col"]), Tree("column", ["col2"])]
    )


def test_resolve_div(tbl):
    assert resolve_parse(tbl.col / tbl.col2) == Tree(
        "div", [Tree("column", ["col"]), Tree("column", ["col2"])]
    )


def test_resolve_fdiv(tbl):
    assert resolve_parse(tbl.col // tbl.col2) == Tree(
        "call",
        ["floor", Tree("div", [Tree("column", ["col"]), Tree("column", ["col2"])])],
    )


def test_resolve_call(tbl):
    assert resolve_parse((tbl.a * tbl.b).sum()) == Tree(
        "call", ["sum", Tree("mul", [Tree("column", ["a"]), Tree("column", ["b"])])]
    )


def test_resolve_metric(Model):
    assert Model._model.metrics["c"].str_expr == "($a / $b)"
