from pathlib import Path
from textwrap import dedent

import pytest
import yaml
from lark import Tree, UnexpectedCharacters

from dictum_core.exceptions import MixedExpressionError
from dictum_core.model.expr.introspection import get_expr_kind, get_expr_total_function
from dictum_core.model.expr.parser import parse_expr

with (Path(__file__).parent / "test_expr_parse.yml").open("r") as fp:
    parse_cases = yaml.safe_load(fp)


@pytest.mark.parametrize("case", parse_cases)
def test_parse(case: dict):
    val = case["val"]
    expected = case["expected"]
    missing = case.get("missing")
    assert parse_expr(val, missing=missing).pretty().strip() == dedent(expected.strip())


def test_measure():
    """Referencing other calculation"""
    assert parse_expr("$calc").children[0] == Tree("measure", ["calc"])
    with pytest.raises(UnexpectedCharacters):
        parse_expr("$table.calc")


def test_dimension():
    """Referencing other dimensions."""
    assert parse_expr(":dim").children[0] == Tree("dimension", ["dim"])
    with pytest.raises(UnexpectedCharacters):
        parse_expr(":table.dim")


def test_expr_kind():
    assert get_expr_kind(parse_expr("1")) == "scalar"
    assert get_expr_kind(parse_expr("'a'")) == "scalar"

    assert get_expr_kind(parse_expr("col")) == "column"
    assert get_expr_kind(parse_expr("col + 1")) == "column"
    assert get_expr_kind(parse_expr("col * 2")) == "column"

    assert get_expr_kind(parse_expr("sum(col)")) == "aggregate"
    assert get_expr_kind(parse_expr("count(col) - 1")) == "aggregate"
    assert get_expr_kind(parse_expr("min(col) * 100")) == "aggregate"
    assert get_expr_kind(parse_expr("max(col) / 100")) == "aggregate"
    assert get_expr_kind(parse_expr("floor(sum(col - 1) + 1) / 10")) == "aggregate"
    assert get_expr_kind(parse_expr("case when max(col) > 1 then 1 else 0 end"))

    assert get_expr_kind(parse_expr("sum(col) in (1, 2, 3)")) == "aggregate"
    assert get_expr_kind(parse_expr("col in (1, 2, 3)")) == "column"
    assert get_expr_kind(parse_expr("1 in (1, 2, 3)")) == "scalar"


def test_expr_kind_invalid():
    with pytest.raises(MixedExpressionError):
        get_expr_kind(parse_expr("sum(col) + col"))
    with pytest.raises(MixedExpressionError):
        get_expr_kind(parse_expr("case when col = 1 then sum(col) end"))
    with pytest.raises(MixedExpressionError):
        get_expr_kind(parse_expr("case when col = 1 then sum(col) end"))
    with pytest.raises(MixedExpressionError):
        get_expr_kind(parse_expr("col in (sum(col), 1, 2)"))


def test_expr_total_function():
    assert get_expr_total_function(parse_expr("sum(col)")) == "sum"
    assert get_expr_total_function(parse_expr("sum(col + 1)")) == "sum"
    assert get_expr_total_function(parse_expr("count(col)")) == "sum"
    assert get_expr_total_function(parse_expr("max(col)")) == "max"
    assert get_expr_total_function(parse_expr("min(col)")) == "min"
    assert get_expr_total_function(parse_expr("avg(col)")) is None
    assert get_expr_total_function(parse_expr("countd(col)")) is None
    assert get_expr_total_function(parse_expr("sum(col) + 1")) is None
