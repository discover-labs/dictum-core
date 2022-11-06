import pytest
from lark import Token, Tree

from dictum_core.interactive_model.expression import (
    InteractiveExpression,
    handle_literals,
    op_list,
    op_map,
)


@pytest.fixture(scope="module")
def one():
    return Token("INTEGER", "1")


@pytest.fixture(scope="module")
def fn():
    @handle_literals
    def _fn(self, other):
        return other

    return _fn


def test_empty_expression(one: Token):
    assert InteractiveExpression([one]).data == "expr"
    assert InteractiveExpression([one]).children == [None, one]


def test_handle_literals_expression(fn, one):
    assert fn(None, InteractiveExpression([one])).children == [None, one]


def test_handle_literals_integer(fn, one):
    assert fn(None, 1).children[1] == one


def test_handle_literals_float(fn):
    assert fn(None, 3.14).children[1] == Token("FLOAT", "3.14")


def test_handle_literals_string(fn):
    assert fn(None, "test").children[1] == Token("STRING", "test")


def test_handle_literals_bool(fn):
    assert fn(None, True).children[1] == Token("TRUE", "True")
    assert fn(None, False).children[1] == Token("FALSE", "False")


@pytest.mark.parametrize("op", op_list)
def test_op_literal(one, op):
    _op = op_map.get(op, op)
    assert (getattr(InteractiveExpression([one]), f"__{op}__")(2)).children[1] == Tree(
        _op, [InteractiveExpression([one]), Token("INTEGER", "2")]
    )


@pytest.mark.parametrize("op", op_list)
def test_rop_literal(one, op):
    _op = op_map.get(op, op)
    assert (getattr(InteractiveExpression([one]), f"__r{op}__")(2)).children[1] == Tree(
        _op, [Token("INTEGER", "2"), InteractiveExpression([one])]
    )
