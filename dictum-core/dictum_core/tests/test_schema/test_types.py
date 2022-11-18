import pytest

from dictum_core.model.types import Type, grains, resolve_type


def test_resolve_basic_types():
    assert resolve_type("bool") == Type(name="bool")
    assert resolve_type("str") == Type(name="str")
    assert resolve_type("int") == Type(name="int")
    assert resolve_type("float") == Type(name="float")


def test_resolve_basic_datetime_types():
    assert resolve_type("datetime") == Type(name="datetime", grain="second")
    assert resolve_type("date") == Type(name="datetime", grain="day")


def test_resolve_grains():
    for grain in grains:
        assert resolve_type(f"datetime:{grain}") == Type(name="datetime", grain=grain)
        assert resolve_type(f"date:{grain}") == Type(name="datetime", grain=grain)


def test_resolve_invalid_grain():
    with pytest.raises(ValueError, match=r"Unknown datetime grain: invalid"):
        resolve_type("datetime:invalid")


def test_resolve_grain_invalid_type():
    with pytest.raises(
        ValueError,
        match=(
            r"Time grains can be specified only for date and datetime types,"
            " got grain for int"
        ),
    ):
        resolve_type("int:day")


def test_resolve_unknown_type():
    with pytest.raises(ValueError, match=r"Unknown type: unknown"):
        resolve_type("unknown")


def test_type_grains():
    T = resolve_type("date:month")
    assert T.grains == ["year", "quarter", "month"]

    assert resolve_type("int").grains == []
