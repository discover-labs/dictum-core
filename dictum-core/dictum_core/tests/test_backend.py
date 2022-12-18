from typing import Optional

import pytest
from dateutil.parser import isoparse
from lark import Tree
from pandas import DataFrame, Timestamp, testing

from dictum_core.backends.base import Backend
from dictum_core.engine.computation import Column, LiteralOrderItem
from dictum_core.model.expr import parse_expr

compiler_scalar_expr_test_cases = [
    ("1.2", 1.2),
    ("1", 1),
    ("42", 42),
    ("-1", -1),
    ("-3.14", -3.14),
    ("'abc'", "abc"),
    ("''", ""),
    ("true", True),
    ("false", False),
    ("@2022-01-01", Timestamp("2022-01-01")),
    ("@2022-01-01T12:34:56", Timestamp("2022-01-01 12:34:56")),
    ("media_types.MediaTypeId", 1),
    ("2 ** 10", 1024),
    ("-media_types.MediaTypeId", -5),
    ("1 / 2", 0.5),
    ("9 // 2", 4),
    ("9 * 9", 81),
    ("-9 * 9", -81),
    ("9 % 2", 1),
    ("66 + 11.4", 77.4),
    ("77.4 - 11.4", 66),
    ("media_types.MediaTypeId is null", False),
    ("0 is null", False),
    ("1 > 2", False),
    ("2 > 1", True),
    ("2 > 2", False),
    ("1 >= 2", False),
    ("2 >= 1", True),
    ("2 >= 2", True),
    ("1 < 2", True),
    ("2 < 1", False),
    ("2 < 2", False),
    ("1 <= 2", True),
    ("2 <= 1", False),
    ("2 <= 2", True),
    ("2 = 2", True),
    ("1 = 2", False),
    ("1 <> 2", True),
    ("2 <> 2", False),
    ("2 in (2, 3, 4)", True),
    ("1 in (2, 3, 4)", False),
    ("media_types.MediaTypeId in (1, 2, 3, 4, 5)", True),
    ("media_types.MediaTypeId in (50)", False),
    ("not true", False),
    ("not false", True),
    ("not 1 = 2", True),
    ("not 2 = 2", False),
    ("2 = 2 and 1 = 1", True),
    ("2 = 2 and 1 = 2", False),
    ("2 = 2 or 1 = 2", True),
    ("2 = 1 or 2 = 2", True),
    ("2 = 1 or 2 = 3", False),
    ("case when 2 = 2 then 0 else 1 end", 0),
    ("case when 2 = 1 then 1 when 3 = 1 then 1 else 0 end", 0),
    ("case when 2 = 1 then 0 when 2 = 2 then 0 end", 0),
    ("case when 2 = 1 then 1 end", None),
    ("abs(-42) = abs(42)", True),
    ("abs(-42)", 42),
    ("abs(42)", 42),
    ("floor(3.14)", 3),
    ("floor(3.99)", 3),
    ("floor(3)", 3),
    ("floor(-3.14)", -4),
    ("floor(0)", 0),
    ("ceil(0)", 0),
    ("ceil(3)", 3),
    ("ceil(3.14)", 4),
    ("ceil(3.99)", 4),
    ("ceil(-3.14)", -3),
    ("coalesce(null, 1)", 1),
    ("coalesce(null, null, 1, 2)", 1),
    ("coalesce(1, null, 2)", 1),
    ("coalesce(null)", None),
    ("coalesce(null, null, null, null)", None),
    ("coalesce(5, media_types.MediaTypeId)", 5),
    ("coalesce(media_types.MediaTypeId, 5)", 1),
    ("tointeger(3.14)", 3),
    ("tointeger(3)", 3),
    ("tointeger(null)", None),
    ("tointeger(-3.14)", -3),
    ("tointeger('3')", 3),
    ("tointeger('-3')", -3),
    ("tofloat('3.14')", 3.14),
    ("tofloat('-3.14')", -3.14),
    ("tofloat(null)", None),
    ("datepart('year', @2022-01-02)", 2022),
    ("datepart('quarter', @2022-01-02)", 1),
    ("datepart('month', @2022-01-02)", 1),
    ("datepart('week', @2022-05-12)", 19),
    ("datepart('day', @2022-01-02)", 2),
    ("datepart('hour', @2022-01-02T12)", 12),
    ("datepart('minute', @2022-01-02T12:34)", 34),
    ("datepart('second', @2022-01-02T12:34:56)", 56),
    ("datepart('dow', @2022-01-02T12:34:56)", 7),
    ("datepart('dow', @2022-01-03T12:34:56)", 1),
    ("datediff('year', @2021-12-31T23:59:59, @2022-01-01)", 1),
    ("datediff('quarter', @2021-12-31T23:59:59, @2022-01-01)", 1),
    ("datediff('month', @2021-12-31T23:59:59, @2022-01-01)", 1),
    ("datediff('week', @2022-01-02T23:59:59, @2022-01-03)", 1),
    ("datediff('day', @2021-12-31T23:59:59, @2022-01-01)", 1),
    ("datediff('hour', @2021-12-31T23:59:59, @2022-01-01)", 1),
    ("datediff('minute', @2021-12-31T23:59:59, @2022-01-01)", 1),
    ("datediff('second', @2021-12-31T23:59:59, @2022-01-01)", 1),
    ("datediff('year', @2021-01-01, @2021-12-31)", 0),
    ("datediff('quarter', @2021-01-01, @2021-03-31)", 0),
    ("datediff('month', @2021-01-01, @2021-01-31)", 0),
    ("datediff('week', @2022-01-03, @2022-01-09)", 0),
    ("datediff('day', @2021-12-31T00:00:00, @2021-12-31T23:59:59)", 0),
    ("datediff('hour', @2021-12-31T23, @2021-12-31T23:59:59)", 0),
    ("datediff('minute', @2021-12-31T04:59, @2021-12-31T04:59:59)", 0),
    ("datediff('second', @2021-12-31T04:59:59, @2021-12-31T04:59:59.999)", 0),
]


def get_compiler_scalar_expr(backend: Backend, expr: str):
    table = backend.table("media_types", "media_types")
    aggregate = backend.aggregate(
        table, groupby=[Column(name="x", expr=parse_expr(expr))], aggregate=[]
    )
    ordered = backend.order_by(aggregate, [LiteralOrderItem(name="x", ascending=True)])
    df = backend.execute(ordered)
    return df.iloc[0, 0]


@pytest.mark.parametrize(["expr", "expected"], compiler_scalar_expr_test_cases)
def test_compiler_scalar_expr(backend: Backend, expr, expected):
    result = get_compiler_scalar_expr(backend, expr)
    assert result == expected


compiler_scalar_expr_dates_test_cases = [
    ("datetrunc('year', @2022-05-06)", isoparse("2022-01-01")),
    ("datetrunc('quarter', @2022-05-06)", isoparse("2022-04-01")),
    ("datetrunc('month', @2022-05-06)", isoparse("2022-05-01")),
    ("datetrunc('week', @2022-05-06)", isoparse("2022-05-02")),
    ("datetrunc('day', @2022-05-06T12)", isoparse("2022-05-06")),
    ("datetrunc('hour', @2022-05-06T12:34)", isoparse("2022-05-06T12")),
    ("datetrunc('minute', @2022-05-06T12:34:56)", isoparse("2022-05-06T12:34")),
    ("datetrunc('second', @2022-05-06T12:34:56.789)", isoparse("2022-05-06T12:34:56")),
    ("dateadd('year', 2, @2022-01-01)", isoparse("2024-01-01")),
    ("dateadd('quarter', 2, @2022-01-01)", isoparse("2022-07-01")),
    ("dateadd('month', 2, @2022-01-01)", isoparse("2022-03-01")),
    ("dateadd('week', 2, @2022-01-01)", isoparse("2022-01-15")),
    ("dateadd('day', 45, @2022-01-01)", isoparse("2022-02-15")),
    ("dateadd('hour', 45, @2022-01-01)", isoparse("2022-01-02 21:00")),
    ("dateadd('minute', 45, @2022-01-01)", isoparse("2022-01-01 00:45")),
    ("dateadd('second', 45, @2022-01-01)", isoparse("2022-01-01 00:00:45")),
]


@pytest.mark.parametrize(["expr", "expected"], compiler_scalar_expr_dates_test_cases)
def test_compiler_scalar_expr_dates(backend: Backend, expr: str, expected):
    """For cases where the backend might return an unknown type, stringify the result
    and compare
    """
    result = get_compiler_scalar_expr(backend, expr)
    assert isoparse(str(result)) == expected


compiler_aggregate_functions_test_cases = [
    ("media_types", "sum(media_types.MediaTypeId)", 15),
    ("media_types", "min(media_types.MediaTypeId)", 1),
    ("media_types", "max(media_types.MediaTypeId)", 5),
    ("media_types", "avg(media_types.MediaTypeId)", 3),
    ("media_types", "count()", 5),
    ("media_types", "count(media_types.MediaTypeId)", 5),
    ("tracks", "countd(tracks.GenreId)", 25),
]


@pytest.mark.parametrize(
    ["table", "expr", "expected"], compiler_aggregate_functions_test_cases
)
def test_compiler_aggregate_functions(
    backend: Backend, table: str, expr: str, expected
):
    table = backend.table(table, table)
    col = Column(name="x", expr=parse_expr(expr))
    aggregate = backend.aggregate(table, groupby=[], aggregate=[col])
    df = backend.execute(aggregate)
    assert df.iloc[0, 0] == expected


# call_window children are: fn_name, args (list), partition_by, order_by, window
# window is always None, not supported yet

_media_type_id = Tree("column", [None, "MediaTypeId"])
_fdiv_3 = Tree("call", ["floor", Tree("div", [_media_type_id, 3])])


def _window_fn(
    fn: str, args: list, partition_by: Optional[list], order_by: Optional[list]
) -> Tree:
    """Helper to build ASTs for window functions"""
    _partition_by = None
    _order_by = None
    if partition_by:
        _partition_by = Tree("partition_by", partition_by)
    if order_by:
        _order_by = Tree(
            "order_by", [Tree("order_by_item", [a, b]) for a, b in order_by]
        )
    return Tree(
        "expr",
        [
            Tree(
                "call_window",
                [
                    fn,
                    *args,
                    _partition_by,
                    _order_by,
                    None,
                ],
            )
        ],
    )


compiler_window_functions_test_cases = [
    (
        _window_fn("row_number", [], None, [(_media_type_id, True)]),
        [1, 2, 3, 4, 5],
        "row_number() over (order by MediaTypeId asc)",
    ),
    (
        _window_fn("row_number", [], [_media_type_id], None),
        [1, 1, 1, 1, 1],
        "row_number() over (partition by MediaTypeId)",
    ),
    (
        _window_fn("row_number", [], None, [(_media_type_id, False)]),
        [5, 4, 3, 2, 1],
        "row_number() over (order by MediaTypeId desc)",
    ),
    (
        _window_fn("row_number", [], [_fdiv_3], [(_media_type_id, True)]),
        [1, 2, 1, 2, 3],
        "row_number() over (partition by MediaTypeId // 3 order by MediaTypeId)",
    ),
    (
        _window_fn("row_number", [], [_fdiv_3], [(_media_type_id, False)]),
        [2, 1, 3, 2, 1],
        "row_number() over (partition by MediaTypeId // 3 order by MediaTypeId desc)",
    ),
    (
        _window_fn("sum", [_media_type_id], None, None),
        [15, 15, 15, 15, 15],
        "sum(MediaTypeId) over ()",
    ),
    (
        _window_fn("sum", [_media_type_id], [_fdiv_3], None),
        [3, 3, 12, 12, 12],
        "sum(MediaTypeId) over (partition by MediaTypeId // 3)",
    ),
    (
        _window_fn("sum", [_media_type_id], [_fdiv_3], [(_media_type_id, True)]),
        [1, 3, 3, 7, 12],
        "sum(MediaTypeId) over (partition by MediaTypeId // 3 order by MediaTypeId)",
    ),
    (
        _window_fn("sum", [_media_type_id], [_fdiv_3], [(_media_type_id, False)]),
        [3, 2, 12, 9, 5],
        (
            "sum(MediaTypeId) over "
            "(partition by MediaTypeId // 3 order by MediaTypeId desc)"
        ),
    ),
]


@pytest.mark.parametrize(
    ["expr", "expected", "comment"], compiler_window_functions_test_cases
)
def test_compiler_window_functions(
    backend: Backend, expr: Tree, expected, comment: str
):
    comment
    table = backend.table("media_types", "media_types")
    id_ = Column(name="id", expr=Tree("expr", [Tree("column", [None, "MediaTypeId"])]))
    x = Column(name="x", expr=expr)
    calculate = backend.calculate(table, [id_, x])
    query = backend.order_by(calculate, [LiteralOrderItem(name="id", ascending=True)])
    df = backend.execute(query)
    expected_df = DataFrame({"id": [1, 2, 3, 4, 5], "x": expected})
    assert df.shape[0] == 5
    testing.assert_frame_equal(df, expected_df)


def test_backend_full_merge(backend: Backend):
    t1 = backend.table("media_types", "t1")
    t1 = backend.filter(t1, parse_expr("t1.MediaTypeId in (1, 2, 3)"))

    t2 = backend.table("media_types", "t2")
    t2 = backend.filter(t2, parse_expr("t2.MediaTypeId in (3, 4, 5)"))

    merged = backend.merge([t1, t2], on=["MediaTypeId"])
    df = backend.execute(merged)

    assert df.shape == (5, 3)
    assert df["MediaTypeId"].to_list() == [1, 2, 3, 4, 5]
    assert df.iloc[:, 1].dropna().size == 3
    assert df.iloc[:, 2].dropna().size == 3


def test_backend_left_merge(backend: Backend):
    t1 = backend.table("media_types", "t1")
    t1 = backend.filter(t1, parse_expr("t1.MediaTypeId in (1, 2, 3)"))

    t2 = backend.table("media_types", "t2")
    t2 = backend.filter(t2, parse_expr("t2.MediaTypeId in (3, 4, 5)"))

    merged = backend.merge([t1, t2], on=["MediaTypeId"], left=True)
    df = backend.execute(merged)

    assert df.shape == (3, 3)
    assert df["MediaTypeId"].to_list() == [1, 2, 3]
    assert df.iloc[:, 1].dropna().size == 3
    assert df.iloc[:, 2].dropna().size == 1
