from pathlib import Path
from textwrap import dedent

import pytest
import yaml
from lark import Tree

from dictum_core.ql import compile_query
from dictum_core.ql.parser import (
    parse_dimension,
    parse_dimension_request,
    parse_metric_request,
    parse_ql,
)
from dictum_core.ql.transformer import (
    compile_dimension,
    compile_dimension_request,
    compile_metric_request,
)
from dictum_core.schema.query import (
    Query,
    QueryDimension,
    QueryDimensionRequest,
    QueryMetricRequest,
)


def _assert_tree(tree: Tree, expected: str):
    assert tree.pretty().strip() == dedent(expected).strip()


with (Path(__file__).parent / "test_ql_parse.yml").open("r") as fp:
    parse_cases = yaml.safe_load(fp)


@pytest.mark.parametrize("case", parse_cases)
def test_ql_parse(case: dict):
    _kind_mapping = {
        "dimension": parse_dimension,
        "dimension_request": parse_dimension_request,
        "metric_request": parse_metric_request,
        "query": parse_ql,
    }
    val = case["val"]
    expected = case["expected"]
    kind = case["kind"]
    func = _kind_mapping[kind]
    assert func(val).pretty().strip() == dedent(expected.strip())


def test_compile_query():
    assert compile_query("select x") == Query.parse_obj(
        {"metrics": [{"metric": {"id": "x"}}]}
    )


def test_compile_groupby():
    q = Query.parse_obj(
        {
            "metrics": [{"metric": {"id": "x"}}],
            "dimensions": [{"dimension": {"id": "y"}}],
        }
    )
    assert compile_query("select x group by y") == q
    assert compile_query("select x by y") == q


def test_compile_groupby_transforms():
    assert compile_query("select x, y by z.p(10)") == Query.parse_obj(
        {
            "metrics": [{"metric": {"id": "x"}}, {"metric": {"id": "y"}}],
            "dimensions": [
                {"dimension": {"id": "z", "transforms": [{"id": "p", "args": [10]}]}}
            ],
        }
    )


def test_compile_where():
    assert compile_query("select x where y.z(10)") == Query.parse_obj(
        {
            "metrics": [{"metric": {"id": "x"}}],
            "filters": [{"id": "y", "transforms": [{"id": "z", "args": [10]}]}],
        }
    )


def test_compile_multiple_groupbys():
    compile_query(
        """
    select x, y
    where z.z(1, 2, 3),
        a.b('a')
    by d.d('d'),
       c,
       f.h(11)
    """
    ) == Query.parse_obj(
        {
            "metrics": [{"metric": {"id": "x"}}, {"metric": {"id": "y"}}],
            "dimensions": [
                {
                    "dimension": {
                        "id": "d",
                        "transforms": [{"id": "d", "args": ["d"]}],
                    },
                },
                {"dimension": {"id": "c"}},
                {"dimension": {"id": "f", "tranforms": [{"id": "h", "args": [11]}]}},
            ],
            "filters": [
                {
                    "id": "z",
                    "transforms": [{"id": "z", "args": [1, 2, 3]}],
                },
                {"id": "a", "transforms": [{"id": "b", "args": ["a"]}]},
            ],
        }
    )


def test_compile_dimension_alias():
    assert compile_query("select metric by dim as dim1") == Query.parse_obj(
        {
            "metrics": [{"metric": {"id": "metric"}}],
            "dimensions": [{"dimension": {"id": "dim"}, "alias": "dim1"}],
        }
    )


def test_compile_dimension_transform_alias():
    assert compile_query("select x by y.a.b(10) as z") == Query.parse_obj(
        {
            "metrics": [{"metric": {"id": "x"}}],
            "dimensions": [
                {
                    "dimension": {
                        "id": "y",
                        "transforms": [{"id": "a"}, {"id": "b", "args": [10]}],
                    },
                    "alias": "z",
                }
            ],
        }
    )


def test_compile_dimension_request():
    assert compile_dimension_request("x") == QueryDimensionRequest.parse_obj(
        {"dimension": {"id": "x"}}
    )
    assert compile_dimension_request("x.y") == QueryDimensionRequest.parse_obj(
        {"dimension": {"id": "x", "transforms": [{"id": "y"}]}}
    )
    assert compile_dimension_request(
        "x.y('z', 1) as a"
    ) == QueryDimensionRequest.parse_obj(
        {
            "dimension": {"id": "x", "transforms": [{"id": "y", "args": ["z", 1]}]},
            "alias": "a",
        }
    )


def test_compile_metric_request():
    assert compile_metric_request(
        "x.y(1, 'f') of (a) within (b, c.n(1)) as al"
    ) == QueryMetricRequest.parse_obj(
        {
            "metric": {
                "id": "x",
                "transforms": [
                    {
                        "id": "y",
                        "args": [1, "f"],
                        "of": [{"id": "a"}],
                        "within": [
                            {"id": "b"},
                            {"id": "c", "transforms": [{"id": "n", "args": [1]}]},
                        ],
                    }
                ],
            },
            "alias": "al",
        }
    )


def test_compile_filter():
    assert compile_dimension("x.y('z')") == QueryDimension.parse_obj(
        {"id": "x", "transforms": [{"id": "y", "args": ["z"]}]}
    )


def test_compile_filter_null():
    assert compile_dimension("x is null") == QueryDimension.parse_obj(
        {"id": "x", "transforms": [{"id": "isnull", "args": []}]}
    )
