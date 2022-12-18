from dictum_core.engine.graph.query import Query, QueryDimension, QueryMetric
from dictum_core.ql.v2 import compile_dimension, compile_metric, compile_query


def test_compile_metric():
    assert compile_metric("$y.t(2, '3') of (a) within (b)") == QueryMetric.parse_obj(
        {
            "id": "y",
            "table_transform": {"id": "t", "args": [2, "3"]},
            "scalar_transforms": [],
            "window": {
                "of": [{"id": "a", "scalar_transforms": []}],
                "within": [{"id": "b", "scalar_transforms": []}],
            },
        }
    )


def test_compile_dimension():
    assert compile_dimension(":d.t('a', 1)") == QueryDimension.parse_obj(
        {"id": "d", "scalar_transforms": [{"id": "t", "args": ["a", 1]}]}
    )


def test_compile_query():
    assert (
        compile_query(
            """
        FROM example:e
        FILTER :x = 1
        DECLARE a = $c.percent
        FILTER (:m, :n) WHERE $s.top(3) WITHIN (:w) = 1
        FILTER :s, :f WHERE $p.total >= 1000
        SELECT :b, $a
        ORDER BY -$z, +:b
        LIMIT 5
        """
        )
        == Query.parse_obj(
            {
                "cube": {
                    "source": {"value": "e", "kind": "example"},
                    "qualifiers": [
                        {
                            "filters": [
                                {
                                    "dimension": {
                                        "id": "x",
                                        "scalar_transforms": [
                                            {"id": "eq", "args": [1]}
                                        ],
                                    }
                                }
                            ]
                        },
                        {
                            "alias": "a",
                            "metric": {
                                "id": "c",
                                "scalar_transforms": [],
                                "table_transform": {"id": "percent", "args": []},
                                "window": None,
                            },
                        },
                        {
                            "filters": [
                                {
                                    "metric": {
                                        "id": "s",
                                        "scalar_transforms": [
                                            {"id": "eq", "args": [1]}
                                        ],
                                        "table_transform": {"id": "top", "args": [3]},
                                        "window": {
                                            "of": [
                                                {"id": "m", "scalar_transforms": []},
                                                {"id": "n", "scalar_transforms": []},
                                            ],
                                            "within": [
                                                {"id": "w", "scalar_transforms": []}
                                            ],
                                        },
                                    }
                                }
                            ]
                        },
                        {
                            "filters": [
                                {"dimension": {"id": "s", "scalar_transforms": []}},
                                {
                                    "metric": {
                                        "id": "p",
                                        "scalar_transforms": [
                                            {"id": "ge", "args": [1000]}
                                        ],
                                        "table_transform": {"id": "total", "args": []},
                                        "window": {
                                            "of": [
                                                {"id": "f", "scalar_transforms": []}
                                            ],
                                            "within": [],
                                        },
                                    }
                                },
                            ]
                        },
                    ],
                },
                "select": [
                    {
                        "id": "b",
                        "scalar_transforms": [],
                        "kind": "dimension",
                        "alias": "b",
                    },
                    {
                        "id": "a",
                        "scalar_transforms": [],
                        "table_transform": None,
                        "window": None,
                        "kind": "metric",
                        "alias": "a",
                    },
                ],
                "order_by": [
                    {
                        "id": "z",
                        "scalar_transforms": [],
                        "table_transform": None,
                        "window": None,
                        "kind": "metric",
                        "ascending": False,
                    },
                    {
                        "id": "b",
                        "scalar_transforms": [],
                        "kind": "dimension",
                        "ascending": True,
                    },
                ],
                "limit": 5,
            }
        )
    )
