from dictum_core.engine.graph.query import QueryMetric


def test_query_metric_name():
    assert (
        QueryMetric.parse_obj(
            {
                "id": "x",
                "table_transform": {"id": "t", "args": [1]},
                "window": {"of": [{"id": "of"}], "within": [{"id": "within"}]},
                "scalar_transforms": [{"id": "eq", "args": ["a"]}],
            }
        ).name
        == "x_t_1_of_of_within_within_eq_a"
    )
