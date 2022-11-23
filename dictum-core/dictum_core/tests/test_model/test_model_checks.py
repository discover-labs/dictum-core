import pytest

from dictum_core import schema
from dictum_core.exceptions import (
    AggregateDimensionError,
    CircularReferenceError,
    ExpressionSyntaxError,
    FilteredMeasureReferenceError,
    InvalidMeasureError,
    InvalidReferenceError,
    MissingAggregationError,
    MissingDimensionError,
    MissingMeasureError,
    MissingPrimaryKeyError,
    MixedExpressionError,
    NonAggregateMeasureError,
    UnknownFunctionError,
)
from dictum_core.model.checks import check_model
from dictum_core.model.model import Model
from dictum_core.schema.model.checks import check_model_config


def build_model(data: dict) -> Model:
    config = schema.Model.parse_obj(data)
    return Model.from_config(config)


@pytest.fixture(scope="function")
def model() -> Model:
    """Provides a valid model that can be broken for tests"""
    config = schema.Model.parse_obj(
        {
            "name": "test",
            "tables": {
                "t1": {
                    "source": "t1",
                    "primary_key": "pk",
                    "measures": {
                        "me1": {"name": "me1", "expr": "sum(col)"},
                        "me3": {"name": "me3", "expr": "sum(col)"},
                    },
                    "dimensions": {
                        "d1": {"name": "d1", "expr": "col", "type": "int"},
                        "d2": {"name": "d2", "expr": "$m1", "type": "int"},
                    },
                    "filters": [":d1 > 0"],
                },
                "t2": {
                    "source": "t2",
                    "measures": {"me2": {"name": "me2", "expr": "count()"}},
                },
            },
            "metrics": {
                "m1": {
                    "name": "m1",
                    "expr": "sum(value)",
                    "table": "t1",
                    "filter": ":d1 = 1 and col = 2",
                },
                "m2": {"name": "m2", "expr": "$m1 / 100"},
                "m3": {"name": "m3", "expr": "$m1 / $m2"},
            },
        }
    )
    check_model_config(config)
    return Model.from_config(config)


def test_model_valid(model: Model):
    check_model(model)


def test_check_expr_parse_checks_dimensions(model: Model):
    model.dimensions["d1"].str_expr = "sum(co"
    with pytest.raises(ExpressionSyntaxError):
        check_model(model)


def test_check_expr_parse_checks_measures(model: Model):
    model.measures["m1"].str_expr = "sum())))"
    with pytest.raises(ExpressionSyntaxError):
        check_model(model)


def test_check_expr_parse_checks_metrics(model: Model):
    model.metrics["m2"].str_expr = "$$test"
    with pytest.raises(ExpressionSyntaxError):
        check_model(model)


def test_check_expr_parse_checks_filters(model: Model):
    model.measures["m1"].str_filter = "Ahahahah!!!!"
    with pytest.raises(ExpressionSyntaxError):
        check_model(model)


def test_check_expr_parse_checks_table_filters(model: Model):
    model.tables["t1"].filters[0].str_expr = "bleep bloop"
    with pytest.raises(ExpressionSyntaxError):
        check_model(model)


# def test_check_measure_measure_refs(model: Model):
#     model.measures["m1"].str_expr = "$m2"
#     with pytest.raises(InvalidReferenceError):
#         check_model(model)


def test_check_metric_column_refs(model: Model):
    model.metrics["m1"].str_expr = "sum(col)"
    with pytest.raises(InvalidReferenceError):
        check_model(model)


def test_check_metric_dimension_refs(model: Model):
    model.metrics["m1"].str_expr = "sum(:d1)"
    with pytest.raises(InvalidReferenceError):
        check_model(model)


@pytest.mark.xfail
def test_check_measure_filter_measure_refs(model: Model):
    """Not sure if this test is needed.
    # TODO: figure out if referencing metrics in measure filters works the same as
            with dimensions
    """
    model.measures["m1"].str_filter = "$m2"
    with pytest.raises(InvalidReferenceError):
        check_model(model)


def test_check_known_functions_metrics(model: Model):
    model.metrics["m2"].str_expr = "spam($m1 / 100)"
    with pytest.raises(UnknownFunctionError):
        check_model(model)


def test_check_known_functions_dimensions(model: Model):
    model.dimensions["d1"].str_expr = "ham(col)"
    with pytest.raises(UnknownFunctionError):
        check_model(model)


def test_check_known_functions_measures(model: Model):
    model.measures["m1"].str_expr = "sam(whoa)"
    with pytest.raises(UnknownFunctionError):
        check_model(model)


def test_check_known_functions_measure_filter(model: Model):
    model.measures["m1"].str_filter = "gram(gg) > 0"
    with pytest.raises(UnknownFunctionError):
        check_model(model)


def test_check_known_functions_table_filters(model: Model):
    model.tables["t1"].filters[0].str_expr = "fam(lala)"
    with pytest.raises(UnknownFunctionError):
        check_model(model)


def test_check_known_calculations_metrics_metrics(model: Model):
    model.metrics["m1"].str_expr = "$mx"
    with pytest.raises(MissingAggregationError):
        check_model(model)


def test_check_known_calculations_dimensions_dimension(model: Model):
    model.dimensions["d1"].str_expr = ":unk // 10"
    with pytest.raises(MissingDimensionError):
        check_model(model)


def test_check_known_calculations_dimensions_metric(model: Model):
    model.dimensions["d2"].str_expr = "$unk"
    with pytest.raises(MissingAggregationError):
        check_model(model)


def test_check_known_calculations_measure_filters(model: Model):
    model.measures["m1"].str_filter = ":unk > 0"
    with pytest.raises(MissingDimensionError):
        check_model(model)


def test_check_known_calculations_table_filters(model: Model):
    model.tables["t1"].filters[0].str_expr = ":unk > 0"
    with pytest.raises(MissingDimensionError):
        check_model(model)


def test_check_kinds_aggregate_dimension(model: Model):
    model.dimensions["d1"].str_expr = "sum(x) + 1"
    with pytest.raises(AggregateDimensionError):
        check_model(model)


def test_check_kinds_scalar_measure(model: Model):
    model.measures["m1"].str_expr = ":d1 - 'scalar'"
    with pytest.raises(NonAggregateMeasureError):
        check_model(model)


def test_check_kinds_aggregate_measure_filter(model: Model):
    model.measures["m1"].str_filter = "sum(:d1) > 0"
    with pytest.raises(AggregateDimensionError):
        check_model(model)


def test_check_kinds_aggregate_table_filter(model: Model):
    model.tables["t1"].filters[0].str_expr = "count() > 0"
    with pytest.raises(AggregateDimensionError):
        check_model(model)


def test_kinds_mixed_measure(model: Model):
    model.measures["m1"].str_expr = "sum(col) + col"
    with pytest.raises(MixedExpressionError):
        check_model(model)


def test_kinds_mixed_dimension(model: Model):
    model.dimensions["d1"].str_expr = "$m1 + col"
    with pytest.raises(MixedExpressionError):
        check_model(model)


def test_kinds_mixed_measure_filter(model: Model):
    model.measures["m1"].str_filter = "$m1 + col"
    with pytest.raises(MixedExpressionError):
        check_model(model)


def test_kinds_mixed_table_filter(model: Model):
    model.tables["t1"].filters[0].str_expr = "$m1 + col"
    with pytest.raises(MixedExpressionError):
        check_model(model)


def test_dimension_self_ref(model: Model):
    model.dimensions["d1"].str_expr = ":d1 + 1"
    with pytest.raises(CircularReferenceError):
        check_model(model)


def test_metric_self_ref(model: Model):
    model.metrics["m2"].str_expr = "$m2"
    with pytest.raises(CircularReferenceError):
        check_model(model)


def test_check_circular_refs_dimensions(model: Model):
    model.dimensions["d1"].str_expr = ":d2"
    model.dimensions["d2"].str_expr = ":d1"
    with pytest.raises(CircularReferenceError):
        check_model(model)


def test_check_circular_refs_metrics(model: Model):
    model.metrics["m2"].str_expr = "$m3"
    model.metrics["m3"].str_expr = "$m2"
    with pytest.raises(CircularReferenceError):
        check_model(model)


def test_check_measure_references_same_table(model: Model):
    model.measures["me2"].str_expr = "$me1 / count()"
    with pytest.raises(MissingMeasureError):
        check_model(model)


def test_check_measures_use_aggregate_functions(model: Model):
    model.measures["me3"].str_expr = "$me1"
    with pytest.raises(InvalidMeasureError):
        check_model(model)


def test_check_measures_dont_reference_filtered_measures(model: Model):
    model.measures["me1"].str_expr = "$m1 / count()"
    with pytest.raises(FilteredMeasureReferenceError):
        check_model(model)


def test_check_aggregate_dimension_table_has_pk(model: Model):
    model.tables["t1"].primary_key = None
    with pytest.raises(MissingPrimaryKeyError):
        check_model(model)
