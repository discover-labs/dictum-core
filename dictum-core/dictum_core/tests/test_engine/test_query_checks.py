import pytest

from dictum_core import Project
from dictum_core.engine.checks import check_query
from dictum_core.exceptions import (
    DuplicateColumnError,
    MissingQueryDimensionError,
    MissingQueryMetricError,
    MissingScalarTransformError,
    MissingTableTransformDimensionError,
    MissingTableTransformError,
    ScalarTransformTypeError,
)
from dictum_core.model import Model
from dictum_core.schema import Query


@pytest.fixture(scope="function")
def query(project: Project):
    """Provides a valid query that can be broken for tests"""
    return project.ql(
        """
    select revenue, revenue.percent within (invoice_date.year)
    where genre = 'Rock'
    by invoice_date.year, invoice_date.quarter
    """
    ).query


@pytest.fixture(scope="module")
def model(project: Project):
    return project.model


def test_query_valid(model: Model, query: Query):
    check_query(model, query)


def test_check_metrics_exist(model: Model, query: Query):
    query.metrics[0].metric.id = "missing"
    with pytest.raises(MissingQueryMetricError):
        check_query(model, query)


def test_check_dimensions_exist(model: Model, query: Query):
    query.dimensions[0].dimension.id = "missing"
    with pytest.raises(MissingQueryDimensionError):
        check_query(model, query)


def test_check_dimensions_exist_checks_filters(model: Model, query: Query):
    query.filters[0].id = "missing"
    with pytest.raises(MissingQueryDimensionError):
        check_query(model, query)


def test_check_dimensions_exist_checks_of_within(model: Model, query: Query):
    query.metrics[1].metric.transforms[0].within[0].id = "missing"
    with pytest.raises(MissingQueryDimensionError):
        check_query(model, query)


def test_check_scalar_transforms_exist(model: Model, query: Query):
    query.dimensions[0].dimension.transforms[0].id = "missing"
    with pytest.raises(MissingScalarTransformError):
        check_query(model, query)


def test_check_table_transforms_exist(model: Model, query: Query):
    query.metrics[1].metric.transforms[0].id = "missing"
    with pytest.raises(MissingTableTransformError):
        check_query(model, query)


def test_check_of_within_in_dimensions(model: Model, query: Query):
    query.dimensions = []
    with pytest.raises(MissingTableTransformDimensionError):
        check_query(model, query)


def test_check_scalar_transform_types(model: Model, query: Query):
    dimension = query.dimensions[0].dimension
    dimension.transforms[0].id = "step"
    dimension.transforms[0].args = [10]
    query.dimensions[0].dimension = dimension
    query.metrics[1].metric.transforms[0].within[0] = dimension
    with pytest.raises(ScalarTransformTypeError):
        check_query(model, query)


def test_check_dimension_join_paths(model: Model):
    query = Query.parse_obj(
        {
            "metrics": [{"metric": {"id": "track_count"}}],
            "dimensions": [{"dimension": {"id": "invoice_date"}}],
        }
    )
    with pytest.raises(MissingQueryDimensionError, match="not available for Measure"):
        check_query(model, query)


def test_check_duplicate_columns(model: Model, query: Query):
    query.dimensions[0].alias = "revenue"
    with pytest.raises(DuplicateColumnError):
        check_query(model, query)
