from itertools import chain

from dictum_core.engine.graph.builders import MetricsGraphBuilder
from dictum_core.engine.query import Query
from dictum_core.exceptions import (
    DuplicateColumnError,
    MissingQueryDimensionError,
    MissingQueryMetricError,
    MissingScalarTransformError,
    MissingTableTransformDimensionError,
    MissingTableTransformError,
    MisusedTableTransformError,
    ScalarTransformTypeError,
)
from dictum_core.model import Model
from dictum_core.ordered_check_caller import OrderedCheckCaller

table_transforms = set(MetricsGraphBuilder.transforms)

check_query = OrderedCheckCaller()


def _check_metrics_exist(model: Model, query: Query):
    for request in query.metrics:
        if request.metric.id not in model.metrics:
            raise MissingQueryMetricError(f"Metric {request.metric.id} does not exist")


def _check_dimension_exists(id_: str, model: Model):
    if id_ not in model.dimensions:
        raise MissingQueryDimensionError(f"Dimension {id_} doesn't exist")


def _check_dimensions_exist(model: Model, query: Query):
    for request in query.dimensions:
        _check_dimension_exists(request.dimension.id, model)
    for dimension in query.filters:
        _check_dimension_exists(dimension.id, model)
    for request in query.metrics:
        if request.metric.transform:
            for dimension in chain(
                request.metric.transform.within, request.metric.transform.of
            ):
                _check_dimension_exists(dimension.id, model)


@check_query.depends_on(_check_dimensions_exist)
def _check_scalar_transforms_exist(model: Model, query: Query):
    for request in query.dimensions:
        for transform in request.dimension.transforms:
            if transform.id not in model.scalar_transforms:
                raise MissingScalarTransformError(
                    f"Scalar transform {transform.id} does not exist"
                )


@check_query.depends_on(_check_metrics_exist)
def _check_table_transforms_exist(model: Model, query: Query):
    for request in query.metrics:
        if (
            request.metric.transform
            and request.metric.transform.id not in table_transforms
        ):
            raise MissingTableTransformError(
                f"Table transform {request.metric.transform.id} does not exist"
            )


@check_query.depends_on(_check_table_transforms_exist)
def _check_top_bottom_usage(model: Model, query: Query):
    top_bottom = {"top", "bottom"}
    for request in query.metrics:
        if (
            request.metric.transform is not None
            and request.metric.transform.id in top_bottom
        ):
            raise MisusedTableTransformError(
                "top/bottom transforms can only be used as table filters, "
                "found in metrics"
            )
    for metric in query.table_filters:
        if (
            metric.transform is not None
            and metric.transform.id in top_bottom
            and len(metric.transforms) > 0
        ):
            raise MisusedTableTransformError(
                "top/bottom table transforms can't be combined with other transforms"
            )


@check_query.depends_on(
    _check_dimensions_exist,
    _check_metrics_exist,
)
def _check_dimension_join_paths(model: Model, query: Query):
    query_dimensions = set(
        chain(
            (r.dimension.id for r in query.dimensions),
            (d.id for d in query.filters),
        )
    )
    for request in query.metrics:
        metric = model.metrics[request.metric.id]
        for measure in metric.measures:
            measure_dimensions = set(measure.dimensions)
            missing = query_dimensions - measure_dimensions
            if len(missing) > 0:
                raise MissingQueryDimensionError(
                    f"Dimensions {missing} requested in the query are not available "
                    f"for {measure} in {metric}"
                )


@check_query.depends_on(
    _check_dimensions_exist,
    _check_metrics_exist,
    _check_table_transforms_exist,
    _check_scalar_transforms_exist,
    _check_dimension_join_paths,
)
def _check_of_within_in_dimensions(model: Model, query: Query):
    # FIXME: with the new engine, the check is only needed with for queried metrics,
    # not filters
    dimension_digests = set()
    for request in query.dimensions:
        dimension_digests.add(request.dimension.digest)

    for request in query.metrics:
        if request.metric.transform:
            transform = request.metric.transform
            for dimension in chain(transform.of, transform.within):
                if dimension.digest not in dimension_digests:
                    raise MissingTableTransformDimensionError(
                        f"Dimension request '{dimension.render()}' is used in "
                        f"'{request.metric.render()}', but not in the group by list"
                    )


@check_query.depends_on(
    _check_dimensions_exist,
    _check_scalar_transforms_exist,
    _check_of_within_in_dimensions,
)
def _check_scalar_transform_types(model: Model, query: Query):
    for request in query.dimensions:
        dimension = model.dimensions[request.dimension.id]
        for transform in request.dimension.transforms:
            if transform.id not in dimension.transforms:
                raise ScalarTransformTypeError(
                    f"Transform {transform.id} is not available for dimension "
                    f"type {dimension.type.name}"
                )


@check_query.register
def _check_duplicate_columns(model: Model, query: Query):
    names = set()
    for request in chain(query.metrics, query.dimensions):
        if request.name in names:
            raise DuplicateColumnError(
                f"Duplicate column name '{request.name}' in the query"
            )
        names.add(request.name)
