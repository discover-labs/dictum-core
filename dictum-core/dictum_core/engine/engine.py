from copy import deepcopy

from lark import Tree
from toolz import compose_left

from dictum_core import model
from dictum_core.engine.aggregate_query_builder import AggregateQueryBuilder
from dictum_core.engine.checks import check_query
from dictum_core.engine.computation import RelationalQuery
from dictum_core.engine.metrics import AddMetric
from dictum_core.engine.metrics import transforms as table_transforms
from dictum_core.engine.operators import (
    FinalizeOperator,
    MaterializeOperator,
    MergeOperator,
)
from dictum_core.engine.query import Query, QueryMetricRequest


def metric_expr(expr: Tree):
    expr = deepcopy(expr)
    for ref in expr.find_data("measure"):
        ref.data = "column"
        ref.children = [None, *ref.children]
    return expr


class Engine:
    def __init__(self, model: "model.Model"):
        self.model = model

    def suggest_dimensions(self, query: Query):
        ...

    def suggest_measures(self, query: Query):
        ...

    def get_range_computation(self, dimension_id: str) -> RelationalQuery:
        ...

    def get_values_computation(self, dimension_id: str) -> RelationalQuery:
        ...

    def get_terminal(self, query: Query) -> MergeOperator:
        builder = AggregateQueryBuilder(
            model=self.model, dimensions=query.dimensions, filters=query.filters
        )

        merge = MergeOperator(query=query)

        # add metrics
        adders = []
        for request in query.metrics:
            if request.metric.transform is None:
                adders.append(AddMetric(request=request, builder=builder))
            else:
                transform_id = request.metric.transform.id
                adder = table_transforms[transform_id]
                adders.append(adder(request=request, builder=builder))

        for metric in query.table_filters:
            if metric.transform is None:
                adders.append(
                    AddMetric(
                        request=QueryMetricRequest(metric=metric),
                        builder=builder,
                        as_filter=True,
                    )
                )
            else:
                transform_id = metric.transform.id
                adder = table_transforms[transform_id]
                adders.append(
                    adder(
                        request=QueryMetricRequest(metric=metric),
                        builder=builder,
                        as_filter=True,
                    )
                )

        if isinstance(query.limit, int):
            merge.limit = query.limit

        return compose_left(*adders)(merge)

    def get_computation(self, query: Query) -> MergeOperator:
        # TODO: turn on
        # check_query(self.model, query)
        terminal = self.get_terminal(query)
        return FinalizeOperator(
            input=MaterializeOperator([terminal]),
            aliases={r.digest: r.name for r in query.metrics + query.dimensions},
        )
