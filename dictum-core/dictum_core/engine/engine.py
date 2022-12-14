from copy import deepcopy
from itertools import chain

from lark import Tree

from dictum_core import model
from dictum_core.engine.checks import check_query
from dictum_core.engine.graph.builders import MetricsGraphBuilder
from dictum_core.engine.graph.operators import FinalizeOperator
from dictum_core.engine.graph.query import (
    Query,
    QueryDimensionRequest,
    QueryMetricRequest,
    QueryTransform,
)
from dictum_core.engine.result import DisplayInfo


def metric_expr(expr: Tree):
    expr = deepcopy(expr)
    for ref in expr.find_data("measure"):
        ref.data = "column"
        ref.children = [None, *ref.children]
    return expr


# TODO: do this at another step so it doesn't affect names
def _prep_query_of_within(query: Query) -> Query:
    """Additionaly prepare query's OF statements and WITHIN statements.
    This is tranform-specific.

    TOP/BOTTOM transforms: OF is everything that's not WITHIN.
    PERCENT transform: fill in OF and WITHIN with missing dimensions if empty.
    """
    query = query.copy(deep=True)
    dims = [r.dimension for r in query.dimensions]
    for metric in chain(query.table_filters, (r.metric for r in query.metrics)):
        if metric.transform is None:
            continue

        transform = metric.transform

        withins = {w.digest for w in metric.transform.within}
        ofs = {o.digest for o in metric.transform.of}

        if transform.id in {"top", "bottom"} and len(transform.of) == 0:
            transform.of = [d for d in dims if d.digest not in withins]

        if metric.transform.id == "percent":
            if len(transform.of) == 0:
                transform.of = [d for d in dims if d.digest not in withins]
                continue
            if len(transform.within) == 0:
                transform.within = [d for d in dims if d.digest not in ofs]
                continue

    for metric in query.table_filters:
        if metric.transform is None:
            # TODO: support this at the level of QL allowing e.g.
            #       HAVING revenue within (genre) > 100
            metric.transform = QueryTransform(id="total", within=dims)

    return query


class Engine:
    def __init__(self, model: "model.Model"):
        self.model = model

    def get_computation(self, query: Query) -> FinalizeOperator:
        # FIXME: turn on
        # check_query(self.model, query)

        # query = _prep_query_of_within(query)

        metrics = [r for r in query.select if isinstance(r, QueryMetricRequest)]
        dimensions = [r for r in query.select if isinstance(r, QueryDimensionRequest)]
        select = [r.digest for r in query.select]

        builder = MetricsGraphBuilder(
            model=self.model,
            cube=query.cube,
            metrics=metrics,
            dimensions=dimensions,
            select=select,
            limit=query.limit,
        )

        graph = builder.get_graph()
        display_info = {
            r.digest: DisplayInfo.from_request(self.model, r) for r in query.select
        }
        return FinalizeOperator(graph, display_info=display_info)
