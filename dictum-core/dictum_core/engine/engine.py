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
    QueryMetricDeclaration,
    QueryMetricRequest,
    QueryMetricWindow,
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
    query: Query = query.copy(deep=True)
    dims = [r for r in query.select if isinstance(r, QueryDimensionRequest)]
    declared = [
        x for x in query.cube.qualifiers if isinstance(x, QueryMetricDeclaration)
    ]
    for request in chain(query.select, declared):
        if (
            isinstance(request, QueryDimensionRequest)
            or request.table_transform is None
        ):
            continue

        t = request.table_transform
        w = request.window or QueryMetricWindow()

        withins = {w.digest for w in w.within}
        ofs = {o.digest for o in w.of}

        # if t.id in {"top", "bottom"} and len(transform.of) == 0:
        #     transform.of = [d for d in dims if d.digest not in withins]

        if t.id == "percent":
            if len(w.of) == 0:
                w.of = [d for d in dims if d.digest not in withins]
            elif len(w.within) == 0:
                w.within = [d for d in dims if d.digest not in ofs]

        if next(chain(w.of, w.within), None) is not None:
            request.window = w

    return query


class Engine:
    def __init__(self, model: "model.Model"):
        self.model = model

    def get_computation(self, query: Query) -> FinalizeOperator:
        # FIXME: turn on
        # check_query(self.model, query)

        query = _prep_query_of_within(query)

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

        declared_requests = {}
        for item in query.cube.qualifiers:
            if isinstance(item, QueryMetricDeclaration):
                declared_requests[item.alias] = item

        digest_requests = {r.digest: r for r in query.select}
        for request in query.select:
            if request.id in declared_requests:
                dreq = declared_requests[request.id].copy(deep=True)
                if request.alias is not None:
                    dreq.alias = request.alias
                digest_requests[request.digest] = dreq

        display_info = {
            d: DisplayInfo.from_request(self.model, r)
            for d, r in digest_requests.items()
        }
        self.model.temp_metrics = {}  # clear the temp metrics
        return FinalizeOperator(graph, display_info=display_info)
