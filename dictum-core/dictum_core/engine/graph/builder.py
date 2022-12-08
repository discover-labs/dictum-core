from dataclasses import dataclass, field
from itertools import chain
from typing import ClassVar, List, Optional, Tuple, Union

from lark import Token, Tree
from toolz import compose_left

from dictum_core.engine.computation import Column
from dictum_core.engine.graph.operators import (
    AggregateOperator,
    CalculateOperator,
    FilterOperator,
    GenerateRecordsOperator,
    LeftJoinOperator,
    MergeOperator,
    Operator,
    RecordsFilterOperator,
    TableOperator,
)
from dictum_core.engine.query import (
    QueryDimension,
    QueryDimensionRequest,
    QueryMetric,
    QueryMetricRequest,
)
from dictum_core.exceptions import ShoudntHappenError
from dictum_core.model import Measure, Model, Table
from dictum_core.model.scalar import DatetruncTransform
from dictum_core.model.scalar import transforms as scalar_transforms
from dictum_core.model.time import GenericTimeDimension


@dataclass
class Join:
    expr: Tree
    table: "JoinedTable"


@dataclass
class UnnestedJoin:
    left_table: Table
    right_table: Union[Table, AggregateOperator]
    left_identity: str
    right_identity: str
    expr: Tree


@dataclass
class JoinedTable:
    model: Model  # required for building subqueries
    table: Union[Table, AggregateOperator]  # AggregateOperator means a subquery
    identity: str
    joins: List[Join] = field(default_factory=list)

    def add_join_path(self, path: List[str], track: tuple = ()):
        if len(path) == 0:
            return

        if len(track) == 0:  # prepend the track with anchor's ID as identity
            track = (self.identity,)

        alias, *path = path
        new_track = (*track, alias)
        new_identity = ".".join(new_track)

        # make sure there are no duplicates
        for existing_join in self.joins:
            if new_identity == existing_join.table.identity:
                existing_join.table.add_join_path(path, track=new_track)
                return

        # special treatment for subquery path
        # don't do recursion because we need current related table
        if alias.startswith("__subquery__"):
            # build an aggregate query
            # measure from subquery name
            measure_id = alias.replace("__subquery__", "")
            measure = self.model.measures[measure_id]

            dimension_id = f"__PK_{self.table.id}"
            # group the measure by previous table's PK
            pk_request = QueryDimensionRequest.parse_obj(
                {"dimension": {"id": dimension_id}}
            )
            builder = MeasureGraphBuilder(
                model=self.model,
                dimensions=[pk_request],
            )
            op = builder.get_graph(measure)
            join = Join(
                table=JoinedTable(model=self.model, table=op, identity=new_identity),
                expr=Tree(
                    "expr",
                    [
                        Tree(
                            "eq",
                            [
                                Tree(
                                    "column",
                                    [".".join(track), self.table.primary_key],
                                ),
                                Tree(
                                    "column",
                                    [new_identity, pk_request.name],
                                ),
                            ],
                        )
                    ],
                ),
            )
            self.joins.append(join)
            return  # subquery is always the last, terminate

        related = self.table.related[alias]
        join_expr = Tree(
            "expr",
            [
                Tree(
                    "eq",
                    [
                        Tree("column", [self.identity, related.foreign_key]),
                        Tree("column", [new_identity, related.related_key]),
                    ],
                )
            ],
        )
        join = Join(
            table=JoinedTable(
                model=self.model, table=related.table, identity=new_identity
            ),
            expr=join_expr,
        )
        self.joins.append(join)
        join.table.add_join_path(path, track=new_track)

    def unnested_joins(self):
        for join in self.joins:
            yield UnnestedJoin(
                left_table=self.table,
                right_table=join.table.table,
                left_identity=self.identity,
                right_identity=join.table.identity,
                expr=join.expr,
            )
            yield from join.table.unnested_joins()

    def add_join_paths(self, paths: List[List[str]]):
        """Convenience method for adding multiple paths"""
        for path in paths:
            self.add_join_path(path)


@dataclass
class MeasureGraphBuilder:
    """Builds an aggregate query execution graph for a single measure

    Parameters:
        model: The underlying Model object to draw the information from
        dimensions: Dimensions to group by
        join_dimensions: A list of additional dimensions to use in the join, but
                         not to group by. To be used at later steps by
                         injected table filters.
        scalar_filters: A list of QueryDimension objects to use as scalar filters
                        (on detailed data)
        table_filters: A list of Operator objects that will be later used for filtering
    """

    model: Model
    dimensions: List[QueryDimensionRequest] = field(default_factory=list)
    join_dimensions: List[QueryDimension] = field(default_factory=list)
    scalar_filters: List[QueryDimension] = field(default_factory=list)

    table_filters: List[GenerateRecordsOperator] = field(default_factory=list)

    def get_dimension_column(
        self, measure: Measure, request: QueryDimensionRequest
    ) -> Column:
        """Construct a Column with a transformed expression relative to
        a given measure's anchor table.
        """
        anchor = measure.table
        dimension = measure.dimensions[request.dimension.id]

        transforms = [
            self.model.scalar_transforms[t.id](*t.args)
            for t in request.dimension.transforms
        ]

        # if generic time, prepend transforms with datetrunc
        # and replace the dimension with measure's time
        if isinstance(dimension, GenericTimeDimension):
            if measure.time is None:
                raise ValueError(
                    f"You requested a generic {dimension} dimension with {measure}, "
                    "but it doesn't have a time dimension specified"
                )
            if dimension.grain is not None:
                transforms = [DatetruncTransform(dimension.type.grain), *transforms]
            dimension = measure.time

        # get the expression with join info
        try:
            join_path = anchor.dimension_join_paths[dimension.id]
        except KeyError:
            raise KeyError(
                f"You requested {dimension}, but it can't be used with "
                f"another measure that you requested on {anchor}"
            )
        expr = dimension.prefixed_expr(join_path)

        result = Column(name=request.name, expr=expr)

        return compose_left(*transforms)(result)

    def get_graph(self, measure: Measure):
        anchor = measure.table
        measure_column = Column(name=measure.id, expr=measure.expr)

        base = TableOperator(anchor)

        # TODO: turn back into graph, unroll in LeftJoinOperator.execute without
        #       actually running any upstream joins

        # build the join tree on the anchor table
        join_tree = JoinedTable(model=self.model, table=anchor, identity=anchor.id)
        join_tree.add_join_paths(measure_column.join_paths)

        for request in self.dimensions:
            column = self.get_dimension_column(measure, request)
            join_tree.add_join_paths(column.join_paths)

        # additional dimensions to join only (not group by), to be used later by
        # table-valued filters
        for dimension in self.join_dimensions:
            column = self.get_dimension_column(
                measure, QueryDimensionRequest(dimension=dimension)
            )
            join_tree.add_join_paths(column.join_paths)

        # get filter expressions and their descriptions
        filters: List[Tuple[Tree, str]] = []
        if measure.filter is not None:
            join_tree.add_join_paths(measure.filter.join_paths)
            filters.append((measure.filter.expr, measure.filter.str_expr))
        for table_filter in anchor.filters:
            join_tree.add_join_paths(table_filter.join_paths)
            filters.append((table_filter.expr, table_filter.str_expr))
        for query_filter in self.scalar_filters:
            column = self.get_dimension_column(
                measure=measure, request=QueryDimensionRequest(dimension=query_filter)
            )
            join_tree.add_join_paths(column.join_paths)
            filters.append((column.expr, column.name))

        # after building the tree, unroll the joins and add operations
        right = []
        right_identities = []
        join_exprs = []
        for join in join_tree.unnested_joins():
            if isinstance(join.right_table, Table):
                right.append(TableOperator(join.right_table))
            elif isinstance(join.right_table, AggregateOperator):
                right.append(join.right_table)
            right_identities.append(join.right_identity)
            join_exprs.append(join.expr)

        # if there were no joins, keep the bare table
        if len(right) > 0:
            base = LeftJoinOperator(
                left=TableOperator(anchor),
                left_identity=join_tree.identity,
                right=right,
                right_identities=right_identities,
                join_exprs=join_exprs,
            )
        else:
            base = TableOperator(anchor)

        # apply filters, if any
        if len(filters) > 0:
            conditions, descriptions = zip(*filters)
            base = FilterOperator(
                base, conditions=conditions, description="\n".join(descriptions)
            )

        # add filters by record
        if len(self.table_filters) > 0:
            field_exprs = []
            for genrec in self.table_filters:
                names = {}
                for name, dimension in genrec.field_dimensions.items():
                    names[name] = self.get_dimension_column(
                        measure=measure,
                        request=QueryDimensionRequest(dimension=dimension),
                    ).expr
                field_exprs.append(names)
            base = RecordsFilterOperator(
                base, filters=self.table_filters, field_exprs=field_exprs
            )

        groupby = [self.get_dimension_column(measure, r) for r in self.dimensions]
        result = AggregateOperator(
            base=base, groupby=groupby, aggregate=[measure_column]
        )
        return result


@dataclass
class MetricBranch:
    """A full graph of requirements for one metric:

    Arguments:
        column: a Column to calculate the resulting metric from merged terminal values
        terminals: a list of terminals, producing measures or metric merges
    """

    column: Column
    terminals: List[Operator]


@dataclass
class MetricsGraphBuilder:
    model: Model

    metrics: List[QueryMetricRequest]
    dimensions: List[QueryDimensionRequest] = field(default_factory=list)

    scalar_filters: List[QueryDimension] = field(default_factory=list)
    table_filters: List[QueryMetric] = field(default_factory=list)

    context_scalar_filters: List[QueryDimension] = field(default_factory=list)
    context_table_filters: List[QueryMetric] = field(default_factory=list)

    # filters to be injected from a previous step
    inject_table_filters: List[Operator] = field(default_factory=list)

    transforms: ClassVar[dict] = {}
    is_transform: ClassVar[bool] = False  # needed to track if transforms are called

    def __init_subclass__(cls) -> None:
        if hasattr(cls, "id"):
            cls.transforms[cls.id] = cls
            cls.is_transform = True

    @classmethod
    def from_query_metric(
        cls,
        model: Model,
        metric: QueryMetric,
        scalar_filters: Optional[List[QueryDimension]] = None,
        inject_table_filters: Optional[List[Operator]] = None,
    ):
        if scalar_filters is None:
            scalar_filters = []
        if inject_table_filters is None:
            inject_table_filters = []

        metrics = [QueryMetricRequest(metric=metric)]
        dimensions = [
            QueryDimensionRequest(dimension=d)
            for d in chain(metric.transform.of, metric.transform.within)
        ]

        return cls(
            model=model,
            metrics=metrics,
            dimensions=dimensions,
            scalar_filters=scalar_filters,
            inject_table_filters=inject_table_filters,
        )

    def get_base_metric_branch(
        self,
        request: QueryMetricRequest,
        scalar_filters: Optional[List[QueryDimension]] = None,
        inject_table_filters: Optional[List[Operator]] = None,
    ) -> MetricBranch:
        """Get a column and a list of measure graphs for a metric."""
        # TODO: get measures that share the same anchor and filters from one query
        if scalar_filters is None:
            scalar_filters = []
        if inject_table_filters is None:
            inject_table_filters = []
        metric = self.model.metrics[request.metric.id]
        measure_builder = MeasureGraphBuilder(
            model=self.model,
            dimensions=self.dimensions,
            scalar_filters=scalar_filters,
            table_filters=inject_table_filters,
        )
        terminals = []
        for measure in metric.measures:
            terminals.append(measure_builder.get_graph(measure))
        column = Column(name=request.name, expr=metric.merged_expr)
        transforms = [
            scalar_transforms[t.id](*t.args) for t in request.metric.transforms
        ]
        column = compose_left(*transforms)(column)
        return MetricBranch(column=column, terminals=terminals)

    def get_transformed_metric_branch(
        self,
        request: QueryMetricRequest,
        scalar_filters: Optional[List[QueryDimension]] = None,
        inject_table_filters: Optional[List[Operator]] = None,
    ) -> MetricBranch:
        """Get a metric calculation graph for a transformed metric"""
        if scalar_filters is None:
            scalar_filters = []
        if inject_table_filters is None:
            inject_table_filters = []
        metric_builder: MetricsGraphBuilder = self.transforms[
            request.metric.transform.id
        ].from_query_metric(
            model=self.model,
            metric=request.metric,
            scalar_filters=scalar_filters,
            inject_table_filters=inject_table_filters,
        )
        graph = metric_builder.get_graph()
        column = Column(
            name=request.name,
            expr=Tree("expr", [Tree("column", [None, request.name])]),
        )
        transforms = [
            scalar_transforms[t.id](*t.args) for t in request.metric.transforms
        ]
        column = compose_left(*transforms)(column)
        return MetricBranch(column=column, terminals=[graph])

    def get_table_filter_graphs(
        self, metrics: List[QueryMetric], inject_table_filters: List[Operator] = None
    ) -> List[Operator]:
        if inject_table_filters is None:
            inject_table_filters = []

        result = []
        for metric in metrics:
            if metric.transform is None:
                raise ShoudntHappenError("Filtering on untransformed metric")
            metric_builder: MetricsGraphBuilder = self.transforms[
                metric.transform.id
            ].from_query_metric(
                model=self.model,
                metric=metric,
                scalar_filters=self.context_scalar_filters,
                inject_table_filters=inject_table_filters,
            )
            field_dimensions = {
                d.name: d for d in chain(metric.transform.of, metric.transform.within)
            }
            result.append(
                GenerateRecordsOperator(
                    input=metric_builder.get_graph(), field_dimensions=field_dimensions
                )
            )
        return result

    def get_final_dimension_columns(self):
        result = []
        for request in self.dimensions:
            column = Column(
                name=request.name,
                expr=Tree("expr", [Tree("column", [None, request.name])]),
            )
            result.append(column)
        return result

    def get_graph(self):
        context_filter_graphs = self.get_table_filter_graphs(self.context_table_filters)
        query_filter_graphs = self.get_table_filter_graphs(
            self.table_filters, inject_table_filters=context_filter_graphs
        )

        # generate graph branches for metrics
        # normal metrics have measures as terminals
        # transformed metrics have metrics as terminals
        metric_branches: List[MetricBranch] = []
        for request in self.metrics:
            if request.metric.transform is None and self.is_transform:
                raise ShoudntHappenError(
                    "Metric transform got an untransformed metric request"
                )
            if request.metric.transform is None or self.is_transform:
                metric_branches.append(
                    self.get_base_metric_branch(
                        request,
                        scalar_filters=[
                            *self.context_scalar_filters,
                            *self.scalar_filters,
                        ],
                        inject_table_filters=[
                            *self.inject_table_filters,
                            *context_filter_graphs,
                            *query_filter_graphs,
                        ],
                    )
                )
            else:
                metric_branches.append(
                    self.get_transformed_metric_branch(
                        request,
                        scalar_filters=[*self.context_scalar_filters],
                        inject_table_filters=[
                            *self.inject_table_filters,
                            *context_filter_graphs,
                        ],
                    )
                )

        merge = MergeOperator(merge_on=[d.name for d in self.dimensions])
        for mg in metric_branches:
            for terminal in mg.terminals:
                merge.add_input(terminal)

        # TODO: calculate metric/dimensions columns
        #       metrics are calculated from prefixed_expr
        #       dimensions are generated as a real coalesce(.., .., ..) expr
        dimension_columns = self.get_final_dimension_columns()
        metric_columns = [br.column for br in metric_branches]
        calculate = CalculateOperator(
            merge, columns=[*dimension_columns, *metric_columns]
        )

        return calculate


@dataclass
class TopGraphBuilder(MetricsGraphBuilder):
    id: ClassVar[str] = "top"
    ascending: ClassVar[bool] = False

    """
    FILTER country WHERE revenue.top(5)
        SELECT revenue BY country
        | row_number() over (order by revenue desc) <= 5

    FILTER city WHERE revenue.top(5) WITHIN (country)
        SELECT revenue BY country, city
        | _rn = row_number() over (partition by country order by revenue desc) <= 5
    """

    def wrap_in_row_number(self, order_by: Tree, partition_by: List[Tree]) -> Tree:
        order_by = order_by.children[0]
        return Tree(
            "expr",
            [
                Tree(
                    "call_window",
                    [
                        "row_number",
                        Tree("partition_by", partition_by),  # partition by
                        Tree(
                            "order_by",
                            [Tree("order_by_item", [order_by, self.ascending])],
                        ),  # order by
                        None,  # window def
                    ],
                )
            ],
        )

    def wrap_in_le(self, expr: Tree) -> Tree:
        expr = expr.children[0]
        n_rows = self.metrics[0].metric.transform.args[0]
        return Tree("expr", [Tree("le", [expr, Token("INTEGER", str(n_rows))])])

    def get_graph(self):
        base = super().get_graph()
        metric = base.columns[-1]  # last column is the metric
        dimensions = base.columns[:-1]

        withins = set(w.digest for w in self.metrics[0].metric.transform.within)
        metric.expr = self.wrap_in_row_number(
            order_by=metric.expr,
            partition_by=[c.expr for c in dimensions if c.name in withins],
        )
        metric.expr = self.wrap_in_le(metric.expr)
        return base


@dataclass
class TotalGraphBuilder(MetricsGraphBuilder):
    """A total is just a metric calculated with OF and WITHIN as its' dimensions."""

    id: ClassVar[str] = "total"
