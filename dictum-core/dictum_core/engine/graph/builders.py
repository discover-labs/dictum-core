from dataclasses import dataclass, field
from itertools import chain
from typing import ClassVar, Dict, List, Optional, Tuple, Union

from lark import Token, Tree
from toolz import compose_left

from dictum_core.engine.computation import Column
from dictum_core.engine.graph.operators import (
    AggregateOperator,
    CalculateOperator,
    FilterOperator,
    GenerateRecordsOperator,
    LeftJoinOperator,
    LimitOperator,
    MergeOperator,
    Operator,
    RecordsFilterOperator,
    TableOperator,
)
from dictum_core.engine.graph.query import (
    QueryCube,
    QueryDimension,
    QueryFilterGroup,
    QueryMetric,
    QueryMetricDeclaration,
    QueryScalarFilter,
    QueryTableFilter,
)
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
        if alias.startswith("__subquery__"):
            # build an aggregate query
            # measure from subquery name
            measure_id = alias.replace("__subquery__", "")
            measure = self.model.measures[measure_id]

            dimension_id = f"__PK_{self.table.id}"
            # group the measure by previous table's PK
            pk_request = QueryDimension(id=dimension_id)
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
                                    [new_identity, pk_request.digest],
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
    dimensions: List[QueryDimension] = field(default_factory=list)
    join_dimensions: List[QueryDimension] = field(default_factory=list)
    scalar_filters: List[QueryDimension] = field(default_factory=list)

    table_filters: List[GenerateRecordsOperator] = field(default_factory=list)

    def get_dimension_column(self, measure: Measure, request: QueryDimension) -> Column:
        """Construct a Column with a transformed expression relative to
        a given measure's anchor table.
        """
        anchor = measure.table
        dimension = measure.dimensions[request.id]

        transforms = [
            self.model.scalar_transforms[t.id](*t.args)
            for t in request.scalar_transforms
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
        expr = compose_left(*transforms)(expr)
        return Column(name=request.digest, expr=expr)

    def get_graph(self, measure: Measure):
        anchor = measure.table
        measure_column = Column(name=measure.id, expr=measure.expr)

        base = TableOperator(anchor)

        # TODO: turn back into graph, unroll in LeftJoinOperator.execute without
        #       actually running any upstream joins

        # build the join tree on the anchor table
        join_tree = JoinedTable(model=self.model, table=anchor, identity=anchor.id)
        join_tree.add_join_paths(measure_column.join_paths)

        groupby = []
        for request in self.dimensions:
            column = self.get_dimension_column(measure, request)
            join_tree.add_join_paths(column.join_paths)
            groupby.append(column)

        # additional dimensions to join only (not group by), to be used later by
        # table-valued filters
        for dimension in self.join_dimensions:
            column = self.get_dimension_column(measure, dimension)
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
            column = self.get_dimension_column(measure, query_filter)
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
                digests = {}
                for digest, dimension in genrec.field_dimensions.items():
                    digests[digest] = self.get_dimension_column(measure, dimension).expr
                field_exprs.append(digests)
            base = RecordsFilterOperator(
                base, filters=self.table_filters, field_exprs=field_exprs
            )

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
    terminals: Union[List[AggregateOperator], List[CalculateOperator]]
    transformed: bool


@dataclass
class MetricsGraphBuilder:
    """Builds a graph to select a list of metrics, up to the terminal operator, which
    can be CalculateOperator, OrderOperator or LimitOperator.

    Parameters:
        model: Source model to get metric/dimension information from.
        cube: Query cube that includes the pipeline of filtering operations, like
              groups of scalar and aggregate filters, intermediate metric declarations.
        select: A list of QueryMetric/QueryDimension digests. Defines order of
                columns in the final CalulateOperator.
        metrics: A list of QueryMetric to be selected. Has to include metrics bot for
                 selection and ordering.
        dimensions: A list of QueryDimension to be selected.
        join_dimensions: A list of QueryDimension to join to the anchor table, but not
                         to group by, to be used by RecordsFilters' filtering
                         expressions.
        order_by: A list of QueryMetric/QueryDimension digests. Defines output's
                  sort order.
        inject_table_filters: A list of operators that will output additional tuples for
                              filtering queries, will be injected into measure graphs.
    """

    model: Model

    cube: QueryCube  # FIXME: decide what to do with "from"
    select: List[str]
    metrics: List[QueryMetric]

    dimensions: List[QueryDimension] = field(default_factory=list)
    order_by: List[str] = field(default_factory=list)
    limit: Optional[int] = None

    join_dimensions: List[QueryDimension] = field(default_factory=list)

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
        scalar_filters: List[QueryDimension],
        inject_table_filters: List[Operator],
        join_dimensions: List[QueryDimension],
    ) -> "MetricsGraphBuilder":
        metrics = [metric]
        dimensions = []
        if metric.window:
            dimensions = list(chain(metric.window.of, metric.window.within))
        scalar_filters = [QueryScalarFilter(dimension=d) for d in scalar_filters]
        return cls(
            model=model,
            cube=QueryCube(qualifiers=[QueryFilterGroup(filters=scalar_filters)]),
            select=[r.digest for r in chain(dimensions, metrics)],  # order is important
            metrics=metrics,
            dimensions=dimensions,
            inject_table_filters=inject_table_filters,
            join_dimensions=join_dimensions,
        )

    def get_metric_branch(
        self,
        metric: QueryMetric,
        scalar_filters: List[QueryDimension],
        inject_table_filters: List[Operator],
        join_dimensions: List[QueryDimension],
    ) -> MetricBranch:
        # basic untransformed vanilla metric
        if (
            metric.window is None
            and metric.table_transform is None
            or self.is_transform  # transforms require base branches
        ):
            return self.get_base_metric_branch(
                metric, scalar_filters, inject_table_filters, join_dimensions
            )

        # metric with window, but without a transform: same as total transform
        if metric.window is not None and metric.table_transform is None:
            return self.get_total_metric_branch(
                metric, scalar_filters, inject_table_filters, join_dimensions
            )

        # transformed metric
        return self.get_transformed_metric_branch(
            metric, scalar_filters, inject_table_filters, join_dimensions
        )

    def get_total_metric_branch(
        self,
        request: QueryMetric,
        scalar_filters: List[QueryDimension],
        inject_table_filters: List[Operator],
        join_dimensions: List[QueryDimension],
    ):
        terminal = TotalGraphBuilder.from_query_metric(
            model=self.model,
            metric=request,
            scalar_filters=scalar_filters,
            inject_table_filters=inject_table_filters,
            join_dimensions=join_dimensions,
        ).get_graph()
        expr = Tree("expr", [Tree("column", [None, request.digest])])
        return MetricBranch(
            column=Column(name=request.digest, expr=expr),
            terminals=[terminal],
            transformed=True,
        )

    def get_base_metric_branch(
        self,
        request: QueryMetric,
        scalar_filters: List[QueryDimension],
        inject_table_filters: List[Operator],
        join_dimensions: List[QueryDimension],
    ) -> MetricBranch:
        """Get a column and a list of measure graphs for a metric."""
        metric = self.model.metrics[request.id]
        measure_builder = MeasureGraphBuilder(
            model=self.model,
            dimensions=self.dimensions,
            scalar_filters=scalar_filters,
            table_filters=inject_table_filters,
            join_dimensions=join_dimensions,
        )
        terminals = []
        for measure in metric.measures:
            terminals.append(measure_builder.get_graph(measure))
        transforms = [
            scalar_transforms[t.id](*t.args) for t in request.scalar_transforms
        ]
        expr = compose_left(*transforms)(metric.merged_expr)
        column = Column(name=request.digest, expr=expr)
        return MetricBranch(column=column, terminals=terminals, transformed=False)

    def get_transformed_metric_branch(
        self,
        request: QueryMetric,
        scalar_filters: List[QueryDimension],
        inject_table_filters: List[Operator],
        join_dimensions: List[QueryDimension],
    ) -> MetricBranch:
        """Get a metric calculation graph for a transformed metric"""
        builder: MetricsGraphBuilder = self.transforms[
            request.table_transform.id
        ].from_query_metric(
            model=self.model,
            metric=request,
            scalar_filters=scalar_filters,
            inject_table_filters=inject_table_filters,
            join_dimensions=join_dimensions,
        )
        graph = builder.get_graph()
        transforms = [
            scalar_transforms[t.id](*t.args) for t in request.scalar_transforms
        ]
        expr = compose_left(*transforms)(
            Tree("expr", [Tree("column", [None, request.digest])])
        )
        column = Column(name=request.digest, expr=expr)
        return MetricBranch(column=column, terminals=[graph], transformed=True)

    def get_table_filter_graph(
        self,
        metric: QueryMetric,
        scalar_filters: List[QueryDimension],
        inject_table_filters: List[Operator],
        join_dimensions: List[QueryDimension],
    ) -> List[Operator]:
        builder_cls = MetricsGraphBuilder
        if metric.table_transform is not None:
            builder_cls = self.transforms[metric.table_transform.id]
        metric_builder: MetricsGraphBuilder = builder_cls.from_query_metric(
            model=self.model,
            metric=metric,
            scalar_filters=scalar_filters,
            inject_table_filters=inject_table_filters,
            join_dimensions=join_dimensions,
        )
        field_dimensions = {
            d.digest: d for d in chain(metric.window.of, metric.window.within)
        }
        return GenerateRecordsOperator(
            input=metric_builder.get_graph(), field_dimensions=field_dimensions
        )

    def left_merge_transformed_metrics(
        self, merge: MergeOperator, metric_branches: List[MetricBranch]
    ) -> MergeOperator:
        return MergeOperator(
            inputs=[merge, *[mb.terminals[0] for mb in metric_branches]],
            merge_on=merge.merge_on,
            left=True,
        )

    def get_graph(self):
        table_filters = [*self.inject_table_filters]
        scalar_filters = []
        join_dimensions = [*self.join_dimensions]
        declared_metrics: Dict[str, Operator] = {}

        # go through qualifiers and collect everything in order
        for item in self.cube.qualifiers:
            if isinstance(item, QueryMetricDeclaration):
                # itermediate metric declaration, build a branch with all the filters
                # collected from the pipeline so far
                declared_metrics[item.alias] = self.get_metric_branch(
                    metric=item.metric,
                    scalar_filters=[*scalar_filters],  # copy lists to avoid nasty stuff
                    inject_table_filters=[*table_filters],
                )
                continue

            # otherwise it's a QueryFilterGroup
            # don't add to global filters until all group filters are computed
            # to keep the AND semantics (filters in the same group don't filter
            # each other)
            group_scalar_filters = []
            group_table_filters = []
            group_join_dimensions = []
            for filter_ in item.filters:

                if isinstance(filter_, QueryScalarFilter):  # scalar filter
                    group_scalar_filters.append(filter_.dimension)

                elif isinstance(filter_, QueryTableFilter):  # table filter
                    # make sure that the filtering dimensions are present in the
                    # detailed queries
                    group_join_dimensions.extend(filter_.metric.window.of)

                    # get the graph
                    filter_graph = self.get_table_filter_graph(
                        metric=filter_.metric,
                        scalar_filters=[*scalar_filters],
                        inject_table_filters=[*table_filters],
                        join_dimensions=[*join_dimensions],
                    )
                    group_table_filters.append(filter_graph)

            scalar_filters.extend(group_scalar_filters)
            table_filters.extend(group_table_filters)
            join_dimensions.extend(group_join_dimensions)

        # generate graph branches for metrics
        metric_branches: List[MetricBranch] = []
        for request in self.metrics:
            metric_branches.append(
                self.get_metric_branch(
                    metric=request,
                    scalar_filters=[*scalar_filters],
                    inject_table_filters=[*table_filters],
                    join_dimensions=[*join_dimensions],
                )
            )

        merged = MergeOperator(merge_on=[d.digest for d in self.dimensions])

        # normal metric branches (always at least one is present) are merged
        transformed_branches = []
        for mb in metric_branches:
            if mb.transformed:
                transformed_branches.append(mb)
            else:
                for terminal in mb.terminals:
                    merged.add_input(terminal)

        # transformed metric branches are left-joined to the resulting merge,
        # because the dimension "spine" could be filtered, but the filters don't
        # always apply to transformed metrics
        if len(transformed_branches) > 0:
            merged = self.left_merge_transformed_metrics(merged, transformed_branches)

        # select/calculate according to the digests in select
        # build a dict of digest -> column
        digest_columns: Dict[str, Column] = {}
        for request in self.dimensions:
            digest_columns[request.digest] = Column(
                name=request.digest,
                expr=Tree("expr", [Tree("column", [None, request.digest])]),
            )
        digest_columns.update({mb.column.name: mb.column for mb in metric_branches})

        # TODO: get rid of the merge if there's only one input
        #       the problem is that the Calculate expects a finalized table, but the
        #       Aggregate still exposes the underlying tables.

        columns = [digest_columns[k] for k in self.select]
        terminal = CalculateOperator(merged, columns=columns)

        # FIXME: implement order by

        if self.limit is not None:
            terminal = LimitOperator(terminal, limit=self.limit)

        return terminal


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
        n_rows = self.metrics[0].table_transform.args[0]
        return Tree("expr", [Tree("le", [expr, Token("INTEGER", str(n_rows))])])

    def get_graph(self):
        base = super().get_graph()

        metric = base.columns[-1]  # last column is the metric
        dimensions = base.columns[:-1]

        withins = set(w.digest for w in self.metrics[0].window.within)
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


@dataclass
class PercentGraphBuilder(MetricsGraphBuilder):
    """Calculate a percentage"""

    id: ClassVar[str] = "percent"

    # percent need to know which queries are selected in the parent query :(
    # can do during preprocessing :clown: (see engine._prep_query_of_within)
    # bare percent: take dimensions to OF, divide by total-total
    # percent OF: within is the rest of the query
    # percent OF + WTIHIN: as it is
    # so: fill in during preprocessing and then
    #   - calculate metric by of + within
    #   - calculate total by within
    #   - divide

    def get_graph(self):
        base = super().get_graph()

        total_metric = self.metrics[0].copy(deep=True)
        total_metric.table_transform.id = "total"
        total_metric.table_transform.of = []

        total = TotalGraphBuilder.from_query_metric(
            model=self.model,
            metric=total_metric,
            scalar_filters=self.scalar_filters,
            inject_table_filters=self.inject_table_filters,
        ).get_graph()

        merged = MergeOperator(
            inputs=[base, total], merge_on=[d.digest for d in self.dimensions]
        )

        expr = Tree(
            "expr",
            [
                Tree(
                    "div",
                    [
                        Tree("column", [None, self.metrics[0].digest]),
                        Tree("column", [None, total_metric.digest]),
                    ],
                )
            ],
        )
        metric_column = Column(name=self.metrics[0].digest, expr=expr)
        columns = [*self.get_final_dimension_columns(), metric_column]
        return CalculateOperator(merged, columns=columns)
