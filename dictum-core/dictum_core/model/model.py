import dataclasses
from typing import Dict, List, Optional

from dictum_core import schema
from dictum_core.format import Format
from dictum_core.model.calculations import (
    Calculation,
    Dimension,
    DimensionsUnion,
    Measure,
    Metric,
    TableCalculation,
)
from dictum_core.model.dicts import DimensionDict, MeasureDict, MetricDict
from dictum_core.model.scalar import transforms as scalar_transforms
from dictum_core.model.table import RelatedTable, Table, TableFilter
from dictum_core.model.time import dimensions as time_dimensions

displayed_fields = {"id", "name", "description", "missing"}

table_calc_fields = displayed_fields | {"str_expr"}


class Model:
    def __init__(self, model: schema.Model):
        self.name = model.name
        self.description = model.description
        self.locale = model.locale

        self.tables: Dict[str, Table] = {}
        self.measures = MeasureDict()
        self.dimensions = DimensionDict()
        self.metrics = MetricDict()
        self.scalar_transforms = scalar_transforms

        self.theme = model.theme

        # add unions
        for union in model.unions.values():
            self.dimensions[union.id] = DimensionsUnion(
                **union.dict(include=displayed_fields),
                format=Format(type=union.type, config=union.format, locale=self.locale),
                type=union.type,
            )

        # add all tables, their relationships and calculations
        for config_table in model.tables.values():
            table = self.create_table(config_table)
            self.tables[table.id] = table

            # add table dimensions
            for dimension in config_table.dimensions.values():
                self.add_dimension(dimension, table)

            # add table measures
            for measure in config_table.measures.values():
                self.add_measure(measure, table)

        # add detached dimensions
        for dimension in model.dimensions.values():
            table = self.tables[dimension.table]
            self.add_dimension(dimension, table)

        # add metrics
        for metric in model.metrics.values():
            self.add_metric(metric)

        # add measure backlinks
        for table in self.tables.values():
            for measure in table.measures.values():
                for target in table.allowed_join_paths:
                    target.measure_backlinks[measure.id] = table

        # add default time dimensions
        for id, time_dimension in time_dimensions.items():
            self.dimensions[id] = time_dimension(locale=self.locale)

    def create_table(self, table: schema.Table):
        result = Table(
            **table.dict(include={"id", "source", "description", "primary_key"})
        )
        for related in table.related.values():
            result.related[related.alias] = RelatedTable(
                parent=result,
                tables=self.tables,
                **related.dict(
                    include={"str_table", "str_related_key", "foreign_key", "alias"}
                ),
            )
        result.filters = [TableFilter(str_expr=f, table=result) for f in table.filters]
        return result

    def add_measure(self, measure: schema.Measure, table: Table) -> Measure:
        result = Measure(
            table=table,
            type=measure.type,
            format=Format(type=measure.type, config=measure.format, locale=self.locale),
            **measure.dict(include=table_calc_fields | {"str_filter", "str_time"}),
        )
        if measure.metric:
            self.metrics.add(Metric.from_measure(result, self))
        table.measures.add(result)
        self.measures.add(result)

    def add_dimension(self, dimension: schema.Dimension, table: Table) -> Dimension:
        result = Dimension(
            table=table,
            type=dimension.type,
            format=Format(
                locale=self.locale,
                type=dimension.type,
                config=dimension.format,
            ),
            **dimension.dict(include=table_calc_fields),
        )
        table.dimensions[result.id] = result
        if dimension.union is not None:
            if dimension.union in table.dimensions:
                raise KeyError(
                    f"Duplicate union dimension {dimension.union} "
                    f"on table {table.id}"
                )
            union = self.dimensions.get(dimension.union)
            table.dimensions[dimension.union] = dataclasses.replace(
                result,
                id=union.id,
                name=union.name,
                description=union.description,
                type=union.type,
                format=union.format,
                missing=union.missing,
                is_union=True,
            )
        self.dimensions.add(result)

    def add_metric(self, metric: schema.Metric):
        if metric.table is not None:
            # table is specified, treat as that table's measure
            table = self.tables.get(metric.table)
            measure = schema.Measure(
                **metric.dict(by_alias=True),
                metric=True,
            )
            return self.add_measure(measure, table)

        # no, it's a real metric
        self.metrics[metric.id] = Metric(
            model=self,
            type=metric.type,
            format=Format(locale=self.locale, type=metric.type, config=metric.format),
            **metric.dict(include=table_calc_fields),
        )

    def get_lineage(
        self, calculation: Calculation, parent: Optional[str] = None
    ) -> List[dict]:
        """Get lineage graph for a calculation. Returns a list of dicts like:

        {"id": "someid", "type": "Metric", "parent": "otherid"}
        """
        _id = f"{calculation.__class__.__name__}:{calculation.id}"
        yield {
            "id": _id,
            "name": calculation.id,
            "parent": parent,
            "type": calculation.__class__.__name__,
        }
        expr = calculation.parsed_expr
        table = None
        has_refs = False
        for k in ("metric", "measure", "dimension"):
            attr = f"{k}s"
            for ref in expr.find_data(k):
                has_refs = True
                yield from self.get_lineage(
                    getattr(self, attr)[ref.children[0]], parent=_id
                )
        if isinstance(calculation, TableCalculation):
            table = calculation.table.id
            prefix = [table] if table is not None else []
            for ref in expr.find_data("column"):
                has_refs = True
                _col = ".".join([*prefix, *ref.children])
                yield {
                    "id": _col,
                    "type": "Column",
                    "parent": _id,
                    "name": _col,
                }
            if not has_refs:
                yield {
                    "id": table,
                    "type": "Column",
                    "parent": _id,
                    "name": f"{table}.*",
                }

    # def suggest_metrics(self, query: schema.Query) -> List[Measure]:
    #     """Suggest a list of possible metrics based on a query.
    #     Only metrics that can be used with all the dimensions from the query
    #     """
    #     result = []
    #     query_dims = set(r.dimension.id for r in query.dimensions)
    #     for metric in self.metrics.values():
    #         if metric.id in query.metrics:
    #             continue
    #         allowed_dims = set(d.id for d in metric.dimensions)
    #         if query_dims < allowed_dims:
    #             result.append(metric)
    #     return sorted(result, key=lambda x: (x.name))

    # def suggest_dimensions(self, query: schema.Query) -> List[Dimension]:
    #     """Suggest a list of possible dimensions based on a query. Only dimensions
    #     shared by all measures that are already in the query.
    #     """
    #     dims = set(self.dimensions) - set(r.dimension.id for r in query.dimensions)
    #     for request in query.metrics:
    #         metric = self.metrics.get(request.metric.id)
    #         for measure in metric.measures:
    #             dims = dims & set(measure.table.allowed_dimensions)
    #     return sorted([self.dimensions[d] for d in dims], key=lambda x: x.name)

    # def get_range_computation(self, dimension_id: str) -> Computation:
    #     """Get a computation that will compute a range of values for a given
    #       dimension.
    #     This is seriously out of line with what different parts of computation mean,
    #     so maybe we need to give them more abstract names.
    #     """
    #     dimension = self.dimensions.get(dimension_id)
    #     table = dimension.table
    #     min_, max_ = dimension.prepare_range_expr([table.id])
    #     return Computation(
    #         queries=[
    #             AggregateQuery(
    #                 join_tree=AggregateQuery(table=table, identity=table.id),
    #                 aggregate={"min": min_, "max": max_},
    #             )
    #         ]
    #     )

    # def get_values_computation(self, dimension_id: str) -> Computation:
    #     """Get a computation that will compute a list of unique possible values for
    #     this dimension.
    #     """
    #     dimension = self.dimensions.get(dimension_id)
    #     table = dimension.table
    #     return Computation(
    #         queries=[
    #             AggregateQuery(
    #                 join_tree=AggregateQuery(table=table, identity=table.id),
    #                 groupby={"values": dimension.prepare_expr([table.id])},
    #             )
    #         ]
    #     )

    def get_currencies_for_query(self, query: schema.Query):
        currencies = set()
        for request in query.metrics:
            metric = self.metrics.get(request.metric.id)
            if metric.format.currency is not None:
                currencies.add(metric.format.currency)
        for request in query.dimensions:
            dimension = self.dimensions.get(request.dimension.id)
            if dimension.format.currency is not None:
                currencies.add(dimension.format.currency)
        return currencies
