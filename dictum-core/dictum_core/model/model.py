import dataclasses
from typing import Any, Dict, List, Optional, Union

from dictum_core import schema
from dictum_core.format import Format
from dictum_core.model.calculations import (
    Calculation,
    Dimension,
    DimensionsUnion,
    Expression,
    Measure,
    Metric,
    TableCalculation,
)
from dictum_core.model.scalar import transforms as scalar_transforms
from dictum_core.model.table import Table, TableFilter
from dictum_core.model.time import GenericTimeDimension
from dictum_core.model.time import dimensions as time_dimensions
from dictum_core.model.types import resolve_type

displayed_fields = {"name", "description", "missing"}

table_calc_fields = displayed_fields | {"str_expr"}


class Model:
    def __init__(self, name: Optional[str] = None) -> None:
        self.name = name
        self.description = None
        self.locale = "en_US"
        self.scalar_transforms = scalar_transforms
        self.theme = None
        self.currency = None

        self.tables: Dict[str, Table] = {}
        self.measures: Dict[str, Measure] = {}
        self.dimensions: Dict[str, Dimension] = {}
        self.metrics: Dict[str, Metric] = {}

    @classmethod
    def from_config(cls, model: schema.Model):
        from dictum_core.model.checks import check_model
        from dictum_core.schema.model.checks import check_model_config

        # TODO: turn on
        # check_model_config(model)

        obj = cls(name=model.name)
        obj.description = model.description
        obj.locale = model.locale
        obj.theme = model.theme
        obj.currency = model.currency

        # add unions
        for id_, union in model.unions.items():
            union_type = resolve_type(union.type)
            obj.dimensions[id_] = DimensionsUnion(
                **union.dict(include=displayed_fields),
                id=id_,
                format=Format(
                    type=union_type,
                    config=union.format,
                    locale=obj.locale,
                    default_currency=obj.currency,
                ),
                type=union_type,
            )

        # add all tables, their relationships and calculations
        for id_, config_table in model.tables.items():
            table = obj.add_table(
                **config_table.dict(
                    include={"source", "description", "primary_key", "filters"}
                ),
                id=id_,
            )

            for alias, related in config_table.related.items():
                table.add_related(
                    str_table=related.str_table,
                    related_key=related.str_related_key,
                    foreign_key=related.foreign_key,
                    alias=alias,
                    tables=obj.tables,
                )

            # add table dimensions
            for id_, dimension in config_table.dimensions.items():
                obj.add_dimension(
                    id=id_,
                    table=table.id,
                    format_config=dimension.format,
                    type=dimension.type,
                    **dimension.dict(include=table_calc_fields | {"union"}),
                )

            # add table measures
            for id_, measure in config_table.measures.items():
                obj.add_measure(
                    table=table.id,
                    id=id_,
                    type=measure.type,
                    str_expr=measure.str_expr,
                    str_filter=measure.str_filter,
                )

        # add metrics
        for id_, metric in model.metrics.items():
            obj.add_metric(
                id=id_,
                type=metric.type,
                format_config=metric.format,
                str_expr=metric.str_expr,
                str_filter=metric.str_filter,
                str_time=metric.str_time,
                name=metric.name,
                description=metric.description,
                missing=metric.missing,
                table=metric.table,
            )

        # add measure backlinks
        for table in obj.tables.values():
            for measure in table.measures.values():
                for target in table.allowed_join_paths:
                    target.measure_backlinks[measure.id] = table

        # add default time dimensions
        for id_, time_dimension in time_dimensions.items():
            obj.dimensions[id_] = time_dimension(locale=obj.locale)

        # TODO: turn on
        # check_model(obj)
        return obj

    def add_table(
        self,
        id: str,
        source: Union[str, Dict[str, str]],
        description: Optional[str] = None,
        primary_key: Optional[str] = None,
        filters: Optional[List[str]] = None,
    ) -> Table:
        result = Table(
            id=id, source=source, description=description, primary_key=primary_key
        )
        self.tables[result.id] = result
        if filters:
            result.filters = [TableFilter(str_expr=f, table=result) for f in filters]
        if primary_key is not None:
            # add a dimension for internal use
            self.add_dimension(
                table=result.id,
                id=f"__PK_{id}",
                name=f"__PK_{id}",
                str_expr=primary_key,
                type="int",
            )
        return result

    def add_measure(
        self,
        table: str,
        id: str,
        str_expr: str,
        description: Optional[str] = None,
        type: Optional[str] = None,
        missing: Optional[Any] = None,
        str_filter: Optional[str] = None,
        str_time: Optional[str] = None,
    ) -> Measure:
        if type is None:
            type = "float"
        type = resolve_type(type)
        table = self.tables.get(table)
        result = Measure(
            model=self,
            table=table,
            id=id,
            type=type,
            missing=missing,
            str_expr=str_expr,
            str_filter=str_filter,
            str_time=str_time,
            description=description,
        )
        table.measures[result.id] = result
        self.measures[result.id] = result
        return result

    def add_dimension(
        self,
        table: str,
        id: str,
        name: str,
        str_expr: str,
        type: str,
        description: Optional[str] = None,
        missing: Optional[Any] = None,
        format_config: Optional[schema.FormatConfig] = None,
        union: Optional[str] = None,
    ) -> Dimension:
        table = self.tables[table]
        type = resolve_type(type)
        result = Dimension(
            table=table,
            id=id,
            name=name,
            str_expr=str_expr,
            description=description,
            missing=missing,
            type=type,
            format=Format(
                type=type,
                config=format_config,
                locale=self.locale,
                default_currency=self.currency,
            ),
        )
        table.dimensions[result.id] = result
        if union is not None:
            if union in table.dimensions:
                raise KeyError(  # TODO: move to checks
                    f"Duplicate union dimension {union} " f"on table {table.id}"
                )
            union: DimensionsUnion = self.dimensions.get(union)
            table.dimensions[union.id] = dataclasses.replace(
                result,
                id=union.id,
                name=union.name,
                description=union.description,
                type=union.type,
                format=union.format,
                missing=union.missing,
                is_union=True,
            )
        self.dimensions[result.id] = result
        return result

    def add_metric(
        self,
        id: str,
        name: str,
        str_expr: str,
        description: Optional[str] = None,
        missing: Optional[Any] = None,
        type: Optional[str] = None,
        format_config: Optional[schema.FormatConfig] = None,
        str_filter: Optional[str] = None,
        str_time: Optional[str] = None,
        table: Optional[str] = None,
    ):
        if table is not None:
            # table is specified, add the measure first
            measure = self.add_measure(
                table=table,
                id=id,
                str_expr=str_expr,
                description=description,
                missing=missing,
                type=type,
                str_time=str_time,
                str_filter=str_filter,
            )
            str_expr = f"${measure.id}"

        # no, it's a real metric
        if type is None:
            type = "float"
        type = resolve_type(type)
        result = Metric(
            model=self,
            id=id,
            name=name,
            str_expr=str_expr,
            description=description,
            missing=missing,
            type=type,
            format=Format(
                locale=self.locale,
                type=type,
                config=format_config,
                default_currency=self.currency,
            ),
        )
        self.metrics[result.id] = result
        return result

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

    @property
    def expressions(self) -> List[Expression]:
        """List all child objects that are instances of Expression.
        Used in checks.
        """
        yield from self.metrics.values()
        yield from (
            d
            for d in self.dimensions.values()
            if not isinstance(d, GenericTimeDimension)
            and not isinstance(d, DimensionsUnion)
        )
        for table in self.tables.values():
            yield from table.filters
        for measure in self.measures.values():
            yield measure
            if measure.filter is not None:
                yield measure.filter

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

    def get_currencies_for_query(self, query):
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
