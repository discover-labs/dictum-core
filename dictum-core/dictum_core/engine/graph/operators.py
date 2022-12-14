from datetime import date, datetime
from hashlib import md5
from typing import Any, Dict, List, Optional, Union

import dateutil
from lark import Token, Tree
from pandas import DataFrame, concat, isna, merge

from dictum_core import model
from dictum_core.backends.base import Backend
from dictum_core.backends.pandas import PandasCompiler
from dictum_core.engine.computation import Column
from dictum_core.engine.query import QueryDimension
from dictum_core.engine.result import DisplayInfo, Result
from dictum_core.exceptions import ShoudntHappenError
from dictum_core.utils.expr import value_to_token


def _digest(s: str) -> str:
    return md5(s.encode("UTF-8")).hexdigest()


class Operator:
    def __init__(self):
        self.result = None
        self._executed = False
        self._upstreams = []
        self._downstreams = []
        for k, attr in self.__dict__.items():
            if k.startswith("_"):
                continue
            if isinstance(attr, Operator):
                attr._downstreams.append(self)
                self._upstreams.append(attr)
            if isinstance(attr, list):
                for item in attr:
                    if isinstance(item, Operator):
                        item._downstreams.append(self)
                        self._upstreams.append(item)

    def execute(self, backend: Backend):
        raise NotImplementedError

    def _materialize(self, backend: Backend, args) -> List[DataFrame]:
        result = []
        for arg in args:
            if isinstance(arg, DataFrame):
                result.append(arg)
            else:
                result.append(backend.execute(arg))
        return result

    def get_result(self, backend: Backend):
        if not self._executed:
            self.result = self.execute(backend)
        return self.result

    @property
    def label(self) -> str:
        return self.__class__.__name__.replace("Operator", "").upper()

    def walk_graph(self):
        for upstream in self._upstreams:
            yield from upstream.walk_graph()
        yield self

    @property
    def upstreams_digest(self) -> str:
        # must sort to avoid ordering issues
        return _digest(":".join(sorted(u.digest for u in self._upstreams)))

    @property
    def digest(self) -> str:
        self_digest = self.get_parameters_digest()
        return _digest(f"{self.upstreams_digest}:{self_digest}")

    def get_parameters_digest(self) -> str:
        raise NotImplementedError

    def graph(self, graph=None, format: str = "png"):
        import graphviz

        self_id = str(id(self))

        if graph is None:
            graph = graphviz.Digraph(
                format=format,
                strict=True,
                node_attr={
                    "fontname": "Monospace",
                    "shape": "box",
                    "fontsize": "9",
                    "height": "0",
                    "width": "0",
                },
                graph_attr={"rankdir": "BT"},
            )
            graph.node(self_id, label=f"{self.label}\n[{self.digest[:6]}]")

        for dependency in self._upstreams:
            dep_id = str(id(dependency))
            graph.node(dep_id, label=f"{dependency.label}\n[{dependency.digest[:6]}]")
            graph.edge(dep_id, self_id)
            dependency.graph(graph=graph)

        return graph

    def __hash__(self):
        return hash(self.digest)

    def __eq__(self, other: "Operator"):
        return self.digest == other.digest

    def __repr__(self):
        return f"{self.__class__.__name__}({self.digest})"

    def _repr_mimebundle_(self, *args, **kwargs):
        return self.graph()._repr_mimebundle_(*args, **kwargs)


class TableOperator(Operator):
    """Select a table"""

    def __init__(self, table: model.Table):
        self.table = table
        super().__init__()

    @property
    def label(self) -> str:
        return f"TABLE\n{self.table.id}"

    def get_parameters_digest(self) -> str:
        return _digest(self.table.id)

    def execute(self, backend: Backend):
        return backend.table(self.table.source, alias=self.table.id)


class LeftJoinOperator(Operator):
    """Left join a selectable to a table"""

    def __init__(
        self,
        left: Union[TableOperator, "MergeOperator"],
        left_identity: str,
        right: List[Union[TableOperator, "AggregateOperator"]],
        join_exprs: List[Tree],
        right_identities: List[str],
    ):
        self.left = left
        self.right = right
        self.join_exprs = join_exprs
        self.left_identity = left_identity
        self.right_identities = right_identities
        super().__init__()

    @property
    def label(self) -> str:
        idx = "\n".join(self.right_identities)
        return f"{self.left_identity} LEFT JOIN\n{idx}"

    def get_parameters_digest(self) -> str:
        right_idx = ":".join(self.right_identities)
        exprs = ":".join(_digest(str(e)) for e in self.join_exprs)
        return _digest(f"{self.left_identity}:{right_idx}:{exprs}")

    def execute(self, backend: Backend):
        result = self.left.get_result(backend)
        for right_identity, right, join_expr in zip(
            self.right_identities, self.right, self.join_exprs
        ):
            right_result = right.get_result(backend)
            result = backend.left_join(
                left=result,
                right=right_result,
                join_expr=join_expr,
                left_identity=self.left_identity,
                right_identity=right_identity,
            )
        return result


class FilterOperator(Operator):
    """Inject one or more conditions into the query (detailed)"""

    def __init__(
        self,
        input: Union[LeftJoinOperator, TableOperator],
        conditions: List[Tree],
        description: Optional[str] = None,
        subquery: bool = False,
    ):
        self.input = input
        self.conditions = conditions
        self.description = description
        self.subquery = subquery
        super().__init__()

    @property
    def label(self) -> str:
        if self.description is None:
            return super().label
        return f"FILTER\n{self.description}"

    def get_parameters_digest(self) -> str:
        return _digest(":".join(str(hash(e)) for e in self.conditions))

    def execute(self, backend: Backend):
        input_result = self.input.get_result(backend)
        result = input_result
        for condition in self.conditions:
            result = backend.filter(input_result, condition, subquery=self.subquery)
        return result


class RecordsFilterOperator(Operator):
    """Filter a query with materialized records from another query.

    Parameters:
        base: a query to filter
        filters: a list of GenerateRecordsOperator that will generate the filtering
                 records
        field_exprs: for each filters, there's a correspondence between the name of a
                     field in each record (dict key) and an expression in the
                     base query.
    """

    def __init__(
        self,
        base: Operator,
        filters: List["GenerateRecordsOperator"],
        field_exprs: List[Dict[str, Tree]],
    ):
        self.base = base
        self.filters = filters
        self.field_exprs = field_exprs
        super().__init__()

    @property
    def label(self) -> str:
        return f"FILTER WITH RECORDS\non {len(self.field_exprs)} dimensions"

    def get_parameters_digest(self) -> str:
        return ""

    def get_record_filter_expr(self, record: dict, exprs: Dict[str, Tree]) -> Tree:
        tree = None
        for k, v in record.items():
            if tree is None:
                tree = Tree("eq", [exprs[k], value_to_token(v)])
                continue
            tree = Tree(
                "AND",
                [tree, Tree("eq", [exprs[k], value_to_token(v)])],
            )

        return tree

    def get_records_filter_expr(
        self, records: List[dict], exprs: Dict[str, Tree]
    ) -> Optional[Tree]:
        tree = None
        for record in records:
            if tree is None:
                tree = self.get_record_filter_expr(record, exprs)
                continue
            tree = Tree("OR", [tree, self.get_record_filter_expr(record, exprs)])
        if tree is None:
            return Tree("expr", [Token("TRUE", "True")])
        return tree

    def execute(self, backend: Backend):
        base = self.base.get_result(backend)
        filters: List[List[Dict[str, Any]]] = [
            f.get_result(backend) for f in self.filters
        ]
        for filter_, exprs in zip(filters, self.field_exprs):
            expr = self.get_records_filter_expr(filter_, exprs)
            base = backend.filter(base, expr)
        return base


class GenerateRecordsOperator(Operator):
    """Generate records from a filtering query. We materialize a calculated metric query
    where the metric is the last column and it's boolean, so we just select all but the
    last columns, where the last column is True, returning a list of dicts.

    RecordsFilterOperator that will do the actual filtering needs to know corresponding
    dimension requests for each of the potential field in the records to be able to
    inject the filters.
    """

    def __init__(
        self, input: "CalculateOperator", field_dimensions: Dict[str, QueryDimension]
    ):
        self.input = input
        self.field_dimensions = field_dimensions
        super().__init__()

    def get_parameters_digest(self) -> str:
        return ""

    def execute(self, backend: Backend):
        base = self.input.get_result(backend)
        condition = Tree("expr", [Tree("column", [None, self.input.columns[-1].name])])
        result = backend.filter(base, condition=condition, subquery=True)

        if not isinstance(result, DataFrame):
            result = backend.execute(result)

        return result.iloc[:, :-1].to_dict(orient="records")


class AggregateOperator(Operator):
    def __init__(
        self,
        base: Union[TableOperator, LeftJoinOperator, FilterOperator],
        groupby: List[Column],
        aggregate: List[Column],
    ):
        self.base = base
        self.groupby = groupby
        self.aggregate = aggregate
        super().__init__()

    def execute(self, backend: Backend):
        base = self.base.get_result(backend)
        return backend.aggregate(
            base=base, groupby=self.groupby, aggregate=self.aggregate
        )

    @property
    def label(self) -> str:
        measures = ", ".join(c.name for c in self.aggregate)
        return f"AGGREGATE {measures}\nBY {len(self.groupby)} dimensions"

    def get_parameters_digest(self) -> str:
        columns = (*self.groupby, *self.aggregate)
        return _digest(":".join(c.name for c in columns))


class MergeOperator(Operator):
    def __init__(
        self,
        inputs: Optional[List[Union[AggregateOperator, "MergeOperator"]]] = None,
        merge_on: Optional[List[str]] = None,
        left: bool = False,
    ):
        self.inputs = inputs or []
        self.merge_on = merge_on or []
        self.left = left
        super().__init__()

    @property
    def label(self) -> str:
        kind = "LEFT " if self.left else "FULL "
        return f"{kind}MERGE on {len(self.merge_on)} columns"

    def get_parameters_digest(self) -> str:
        return _digest(":".join(self.merge_on))

    def add_aggregate(self, op: AggregateOperator):
        """Custom logic for adding aggregates.

        This is to avoid duplicating select queries for measures that are selected from
        the same anchor table and with the same filters.

        If two aggregates have the same upstreams, then the new aggregate expression is
        just appended to an existing aggregation.
        """
        for existing_op in self.inputs:
            if (
                isinstance(existing_op, AggregateOperator)
                and existing_op.upstreams_digest == op.upstreams_digest
            ):
                existing_op.aggregate.extend(op.aggregate)
                return

        # this is the first such aggregation, skip
        self.inputs.append(op)
        self._upstreams.append(op)

    def add_input(self, op):
        if op not in self.inputs:
            if isinstance(op, AggregateOperator):
                self.add_aggregate(op)
            else:
                self.inputs.append(op)
                self._upstreams.append(op)

    def execute(self, backend: Backend):
        results = []
        for input in self.inputs:
            results.append(input.get_result(backend))
        if len(results) == 1:
            return results[0]
        return backend.merge(results, on=self.merge_on, left=self.left)


class MaterializedLeftMergeOperator(Operator):
    def __init__(self, left: Operator, right: List[Operator], on: List[str]):
        self.left = left
        self.right = right
        self.on = on
        super().__init__()

    def get_parameters_digest(self) -> str:
        return _digest(":".join(self.on))

    def execute(self, backend: Backend) -> DataFrame:
        left, *right = self._materialize(
            backend,
            [
                self.left.get_result(backend),
                *(r.get_result(backend) for r in self.right),
            ],
        )
        result = left
        for df in right:
            on_columns = list(set(self.on) & set(result.columns) & set(df.columns))
            if len(on_columns) > 0:
                result = merge(
                    result,
                    df,
                    left_on=on_columns,
                    right_on=on_columns,
                    how="left",
                    suffixes=("", "__joined"),  # avoid name clashes
                )
            else:
                result = merge(result, df, how="cross", suffixes=("", "__joined"))
        return result


class CalculateOperator(Operator):
    def __init__(self, input: Operator, columns: List[Column]):
        self.input = input
        self.columns = columns
        super().__init__()

    @property
    def label(self) -> str:
        return f"CALCULATE\n{len(self.columns)} columns"

    def get_parameters_digest(self) -> str:
        columns = ":".join(sorted(c.name for c in self.columns))
        return _digest(f"{columns}")

    def calculate_pandas(self, df: DataFrame) -> DataFrame:
        compiler = PandasCompiler()
        tables = {None: df}
        result = []
        for column in self.columns:
            series = compiler.compile(column.expr, tables).rename(column.name)
            result.append(series)
        return concat(result, axis=1)

    def execute(self, backend: Backend):
        base = self.input.get_result(backend)
        if isinstance(base, DataFrame):
            return self.calculate_pandas(base)
        return backend.calculate(base, columns=self.columns)


def _date_mapper(v):
    if isinstance(v, date):
        return v
    if isinstance(v, str):
        return dateutil.parser.parse(v).date()
    if isinstance(v, datetime):
        return v.date()
    return v


def _datetime_mapper(v):
    if isinstance(v, datetime):
        return v
    if isinstance(v, date):
        return datetime.combine(v, datetime.min.time())
    if isinstance(v, str):
        return dateutil.parser.parse(v)
    return v


_type_mappers = {
    "bool": bool,
    "float": float,
    "int": int,
    "str": str,
    "date": _date_mapper,
    "datetime": _datetime_mapper,
}


class FinalizeOperator(Operator):
    """Materialize, convert DataFrame to List[dict], coerce types, add display info"""

    def __init__(self, input: Operator, display_info: Dict[str, DisplayInfo]):
        self.input = input
        self.display_info = display_info
        super().__init__()

    @property
    def names(self) -> List[str]:
        return {k: v.column_name for k, v in self.display_info.items()}

    def get_parameters_digest(self) -> str:
        return ""

    def coerce_types(self, data: List[dict]):
        types = {k: v.type for k, v in self.display_info.items()}
        for row in data:
            for k, v in row.items():
                if row[k] is None:
                    continue
                if isna(row[k]):
                    row[k] = None
                else:
                    row[k] = _type_mappers[types[k].name](v)
        return data

    def rename_fields(self, data: List[dict]):
        result = []
        for row in data:
            result.append({self.names[k]: v for k, v in row.items()})
        return result

    def execute(self, backend: Backend) -> Result:
        result = self.input.get_result(backend)

        # materialize if not materialized
        if not isinstance(result, DataFrame):
            result = backend.execute(result)

        result = result.to_dict(orient="records")
        result = self.coerce_types(result)
        result = self.rename_fields(result)
        return Result(
            data=result,
            display_info={self.names[k]: v for k, v in self.display_info.items()},
            executed_queries=[],  # TODO: add
        )


class LimitOperator(Operator):
    def __init__(self, input: Operator, limit: int):
        self.input = input
        self.limit = limit
        super().__init__()

    def execute(self, backend: Backend):
        result = self.input.get_result(backend)
        return backend.limit(result, self.limit)
