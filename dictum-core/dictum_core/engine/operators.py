import time
from datetime import date, datetime
from typing import Dict, FrozenSet, List, Optional, Union

import dateutil
from lark import Tree
from pandas import DataFrame, concat, isna, merge

from dictum_core import engine
from dictum_core.backends.base import Backend
from dictum_core.backends.pandas import PandasColumnTransformer, PandasCompiler
from dictum_core.engine.computation import Column
from dictum_core.engine.result import ExecutedQuery, Result
from dictum_core.schema import Query


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


def _set_index(df: DataFrame, columns: List[str]):
    index = list(set(df.columns) & set(columns))
    if index:
        return df.set_index(index)
    return df


class Operator:
    def __init__(self):
        super().__init__()
        self.result = None
        self.ready = False
        self.executed_queries = []

    def execute(self, backend: Backend):
        """Actual operator logic. Execute any upstreams."""
        raise NotImplementedError

    def get_result(self, backend: Backend):
        if self.ready:
            return self.result
        self.result = self.execute(backend)
        self.ready = True
        return self.result

    @property
    def level_of_detail(self) -> List[str]:
        raise NotImplementedError

    @property
    def types(self) -> Dict[str, str]:
        raise NotImplementedError

    @property
    def _dependencies(self) -> List["Operator"]:
        for name in dir(self):
            attr = getattr(self, name)
            if isinstance(attr, Operator):
                yield attr
            elif isinstance(attr, list) and all(isinstance(i, Operator) for i in attr):
                yield from attr

    def walk_graph(self):
        for dep in self._dependencies:
            yield from dep.walk_graph()
        yield self

    def create_graph_label(self, backend: Backend):
        return self.__class__.__name__.replace("Operator", "")

    def graph(self, backend: Backend, *, graph=None, format: str = "png"):
        import graphviz

        self_id = str(id(self))

        if graph is None:
            graph = graphviz.Digraph(format=format, strict=True)
            graph.node(self_id, label=self.create_graph_label(backend))

        for dependency in self._dependencies:
            dep_id = str(id(dependency))
            graph.node(dep_id, label=dependency.create_graph_label(backend))
            graph.edge(dep_id, self_id)
            dependency.graph(backend=backend, graph=graph)

        graph.graph_attr["rankdir"] = "LR"
        return graph


class MaterializeMixin:
    def materialize(self, inputs: list, backend: Backend) -> List[DataFrame]:
        """Materialize multiple queries on the backend, return a list of DataFrames."""
        results = []
        for input in inputs:  # TODO: parallelize
            if not isinstance(input, DataFrame):
                start = time.time()
                query = backend.display_query(input)
                input = backend.execute(input)
                self.executed_queries.append(
                    ExecutedQuery(query=query, time=time.time() - start)
                )
            results.append(input)
        return results


class QueryOperator(Operator):
    def __init__(self, input: engine.RelationalQuery):
        self.input = input
        self.limit: Optional[int] = None
        self.order: Optional[List[engine.LiteralOrderItem]] = None
        super().__init__()

    def execute(self, backend: Backend):
        result = backend.compile_query(self.input)
        if self.limit is not None:
            result = backend.limit(result, self.limit)
        if self.order is not None:
            result = backend.order(result, self.order)
        return result

    @property
    def level_of_detail(self) -> List[str]:
        return [c.name for c in self.input._groupby]

    @property
    def types(self) -> Dict[str, str]:
        return {c.name: c.type for c in self.input.columns}


class InnerJoinOperator(Operator, MaterializeMixin):
    def __init__(
        self,
        query: Operator,
        to_join: Operator,
    ):
        super().__init__()
        self.query = query
        self.to_join = to_join

    def execute(self, backend: Backend):
        query = self.query.get_result(backend)
        to_join = self.to_join.get_result(backend)

        if not isinstance(query, DataFrame) and not isinstance(to_join, DataFrame):
            return backend.inner_join(query, to_join, self.level_of_detail)

        # join in pandas
        query, to_join = self.materialize([query, to_join], backend)
        query = _set_index(query, self.level_of_detail)
        to_join = _set_index(to_join, self.level_of_detail)
        return merge(
            query,
            to_join,
            left_index=True,
            right_index=True,
            how="inner",
            suffixes=["", "__joined"],
        )[query.columns].reset_index()

    @property
    def level_of_detail(self) -> List[str]:
        return self.query.level_of_detail

    @property
    def types(self) -> Dict[str, str]:
        return self.query.types


class MergeOperator(Operator, MaterializeMixin):
    def __init__(
        self,
        query: Query,
        inputs: Optional[List[Union[QueryOperator, "MergeOperator"]]] = None,
    ):
        super().__init__()
        self.query = query
        self.inputs = inputs if inputs is not None else []
        self.dimensions: List[Column] = []
        self.metrics: List[Column] = []
        self.limit: Optional[int] = []
        self.order: Optional[List[engine.LiteralOrderItem]] = []

    @property
    def digest(self) -> str:
        return self.query.digest

    @property
    def digests(self) -> FrozenSet[str]:
        return frozenset(i.digest for i in self.inputs if isinstance(i, MergeOperator))

    @property
    def columns(self):
        return self.dimensions + self.metrics

    @property
    def level_of_detail(self) -> List[str]:
        result = set()
        for item in self.inputs:
            result |= set(item.level_of_detail)
        return list(result)

    @property
    def types(self) -> Dict[str, str]:
        return {c.name: c.type for c in self.columns}

    def add_query(self, query: QueryOperator):
        """It's important that dimension columns are added to the merge
        as they appear in the original queries if they are transformed.
        We need to keep the transformed format, name and other display info.
        So, adding these needs to happen automatically, without the engine
        handling getting them out of queries.
        """
        names = {c.name for c in self.dimensions}
        for column in query.input._groupby:
            if column.name not in names:
                self.dimensions.append(
                    Column(
                        name=column.name,
                        expr=Tree("expr", [Tree("column", [None, column.name])]),
                        type=column.type,
                        display_info=column.display_info,
                    )
                )
        self.inputs.append(query)

    def add_merge(self, merge: "MergeOperator"):
        """Add an input merge operator. Checks that the digest doesn't already exist
        in the inputs.
        """
        if merge.digest not in self.digests:
            self.inputs.append(merge)

    def calculate_pandas(self, input: List[list]) -> List[list]:
        column_transformer = PandasColumnTransformer({None: input})
        compiler = PandasCompiler()
        columns = []
        for column in self.columns:
            expr_with_columns = column_transformer.transform(column.expr)
            result = compiler.transformer.transform(expr_with_columns).rename(
                column.name
            )
            columns.append(result)

        result = concat(columns, axis=1)

        if self.order:
            result = result.sort_values(
                by=[i.name for i in self.order],
                ascending=[i.ascending for i in self.order],
            )
        if self.limit:
            result = result.head(self.limit)

        return result

    def merge_pandas(self, inputs: List[DataFrame]) -> DataFrame:
        """Merge multiple DataFrames with Pandas."""
        if len(inputs) == 1:
            return inputs[0]

        result, *rest = inputs
        for df in rest:
            on = list(set(self.level_of_detail) & set(result.columns) & set(df.columns))
            if on:
                result = merge(
                    result,
                    df,
                    left_on=on,
                    right_on=on,
                    how="outer",
                    suffixes=["", "__joined"],
                )
            else:
                result = merge(result, df, how="cross")

        return result

    def merge_queries(self, inputs: list, backend: Backend):
        """Merge multiple queries on the backend."""
        if len(inputs) == 1:
            return inputs[0]
        return backend.merge_queries(inputs, self.level_of_detail)

    def execute(self, backend: Backend) -> List[DataFrame]:
        """Merge multiple inputs. If backend implements merge method and there are
        multiple queries that aren't materialized, use that. If there are any results
        that are DataFrame, materialize the rest and merge in Pandas. Otherwise return
        as is.
        """
        results = []
        for input in self.inputs:
            results.append(input.get_result(backend))

        if len(results) > 1:
            dfs = list(filter(lambda x: isinstance(x, DataFrame), results))
            queries = list(filter(lambda x: not isinstance(x, DataFrame), results))

            result = None
            if len(dfs) > 0:
                materialized = self.materialize(queries, backend)
                result = self.merge_pandas([*dfs, *materialized])
            else:
                try:
                    result = self.merge_queries(queries, backend)
                except NotImplementedError:  # backend doesn't support merge
                    materialized = self.materialize(queries, backend)
                    result = self.merge_pandas(materialized)
        else:
            result = results[0]

        if isinstance(result, DataFrame):
            return self.calculate_pandas(result)

        result = backend.calculate(result, self.columns)
        if self.limit:
            result = backend.limit(result, self.limit)

        # TODO: support order in the query
        if self.order:
            result = backend.order(result, self.order)

        return result


class MaterializeOperator(Operator, MaterializeMixin):
    def __init__(self, inputs: List[Operator]):
        super().__init__()
        self.inputs = inputs

    def execute(self, backend: Backend):
        inputs = [i.get_result(backend) for i in self.inputs]
        return self.materialize(inputs, backend)

    @property
    def level_of_detail(self) -> List[str]:
        result = set()
        for item in self.inputs:
            result |= set(item.level_of_detail)
        return list(result)

    @property
    def types(self) -> Dict[str, str]:
        result = {}
        for item in self.inputs:
            result.update(item.types)
        return result


class FilterOperator(Operator):
    def __init__(self, input: Operator, conditions: List[Tree]):
        self.input = input
        self.conditions = conditions
        super().__init__()

    def execute(self, backend: Backend):
        input = self.input.get_result(backend)
        if not isinstance(input, DataFrame):
            return backend.filter(input, self.conditions)

        # pandas
        column_transformer = PandasColumnTransformer({None: input})
        compiler = PandasCompiler()
        for expr in self.conditions:
            condition = compiler.transformer.transform(
                column_transformer.transform(expr)
            )
            input = input[condition]
        return input

    @property
    def level_of_detail(self) -> List[str]:
        return self.input.level_of_detail

    @property
    def types(self) -> Dict[str, str]:
        return self.input.types


class RecordsFilterOperator(Operator):
    def __init__(
        self,
        query: Operator,
        materialized: MaterializeOperator,
        drop_last_column: bool = False,
    ):
        self.query = query
        self.materialized = materialized
        self.drop_last_column = drop_last_column
        super().__init__()

    def filter_pandas(self, df: DataFrame, filters: List[DataFrame]):
        result = df
        for f in filters:
            columns = list(set(f.columns) & set(result.columns))
            # in case there are duplicate tuples in columns subset
            _f = f[columns].drop_duplicates()
            result = merge(
                result, _f, how="inner", on=columns, suffixes=["", "___drop"]
            )[df.columns]
        return result

    def execute(self, backend: Backend):
        filters: List[DataFrame] = self.materialized.get_result(backend)
        if self.drop_last_column:
            filters = [f.iloc[:, :-1] for f in filters]

        result = self.query.get_result(backend)

        if isinstance(result, DataFrame):
            return self.filter_pandas(result, filters)

        records = [df.to_dict(orient="records") for df in filters]
        return backend.filter_with_records(result, records)

    @property
    def level_of_detail(self) -> List[str]:
        return self.query.level_of_detail

    @property
    def types(self) -> Dict[str, str]:
        return self.query.types


class FinalizeOperator(Operator):
    def __init__(self, input: MaterializeOperator, aliases: Dict[str, str]):
        super().__init__()
        if not isinstance(input, MaterializeOperator):
            raise TypeError(
                "FinalizeOperator expects an instance of MaterializeOperator as input"
            )
        self.input = input
        self.aliases = aliases

    def coerce_types(self, data: List[dict]):
        for row in data:
            for k, v in row.items():
                if row[k] is None:
                    continue
                if isna(row[k]):
                    row[k] = None
                else:
                    row[k] = _type_mappers[self.types[k].name](v)
        return data

    def execute(self, backend: Backend):
        """Pluck the materialized df, convert to dicts, coerce types."""
        data = self.input.get_result(backend)[0]
        data = self.coerce_types(
            data.rename(columns=self.aliases).to_dict(orient="records")
        )
        display_info = {}
        for column in self.input.inputs[0].columns:
            info = column.display_info
            if column.name in self.aliases:
                column.name = self.aliases[column.name]
                info.name = column.name
                info.keep_name = True
            display_info[column.name] = info
        return Result(
            data=data,
            executed_queries=self.get_executed_queries(),
            display_info=display_info,
        )

    def get_executed_queries(self) -> List[ExecutedQuery]:
        result = []
        for item in self.walk_graph():
            result.extend(item.executed_queries)
        return result

    @property
    def level_of_detail(self) -> List[str]:
        return self.input.level_of_detail

    @property
    def types(self) -> Dict[str, str]:
        result = {}
        for k, v in self.input.types.items():
            k = self.aliases.get(k, k)
            result[k] = v
        return result
