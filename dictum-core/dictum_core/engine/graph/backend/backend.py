import logging
from typing import Dict, List, Union

from lark import Transformer, Tree
from lark.exceptions import VisitError
from pandas import DataFrame, concat, merge
from sqlalchemy import Table, select
from sqlalchemy.sql import Alias, Join, Select, Subquery

from dictum_core.backends.pandas import PandasColumnTransformer, PandasCompiler
from dictum_core.backends.sqlite import SQLiteBackend
from dictum_core.engine import Column
from dictum_core.exceptions import ShoudntHappenError

logger = logging.getLogger(__name__)


def _get_tables(base: Union[Alias, Join, Select, Subquery]) -> dict:
    if isinstance(base, Alias):
        return {base.name: base}

    if isinstance(base, Select):
        return _get_tables(base.get_final_froms()[0])

    result = {}
    if isinstance(base, Join):
        if isinstance(base.left, Join):
            result.update(_get_tables(base.left))
        elif isinstance(base.left, Alias):
            result[base.left.name] = base.left

        result[base.right.name] = base.right
        return result

    raise ShoudntHappenError(
        f"Error in SQLAlchemy _get_tables: unhandled input type {type(base)}"
    )


def get_case_insensitive_column(obj, column: str):
    """SQL is case-insensitive, but SQLAlchemy isn't, because columns are stored
    in a dict-like structure, keys exactly as specified in the database. That's
    why this function is needed.
    """
    columns = obj.selected_columns if isinstance(obj, Select) else obj.columns
    result = None
    if column in columns:
        result = columns[column]
    for k, v in columns.items():
        if k.lower() == column.lower():
            result = v

    if result is None:
        raise KeyError(f"Can't find column {column} in {obj}")

    return result


class ColumnTransformer(Transformer):
    """Replaces columns in the expressions with actual column objects from the join."""

    def __init__(self, tables, visit_tokens: bool = True) -> None:
        self._tables = tables
        super().__init__(visit_tokens=visit_tokens)

    def column(self, children: list):
        *path, field = children
        if path == [None]:
            identity = None
        else:
            identity = ".".join(path)
        return get_case_insensitive_column(self._tables[identity], field)


class Backend(SQLiteBackend):
    def compile(self, expr: Tree, tables: Dict[str, Table]):
        """Compile an expr to SQL"""
        try:
            column_transformer = ColumnTransformer(tables)
            expr_with_columns = column_transformer.transform(expr)
        except VisitError as e:
            raise e.orig_exc
        return self.compiler.transformer.transform(expr_with_columns)

    def left_join(
        self,
        left: Union[Table, Select],
        right: Table,
        left_identity: str,
        right_identity: str,
        join_expr: Tree,
    ) -> Select:
        right = right.alias(right_identity)

        if isinstance(right, Subquery):
            tables = {right_identity: right}
        else:
            tables = _get_tables(right)

        if isinstance(left, Table):
            left = left.alias(left_identity)

        if isinstance(left, Select):
            left = left.get_final_froms()[0]

        tables.update(_get_tables(left))

        onclause = self.compile(join_expr, tables)
        return left.join(right, onclause=onclause, isouter=True).select()

    def aggregate(
        self, base: Select, groupby: List[Column], aggregate: List[Column]
    ) -> Select:
        tables = _get_tables(base)

        groupby_columns = []
        for column in groupby:
            groupby_columns.append((column.name, self.compile(column.expr, tables)))
        aggregate_columns = []
        for column in aggregate:
            aggregate_columns.append((column.name, self.compile(column.expr, tables)))

        select_columns = [*groupby_columns, *aggregate_columns]

        whereclause = base.whereclause
        base = base.get_final_froms()[0]

        result = select(
            *(col.label(label) for label, col in select_columns)
        ).select_from(base)
        if whereclause is not None:
            result = result.where(whereclause)
        return result.group_by(*(col for _, col in groupby_columns))

    def filter_pandas(self, df: DataFrame, condition: Tree) -> DataFrame:
        column_transformer = PandasColumnTransformer({None: df})
        compiler = PandasCompiler()
        filter_ = compiler.transformer.transform(
            column_transformer.transform(condition)
        )
        return df.loc[filter_]

    def filter(
        self, base: Union[Select, DataFrame], condition: Tree, subquery: bool = False
    ) -> Union[Select, DataFrame]:
        if isinstance(base, DataFrame):
            return self.filter_pandas(base, condition=condition)

        if subquery:
            base = base.subquery()
        tables = _get_tables(base)
        clause = self.compile(condition, tables)
        return base.where(clause)

    def calculate_pandas(self, df: DataFrame, columns: List[Column]):
        column_transformer = PandasColumnTransformer({None: df})
        compiler = PandasCompiler()
        result = []
        for column in columns:
            series = compiler.transformer.transform(
                column_transformer.transform(column.expr)
            ).rename(column.name)
            result.append(series)

        return concat(result, axis=1)

    def calculate(
        self, base: Union[Select, DataFrame], columns: List[Column]
    ) -> Select:
        if isinstance(base, DataFrame):
            return self.calculate_pandas(df=base, columns=columns)
        selected_columns = []
        tables = _get_tables(base)
        for column in columns:
            selected_columns.append(
                self.compile(expr=column.expr, tables=tables).label(column.name)
            )
        return base.with_only_columns(*selected_columns)

    def full_outer_join(self, bases: List[Select], on=List[str]):
        dfs = []
        for base in bases:
            if isinstance(base, DataFrame):
                dfs.append(base)
            else:
                dfs.append(self.execute(base))

        if len(dfs) == 1:
            return dfs[0]

        result, *rest = dfs
        for df in rest:
            join_columns = list(set(on) & set(result.columns) & set(df.columns))
            if join_columns:
                result = merge(
                    result,
                    df,
                    left_on=join_columns,
                    right_on=join_columns,
                    how="outer",
                    suffixes=["", "__joined"],
                )
            else:
                result = merge(result, df, how="cross")

        return result

    def table(self, source: Union[str, dict], alias: str) -> Select:
        if isinstance(source, str):
            return Table(source, self.metadata, autoload=True).alias(alias).select()
        return (
            Table(
                source["table"],
                self.metadata,
                schema=source.get("schema"),
                autoload=True,
            )
            .alias(alias)
            .select()
        )
