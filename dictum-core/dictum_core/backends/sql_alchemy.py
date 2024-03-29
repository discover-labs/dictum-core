import logging
from functools import cached_property
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import sqlparse
from lark import Transformer, Tree
from sqlalchemy import (
    Date,
    DateTime,
    Float,
    Integer,
    MetaData,
    Table,
    and_,
    case,
    cast,
    create_engine,
    distinct,
    func,
    not_,
    or_,
    select,
    true,
)
from sqlalchemy.engine.url import URL
from sqlalchemy.sql import Select
from sqlalchemy.sql.functions import coalesce

import dictum_core.model
from dictum_core.backends.base import Backend, Compiler
from dictum_core.backends.mixins.arithmetic import ArithmeticCompilerMixin
from dictum_core.engine import Column, LiteralOrderItem, RelationalQuery

logger = logging.getLogger(__name__)


def get_case_insensitive_column(obj, column: str):
    """SQL is case-insensitive, but SQLAlchemy isn't, because columns are stored
    in a dict-like structure, keys exactly as specified in the database. That's
    why this function is needed.
    """
    columns = obj.selected_columns if isinstance(obj, Select) else obj.columns
    if column in columns:
        return columns[column]
    for k, v in columns.items():
        if k.lower() == column.lower():
            return v
    raise KeyError(f"Can't find column {column} in {obj.name}")


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


class SQLAlchemyCompiler(ArithmeticCompilerMixin, Compiler):
    def __init__(self, backend: "SQLAlchemyBackend"):
        self.backend = backend
        super().__init__()

    def column(self, _):
        """A no-op. SQLAlchemy columns need to know about tables,
        so this transformation is done beforehand with ColumnTransformer.
        """

    def IN(self, a, b):
        return a.in_(b)

    def NOT(self, x):
        return not_(x)

    def AND(self, a, b):
        return and_(a, b)

    def OR(self, a, b):
        return or_(a, b)

    def isnull(self, value):
        return value == None  # noqa: E711

    def case(self, whens, else_=None):
        return case(*(tuple(w) for w in whens), else_=else_)

    def sum(self, arg):
        return func.sum(arg)

    def avg(self, arg):
        return func.avg(arg)

    def min(self, arg):
        return func.min(arg)

    def max(self, arg):
        return func.max(arg)

    def count(self, arg=None):
        args = []
        if arg is not None:
            args = [arg]
        return func.count(*args)

    def countd(self, arg):
        return func.count(distinct(arg))

    def call_window(
        self, fn: str, args: list, partition: list, order: list, rows: list
    ):
        if order:
            order_by = []
            for item in order:
                col, asc = item.children
                col = col.asc() if asc else col.desc()
                order_by.append(col)
            order = order_by
        return super().call_window(fn, *args, partition, order, rows)

    def window_sum(self, arg, partition, order, rows):
        return func.sum(arg).over(partition_by=partition, order_by=order)

    def window_row_number(self, _, partition, order, rows):
        return func.row_number().over(partition_by=partition, order_by=order)

    def floor(self, arg):
        return func.floor(arg)

    def ceil(self, arg):
        return func.ceil(arg)

    def abs(self, arg):
        return func.abs(arg)

    def datepart(self, part, arg):
        return func.date_part(part, arg)

    def datetrunc(self, part, arg):
        return func.date_trunc(part, arg)

    def datediff(self, part, start, end):
        return func.datediff(part, start, end)

    def now(self):
        return func.now()

    def today(self):
        return self.todate(self.now())

    def coalesce(self, *args):
        return func.coalesce(*args)

    def tointeger(self, arg):
        return cast(arg, Integer)

    def tofloat(self, arg):
        return cast(arg, Float)

    def todate(self, arg):
        return cast(arg, Date)

    def todatetime(self, arg):
        return cast(arg, DateTime)

    def _table(self, source: Union[str, Dict]):
        if isinstance(source, str):
            return self.backend.table(source)
        if isinstance(source, dict):
            schema = source.get("schema")
            table = source.get("table")
            if table is None:
                raise ValueError(f"table is required for a {self.type} backend")
            return self.backend.table(table, schema)
        raise ValueError(f"Source must be a str or a dict for a {self.type} backend")

    def compile_query(self, query: RelationalQuery):
        if isinstance(query.source, dictum_core.model.Table):
            id_ = query.source.id
            table = self._table(query.source.source)
        elif isinstance(query.source, RelationalQuery):
            id_ = None
            table = self.compile_query(query.source)

        tables = {id_: table}

        # replaces column refs with actual SQLA column objs
        column_transformer = ColumnTransformer(tables)

        # join the tables
        for join in query.joins:
            if isinstance(join.right, RelationalQuery):
                right_table = self.compile_query(join.right)
            else:
                right_table = self._table(join.right.source)
            right_table = right_table.alias(join.right_identity)
            tables[join.right_identity] = right_table
            join_expr = self.transformer.transform(
                column_transformer.transform(join.expr)
            )
            table = table.join(right_table, join_expr, isouter=(not join.inner))

        # add calcs
        columns = {}
        for column in query.columns:
            columns[column.name] = self.transformer.transform(
                column_transformer.transform(column.expr)
            )
        groupby = []
        for expr in query.groupby:
            compiled = self.transformer.transform(column_transformer.transform(expr))
            groupby.append(compiled)

        stmt = (
            select(*(v.label(k) for k, v in columns.items()))
            .select_from(table)
            .group_by(*groupby)
        )

        # apply filters
        for expr in query.filters:
            condition = self.transformer.transform(column_transformer.transform(expr))
            stmt = stmt.where(condition)

        # apply order and limit
        # order only makes sense at this stage if there's a limit
        if query.limit and query.order:
            order_clauses = []
            for item in query.order:
                expr = self.transformer.transform(
                    column_transformer.transform(item.expr)
                )
                if not item.ascending:
                    expr = expr.desc()
                order_clauses.append(expr)
            stmt = stmt.order_by(*order_clauses).limit(query.limit)

        return stmt

    def merge_queries(self, queries: List[Select], merge_on: List[str]):
        """Merge multiple queries. Wrap each in subquery and then full outer join them.
        The join condition is constructed by chaining the coalesced statements:

        If A and B are dimension columns and x, y, z are measures:

        SELECT coalesce(coalesce(t1.A, t2.A), t3.A) as A,
            coalesce(coalesce(t1.b, t2.B), t3.B) as B,
            t1.x, t2.y, t3.z
        FROM (...) as t1
        FULL OUTER JOIN (...) as t2
            ON t1.A = t2.A
            AND t1.B = t2.B
        FULL OUTER JOIN (...) as t3
            ON coalesce(t1.A, t2.A) = t3.A
            AND coalesce(t1.B, t2.B) = t3.B
        """
        subqueries = [q.subquery() for q in queries]
        joined, *joins = subqueries

        coalesced = {k: v for k, v in joined.c.items() if k in merge_on}

        for join in joins:
            # join on columns that are common between merge_on, the table that's being
            # joined and the merge_on columns that were present so far in the already
            # joined tables
            on = set(merge_on) & set(coalesced) & set(join.c.keys())

            # build the join condition
            cond = (
                and_(*(coalesced[c] == join.c[c] for c in on))
                if len(on) > 0
                else true()
                # true() is when there's no merge_on
                # or no common columns just cross join
            )

            # add the new join
            joined = joined.outerjoin(join, cond, full=True)

            # update the coalesced column list
            for column in set(merge_on) & set(join.c.keys()):
                if column in coalesced:
                    coalesced[column] = coalesce(coalesced[column], join.c[column])
                else:
                    coalesced[column] = join.c[column]

        # at this point we just need to select the coalesced columns
        columns = []
        for k, v in coalesced.items():
            columns.append(v.label(k))

        # and any other columns that are not in the coalesced mapping
        for s in subqueries:
            columns.extend(c for c in s.c if c.name not in coalesced)

        return select(*columns).select_from(joined)

    def calculate(self, query: Select, columns: List[Column]) -> Select:
        result_columns = []
        subquery = query.subquery()
        transformer = ColumnTransformer({None: subquery})
        for column in columns:
            resolved_column = transformer.transform(column.expr)
            sql_expr = self.transformer.transform(resolved_column).label(column.name)
            result_columns.append(sql_expr)
        return select(*result_columns).select_from(subquery)

    def inner_join(self, query: Select, to_join: Select, join_on: List[str]):
        to_join = to_join.subquery()
        conditions = and_(
            *(
                query.selected_columns[col] == to_join.c[col]
                for col in join_on
                if col in to_join.c  # support uneven level of detail
            )
        )
        return query.join(to_join, conditions)

    def limit(self, query: Select, limit: int):
        return query.limit(limit)

    def order(self, query: Select, items: List[LiteralOrderItem]) -> Select:
        clauses = []
        for item in items:
            clause = query.selected_columns[item.name]
            clause = clause.asc() if item.ascending else clause.desc()
            clauses.append(clause)
        return query.order_by(*clauses)

    def filter(self, query: Select, conditions: List[Tree]) -> Select:
        query = query.subquery().select()
        column_transformer = ColumnTransformer({None: query})
        for expr in conditions:
            condition = self.transformer.transform(column_transformer.transform(expr))
            query = query.where(condition)
        return query

    def filter_with_records(self, query: Select, records: List[List[Dict[str, Any]]]):
        if len(records) == 0:
            return query

        for recordset in records:
            conditions = []
            for record in recordset:
                # some filter columns might be missing from the query
                # for example if the query calculated a total
                condition_list = [
                    query.selected_columns[k] == v
                    for k, v in record.items()
                    if k in query.selected_columns
                ]
                if len(condition_list) > 0:
                    conditions.append(and_(*condition_list))
            if len(conditions) > 0:
                query = query.where(or_(*conditions))

        return query


class SQLAlchemyBackend(Backend):
    compiler_cls = SQLAlchemyCompiler

    def __init__(
        self,
        pool_size: Optional[int] = None,
        default_schema: Optional[str] = None,
        **kwargs,
    ):
        self.pool_size = pool_size
        self.default_schema = default_schema
        super().__init__(pool_size=pool_size, default_schema=default_schema, **kwargs)

    @property
    def url(self) -> str:
        url_params = {
            k: v
            for k, v in self.parameters.items()
            if k not in {"default_schema", "pool_size"}
        }
        return URL.create(**url_params)

    @cached_property
    def engine(self):
        return create_engine(self.url, pool_size=self.pool_size)

    @cached_property
    def metadata(self):
        return MetaData()

    def __str__(self):
        return repr(self.engine.url)

    def display_query(self, query: Select) -> str:
        return sqlparse.format(
            str(query.compile(bind=self.engine)),
            reindent=True,
            wrap_after=60,
        )

    def execute(self, query: Select) -> pd.DataFrame:
        return pd.read_sql(query, self.engine, coerce_float=True)

    def table(self, name: str, schema: Optional[str] = None) -> Table:
        if schema is None:
            schema = self.default_schema
        return Table(name, self.metadata, schema=schema, autoload_with=self.engine)
