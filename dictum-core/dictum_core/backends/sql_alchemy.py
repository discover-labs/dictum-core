import logging
from functools import cached_property
from typing import List, Optional, Union

import sqlparse
from dateutil.parser import isoparse
from lark import Transformer, Tree
from pandas import DataFrame, read_sql
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
from sqlalchemy.sql import Alias, Join, Select, Subquery

from dictum_core.backends.base import Backend, Compiler
from dictum_core.backends.mixins.arithmetic import ArithmeticCompilerMixin
from dictum_core.engine import Column
from dictum_core.exceptions import ShoudntHappenError

logger = logging.getLogger(__name__)


def _get_case_insensitive_column(obj, column: str):
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
    raise KeyError(f"Can't find column {column} in {obj}")


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
        return _get_case_insensitive_column(self._tables[identity], field)


class SQLAlchemyCompiler(ArithmeticCompilerMixin, Compiler):
    def column(self, _):
        """A no-op. SQLAlchemy columns need to know about tables,
        so this transformation is done beforehand with ColumnTransformer.
        """

    def DATETIME(self, value: str):
        return isoparse(value)

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
        return case(whens, else_=else_)

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
        return super().call_window(fn, args, partition, order, rows)

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

    def dateadd(self, part, interval, value):
        return func.dateadd(part, interval, value)

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

    def compile(self, expr: Tree, tables: dict):
        ct = ColumnTransformer(tables)
        return self.transformer.transform(ct.transform(expr))


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
    def metadata(self) -> MetaData:
        return MetaData(self.engine)

    def __str__(self):
        return repr(self.engine.url)

    def display_query(self, query: Select) -> str:
        return sqlparse.format(
            str(query.compile(bind=self.engine)),
            reindent=True,
            wrap_after=60,
        )

    def execute(self, query: Select) -> DataFrame:
        return read_sql(query, self.engine, coerce_float=True)

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

    def left_join(
        self,
        left: Select,
        right: Select,
        left_identity: str,
        right_identity: str,
        join_expr: Tree,
    ) -> Select:
        """
        Left can be
        - Select from an aliased table (inner: Alias)
        - Select from a previous step in adding the joins (inner: Join)
        - Select from the result of a merge (inner: Subquery)

        Right can be
        - Select from an aliased Table (inner: Alias)
        - Select from a dimension subaggregation (inner: Join)
        """
        tables = {}

        # deal with the left side
        inner_left = left.get_final_froms()[0]
        if isinstance(inner_left, Alias):  # just a table
            tables.update({left_identity: inner_left})
            left = inner_left
        elif isinstance(inner_left, Join):  # previously joined
            tables.update(_get_tables(inner_left))
            left = inner_left
        elif isinstance(inner_left, Subquery):  # merge result
            left = left.alias(left_identity)  # add another subq
            tables.update({left_identity: left})
        else:
            raise ShoudntHappenError(
                f"Unexpected inner type in left item of LEFT JOIN: {type(inner_left)}"
            )

        # TODO: factorize duplicate code
        # right side
        inner_right = right.get_final_froms()[0]
        if isinstance(inner_right, Alias):  # just a table
            right = inner_right.alias(right_identity)
            tables.update({right_identity: right})
        elif isinstance(inner_right, Join):  # subagg
            right = right.alias(right_identity)
            tables.update({right_identity: right})
        else:
            raise ShoudntHappenError(
                f"Unexpected inner type in right item of LEFT JOIN: {type(inner_right)}"
            )

        onclause = self.compile(join_expr, tables)
        return left.join(right, onclause=onclause, isouter=True).select()

    def aggregate(
        self, base: Select, groupby: List[Column], aggregate: List[Column]
    ) -> Select:
        """Aggregate. The input can be a left join result or a Table select"""
        tables = {}

        whereclause = base.whereclause

        inner = base.get_final_froms()[0]
        if isinstance(inner, Alias):
            tables.update({inner.name: inner})
            base = inner
        elif isinstance(inner, Join):
            tables.update(_get_tables(inner))
            base = inner
        else:
            raise ShoudntHappenError(f"Unknown AGGREGATE input type: {type(base)}")

        groupby_columns = []
        for column in groupby:
            groupby_columns.append((column.name, self.compile(column.expr, tables)))
        aggregate_columns = []
        for column in aggregate:
            aggregate_columns.append((column.name, self.compile(column.expr, tables)))

        select_columns = [*groupby_columns, *aggregate_columns]

        result = select(
            *(col.label(label) for label, col in select_columns)
        ).select_from(base)
        if whereclause is not None:
            result = result.where(whereclause)
        return result.group_by(*(col for _, col in groupby_columns))

    def filter(self, base: Select, condition: Tree, subquery: bool = False) -> Select:
        if subquery:
            base = base.subquery()
            tables = {None: base}
        else:
            tables = _get_tables(base)

        clause = self.compile(condition, tables)

        if subquery:
            return base.select().where(clause)

        return base.where(clause)

    def calculate(self, base: Select, columns: List[Column]) -> Select:
        selected_columns = []
        tables = {None: base}
        for column in columns:
            selected_columns.append(
                self.compile(expr=column.expr, tables=tables).label(column.name)
            )
        return base.with_only_columns(*selected_columns)

    def merge(self, bases: List[Select], on: List[str], left: bool = False):
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

        A left merge is the same, but with a left join
        """
        subqueries = [q.subquery() for q in bases]
        joined, *joins = subqueries

        coalesced = {k: v for k, v in joined.c.items() if k in on}

        full = not left

        for join in joins:
            # join on columns that are common between merge_on, the table that's being
            # joined and the merge_on columns that were present so far in the already
            # joined tables
            on = set(on) & set(coalesced) & set(join.c.keys())

            # build the join condition
            cond = (
                and_(*(coalesced[c] == join.c[c] for c in on))
                if len(on) > 0
                else true()
                # true() is when there's no merge_on
                # or no common columns just cross join
            )

            # add the new join
            joined = joined.outerjoin(join, cond, full=full)

            # update the coalesced column list
            for column in set(on) & set(join.c.keys()):
                if column in coalesced:
                    coalesced[column] = func.coalesce(coalesced[column], join.c[column])
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

    def limit(self, base: Select, limit: int) -> Select:
        return base.limit(limit)
