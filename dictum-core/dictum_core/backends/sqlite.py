import warnings
from functools import cached_property
from inspect import signature
from typing import List

from pandas import DataFrame
from sqlalchemy import (
    Integer,
    String,
    and_,
    case,
    cast,
    create_engine,
    event,
    func,
    select,
)
from sqlalchemy.exc import SAWarning
from sqlalchemy.sql import Select

from dictum_core.backends.mixins.datediff import DatediffCompilerMixin
from dictum_core.backends.pandas import PandasBackendFallbackMixin
from dictum_core.backends.sql_alchemy import SQLAlchemyBackend, SQLAlchemyCompiler

trunc_modifiers = {
    "year": ["start of year"],
    "month": ["start of month"],
    "week": ["1 day", "weekday 1", "-7 days", "start of day"],
    "day": ["start of day"],
}

trunc_formats = {
    "hour": r"%Y-%m-%d %H:00:00",
    "minute": r"%Y-%m-%d %H:%M:00",
    "second": r"%Y-%m-%d %H:%M:%S",
}

part_formats = {
    "year": r"%Y",
    "month": r"%m",
    "week": r"%W",
    "day": r"%d",
    "hour": r"%H",
    "minute": r"%M",
    "second": r"%S",
}


class SQLiteCompiler(DatediffCompilerMixin, SQLAlchemyCompiler):
    def coalesce(self, *args):
        """SQLite coalesce doesn't support a single arg, so add None if there's
        only one arg
        """
        if len(args) == 1:
            args = [*args, None]
        return super().coalesce(*args)

    # there's not floor and ceil in SQLite
    def floor(self, arg):
        return case(
            (arg < 0, cast(arg, Integer) - 1),
            else_=cast(arg, Integer),
        )

    def ceil(self, arg):
        return case(
            (arg == cast(arg, Integer), arg),
            (arg > 0, cast(arg, Integer) + 1),
            else_=cast(arg, Integer),
        )

    def div(self, a, b):
        """Fix integer division semantics"""
        return a / self.tofloat(b)

    # there's not date/datetime type in SQLite, so casting won't work
    # use built-in conversion functions
    def todate(self, arg):
        return func.date(arg)

    def todatetime(self, arg):
        return func.datetime(arg)

    # datetrunc wizardry
    def datetrunc(self, part, arg):
        if part in trunc_modifiers:
            modifiers = trunc_modifiers[part]
            return func.datetime(arg, *modifiers)
        if part in trunc_formats:
            fmt = trunc_formats[part]
            return func.strftime(fmt, arg)
        if part == "quarter":
            year = self.datetrunc("year", arg)
            quarter_part = self.datepart_quarter(arg)
            return func.datetime(
                year, "start of year", cast((quarter_part - 1) * 3, String) + " months"
            )
        raise ValueError(
            "Valid values for datetrunc part are year, quarter, "
            "month, week, day, hour, minute, second — "
            f"got '{part}'."
        )

    # datepart wizardry (with string formatting)
    def datepart_quarter(self, arg):
        return cast(
            (func.strftime("%m", arg) + 2) / 3,
            Integer,
        )

    def datepart_dayofweek(self, arg):
        value = cast(func.strftime("%w", arg), Integer)
        return case({0: 7}, value=value, else_=value)

    def datepart(self, part, arg):
        fmt = part_formats.get(part)
        if fmt is not None:
            return cast(func.strftime(fmt, arg), Integer)
        if part == "quarter":
            return self.datepart_quarter(arg)
        if part in {"dow", "dayofweek"}:
            return self.datepart_dayofweek(arg)
        raise ValueError(
            "Valid values for datepart part are year, quarter, "
            "month, week, day, hour, minute, second — "
            f"got '{part}'."
        )

    # for day diff datediff mixin
    def datediff_day(self, start, end):
        start_day = func.julianday(self.datetrunc("day", start))
        end_day = func.julianday(self.datetrunc("day", end))
        return cast(end_day - start_day, Integer)

    def datediff(self, part, start, end):
        return super().datediff(part, start, end)

    def dateadd(self, part, interval, value):
        """In sqlite adding dates/datetimes is done this way:
        DATETIME(value, '+7 day')

        The only parts that don't work as-is are quarter and week. We have to replace
        them with 7 days and 3 months.
        """
        if part == "week":
            part = "day"
            interval = interval * 7

        if part == "quarter":
            part = "month"
            interval = interval * 3

        return func.datetime(value, f"{interval} {part}")

    def now(self):
        return func.datetime()

    def today(self):
        return func.date()


class SQLiteBackend(SQLAlchemyBackend, PandasBackendFallbackMixin):

    type = "sqlite"
    compiler_cls = SQLiteCompiler

    def __init__(self, database: str):
        super().__init__(drivername="sqlite", database=database)

    def mount_udfs(self, conn, rec):
        """Mount user-defined functions not supported natively by SQLite.
        Functions are implemented as static methods on the backend class.
        """
        for name in dir(self):
            if name.startswith("sqlite_udf_"):
                fn_name = name.replace("sqlite_udf_", "")
                attr = getattr(self, name)
                n_params = len(signature(attr).parameters)
                conn.create_function(fn_name, n_params, attr)

    @staticmethod
    def sqlite_udf_power(a, b):
        return a**b

    @cached_property
    def engine(self):
        """SQLite doesn't support connection pooling, so have to redefine this"""
        engine = create_engine(self.url)
        event.listen(engine, "connect", self.mount_udfs)
        return engine

    def execute(self, query: Select) -> DataFrame:
        # hide SQLAlchemy warnings about decimals
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SAWarning)
            return super().execute(query)

    def union_all_full_outer_join(
        self, left: Select, right: Select, on: List[str], left_join: bool = False
    ) -> Select:
        """Perform a full outer join on two tables using UNION ALL of two LEFT JOINs
        or just a left join. Merge "on" columns with COALESCE.
        """
        on_columns = (
            set(left.selected_columns.keys())
            & set(right.selected_columns.keys())
            & set(on)
        )

        left = left.subquery()
        right = right.subquery()

        onclause = True
        if len(on_columns) > 0:
            onclause = and_(*(left.c[c] == right.c[c] for c in on_columns))

        # on_columns: columns that should be coalesce'd
        # other columns: left alone in both
        coalesced_columns = [
            func.coalesce(left.c[name], right.c[name]).label(name)
            for name in on_columns
        ]
        other_left_columns = [c for c in left.c if c.name not in on_columns]
        other_right_columns = [c for c in right.c if c.name not in on_columns]
        other_columns = [*other_left_columns, *other_right_columns]

        first: Select = select(*coalesced_columns, *other_columns).select_from(
            left.join(right, onclause=onclause, isouter=True)
        )

        if left_join:
            return first

        join_check = list(left.c)[-1]  # last column is the metric
        second = (
            select(*coalesced_columns, *other_columns)
            .select_from(right.join(left, onclause=onclause, isouter=True))
            .where(join_check == None)  # noqa: E711
        )
        return first.union_all(second)

    def merge(self, bases: List[Select], on: List[str], left: bool = False):
        """SQLite doesn't support full outer join, so we have to emulate it with
        UNION ALL and LEFT JOIN
        """
        # materialize bases that aren't materialized yet
        result, *rest = bases
        for add in rest:
            result = self.union_all_full_outer_join(result, add, on, left_join=left)
        return result.subquery().select()
