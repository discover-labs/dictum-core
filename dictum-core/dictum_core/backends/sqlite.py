import warnings
from functools import cached_property
from typing import List

from pandas import DataFrame
from sqlalchemy import Integer, String, create_engine
from sqlalchemy.engine.url import URL
from sqlalchemy.exc import SAWarning
from sqlalchemy.sql import Select, case, cast, func

from dictum_core.backends.mixins.datediff import DatediffCompilerMixin
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


class SQLiteFunctionsMixin:
    def floor(self, arg):
        return case(
            (arg < 0, cast(arg, Integer) - 1),
            else_=cast(arg, Integer),
        )

    def ceil(self, arg):
        return case(
            (arg > 0, cast(arg, Integer) + 1),
            else_=cast(arg, Integer),
        )

    def todate(self, arg):
        return func.date(arg)

    def todatetime(self, arg):
        return func.datetime(arg)

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

    def datepart_quarter(self, arg):
        return cast(
            (func.strftime("%m", arg) + 2) / 3,
            Integer,
        )

    def datepart(self, part, arg):
        part = part.lower()
        fmt = part_formats.get(part)
        if fmt is not None:
            return cast(func.strftime(fmt, arg), Integer)
        if part == "quarter":
            return self.datepart_quarter(arg)
        raise ValueError(
            "Valid values for datepart part are year, quarter, "
            "month, week, day, hour, minute, second — "
            f"got '{part}'."
        )

    def datediff_day(self, start, end):
        start_day = func.julianday(self.datetrunc("day", start))
        end_day = func.julianday(self.datetrunc("day", end))
        return cast(end_day - start_day, Integer)

    def now(self):
        return func.datetime()

    def today(self):
        return func.date()


class SQLiteCompiler(SQLiteFunctionsMixin, DatediffCompilerMixin, SQLAlchemyCompiler):
    pass


class SQLiteBackend(SQLAlchemyBackend):

    type = "sqlite"
    compiler_cls = SQLiteCompiler

    def __init__(self, database: str):
        super().__init__(pool_size=None, default_schema=None, database=database)

    @property
    def url(self) -> str:
        return URL.create(drivername="sqlite", **self.parameters)

    @cached_property
    def engine(self):
        """SQLite doesn't support connection pooling, so have to redefine this"""
        return create_engine(self.url)

    def execute(self, query: Select) -> DataFrame:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SAWarning)
            return super().execute(query)

    def merge_queries(self, queries: List[Select], merge_on: List[str]):
        raise NotImplementedError
