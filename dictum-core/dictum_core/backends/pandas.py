from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from lark import Transformer

from dictum_core.backends.base import Compiler
from dictum_core.backends.mixins.arithmetic import ArithmeticCompilerMixin
from dictum_core.backends.mixins.datediff import DatediffCompilerMixin
from dictum_core.engine import Column, LiteralOrderItem, RelationalQuery


class PandasColumnTransformer(Transformer):
    """Replace column references with pd.Series"""

    def __init__(
        self, tables: Dict[str, pd.DataFrame], visit_tokens: bool = True
    ) -> None:
        self._tables = tables
        super().__init__(visit_tokens=visit_tokens)

    def column(self, children: list):
        identity, column = children
        return self._tables[identity][column]


class PandasCompiler(ArithmeticCompilerMixin, DatediffCompilerMixin, Compiler):
    def column(self, table: str, name: str):
        """
        Not required, columns are replaced by pd.Series
        with PandasColumnTransformer
        """

    def IN(self, value, values):
        return value.isin(values)

    def NOT(self, value):
        return ~value

    def AND(self, a, b):
        return a & b

    def OR(self, a, b):
        return a | b

    def isnull(self, value):
        return value.isna()

    def case(self, *whens, else_=None):
        return pd.Series(
            np.select(*zip(*whens), default=else_),
            index=whens[0][0].index,
        )

    # built-in functions
    # aggregate

    def sum(self, arg):
        return arg.sum()

    def avg(self, arg):
        return arg.mean()

    def min(self, arg):
        return arg.min()

    def max(self, arg):
        return arg.max()

    def count(self, args: list):
        """Aggregate count, with optional argument"""
        raise NotImplementedError  # not needed for now

    def countd(self, arg):
        return arg.unique().shape[0]

    # window functions

    def window_sum(self, arg, partition, order, rows):
        if partition:
            return arg.groupby(partition).transform(sum)
        return arg.groupby(pd.Series(0, index=arg.index)).transform(sum)

    def window_row_number(self, args, partition, order, rows):
        if order is None and partition is None:
            return args[0].cumcount() + 1
        if order:
            cols, asc = zip(*(i.children for i in order))
            df = pd.concat([*cols], axis=1)
            df = df.sort_values(by=list(df.columns), ascending=asc)
        if partition == []:
            # create empty groupby
            partition = [pd.Series(data=0, index=df.index)]
        return df.groupby(partition).cumcount() + 1

    # scalar functions

    def abs(self, arg):
        return arg.abs()

    def floor(self, arg):
        return arg.floor()

    def ceil(self, arg):
        return arg.ceil()

    def coalesce(self, *args):
        result, *rest = args
        for item in rest:
            result = result.fillna(item)
        return result

    # type casting

    def tointeger(self, arg):
        return arg.astype(int)

    def tofloat(self, arg):
        return arg.astype(float)

    def todate(self, arg):
        return pd.to_datetime(arg).dt.round("D")

    def todatetime(self, arg):
        return pd.to_datetime(arg)

    # dates

    def datepart(self, part, arg):
        """Part of a date as an integer. First arg is part as a string, e.g. 'month',
        second is date/datetime.
        """
        return getattr(arg.dt, part)

    def datetrunc(self, part, arg):
        """Date truncated to a given part. Args same as datepart."""
        # TODO: support other parts
        mapping = {
            "year": "YS",
            "month": "MS",
            "day": "D",
        }
        return arg.dt.round(mapping[part])

    # for DatediffCompilerMixin
    def datediff_day(self, start, end):
        return (end - start).days

    def now(self):
        return datetime.now()

    def today(self):
        return datetime.today()

    # compilation

    def compile_query(self, query: RelationalQuery):
        raise NotImplementedError

    def merge_queries(self, queries: List, merge_on: List[str]):
        raise NotImplementedError

    def calculate(self, query, columns: List[Column]):
        raise NotImplementedError

    def filter(self, query, conditions: Dict[str, Any]):
        raise NotImplementedError

    def filter_with_records(self, query, records: List[List[Dict[str, Any]]]):
        raise NotImplementedError

    def inner_join(self, query, to_join, join_on: List[str]):
        raise NotImplementedError

    def limit(self, query, limit: int):
        raise NotImplementedError

    def order(self, query, items: List[LiteralOrderItem]):
        raise NotImplementedError
