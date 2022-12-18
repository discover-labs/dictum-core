from datetime import datetime
from typing import Dict, List

import numpy as np
from lark import Transformer, Tree
from pandas import DataFrame, Series, concat, merge, to_datetime

from dictum_core.backends.base import Compiler
from dictum_core.backends.mixins.arithmetic import ArithmeticCompilerMixin
from dictum_core.backends.mixins.datediff import DatediffCompilerMixin
from dictum_core.engine import Column


class PandasColumnTransformer(Transformer):
    """Replace column references with pd.Series"""

    def __init__(self, tables: Dict[str, DataFrame], visit_tokens: bool = True) -> None:
        self._tables = tables
        super().__init__(visit_tokens=visit_tokens)

    def column(self, children: list):
        *path, column = children
        if path == [None]:
            identity = None
        else:
            identity = ".".join(path)
        return self._tables[identity][column]


class PandasCompiler(ArithmeticCompilerMixin, DatediffCompilerMixin, Compiler):
    def column(self, table: str, name: str):
        """
        Not required, columns are replaced by pd.Series
        with PandasColumnTransformer
        """

    def DATETIME(self, value: str):
        raise NotImplementedError

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
        return Series(
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

    def window_sum(
        self, args: List[Series], partition: List[Series], order: List[Tree], rows
    ):
        arg = args[0]  # there's only one arg

        # running sum
        if order:
            by, ascending = zip(*(i.children for i in order))
            sort_df = DataFrame(by).T
            sort_cols = sort_df.columns.to_list()
            ix_order = sort_df.sort_values(
                sort_cols, ascending=ascending
            ).index.tolist()
            return arg.loc[ix_order].groupby(partition).cumsum()

        # unordered partitioned sum
        if partition:
            return arg.groupby(partition).transform(sum)

        # just a sum
        return arg.groupby(Series(0, index=arg.index)).transform(sum)

    def window_row_number(self, args, partition, order, rows):
        if order is None and partition is None:
            return args[0].cumcount() + 1
        if order:
            cols, asc = zip(*(i.children for i in order))
            df = concat([*cols], axis=1)
            df = df.sort_values(by=list(df.columns), ascending=asc)
        if partition == []:
            # create empty groupby
            partition = [Series(data=0, index=df.index)]
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
        return to_datetime(arg).dt.round("D")

    def todatetime(self, arg):
        return to_datetime(arg)

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

    def dateadd(self, part, interval, value):
        raise NotImplementedError

    # for DatediffCompilerMixin
    def datediff_day(self, start, end):
        return (end - start).days

    def now(self):
        return datetime.now()

    def today(self):
        return datetime.today()

    def compile(self, expr: Tree, tables: dict) -> Series:
        ct = PandasColumnTransformer(tables)
        return self.transformer.transform(ct.transform(expr))


class PandasBackendFallbackMixin:
    _pd_compiler = PandasCompiler()

    def filter_pandas(self, df: DataFrame, condition: Tree) -> DataFrame:
        filter_ = self._pd_compiler.compile(condition, {None: df})
        return df.loc[filter_]

    def full_outer_join_pandas(self, bases: List[DataFrame], on=List[str]) -> DataFrame:
        if len(bases) == 1:
            return bases[0]

        result, *rest = bases
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

    def left_join_pandas(
        self,
        left: DataFrame,
        right: DataFrame,
        join_expr: Tree,
    ):
        # pandas can't do joins on expressions
        # so we use a nasty hack by knowing that the expression is always an eq and by
        # ignoring table identities altogether
        # this only works because we know that materialied left joins happen only when
        # transformed metrics are left-joined to the dimension "spine"
        left_on = []
        right_on = []
        for eq in join_expr.find_data("eq"):
            lc, rc = eq.children
            left_on.append(lc.children[1])
            right_on.append(rc.children[1])
        return merge(left, right, left_on=left_on, right_on=right_on, how="left")

    def calculate_pandas(self, base: DataFrame, columns: List[Column]) -> DataFrame:
        tables = {None: base}
        result = []
        for column in columns:
            series = self._pd_compiler.compile(column.expr, tables).rename(column.name)
            result.append(series)

        return concat(result, axis=1)
