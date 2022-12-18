from typing import List

from dateutil.parser import isoparse
from lark import Lark, Token, Transformer, Tree

from dictum_core import grammars
from dictum_core.engine.graph.query import (
    Query,
    QueryCube,
    QueryDimension,
    QueryDimensionOrderItem,
    QueryDimensionRequest,
    QueryFilterGroup,
    QueryMetric,
    QueryMetricDeclaration,
    QueryMetricOrderItem,
    QueryMetricRequest,
    QueryMetricWindow,
    QueryScalarFilter,
    QuerySource,
    QueryTableFilter,
    QueryTransform,
)


class Preprocessor(Transformer):
    def IDENTIFIER(self, token: Token):
        return token.value

    def QUOTED_IDENTIFIER(self, token: Token):
        return token.value.strip('"')  # unquote

    def INT(self, token: Token):
        return int(token.value)

    def UINT(self, token: Token):
        return int(token.value)

    def FLOAT(self, token: Token):
        return float(token.value)

    def PERCENTAGE(self, token: Token):
        return float(token.value.strip("%")) / 100

    def STRING(self, token: Token):
        return token.value.strip("'")

    def DATETIME(self, token: Token):
        return isoparse(token.value[1:])

    def requested_metric(self, children: list):
        return Tree("metric", children)

    def requested_dimension(self, children: list):
        return Tree("dimension", children)

    def identifier(self, children: list):
        return children[0]


stmt = Lark.open("ql_v2.lark", rel_to=grammars.__file__, start="stmt")
metric = Lark.open("ql_v2.lark", rel_to=grammars.__file__, start="metric")
dimension = Lark.open("ql_v2.lark", rel_to=grammars.__file__, start="dimension")
preprocessor = Preprocessor()


def parse_stmt(text: str) -> Tree:
    return preprocessor.transform(stmt.parse(text))


def parse_metric(text: str) -> Tree:
    return preprocessor.transform(metric.parse(text))


def parse_dimension(text: str) -> Tree:
    return preprocessor.transform(dimension.parse(text))


class TransformInjector:
    def __init__(self):
        self.name = None

    def __set_name__(self, owner, name: str):
        self.name = name

    def __get__(self, obj, objtype=None):
        return self

    def __call__(self, children: list):
        dim, *args = children
        dim.scalar_transforms.append(QueryTransform(id=self.name, args=args))
        return dim


class Compiler(Transformer):

    eq = TransformInjector()
    ne = TransformInjector()
    gt = TransformInjector()
    ge = TransformInjector()
    lt = TransformInjector()
    le = TransformInjector()
    isin = TransformInjector()

    def stmt(self, children: list):
        return children[0]

    def method(self, children: list):
        id_, *args = children
        return QueryTransform(id=id_, args=args)

    def window(self, children: List[Tree]):
        if len(children) == 0:
            return None
        kwargs = {}
        for item in children:
            kwargs[item.data] = item.children
        return QueryMetricWindow(**kwargs)

    def dimension(self, children: list):
        id_, *transforms = children
        return QueryDimension(id=id_, scalar_transforms=transforms)

    def dimension_request(self, children: list):
        *alias, dimension = children
        alias = alias[0].children[0] if alias else None
        return QueryDimensionRequest.parse_obj({**dimension.dict(), "alias": alias})

    def metric(self, children: list):
        id_, *rest = children
        kwargs = {}
        for item in rest:
            if isinstance(item, QueryMetricWindow):
                kwargs["window"] = item
                continue
            if isinstance(item, QueryTransform):
                kwargs["table_transform"] = item
        return QueryMetric(id=id_, **kwargs)

    def metric_request(self, children: list):
        *alias, metric = children
        alias = alias[0].children[0] if alias else None
        return QueryMetricRequest.parse_obj({**metric.dict(), "alias": alias})

    def scalar_filter(self, children: list):
        return QueryScalarFilter(dimension=children[0])

    def table_filter_window(self, children: list):
        return QueryMetricWindow(within=children[0].children)

    def table_filter_metric(self, children: list):
        return self.metric(children)

    def table_filter_where(self, children: list):
        return children[0]

    def table_filter(self, children: list):
        of, metric = children
        if metric.window is None:
            metric.window = QueryMetricWindow()
        metric.window.of = of.children
        return QueryTableFilter(metric=metric)

    def filter_group(self, children: list):
        return QueryFilterGroup(filters=children)

    def declare(self, children: list):
        alias, metric = children
        return QueryMetricDeclaration(alias=alias, **metric.dict())

    def source(self, children: list):
        kind, value = children
        return QuerySource(kind=kind, value=value)

    def cube(self, children: list):
        source, *qualifiers = children
        return QueryCube(source=source, qualifiers=qualifiers)

    def metric_orderby_item(self, children: list):
        *order, metric = children
        ascending = (order[0] == "+") if order else True
        return QueryMetricOrderItem.parse_obj({**metric.dict(), "ascending": ascending})

    def dimension_orderby_item(self, children: list):
        *order, dimension = children
        ascending = (order[0] == "+") if order else True
        return QueryDimensionOrderItem.parse_obj(
            {**dimension.dict(), "ascending": ascending}
        )

    def query(self, children: list):
        cube, select, *rest = children
        limit = None
        order_by = []
        for item in rest:
            if item.data == "limit":
                limit = item.children[0]
            if item.data == "orderby":
                order_by = item.children
        return Query(
            cube=cube,
            select=select.children,
            order_by=order_by,
            limit=limit,
        )


compiler = Compiler()


def compile_metric(text: str) -> QueryMetric:
    return compiler.transform(parse_metric(text))


def compile_dimension(text: str) -> QueryDimension:
    return compiler.transform(parse_dimension(text))


def compile_query(text: str) -> Query:
    return compiler.transform(parse_stmt(text))
