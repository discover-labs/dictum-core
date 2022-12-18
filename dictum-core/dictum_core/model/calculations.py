from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from lark import Transformer, Tree

import dictum_core.model
from dictum_core.format import Format
from dictum_core.model import utils
from dictum_core.model.expr import get_expr_kind, parse_expr
from dictum_core.model.scalar import ScalarTransformMeta, transforms_by_input_type
from dictum_core.model.time import GenericTimeDimension
from dictum_core.model.time import dimensions as time_dimensions
from dictum_core.model.types import Type
from dictum_core.utils import value_to_token


@dataclass
class Displayed:
    name: str
    description: str
    format: Optional[Format]


@dataclass
class Expression:
    str_expr: str

    @property
    def parsed_expr(self) -> Tree:
        return parse_expr(self.str_expr)

    @property
    def expr(self) -> Tree:
        raise NotImplementedError

    @property
    def join_paths(self) -> List[str]:
        result = []
        for ref in self.expr.find_data("column"):
            path = ref.children[1:-1]
            if path:
                result.append(path)
        return result


@dataclass(eq=False, repr=False)
class Calculation:
    """Parent class for measures and dimensions."""

    id: str
    type: Type
    missing: Optional[Any]

    @property
    def expr_tree(self) -> str:
        return self.expr.pretty()

    @property
    def kind(self) -> str:
        try:
            return get_expr_kind(self.parsed_expr)
        except ValueError as e:
            raise ValueError(
                f"Error in {self} expression {self.str_expr}: {e}"
            ) from None

    def prefixed_expr(self, prefix: List[str]) -> Tree:
        return utils.prefixed_expr(self.expr, prefix)

    def prepare_expr(self, prefix: List[str]) -> Tree:
        return utils.prepare_expr(self.expr, prefix)

    def prepare_range_expr(self, base_path: List[str]) -> Tuple[Tree, Tree]:
        return (
            Tree("call", ["min", self.prepare_expr(base_path)]),
            Tree("call", ["max", self.prepare_expr(base_path)]),
        )

    def __str__(self):
        return f"{self.__class__.__name__}({self.id})"

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id


@dataclass(eq=False, repr=False)
class TableCalculation(Calculation):
    table: "dictum_core.model.Table"


class DimensionTransformer(Transformer):
    def __init__(
        self,
        table: "dictum_core.model.Table",
        measures: Dict[str, "Measure"],
        dimensions: Dict[str, "Dimension"],
        visit_tokens: bool = True,
    ) -> None:
        self._table = table
        self._measures = measures
        self._dimensions = dimensions
        super().__init__(visit_tokens=visit_tokens)

    def column(self, children: list):
        return Tree("column", [self._table.id, *children])

    def measure(self, children: list):
        measure_id = children[0]
        return Tree("column", [self._table.id, f"__subquery__{measure_id}", measure_id])

    def dimension(self, children: list):
        dimension_id = children[0]
        dimension = self._dimensions[dimension_id]
        path = self._table.dimension_join_paths.get(dimension_id, [])
        return dimension.prefixed_expr(path).children[0]


@dataclass(eq=False, repr=False)
class Dimension(Displayed, TableCalculation, Expression):
    is_union: bool = False

    @property
    def expr(self) -> Tree:
        transformer = DimensionTransformer(
            self.table, self.table.measure_backlinks, self.table.allowed_dimensions
        )
        expr = transformer.transform(self.parsed_expr)
        if self.missing is not None:
            expr = Tree(
                "expr",
                [
                    Tree(
                        "call",
                        [
                            "coalesce",
                            expr.children[0],
                            value_to_token(self.missing),
                        ],
                    )
                ],
            )
        return expr

    @property
    def transforms(self) -> Dict[str, ScalarTransformMeta]:
        return transforms_by_input_type[self.type.name]


@dataclass
class TableFilter(Expression):
    table: "dictum_core.model.Table"

    @property
    def expr(self) -> Tree:
        expr = parse_expr(self.str_expr)
        transformer = DimensionTransformer(
            table=self.table,
            measures=self.table.measure_backlinks,
            dimensions=self.table.allowed_dimensions,
        )
        return transformer.transform(expr)

    def __str__(self) -> str:
        return f"TableFilter({self.str_expr}) on {self.table}"

    def __hash__(self):
        return hash(id(self))


@dataclass
class DimensionsUnion(Displayed, Calculation):
    def __str__(self):
        return f"Union({self.id})"


class MeasureTransformer(Transformer):
    def __init__(
        self,
        table: "dictum_core.model.Table",
        measures: Dict[str, "Measure"],
        dimensions: Dict[str, "Dimension"],
        visit_tokens: bool = True,
    ) -> None:
        self._table = table
        self._measures = measures
        self._dimensions = dimensions
        super().__init__(visit_tokens=visit_tokens)

    def column(self, children: list):
        return Tree("column", [self._table.id, *children])

    def measure(self, children: list):
        ref_id = children[0]
        measure = self._measures[ref_id]
        expr = measure.expr.children[0]
        if measure.missing is not None:
            expr = Tree("call", ["coalesce", expr, value_to_token(measure.missing)])
        return expr

    def dimension(self, children: list):
        dimension_id = children[0]
        path = self._table.dimension_join_paths[dimension_id]
        return self._dimensions[dimension_id].prefixed_expr(path).children[0]


@dataclass
class MeasureFilter(Expression):
    measure: "Measure"

    @property
    def expr(self) -> Tree:
        table = self.measure.table
        transformer = DimensionTransformer(
            table, table.measure_backlinks, table.allowed_dimensions
        )
        return transformer.transform(self.parsed_expr)

    def __str__(self) -> str:
        return f"Filter({self.str_expr} on {self.measure})"


@dataclass(repr=False)
class Measure(TableCalculation, Expression):
    model: "dictum_core.model.Model"
    str_filter: Optional[str] = None
    str_time: Optional[str] = None
    description: Optional[str] = None

    @property
    def expr(self) -> Tree:
        transformer = MeasureTransformer(
            self.table, self.table.measures, self.table.allowed_dimensions
        )
        return transformer.transform(self.parsed_expr)

    @property
    def filter(self) -> TableFilter:
        if self.str_filter is None:
            return None
        return MeasureFilter(measure=self, str_expr=self.str_filter)

    @property
    def time(self) -> Dimension:
        if self.str_time is not None:  # explicit time
            return self.table.allowed_dimensions[self.str_time]

    @property
    def dimensions(self) -> Dict[str, Dimension]:
        result = self.table.allowed_dimensions.copy()
        if self.str_time is not None:
            allowed_grains = set(self.time.type.grains)
            for TimeDimension in time_dimensions.values():
                if TimeDimension.grain in allowed_grains or TimeDimension.id == "Time":
                    dimension = TimeDimension(locale=self.model.locale)
                    result[dimension.id] = dimension
        return result

    def __eq__(self, other: "Metric"):
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)


class MetricTransformer(Transformer):
    def __init__(
        self,
        metrics: Dict[str, "Metric"],
        measures: Dict[str, "Measure"],
        visit_tokens: bool = True,
    ) -> None:
        self._metrics = metrics
        self._measures = measures
        super().__init__(visit_tokens=visit_tokens)

    def column(self, children: list):
        raise ValueError("Column references are not allowed in metrics")

    def dimension(self, children: list):
        raise ValueError("Dimension references are not allowed in metrics")

    def measure(self, children: list):
        ref_id = children[0]
        if ref_id in self._metrics:
            return self._metrics[ref_id].expr.children[0]
        if ref_id in self._measures:
            return Tree("measure", children)
        raise KeyError(f"reference {ref_id} not found")


@dataclass(repr=False)
class Metric(Displayed, Calculation, Expression):
    model: "dictum_core.model.Model"
    is_measure: bool = False

    @classmethod
    def from_measure(
        cls, measure: Measure, model: "dictum_core.model.Model"
    ) -> "Metric":
        return cls(
            model=model,
            id=measure.id,
            name=measure.name,
            description=measure.description,
            str_expr=f"${measure.id}",
            type=measure.type,
            format=measure.format,
            missing=measure.missing,
            is_measure=True,
        )

    @property
    def expr(self) -> Tree:
        metrics = self.model.metrics.copy()
        del metrics[self.id]
        transformer = MetricTransformer(metrics, self.model.measures)
        expr = transformer.transform(self.parsed_expr)
        if self.missing is not None:
            expr = Tree(
                "expr",
                [
                    Tree(
                        "call",
                        [
                            "coalesce",
                            expr.children[0],
                            value_to_token(self.missing),
                        ],
                    )
                ],
            )
        return expr

    @property
    def merged_expr(self) -> Tree:
        """Same as expr, but measures are selected as columns from the merged table"""
        expr = deepcopy(self.expr)
        for ref in expr.find_data("measure"):
            ref.data = "column"
            ref.children = [None, *ref.children]
        return expr

    @property
    def measures(self) -> List[Measure]:
        result = []
        for ref in self.expr.find_data("measure"):
            result.append(self.model.measures.get(ref.children[0]))
        return result

    @property
    def dimensions(self) -> List[Dimension]:
        ids = set.intersection(*(set(d for d in m.dimensions) for m in self.measures))
        return [
            d
            for d in self.model.dimensions.values()
            if d.id in ids and not d.id.startswith("__")
        ]

    @property
    def generic_time_dimensions(self) -> List[GenericTimeDimension]:
        """Return a list of generic time dimensions available for this metric. All of
        them are available if generic time is defined for all measures used here.
        """
        return list(
            sorted(
                (d for d in self.dimensions if isinstance(d, GenericTimeDimension)),
                key=lambda x: x.sort_order,
            )
        )

    @property
    def lineage(self) -> List[dict]:
        return list(self.model.get_lineage(self))
