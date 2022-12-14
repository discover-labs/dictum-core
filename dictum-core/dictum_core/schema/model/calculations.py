from typing import Any, Optional

from pydantic import Field

from dictum_core.schema.model.format import Formatted


class Displayed(Formatted):
    name: str
    description: Optional[str]
    type: str
    missing: Optional[Any]


class Calculation(Displayed):
    str_expr: str = Field(..., alias="expr")


class AggregateCalculation(Calculation):
    type: str = "float"
    str_filter: Optional[str] = Field(alias="filter")
    str_time: Optional[str] = Field(alias="time")


class Measure(AggregateCalculation):
    metric: bool = False


class Metric(AggregateCalculation):
    table: Optional[str]  # this one is for metric-measures


class Dimension(Calculation):
    union: Optional[str]


class DetachedDimension(Dimension):
    """Just a dimension not defined on a table, the user has to explicitly
    specify which table it is.
    """

    table: str


class DimensionsUnion(Displayed):
    pass
