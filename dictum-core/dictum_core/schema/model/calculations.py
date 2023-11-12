from typing import Any, Optional

from pydantic import Field

from dictum_core.schema.model.format import Formatted


class Displayed(Formatted):
    name: str
    description: Optional[str] = None
    type: str
    missing: Optional[Any] = None


class Calculation(Displayed):
    str_expr: str = Field(..., alias="expr")


class AggregateCalculation(Calculation):
    type: str = "float"
    str_filter: Optional[str] = Field(None, alias="filter")
    str_time: Optional[str] = Field(None, alias="time")


class Measure(AggregateCalculation):
    metric: bool = False


class Metric(AggregateCalculation):
    table: Optional[str] = None  # this one is for metric-measures


class Dimension(Calculation):
    union: Optional[str] = None


class DetachedDimension(Dimension):
    """Just a dimension not defined on a table, the user has to explicitly
    specify which table it is.
    """

    table: str


class DimensionsUnion(Displayed):
    pass
