from typing import Any, Optional

from pydantic import Field, validator

from dictum_core.schema.id import ID
from dictum_core.schema.model.format import Formatted
from dictum_core.schema.model.types import Type, resolve_type


class Displayed(Formatted):
    id: ID
    name: str
    description: Optional[str]
    type: Type
    missing: Optional[Any]

    @validator("type", pre=True)
    def resolve_type(cls, value):
        if isinstance(value, str):
            return resolve_type(value)
        return value


class Calculation(Displayed):
    str_expr: str = Field(..., alias="expr")


class AggregateCalculation(Calculation):
    type: Type = Type(name="float")
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
