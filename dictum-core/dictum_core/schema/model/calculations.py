from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field, root_validator

FormatKind = Literal[
    "number", "decimal", "percent", "currency", "date", "datetime", "string"
]


class FormatConfig(BaseModel):
    kind: FormatKind
    pattern: Optional[str]
    skeleton: Optional[str]
    currency: Optional[str]

    @root_validator(skip_on_failure=True)
    def validate_pattern_skeleton(cls, values):
        pat = values.get("pattern")
        skel = values.get("skeleton")
        if pat is not None and skel is not None:
            raise ValueError("pattern and skeleton options are mutually exclusive")
        if skel is not None and values["kind"] not in {"date", "datetime"}:
            raise ValueError(
                "skeletons can only be used with date and datetime formats"
            )
        return values


Format = Union[FormatKind, FormatConfig]


class Formatted(BaseModel):

    format: Optional[Format]


class Displayed(Formatted):
    name: str
    description: Optional[str]


class Calculation(BaseModel):
    type: str
    str_expr: str = Field(..., alias="expr")
    missing: Optional[Any] = None


class Aggregation(Calculation):
    type: str = "float"
    str_filter: Optional[str] = Field(alias="filter")
    str_time: Optional[str] = Field(alias="time")


class Measure(Aggregation):
    description: Optional[str]


class Metric(Displayed, Aggregation):
    table: Optional[str]  # this one is for metric-measures


class Dimension(Displayed, Calculation):
    union: Optional[str]


class DetachedDimension(Dimension):
    """Just a dimension not defined on a table, the user has to explicitly
    specify which table it is.
    """

    table: str


class DimensionsUnion(Displayed):
    type: str
    missing: Optional[Any]
