from typing import Literal, Optional, Union

from pydantic import BaseModel, root_validator

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
