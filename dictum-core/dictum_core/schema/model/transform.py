from typing import Optional

from pydantic import Field

from dictum_core.schema.id import ID
from dictum_core.schema.model.format import Formatted
from dictum_core.schema.model.type import Type


class Transform(Formatted):
    id: ID
    name: str
    description: Optional[str]
    args: list = []
    str_expr: str = Field(..., alias="expr")
    return_type: Optional[Type]
