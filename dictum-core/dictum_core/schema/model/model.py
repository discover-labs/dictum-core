from typing import Dict, Optional

from pydantic import BaseModel

from dictum_core.schema.id import ID
from dictum_core.schema.model.calculations import (
    DetachedDimension,
    DimensionsUnion,
    Metric,
)
from dictum_core.schema.model.table import Table
from dictum_core.schema.model.transform import Transform

root_keys = {"tables", "metrics", "unions"}


class Model(BaseModel):
    name: str
    description: Optional[str]
    locale: str = "en_US"
    currency: str = "USD"

    dimensions: Dict[ID, DetachedDimension] = {}
    metrics: Dict[ID, Metric] = {}
    unions: Dict[ID, DimensionsUnion] = {}

    tables: Dict[ID, Table] = {}
    transforms: Dict[
        ID, Transform
    ] = {}  # ignored for now, TODO: load as LiteralTransform

    theme: Optional[dict]
