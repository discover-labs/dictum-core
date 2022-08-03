from typing import Dict, Optional

from pydantic import BaseModel, root_validator, validator

from dictum_core.schema import utils
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

    set_metrics_ids = validator("metrics", allow_reuse=True, pre=True)(utils.set_ids)
    set_dimensions_ids = validator("dimensions", allow_reuse=True, pre=True)(
        utils.set_ids
    )
    set_tables_ids = validator("tables", allow_reuse=True, pre=True)(utils.set_ids)
    set_unions_ids = validator("unions", allow_reuse=True, pre=True)(utils.set_ids)
    set_transforms_ids = validator("transforms", allow_reuse=True, pre=True)(
        utils.set_ids
    )

    @root_validator(pre=True)
    def set_default_currency(cls, value: dict):
        currency = value.get("currency", "USD")

        def _set(calculation: dict):
            if calculation.get("format") == "currency":
                calculation["format"] = dict(kind="currency", currency=currency)
            if calculation.get("format", {}).get("kind") == "currency":
                calculation["format"].setdefault("currency", currency)

        for table in value.get("tables", {}).values():
            for key in ["measures", "dimensions"]:
                for calculation in table.get(key, {}).values():
                    _set(calculation)
        for key in ["metrics", "dimensions", "unions"]:
            for calculation in value.get(key, {}).values():
                _set(calculation)
        return value
