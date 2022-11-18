from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field

from dictum_core.schema.id import ID
from dictum_core.schema.model.calculations import Dimension, Measure


class RelatedTable(BaseModel):
    str_table: str = Field(alias="table")
    foreign_key: str
    str_related_key: Optional[str] = Field(alias="related_key")


class Table(BaseModel):
    description: Optional[str]
    source: Union[str, Dict]
    primary_key: Optional[str]
    related: Dict[ID, RelatedTable] = {}
    measures: Dict[ID, Measure] = {}
    dimensions: Dict[ID, Dimension] = {}
    filters: List[str] = []
