from typing import List, Optional

from pydantic import BaseModel

from dictum_core.schema.catalog.calculations import CatalogMetric


class Catalog(BaseModel):
    name: str
    description: Optional[str]
    metrics: List[CatalogMetric]
