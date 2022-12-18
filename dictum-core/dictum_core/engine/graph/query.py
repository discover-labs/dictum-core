from hashlib import md5
from typing import List, Literal, Optional, Union

from pydantic import BaseModel


class Base(BaseModel):
    @property
    def digest(self) -> str:
        return md5(self._get_digest_json().encode("UTF-8")).hexdigest()

    def _get_digest_json(self) -> str:
        return self.json(sort_keys=True)

    def __hash__(self):
        return hash(self.digest)

    def __eq__(self, other: "Base") -> bool:
        return self.digest == other.digest


class QueryTransform(Base):
    id: str
    args: list

    @property
    def name(self) -> str:
        return "_".join((self.id, *(str(a) for a in self.args)))


class QueryCalculation(Base):
    id: str
    scalar_transforms: List[QueryTransform] = []

    @property
    def name(self) -> str:
        """Generate a human-readable column name."""
        suffix = self.get_suffix()
        if not suffix:
            return self.id
        return f"{self.id}_{suffix}"

    def get_scalar_transforms_suffix(self) -> str:
        return "_".join(t.name for t in self.scalar_transforms)

    def get_suffix(self) -> str:
        scalars_suffix = self.get_scalar_transforms_suffix()
        if scalars_suffix:
            return f"_{scalars_suffix}"
        return ""


class QueryDimension(QueryCalculation):
    pass


class QueryMetricWindow(Base):
    of: List[QueryDimension] = []
    within: List[QueryDimension] = []

    @property
    def name(self) -> str:
        of = "_".join(d.name for d in self.of)
        within = "_".join(d.name for d in self.within)
        if of:
            of = f"of_{of}"
        if within:
            within = f"within_{within}"
        return "_".join([x for x in (of, within) if x])


class QueryMetric(QueryCalculation):
    table_transform: Optional[QueryTransform] = None
    window: Optional[QueryMetricWindow] = None

    def get_suffix(self) -> str:
        suffixes = [
            ("" if self.table_transform is None else self.table_transform.name),
            (self.window.name if self.window is not None else ""),
            self.get_scalar_transforms_suffix(),
        ]
        return "_".join(s for s in suffixes if s)


class QueryScalarFilter(Base):
    dimension: QueryDimension


class QueryTableFilter(Base):
    metric: QueryMetric


Filter = Union[QueryScalarFilter, QueryTableFilter]


class QueryFilterGroup(Base):
    filters: List[Filter]


class QueryMetricDeclaration(QueryMetric):
    alias: str

    @property
    def name(self) -> str:
        return self.alias


Qualifier = Union[QueryFilterGroup, QueryMetricDeclaration]


class QuerySource(Base):
    value: str
    kind: Literal["example", "path"]


class QueryCube(Base):
    source: Optional[QuerySource] = None  # from
    qualifiers: List[Qualifier] = []


class QueryDimensionRequest(QueryDimension):
    kind: Literal["dimension"] = "dimension"
    alias: Optional[str] = None

    def _get_digest_json(self) -> str:
        return self.json(sort_keys=True, exclude={"kind", "alias"})

    @property
    def name(self) -> str:
        if self.alias is not None:
            return self.alias
        return super().name


class QueryMetricRequest(QueryMetric):
    kind: Literal["metric"] = "metric"
    alias: Optional[str] = None

    @property
    def name(self) -> str:
        if self.alias is not None:
            return self.alias
        return super().name

    def _get_digest_json(self) -> str:
        return self.json(sort_keys=True, exclude={"kind", "alias"})


class QueryMetricOrderItem(QueryMetric):
    kind: Literal["metric"] = "metric"
    ascending: bool = False


class QueryDimensionOrderItem(QueryDimension):
    kind: Literal["dimension"] = "dimension"
    ascending: bool = False


class Query(Base):
    cube: QueryCube
    select: List[Union[QueryMetricRequest, QueryDimensionRequest]]
    order_by: List[Union[QueryMetricOrderItem, QueryDimensionOrderItem]] = []
    limit: Optional[int] = None
