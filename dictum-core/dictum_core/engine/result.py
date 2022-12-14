from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Union

from dictum_core.engine.graph.query import (
    QueryDimensionRequest,
    QueryMetricRequest,
    QueryTransform,
)
from dictum_core.format import Format
from dictum_core.model import Model
from dictum_core.model.scalar import transforms as scalar_transforms
from dictum_core.model.types import Type


@dataclass
class ExecutedQuery:
    query: str
    time: float


AltairTimeUnit = Literal[
    "year",
    "yearquarter",
    "yearmonth",
    "yearmonthdate",
    "yearmonthdatehours",
    "yearmonthdatehoursminutes",
    "yearmonthdatehoursminutesseconds",
]

DisplayColumnKind = Literal["metric", "dimension"]


grain_to_altair_time_unit = {
    "year": "year",
    "quarter": "yearquarter",
    "month": "yearmonth",
    "day": "yearmonthdate",
    "hour": "yearmonthdatehours",
    "minute": "yearmonthdatehoursminutes",
    "second": "yearmonthdatehoursminutesseconds",
}


@dataclass
class DisplayInfo:
    """Information for the displaying code:
    either data formatter or Altair
    """

    display_name: str
    column_name: str
    format: Format
    kind: Literal["metric", "dimension"]
    type: Optional[Type] = None
    keep_display_name: bool = False  # needed to keep the aliases from being transformed
    altair_time_unit: Optional[str] = None

    @staticmethod
    def transform_display_info(
        transforms: List[QueryTransform], info: "DisplayInfo"
    ) -> "DisplayInfo":
        """Run a pipeline of scalar transforms over display info"""
        for transform in transforms:
            info = scalar_transforms[transform.id](*transform.args).get_display_info(
                info
            )
        return info

    @classmethod
    def from_dimension_request(
        cls, model: Model, request: QueryDimensionRequest
    ) -> "DisplayInfo":
        dimension = model.dimensions[request.id]
        altair_time_unit = grain_to_altair_time_unit.get(dimension.type.grain)
        base = DisplayInfo(
            display_name=(dimension.name if request.alias is None else request.alias),
            column_name=request.name,
            type=dimension.type,
            format=dimension.format,
            kind="dimension",
            keep_display_name=(request.alias is not None),
            altair_time_unit=altair_time_unit,
        )
        return cls.transform_display_info(request.scalar_transforms, base)

    @classmethod
    def from_metric_request(
        cls, model: Model, request: QueryMetricRequest
    ) -> "DisplayInfo":
        metric = model.metrics[request.id]
        base = DisplayInfo(
            display_name=(metric.name if request.alias is None else request.alias),
            column_name=request.name,
            type=metric.type,
            format=metric.format,
            kind="metric",
            keep_display_name=(request.alias is not None),
        )
        if request.table_transform is not None:
            # TODO: figure out how to put this logic into graph builders
            if request.table_transform.id == "percent":
                base.type = Type(name="float")
                base.format = Format(
                    locale=base.format.locale,
                    default_currency=base.format.default_currency,
                    type=Type(name="float"),
                    config="percent",
                )
            elif request.table_transform.id in {"top", "bottom"}:
                base.type = Type(name="bool")
                base.format = Format(
                    locale=base.format.locale,
                    default_currency=base.format.default_currency,
                    type=Type(name="bool"),
                )
        return cls.transform_display_info(request.scalar_transforms, base)

    @classmethod
    def from_request(
        cls, model: Model, request: Union[QueryDimensionRequest, QueryMetricRequest]
    ) -> "DisplayInfo":
        if isinstance(request, QueryDimensionRequest):
            return cls.from_dimension_request(model, request)
        return cls.from_metric_request(model, request)


@dataclass
class Result:
    data: List[dict]
    display_info: Dict[str, DisplayInfo]
    executed_queries: List[ExecutedQuery] = field(default_factory=list)
