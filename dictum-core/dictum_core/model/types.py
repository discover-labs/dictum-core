from dataclasses import dataclass
from typing import List, Literal, Optional

grain_grains = {
    "year": ["year"],
    "quarter": ["year", "quarter"],
    "month": ["year", "quarter", "month"],
    "week": ["year", "week"],
    "day": ["year", "quarter", "month", "week", "day"],
    "hour": ["year", "quarter", "month", "week", "day", "hour"],
    "minute": ["year", "quarter", "month", "week", "day", "hour", "minute"],
    "second": ["year", "quarter", "month", "week", "day", "hour", "minute", "second"],
}


grains = set(grain_grains)


@dataclass
class Type:
    name: Literal["bool", "int", "float", "datetime", "str"]
    grain: Optional[
        Literal["year", "quarter", "month", "week", "day", "hour", "minute", "second"]
    ] = None

    @property
    def grains(self) -> List[str]:
        if self.name != "datetime":
            return []
        return grain_grains[self.grain]


def resolve_type(name: str) -> Type:
    if name in {"bool", "int", "float", "str"}:
        return Type(name=name)
    if name == "date":
        return Type(name="datetime", grain="day")
    if name == "datetime":
        return Type(name="datetime", grain="second")
    if ":" in name:
        _type, grain = name.split(":", maxsplit=1)
        if grain not in grains:
            raise ValueError(f"Unknown datetime grain: {grain}")
        if _type not in {"datetime", "date"}:
            raise ValueError(
                "Time grains can be specified only for date and datetime types,"
                f" got grain for {_type}"
            )
        return Type(name="datetime", grain=grain)
    raise ValueError(f"Unknown type: {name}")
