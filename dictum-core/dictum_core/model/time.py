from dictum_core.format import Format
from dictum_core.model.types import Type


class GenericTimeDimension:
    type: Type
    grain: str

    _sort_counter = 0

    def __init__(self, locale: str):
        self.locale = locale

    def __init_subclass__(cls):
        cls.id = cls.__name__
        cls.sort_order = GenericTimeDimension._sort_counter
        GenericTimeDimension._sort_counter += 1

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return str(self)

    @property
    def name(self) -> str:
        return str(self)

    @property
    def format(self) -> Format:
        return Format(locale=self.locale, type=self.type)

    @property
    def type(self) -> Type:
        return Type(name="datetime", grain=self.grain)

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)


class Time(GenericTimeDimension):
    grain = "second"


class Year(GenericTimeDimension):
    grain = "year"


class Quarter(GenericTimeDimension):
    grain = "quarter"


class Month(GenericTimeDimension):
    grain = "month"


class Week(GenericTimeDimension):
    grain = "week"


class Day(GenericTimeDimension):
    grain = "day"


class Date(Day):
    pass


class Hour(GenericTimeDimension):
    grain = "hour"


class Minute(GenericTimeDimension):
    grain = "minute"


class Second(GenericTimeDimension):
    grain = "second"


dimensions = {
    "Time": Time,
    "Year": Year,
    "Quarter": Quarter,
    "Month": Month,
    "Week": Week,
    "Day": Day,
    "Date": Date,
    "Hour": Hour,
    "Minute": Minute,
    "Second": Second,
}
