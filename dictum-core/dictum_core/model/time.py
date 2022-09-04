from dictum_core.schema import FormatConfig, Type


class TimeDimension:
    def __init__(cls, name, bases, attrs):
        cls.id = name
        cls.name = name
        cls.type = attrs.get("type")
        cls.period = attrs.get("period")
        cls.pattern = attrs.get("pattern")
        cls.skeleton = attrs.get("skeleton")

    def __repr__(self):
        return self.name

    @property
    def format(self) -> FormatConfig:
        if self.skeleton:
            return FormatConfig(kind="datetime", skeleton=self.skeleton)
        if self.pattern:
            return FormatConfig(kind="datetime", pattern=self.pattern)
        raise ValueError  # this shouldn't happen


class BaseTimeDimension(metaclass=TimeDimension):
    type: Type
    pattern: str = None
    skeleton: str = None

    def __init__(self):
        raise ValueError("Time dimensions are singletons, don't instantiate them")


class Time(BaseTimeDimension):
    type: Type = Type(name="datetime", grain="second")
    period = None
    skeleton = "yyMMdHmmss"


class Year(BaseTimeDimension):
    type: Type = Type(name="datetime", grain="year")
    period = "year"
    pattern = "yy"


class Quarter(BaseTimeDimension):
    type: Type = Type(name="datetime", grain="quarter")
    period = "quarter"
    pattern = "qqqq yy"


class Month(BaseTimeDimension):
    type: Type = Type(name="datetime", grain="month")
    period = "month"
    skeleton = "MMMMy"


class Week(BaseTimeDimension):
    type: Type = Type(name="datetime", grain="week")
    period = "week"
    skeleton = "w Y"


class Day(BaseTimeDimension):
    type: Type = Type(name="datetime", grain="day")
    period = "day"
    skeleton = "yMd"


class Date(Day):
    type: Type = Type(name="datetime", grain="day")
    period = "day"


class Hour(BaseTimeDimension):
    type: Type = Type(name="datetime", grain="hour")
    period = "hour"
    skeleton = "yMd hh"


class Minute(BaseTimeDimension):
    type: Type = Type(name="datetime", grain="minute")
    period = "minute"
    skeleton = "yMd hm"


class Second(BaseTimeDimension):
    type: Type = Type(name="datetime", grain="second")
    period = "minute"
    period = "second"
    skeleton = "yMd hms"


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
