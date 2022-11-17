from datetime import datetime

from dictum_core.format import Format
from dictum_core.model.types import Type
from dictum_core.schema import FormatConfig

locale = "en_US"


def test_format_default_float():
    fmt = Format(locale=locale, type=Type(name="float"))
    assert fmt.format_value(1.0) == "1"
    assert fmt.format_value(None) == ""


def test_format_default_int():
    fmt = Format(locale=locale, type=Type(name="int"))
    assert fmt.format_value(1) == "1"
    assert fmt.format_value(None) == ""


def test_format_default_datetime():
    fmt = Format(locale=locale, type=Type(name="datetime", grain="year"))
    assert fmt.format_value(None) == ""

    dt = datetime(2022, 11, 17, 12, 33, 11)

    def _grain(grain: str):
        return Format(
            locale=locale, type=Type(name="datetime", grain=grain)
        ).format_value(dt)

    assert _grain("year") == "2022"
    assert _grain("quarter") == "Q4 2022"
    assert _grain("month") == "November 2022"
    assert _grain("week") == "week 47 of 2022"
    assert _grain("day") == "11/17/2022"
    assert _grain("hour") == "11/17/22, 12:33 PM"
    assert _grain("minute") == "11/17/22, 12:33 PM"
    assert _grain("second") == "Nov 17, 2022, 12:33:11 PM"


def test_format_default_bool():
    fmt = Format(locale=locale, type=Type(name="bool"))
    assert fmt.format_value(None) == ""
    assert fmt.format_value(True) == "True"


def test_format_default_str():
    fmt = Format(locale=locale, type=Type(name="str"))
    assert fmt.format_value(None) == ""
    assert fmt.format_value("abc") == "abc"


def test_d3_format():
    fmt = Format(
        locale=locale, type=Type(name="float"), config=FormatConfig(kind="percent")
    )
    assert fmt.d3_format == "01,.0%"
