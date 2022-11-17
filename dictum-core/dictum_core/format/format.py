from datetime import datetime
from functools import lru_cache
from numbers import Number
from typing import Any, Optional

from babel import Locale
from babel.dates import format_datetime, match_skeleton
from babel.numbers import format_currency, format_decimal, format_percent

from dictum_core.format import d3
from dictum_core.model.types import Type
from dictum_core.schema.model.format import Format, FormatConfig

grain_skeletons = {
    "year": "y",
    "quarter": "yQQQ",
    "month": "MMMMy",
    "week": "yw",
    "day": "yMd",
}

cache = lru_cache(maxsize=None)


@cache
def get_locale(locale: str) -> Locale:
    return Locale(locale)


class BaseFormatKind:
    def __init__(self, locale: str):
        self.locale = locale

    @property
    def localedata(self) -> Locale:
        return get_locale(self.locale)

    def format_value(self, value: Any) -> str:
        raise NotImplementedError

    def get_d3_format(self) -> str:
        return None


class StringFormat(BaseFormatKind):
    def __init__(self, locale: str):
        super().__init__(locale)

    def format_value(self, value: str) -> str:
        return value


class BoolFormat(BaseFormatKind):
    def __init__(self, locale: str):
        super().__init__(locale)

    def format_value(self, value: bool) -> str:
        return str(value)  # TODO: format based on locale


class NumberFormat(BaseFormatKind):
    def __init__(self, locale: str, pattern: Optional[str] = None):
        super().__init__(locale)
        self.pattern = pattern

    def format_value(self, value: Number) -> str:
        return super().format_value(value)


class DecimalFormat(NumberFormat):
    def format_value(self, value: Number) -> str:
        return format_decimal(value, format=self.pattern, locale=self.locale)

    def get_d3_format(self) -> str:
        pattern = self.pattern
        if pattern is None:
            pattern = self.localedata.decimal_formats[None].pattern
        return d3.ldml_number_to_d3_format(pattern)


class PercentFormat(NumberFormat):
    def format_value(self, value: Number) -> str:
        if self.pattern is None:
            return format_percent(value, locale=self.locale)
        return format_decimal(value, format=self.pattern, locale=self.locale)

    def get_d3_format(self) -> str:
        pattern = self.pattern
        if pattern is None:
            pattern = self.localedata.percent_formats[None].pattern
        return d3.ldml_number_to_d3_format(pattern)


class CurrencyFormat(NumberFormat):
    def __init__(self, locale: str, currency: str, pattern: Optional[str] = None):
        super().__init__(locale, pattern)
        self.currency = currency

    def format_value(self, value: Number) -> str:
        return format_currency(
            value, currency=self.currency, format=self.pattern, locale=self.locale
        )

    def get_d3_format(self) -> str:
        pattern = self.pattern
        if pattern is None:
            pattern = self.localedata.currency_formats["standard"].pattern
        return d3.ldml_number_to_d3_format(pattern)


class DatetimeFormat(BaseFormatKind):
    def __init__(
        self,
        locale: str,
        pattern: Optional[str] = None,
        skeleton: Optional[str] = None,
        grain: Optional[str] = None,
    ):
        super().__init__(locale)
        self.pattern = pattern
        self.skeleton = skeleton
        self.grain = grain

    def get_skeleton_pattern(self, skeleton: str):
        key = match_skeleton(skeleton, self.localedata.datetime_skeletons)
        if key is None:
            return None
        return self.localedata.datetime_skeletons[key]

    def get_babel_format(self):
        if self.pattern is not None:
            return self.pattern

        if self.skeleton is not None:
            return self.get_skeleton_pattern(self.skeleton)

        grain_skeleton = grain_skeletons.get(self.grain)
        if grain_skeleton is not None:
            return self.get_skeleton_pattern(grain_skeleton)

        format = "short"
        if self.grain == "second":
            format = "medium"

        date_format = self.localedata.date_formats[format]
        time_format = self.localedata.time_formats[format]
        return self.localedata.datetime_formats[format].format(time_format, date_format)

    def format_value(self, value: datetime) -> str:
        return format_datetime(
            value, format=self.get_babel_format(), locale=self.locale
        )

    def get_d3_format(self) -> str:
        pattern = self.get_babel_format()
        return d3.ldml_date_to_d3_time_format(pattern)


class Format:
    def __init__(
        self,
        locale: str,
        type: Type,
        default_currency: Optional[str] = None,
        config: Optional[Format] = None,
    ):
        self.locale = locale
        self.type = type
        self.default_currency = default_currency
        if isinstance(config, str):
            config = FormatConfig(kind=config)
        if config is not None and config.kind == "currency" and config.currency is None:
            config.currency = default_currency
        self.config = config

    @property
    def kind(self) -> BaseFormatKind:
        if not isinstance(self.config, FormatConfig):
            if self.type.name == "str":
                return StringFormat(locale=self.locale)
            if self.type.name == "bool":
                return BoolFormat(locale=self.locale)
            if self.type.name in {"int", "float"}:
                return DecimalFormat(locale=self.locale)
            if self.type.name == "datetime":
                return DatetimeFormat(locale=self.locale, grain=self.type.grain)

        if self.config.kind == "string":
            return StringFormat(locale=self.locale)
        if self.config.kind in {"number", "decimal"}:
            return DecimalFormat(locale=self.locale, pattern=self.config.pattern)
        if self.config.kind == "currency":
            return CurrencyFormat(
                locale=self.locale,
                currency=self.config.currency,
                pattern=self.config.pattern,
            )
        if self.config.kind == "percent":
            return PercentFormat(locale=self.locale, pattern=self.config.pattern)
        if self.config.kind in {"datetime", "date"}:
            return DatetimeFormat(
                locale=self.locale,
                pattern=self.config.pattern,
                skeleton=self.config.skeleton,
                grain=self.type.grain,
            )

    @property
    def currency(self) -> str:
        if self.config is None:
            return None
        return self.config.currency

    @property
    def d3_format(self) -> Optional[str]:
        return self.kind.get_d3_format()

    def format_value(self, value: Any) -> str:
        if value is None:
            return ""
        return self.kind.format_value(value)
