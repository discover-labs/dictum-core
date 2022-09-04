from typing import Dict, List

from dictum_core.format import Format


class Formatter:
    """Intelligently format the data based on the information in the query."""

    def __init__(self, formats: Dict[str, Format]):
        self.formats = formats

    def format(self, data: List[dict]):
        for row in data:
            yield self.format_row(row)

    def format_row(self, row: dict):
        return {k: self.format_value(v, self.formats[k]) for k, v in row.items()}

    def format_value(self, value, format: Format) -> str:
        return format.format_value(value)
