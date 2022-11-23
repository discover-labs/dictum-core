from functools import cached_property
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from dictum_core import examples
from dictum_core.backends.sqlite import SQLiteBackend

sql_path = Path(examples.__file__).parent / "chinook" / "chinook.sqlite.sql"


class ChinookBackend(SQLiteBackend):
    """One-database backend for tutorials and tests.
    Connects to an in-memory SQLite DB.
    """

    type = "chinook"

    def __init__(self):
        super().__init__(database="")  # in-memory

    @cached_property
    def engine(self) -> Engine:
        engine = create_engine(self.url)
        with engine.connect() as conn:
            conn.connection.executescript(sql_path.read_text())
        return engine
