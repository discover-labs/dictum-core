from functools import cached_property
from pathlib import Path

import requests
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from dictum_core.backends.sqlite import SQLiteBackend

cache_path = Path(__file__).parent / "sample_cache"
cache_path.mkdir(exist_ok=True)

download_url = (
    "https://github.com/discover-labs/dictum-sample-databases/blob/master/"
    "{name}.sqlite.sql?raw=true"
)


def get_dataset_path(name: str) -> Path:
    return cache_path / f"{name}.sqlite.sql"


def download_dataset(name: str):
    url = download_url.format(name=name)
    res = requests.get(url)
    res.raise_for_status()
    get_dataset_path(name).write_text(res.text)


def get_dataset_sql(name: str):
    dataset_path = get_dataset_path(name)
    if not dataset_path.is_file():
        download_dataset(name)
    return dataset_path.read_text()


class SampleBackend(SQLiteBackend):
    """Sample SQLite backends for tutorials and tests.
    Connects to an in-memory SQLite DB.
    """

    type = "sample"

    def __init__(self, database: str):
        self.database = database
        super().__init__(database="")  # in-memory

    @cached_property
    def engine(self) -> Engine:
        engine = create_engine(self.url)
        with engine.connect() as conn:
            conn.connection.executescript(get_dataset_sql(self.database))
        return engine
