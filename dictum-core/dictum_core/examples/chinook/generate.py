from pathlib import Path

from dictum_core import Project
from dictum_core.schema.project import env


def generate():
    path = Path(__file__).parent
    env.data["CHINOOK_DATABASE"] = ""
    project = Project.from_path(path)
    with project.backend.engine.connect() as conn:
        conn.connection.executescript((path / "chinook.sqlite.sql").read_text())
    return project
