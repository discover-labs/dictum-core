import os
from pathlib import Path

import pytest

from dictum_core.examples import chinook

chinook_path = Path(chinook.__file__).parent


@pytest.fixture(scope="session")
def backend():
    from dictum_core.examples.chinook.generate import generate

    yield generate().backend


@pytest.fixture(scope="session")
def project(backend):
    from dictum_core import Project

    project = Project.example("chinook")
    project.backend = backend
    yield project


@pytest.fixture(scope="session")
def chinook():
    os.environ["CHINOOK_DATABASE"] = ""
    from dictum_core import Project

    return Project.example("chinook").model


@pytest.fixture(scope="session")
def engine(chinook):
    from dictum_core.engine import Engine

    return Engine(chinook)
