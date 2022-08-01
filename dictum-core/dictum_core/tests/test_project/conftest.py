import pytest

from dictum_core import Project


@pytest.fixture(scope="module")
def project() -> Project:
    return Project.example("chinook")
