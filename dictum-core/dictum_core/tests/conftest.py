import pytest

from dictum_core import Project


@pytest.fixture(scope="session")
def project():
    return Project.example("chinook")


@pytest.fixture(scope="session")
def backend(project: Project):
    return project.backend


@pytest.fixture(scope="session")
def chinook(project: Project):
    return project.model


@pytest.fixture(scope="session")
def engine(project: Project):
    return project.engine
