from dictum_core import Project


def test_pivot_in(project: Project):
    project.pivot("revenue").where(project.d.genre.isin("Rock"))
