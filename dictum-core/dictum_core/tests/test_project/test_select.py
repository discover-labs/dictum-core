from dictum_core import Project
from dictum_core.project import Select


def test_select(project: Project):
    select = project.select(project.m.revenue)
    assert isinstance(select, Select)
    assert select.df().iloc[0, 0] == 2328.6


def test_select_by(project: Project):
    select = project.select(project.m.revenue).by(project.d.genre)
    assert select.df().shape == (24, 2)


def test_select_where(project: Project):
    select = project.select(project.m.track_count).where(project.d.genre == "Rock")
    assert select.df().iloc[0, 0] == 1297


def test_select_alias(project: Project):
    select = project.select(project.m.revenue).by(
        project.d.invoice_date.year.name("year")
    )
    assert "year" in select.df().columns


def test_select_str(project: Project):
    select = project.select("track_count").by("genre as g").where("genre = 'Rock'")
    df = select.df()
    assert "g" in df.columns
    assert "track_count" in df.columns
    assert df.iloc[0, 1] == 1297
