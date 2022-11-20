from pathlib import Path

import lark.exceptions
import pytest
import yaml
from lark import Tree

from dictum_core.project.magics.magics import ProjectMagics
from dictum_core.project.magics.parser import (
    parse_shorthand_dimension,
    parse_shorthand_format,
    parse_shorthand_metric,
    parse_shorthand_table,
)
from dictum_core.project.project import Project


def test_parse_raw_table():
    parse_shorthand_table("test") == Tree("table_full", [Tree("table", ["test"])])


def test_parse_table_with_pk():
    parse_shorthand_table("test[pk]") == Tree(
        "table_full", [Tree("table", ["test"]), Tree("pk", ["pk"])]
    )


def test_parse_table_with_pk_source():
    result = parse_shorthand_table("test[pk] src")
    assert result == Tree(
        "table_full",
        [
            Tree(
                "table_def",
                [
                    Tree("table", ["test"]),
                    Tree("pk", ["pk"]),
                    Tree("source", ["src"]),
                ],
            )
        ],
    )


def test_parse_table_with_pk_source_kv():
    result = parse_shorthand_table("test[pk] src=x y=z")
    assert result == Tree(
        "table_full",
        [
            Tree(
                "table_def",
                [
                    Tree("table", ["test"]),
                    Tree("pk", ["pk"]),
                    Tree("source", [{"src": "x", "y": "z"}]),
                ],
            )
        ],
    )


def test_parse_table_with_source():
    parse_shorthand_table("test schema=public table=what").children == [
        Tree("table", ["test"]),
        Tree("source", [{"schema": "public", "table": "what"}]),
    ]


@pytest.mark.parametrize("tbl", ["", " @ tbl ", " @ tbl where :dim = 'val' "])
@pytest.mark.parametrize("type", ["", " :: int "])
@pytest.mark.parametrize("alias", ["", " as z "])
def test_parse_metric(tbl, type, alias):
    """Basic tests for various combinations of metric parameters"""
    definition = f"x = sum(y) {tbl} {type} {alias}"
    parse_shorthand_metric(definition)


@pytest.mark.parametrize("alias", ["", " as z "])
@pytest.mark.parametrize("type", [" :: int "])
@pytest.mark.parametrize("tbl", ["", " @ tbl "])
def test_parse_dimension(tbl, type, alias):
    """Basic tests for various combinations of dimension parameters"""
    definition = f"x = y {tbl} {type} {alias}"
    parse_shorthand_dimension(definition)


def test_dimension_without_type_fails():
    with pytest.raises(lark.exceptions.UnexpectedEOF):
        parse_shorthand_dimension("x = y")


def test_parse_table_with_related():
    result = parse_shorthand_table("test related | column -> other.column")
    _, related = result.children
    assert related.data == "related"


def test_parse_format_str():
    result = parse_shorthand_format("currency")
    assert result == Tree("format", ["currency"])


def test_parse_format_kv():
    result = parse_shorthand_format("kind=currency currency=USD")
    assert result == Tree("format", [("kind", "currency"), ("currency", "USD")])


def test_standalone_table(tmp_path: Path, project):
    project = Project.new(backend=project.backend, path=tmp_path)
    magics = ProjectMagics(project)
    magics.table("genres")


@pytest.fixture(scope="function")
def empty():
    return Project.example("empty")


def test_project_create_table(empty: Project):
    empty.update_shorthand_table("invoice_items")
    assert "invoice_items" in empty.model_data["tables"]


def test_project_create_metric_with_filter(empty: Project):
    empty.update_shorthand_table("tbl")
    empty.update_shorthand_metric("x = sum(x * y) @ tbl where z > 0")
    assert "x" in empty.model_data["metrics"]
    assert "filter" in empty.model_data["metrics"]["x"]
    assert empty.model_data["metrics"]["x"]["filter"] == "z > 0"


def test_metric_properties(empty: Project):
    empty.update_shorthand_table("tbl")
    empty.update_shorthand_dimension("d = d @ tbl ::date")
    empty.update_shorthand_metric(
        'x = sum(x * y) @ tbl | time=d name="something something"'
    )
    assert "x" in empty.model.metrics
    assert empty.model.metrics["x"].name == "something something"
    assert len(empty.model.metrics["x"].generic_time_dimensions) > 0


def test_project_create_from_scratch_write(tmp_path: Path, project: Project):
    project = Project.new(backend=project.backend, path=tmp_path)
    magics = ProjectMagics(project)
    magics.table("invoice_items")
    magics.metric("revenue = sum(Quantity * UnitPrice) @ invoice_items")
    magics.format("currency")
    project.write()
    assert (
        yaml.safe_load((tmp_path / "metrics" / "revenue.yml").read_text())["format"]
        == "currency"
    )
    magics.table("invoices")
    magics.related("invoice_items invoice | InvoiceId -> invoices.InvoiceId")
    magics.dimension("date = InvoiceDate @ invoices ::date")
    magics.table("genres")
    magics.dimension("genre = Name @ genres ::str")
    cell = """
    dimension media_type = Name ::str
    dimension music = MediaTypeId in (1, 2, 4, 5) ::bool as "Is Music"
    """
    magics.table(line="media_types", cell=cell)
    cell = """
    genre | GenreId -> genres.GenreId
    media_type | MediaTypeId -> media_types.MediaTypeId
    metric tracks = count()
    dimension track_length = Milliseconds / 1000 / 60 ::float
    """
    magics.table(line="tracks[TrackId]", cell=cell)
    magics.related("invoice_items track | TrackId -> tracks")
    magics.metric(
        "unique_paying_customers = countd(invoice.CustomerId) @ invoice_items"
    )
    magics.metric("arppu = $revenue / $unique_paying_customers as ARPPU")
    project.write()

    assert (tmp_path / "project.yml").is_file()
    assert (tmp_path / "metrics" / "arppu.yml").is_file()

    # edit
    magics.metric(
        "arppu = $revenue / $unique_paying_customers "
        'as "Average Revenue per Paying User"'
    )
    project.write()
    arppu = yaml.safe_load((tmp_path / "metrics" / "arppu.yml").read_text())
    assert arppu["name"] == "Average Revenue per Paying User"
