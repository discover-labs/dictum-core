from pathlib import Path

import pytest
import yaml
from lark import Tree

from dictum_core.exceptions import ShorthandSyntaxError
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
    result = parse_shorthand_table("test[pk] { src=x y=z }")
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
    parse_shorthand_table("test { schema=public table=what }").children == [
        Tree("table", ["test"]),
        Tree("source", [{"schema": "public", "table": "what"}]),
    ]


@pytest.mark.parametrize("tbl", ["", " tbl. "])
@pytest.mark.parametrize("type", ["", " :: int "])
@pytest.mark.parametrize("alias", ["", " as z "])
@pytest.mark.parametrize("filter", ["", "where :dim = 'val'"])
@pytest.mark.parametrize("props", ["", "{ k=v a=b }"])
def test_parse_metric(tbl, type, alias, filter, props):
    """Basic tests for various combinations of metric parameters"""
    definition = f"{tbl} x = sum(y) {filter} {type} {alias} {props}"
    parse_shorthand_metric(definition)


@pytest.mark.parametrize("alias", ["", " as z "])
@pytest.mark.parametrize("type", [" :: int "])
@pytest.mark.parametrize("tbl", ["", " tbl. "])
@pytest.mark.parametrize("props", ["", "{ k=v a=b }"])
def test_parse_dimension(tbl, type, alias, props):
    """Basic tests for various combinations of dimension parameters"""
    definition = f"{tbl} x = y {type} {alias} {props}"
    parse_shorthand_dimension(definition)


def test_dimension_without_type_fails():
    with pytest.raises(ShorthandSyntaxError):
        parse_shorthand_dimension("x = y")


def test_parse_table_with_related():
    result = parse_shorthand_table("table related alias = column -> other.column")
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
    return Project.example("chinook", empty=True)


def test_project_create_table(empty: Project):
    empty.update_shorthand_table("invoice_items")
    assert "invoice_items" in empty.model_data["tables"]


def test_project_create_metric_with_filter(empty: Project):
    empty.update_shorthand_table("tbl")
    empty.update_shorthand_metric("tbl.x = sum(x * y) where z > 0")
    assert "x" in empty.model_data["metrics"]
    assert "filter" in empty.model_data["metrics"]["x"]
    assert empty.model_data["metrics"]["x"]["filter"] == "z > 0"


def test_metric_properties(empty: Project):
    empty.update_shorthand_table("tbl")
    empty.update_shorthand_dimension("tbl.d = d ::date")
    empty.update_shorthand_metric(
        'tbl.x = sum(x * y) { time=d name="something something" }'
    )
    assert "x" in empty.model.metrics
    assert empty.model.metrics["x"].name == "something something"
    assert len(empty.model.metrics["x"].generic_time_dimensions) > 0


def test_project_create_from_scratch_write(tmp_path: Path, project: Project):
    project = Project.new(backend=project.backend, path=tmp_path)
    magics = ProjectMagics(project)
    magics.table("invoice_items")
    magics.metric(
        "invoice_items.revenue = sum(Quantity * UnitPrice) { format=currency }"
    )
    project.write()
    assert (
        yaml.safe_load((tmp_path / "metrics" / "revenue.yml").read_text())["format"]
        == "currency"
    )
    magics.table("invoices")
    magics.related("invoice_items.invoice = InvoiceId -> invoices.InvoiceId")
    magics.dimension("invoices.date = InvoiceDate ::date")
    magics.table("genres")
    magics.dimension("genres.genre = Name ::str")
    cell = """
    dimension media_type = Name ::str
    dimension music = MediaTypeId in (1, 2, 4, 5) ::bool as "Is Music"
    """
    magics.table(line="media_types", cell=cell)
    cell = """
    related genre = GenreId -> genres.GenreId
    related media_type = MediaTypeId -> media_types.MediaTypeId
    metric tracks = count()
    dimension track_length = Milliseconds / 1000 / 60 ::float
    """
    magics.table(line="tracks[TrackId]", cell=cell)
    magics.related("invoice_items.track = TrackId -> tracks")
    magics.metric("invoice_items.unique_paying_customers = countd(invoice.CustomerId)")
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
