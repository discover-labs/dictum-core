from pathlib import Path

import yaml
from lark import Tree

from dictum_core.project.magics.magics import ProjectMagics
from dictum_core.project.magics.parser import (
    parse_shorthand_format,
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


def test_create_project_from_scratch(tmp_path: Path, project: Project):
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
