from lark import Tree

from dictum_core.project.magics.parser import parse_shorthand_table


def test_parse_raw_table():
    result = parse_shorthand_table("test")
    assert result.children == [Tree("table", ["test"]), Tree("source", [])]


def test_parse_table_with_source():
    result = parse_shorthand_table("test schema=public table=what")
    assert result.children == [
        Tree("table", ["test"]),
        Tree("source", [("schema", "public"), ("table", "what")]),
    ]


def test_parse_table_with_related():
    result = parse_shorthand_table("test related | column -> other.column")
    _, _, related = result.children
    assert related.data == "related"
