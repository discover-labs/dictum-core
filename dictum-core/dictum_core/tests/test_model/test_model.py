import pytest
from lark import Token, Tree

from dictum_core.model import Dimension, Model


@pytest.fixture(scope="module")
def store(store_full):
    return store_full


def test_store_loads(chinook: Model):
    assert chinook.metrics
    assert chinook.measures
    assert chinook.dimensions
    assert chinook.tables


def test_store_parses_expr(chinook: Model):
    assert isinstance(chinook.measures["revenue"].expr, Tree)
    assert isinstance(chinook.dimensions["leap_year"].expr, Tree)
    assert isinstance(chinook.metrics["revenue_per_track"].expr, Tree)


def test_dimension_same_table_as_measures(chinook: Model):
    """There was a bug where the table couldn't find a join path from a self to
    a dimension declared on self :-/
    """
    assert (
        chinook.tables["tracks"].dimension_join_paths.get("track_length_10s_bins")
        is not None
    )


def test_resolve_measures(chinook: Model):
    assert chinook.measures["test_arppu"].expr == Tree(
        "expr",
        [
            Tree(
                "div",
                [
                    Tree(
                        "call",
                        [
                            "coalesce",
                            Tree(
                                "call",
                                [
                                    "sum",
                                    Tree(
                                        "mul",
                                        [
                                            Tree(
                                                "column", ["invoice_items", "UnitPrice"]
                                            ),
                                            Tree(
                                                "column", ["invoice_items", "Quantity"]
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                            Token("INTEGER", "0"),
                        ],
                    ),
                    Tree(
                        "call",
                        [
                            "countd",
                            Tree("column", ["invoice_items", "invoice", "CustomerId"]),
                        ],
                    ),
                ],
            )
        ],
    )


def test_resolve_metrics(chinook: Model):
    assert chinook.metrics.get("revenue_per_track").expr.children[0] == Tree(
        "div",
        [
            Tree(
                "call",
                ["coalesce", Tree("measure", ["revenue"]), Token("INTEGER", "0")],
            ),
            Tree("measure", ["track_count"]),
        ],
    )


def test_union_is_dimension(chinook: Model):
    union = chinook.tables["customers"].dimensions["country"]
    assert isinstance(union, Dimension)
    assert hasattr(union, "prefixed_expr")


def test_union_allowed(chinook: Model):
    assert "country" in chinook.tables["customers"].dimension_join_paths
    assert "country" in chinook.tables["customers"].allowed_dimensions
    assert "country" not in chinook.tables["invoice_items"].dimension_join_paths
    assert "country" not in chinook.tables["invoice_items"].allowed_dimensions
    assert "customer_country" in chinook.tables["invoice_items"].dimension_join_paths
    assert "customer_country" in chinook.tables["invoice_items"].allowed_dimensions


def test_resolve_aggregate_dimension(chinook: Model):
    assert chinook.dimensions.get("customer_orders_amount").expr.children[0] == Tree(
        "column", ["customers", "__subquery__revenue", "revenue"]
    )


def test_resolve_related_dimension(chinook: Model):
    assert chinook.dimensions.get("order_customer_country").expr.children[0] == Tree(
        "column", ["invoice_items", "invoice", "customer", "Country"]
    )


def test_resolve_dimension_on_anchor(chinook: Model):
    assert chinook.dimensions.get("genre").expr.children[0] == Tree(
        "column", ["genres", "Name"]
    )


def test_resolve_related_aggregate_dimension(chinook: Model):
    assert chinook.dimensions.get("first_order_cohort_month").expr.children[0] == Tree(
        "call",
        [
            "datediff",
            Token("STRING", "month"),
            Tree(
                "column",
                [
                    "invoice_items",
                    "invoice",
                    "customer",
                    "__subquery__min_sale_date",
                    "min_sale_date",
                ],
            ),
            Tree("column", ["invoice_items", "invoice", "InvoiceDate"]),
        ],
    )


def test_inject_default_filters_and_transforms(chinook: Model):
    assert len(chinook.scalar_transforms) == 26


def test_metric_missing(chinook: Model):
    assert chinook.metrics.get("revenue").expr.children[0] == Tree(
        "call", ["coalesce", Tree("measure", ["revenue"]), Token("INTEGER", 0)]
    )


def test_get_lineage(chinook: Model):
    L = list(chinook.get_lineage(chinook.metrics["revenue_per_track"]))
    assert len(L) == 6
    assert L[0]["type"] == "Metric"
    assert L[0]["id"].startswith("Metric:")


def test_replace_table():
    model = Model()
    test = model.add_table(id="test", source="test")
    model.add_table(id="related", source="related")
    test.add_related(
        str_table="related",
        related_key="x",
        foreign_key="y",
        alias="related",
        tables=model.tables,
    )
