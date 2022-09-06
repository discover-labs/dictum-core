from dictum_core.model import Model


def test_measure_dimensions_union(chinook: Model):
    assert "country" in set(chinook.measures.get("n_customers").dimensions)
    assert "country" in set(d.id for d in chinook.metrics.get("n_customers").dimensions)


def test_metric_dimensions(chinook: Model):
    dimensions = chinook.metrics["revenue_per_track"].dimensions
    assert isinstance(dimensions, list)
    assert len(dimensions) == 6


def test_metric_generic_time_dimensions(chinook: Model):
    assert chinook.metrics["revenue_per_track"].generic_time_dimensions == []
    assert len(chinook.metrics["avg_sold_unit_price"].generic_time_dimensions) == 7
    assert (
        chinook.metrics["avg_sold_unit_price"].generic_time_dimensions[1].id == "Year"
    )
    assert (
        chinook.metrics["avg_sold_unit_price"].generic_time_dimensions[1].sort_order
        == 1
    )
