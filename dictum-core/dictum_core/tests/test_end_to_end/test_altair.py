"""Mostly tests that Altair integration works without errors. Tests must return the
chart, which will be rendered and saved to altair_output folder (see conftest.py).

When possible, the tests check that the output is correct, but generally it's better
to visually inspect the output.
"""

import altair as alt

from dictum_core import Project


def test_metric(project: Project):
    return project.chart().mark_bar().encode(x=project.m.revenue)


def test_metric_dimension(project: Project):
    return (
        project.chart()
        .mark_bar()
        .encode(
            x=project.d.invoice_date.year.name("Year"),
            y=project.m.revenue,
        )
    )


def test_metric_dimension_year(project: Project):
    return (
        project.chart()
        .mark_bar()
        .encode(
            x=alt.X(project.d.Year, type="ordinal"),
            y=project.m.revenue,
        )
    )


def test_override_type(project: Project):
    chart = (
        project.chart()
        .mark_bar()
        .encode(
            x=alt.X(project.d.invoice_date.year.name("Year"), type="ordinal"),
            y=alt.Y(project.m.revenue),
        )
    )
    assert chart._render_self().encoding.x.type == "ordinal"
    return chart


def test_override_axis_title(project: Project):
    chart = (
        project.chart()
        .mark_bar()
        .encode(x=alt.X(project.m.revenue, axis=alt.Axis(title="Revenue ($)")))
    )
    assert chart._render_self().encoding.x.axis["title"] == "Revenue ($)"
    return chart


def test_inject_format(project: Project):
    chart = project.chart().mark_bar().encode(x=project.m.revenue)
    assert chart._render_self().encoding.x.axis["format"] == "$01,.2f"
    return chart


def test_facet(project: Project):
    chart = (
        project.chart()
        .mark_bar()
        .encode(
            x=alt.X(project.d.invoice_date.year, type="ordinal"),
            y=alt.Y(project.m.revenue),
        )
        .properties(width=150, height=100)
        .facet(project.d.genre, columns=3)
    )
    assert chart._render_self().facet["header"]["title"] == "Genre"
    return chart


def test_facet_row(project: Project):
    chart = (
        project.chart()
        .mark_bar()
        .encode(
            x=alt.X(project.d.invoice_date.year, type="ordinal"),
            y=alt.Y(project.m.revenue),
        )
        .properties(width=150, height=100)
        .facet(row=project.d.genre)
    )
    assert chart._render_self().facet.row["header"]["title"] == "Genre"
    return chart


def test_facet_column(project: Project):
    chart = (
        project.chart()
        .mark_bar()
        .encode(
            x=alt.X(project.d.invoice_date.year, type="ordinal"),
            y=alt.Y(project.m.revenue),
        )
        .properties(width=150, height=100)
        .facet(column=project.d.genre)
    )
    assert chart._render_self().facet.column["header"]["title"] == "Genre"
    return chart


def test_repeat(project: Project):
    chart = (
        project.chart()
        .mark_bar()
        .encode(
            x=alt.X(project.d.invoice_date.year, type="ordinal"),
            y=alt.Y(alt.repeat(), type="quantitative"),
        )
        .properties(width=150, height=100)
        .repeat([project.m.revenue, project.m.items_sold])
    )
    rendered = chart._rendered_dict()
    assert len(rendered["spec"]["encoding"]["tooltip"]) == 3
    return chart


def test_repeat_row(project: Project):
    chart = (
        project.chart()
        .mark_bar()
        .encode(
            x=alt.X(project.d.invoice_date.year, type="ordinal"),
            y=alt.Y(alt.repeat("row"), type="quantitative"),
        )
        .properties(width=150, height=100)
        .repeat(row=[project.m.revenue, project.m.items_sold])
    )
    return chart


def test_repeat_column(project: Project):
    chart = (
        project.chart()
        .mark_bar()
        .encode(
            x=alt.X(project.d.invoice_date.year, type="ordinal"),
            y=alt.Y(alt.repeat("column"), type="quantitative"),
        )
        .properties(width=150, height=100)
        .repeat(column=[project.m.revenue, project.m.items_sold])
    )
    return chart


def test_sort(project: Project):
    chart = (
        project.chart()
        .mark_bar()
        .encode(
            x=alt.X(project.m.revenue),
            y=alt.Y(project.d.genre, sort=project.m.revenue),
        )
    )
    return chart


def test_sort_field(project: Project):
    chart = (
        project.chart()
        .mark_bar()
        .encode(
            x=alt.X(project.m.revenue),
            y=alt.Y(
                project.d.genre,
                sort=alt.EncodingSortField(project.m.track_count, order="descending"),
            ),
        )
    )
    return chart


def test_detail(project: Project):
    chart = (
        project.chart()
        .mark_bar(stroke="white")
        .encode(
            x=alt.X(project.d.invoice_date.year, type="ordinal"),
            y=alt.Y(project.m.revenue),
            detail=project.d.genre,
        )
    )
    return chart


def test_detail_list(project: Project):
    chart = (
        project.chart()
        .mark_bar(stroke="white")
        .encode(
            x=alt.X(project.d.invoice_date.year, type="ordinal"),
            y=alt.Y(project.m.revenue),
            detail=[project.d.genre],
        )
    )
    return chart


def test_calculate_transform(project: Project):
    """Test that normal encodings derived from transform_calculate'd fields work as
    expected.
    """
    chart = (
        project.chart(project.m.revenue)
        .mark_bar()
        .encode(x=project.d.invoice_date.year, y="max_pct:Q")
        .transform_joinaggregate(
            groupby=[],
            max="sum(revenue)",
        )
        .transform_calculate(max_pct="datum.revenue / datum.max")
    )
    return chart


def test_custom_calculation_format(project: Project):
    return (
        project.chart()
        .mark_bar()
        .encode(
            x=project.m.revenue_per_track,
            y=project.d.genre,
        )
    )


def test_x2_type_time_unit(project: Project):
    """Make sure that the x2 timeUnit is correctly handled (not added)"""
    (
        project.chart(project.m.revenue)
        .mark_rect()
        .encode(
            x=project.d.invoice_date.datetrunc("year"),
            x2=project.d.invoice_date.datetrunc("year"),
        )
    )._rendered_dict()
