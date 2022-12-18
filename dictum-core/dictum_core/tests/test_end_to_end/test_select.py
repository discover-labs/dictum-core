import pandas as pd
import pytest

from dictum_core.project import Project


def test_single_measure(project: Project):
    result = project.ql("from example:chinook select $revenue").df()
    assert result.columns[0] == "revenue"
    assert result.iloc[0, 0] == 2328.6


def test_metric(project: Project):
    result = project.ql("from example:chinook select $revenue_per_track").df()
    result.round(2).loc[0, "revenue_per_track"] == 0.66


def test_metric_groupby(project: Project):
    result = project.ql("from example:chinook select :genre, $revenue_per_track").df()
    assert result.shape == (25, 2)


def test_multiple_anchors(project: Project):
    result = project.ql("from example:chinook select $revenue, $track_count").df()
    assert next(result.itertuples()) == (0, 2328.6, 3503)


def test_multiple_anchors_by(project: Project, engine):
    result = project.ql(
        "from example:chinook select :genre, $revenue, $track_count"
    ).df()
    assert result.shape == (25, 3)


def test_select_aggregate_dimension(project: Project):
    result = project.ql("from example:chinook select :track_revenue, $revenue").df()
    assert result.shape == (4, 2)


def test_groupby(project: Project):
    result = project.ql(
        "from example:chinook select :genre, $revenue, $track_count"
    ).df()
    assert result.shape == (25, 3)


def test_filter_eq(project: Project):
    result = project.ql(
        """
    from example:chinook
    filter artist = 'Iron Maiden'
    select :genre, $revenue, $track_count
    """
    ).df()
    assert result.shape == (4, 3)


def test_filter_ne(project: Project):
    result = project.ql(
        """
    from example:chinook
    filter genre <> 'Rock'
    select :artist, $revenue, $track_count
    """
    ).df()
    assert result.shape == (165, 3)


def test_filter_gt(project: Project):
    result = project.ql(
        """
    from example:chinook
    filter order_amount > 5
    select $revenue
    """
    ).df()
    assert result.iloc[0, 0] == 1797.81


def test_filter_ge(project: Project):
    result = project.ql(
        """
    from example:chinook
    filter order_amount >= 5
    select $revenue
    """
    ).df()
    assert result.iloc[0, 0] == 1797.81


def test_filter_lt(project: Project):
    result = project.ql(
        """
    from example:chinook
    filter order_amount < 5
    select $revenue
    """
    ).df()
    assert result.iloc[0, 0] == 530.79


def test_filter_le(project: Project):
    result = project.ql(
        """
    from example:chinook
    filter order_amount <= 5
    select $revenue
    """
    ).df()
    assert result.iloc[0, 0] == 530.79


def test_filter_isin(project: Project):
    result = project.ql(
        """
    from example:chinook
    filter genre in ('Alternative', 'Rock')
    select :genre, $revenue
    """
    ).df()
    assert result.shape == (2, 2)


def test_date_unit(project: Project):
    result = project.ql(
        """
    from example:chinook
    select :invoice_date.year, $revenue
    """
    ).df()
    assert result.shape == (5, 2)


def test_step_transform(project: Project):
    result = project.ql(
        """
    from example:chinook
    select :order_amount.step(10), $revenue
    """
    ).df()
    assert result.shape == (3, 2)


def test_dimension_alias(project: Project):
    result = project.ql(
        """
    from example:chinook
    select year = :invoice_date.year, $revenue
    """
    ).df()
    assert tuple(result.columns) == ("year", "revenue")


def test_metric_alias(project: Project):
    result = project.ql(
        """
    from example:chinook
    select :genre, test = $revenue
    """
    ).df()
    assert tuple(result.columns) == ("genre", "test")


def test_datetrunc_and_inrange(project: Project):
    result = project.ql(
        """
    from example:chinook
    filter invoice_date.inrange(@2010-01-01, @2011-12-31)
    select :invoice_date.datetrunc('week'), $revenue
    """
    ).df()
    assert result.shape == (81, 2)


def test_top_with_measure_basic(project: Project):
    result = project.ql(
        """
    from example:chinook
    filter genre where revenue.top(5)
    select :genre, $revenue
    """
    ).df()
    assert set(result.genre) == {
        "Metal",
        "Latin",
        "Rock",
        "Alternative & Punk",
        "TV Shows",
    }


def test_top_with_metrics_basic(project: Project):
    result = project.ql(
        """
    from example:chinook
    filter genre where revenue_per_track.top(5)
    select :genre, $revenue
    """
    ).df()
    assert set(result.genre) == {
        "Bossa Nova",
        "Sci Fi & Fantasy",
        "TV Shows",
        "Comedy",
        "Science Fiction",
    }


def test_top_with_multiple_basic(project: Project):
    result = project.ql(
        """
    from example:chinook
    filter
        genre where revenue.top(10),
        genre where revenue_per_track.top(10),
    select :genre, $revenue
    """
    ).df()
    assert set(result.genre) == {
        "Blues",
        "TV Shows",
        "Drama",
        "Alternative & Punk",
        "Metal",
        "R&B/Soul",
    }


def test_top_with_multiple_basic_reverse_order(project: Project):
    result = project.ql(
        """
    from example:chinook
    filter
        genre where revenue_per_track.top(10),
        genre where revenue.top(10),
    select :genre, $revenue
    """
    ).df()
    assert set(result.genre) == {
        "Blues",
        "TV Shows",
        "Drama",
        "Alternative & Punk",
        "Metal",
        "R&B/Soul",
    }


def test_top_with_measure_basic_metric(project: Project):
    result = project.ql(
        """
    from example:chinook
    filter genre where revenue.top(5)
    select :genre, $revenue
    """
    ).df()
    assert set(result.genre) == {
        "Metal",
        "Latin",
        "Rock",
        "Alternative & Punk",
        "TV Shows",
    }


def test_top_with_metric_basic_metric(project: Project):
    result = project.ql(
        """
    from example:chinook
    filter genre where revenue_per_track.top(5)
    select :genre, $revenue
    """
    ).df()
    assert set(result.genre) == {
        "Bossa Nova",
        "Sci Fi & Fantasy",
        "TV Shows",
        "Comedy",
        "Science Fiction",
    }


def test_top_with_multiple_basic_metric(project: Project):
    result = project.ql(
        """
    from example:chinook
    filter
        genre where revenue.top(10),
        genre where revenue_per_track.top(10),
    select :genre, $revenue
    """
    ).df()
    assert set(result.genre) == {
        "Blues",
        "TV Shows",
        "Drama",
        "Alternative & Punk",
        "Metal",
        "R&B/Soul",
    }


def test_top_with_multiple_basic_metric_reverse_order(project: Project):
    result = project.ql(
        """
    from example:chinook
    filter
        genre where revenue_per_track.top(10),
        genre where revenue.top(10),
    select :genre, $revenue
    """
    ).df()
    assert set(result.genre) == {
        "Blues",
        "TV Shows",
        "Drama",
        "Alternative & Punk",
        "Metal",
        "R&B/Soul",
    }


def test_top_with_measure_within(project: Project):
    result = project.ql(
        """
    from example:chinook
    filter (artist) where revenue.top(3) within (genre)
    select :genre, :artist, $revenue
    """
    ).df()
    assert result.shape == (56, 3)


def test_top_with_measure_within_of(project: Project):
    result = project.ql(
        """
    from example:chinook
    filter genre where revenue.top(3),
        artist where revenue.top(3) within (genre)
    select :genre, :artist, $revenue
    """
    ).df()
    assert result.shape == (9, 3)
    assert set(result.genre) == {"Latin", "Metal", "Rock"}


def test_top_with_metric_within_of(project: Project):
    result = project.ql(
        """
    from example:chinook
    filter
        genre where revenue_per_track.top(3),
        artist where revenue_per_track.top(3) within (genre),
    select :genre, :artist, $revenue, $revenue_per_track
    """
    ).df()
    assert result.shape == (6, 4)


def test_tops_with_total(project: Project):
    result = project.ql(
        """
    from example:chinook
    filter
        customer_country where revenue.top(3),
        customer_city where revenue.top(1) within (customer_country),
    select :customer_country, :customer_city, $revenue, $revenue.total
    """
    ).df()
    assert result.shape == (3, 4)


def test_tops_with_total_within(project: Project):
    result = project.ql(
        """
    from example:chinook
    filter
        customer_country where revenue.top(5),
        customer_city where revenue.top(1) within (customer_country)
    select :customer_country,
        :customer_city,
        $revenue,
        $revenue.total within (customer_country),
    """
    ).df()
    assert result.shape == (5, 4)


def test_tops_with_total_without_total(project: Project):
    result = project.ql(
        """
        from example:chinook
    filter
        customer_country where revenue.top(5),
        customer_city where revenue.top(1) within (customer_country)
    select :customer_country,
        :customer_city,
        $revenue,
        $revenue within (customer_country),
    """
    ).df()
    assert (
        result.groupby("customer_country")["revenue"].transform("sum")
        == result["revenue_within_customer_country"]
    ).all()


def test_total_basic(project: Project):
    result = project.ql(
        """
    from example:chinook
    select :genre, $revenue, $revenue.total
    """
    ).df()
    assert result.shape == (24, 3)
    assert result["revenue_total"].unique().tolist() == [2328.6]


def test_total_metric(project: Project):
    result = project.ql(
        """
    from example:chinook
    filter artist where revenue.top(3) within (genre)
    select :genre, :artist,
        $revenue_per_track,
        $revenue_per_track.total within (genre)
    """
    ).df()
    assert result.shape == (56, 4)
    assert len(set(result.revenue_per_track_total_within_genre)) == 23


def test_total_transformed_dimension(project: Project):
    result = project.ql(
        """
    from example:chinook
    select :Time.year, :Time.month,
        $revenue.total within (Time.year)
    """
    ).df()
    assert len(result.iloc[:, -1].unique()) == 5


def test_total_filters(project: Project):
    """Test that filters are applied to the table transforms too (was a bug)"""
    result = project.ql(
        """
    from example:chinook
    filter genre in ('Rock', 'Alternative & Punk')
    select :genre, $revenue.total
    """
    ).df()
    assert result["revenue_total"].unique()[0] == 1068.21


def test_percent_basic(project: Project):
    result = project.ql(
        """
    from example:chinook
    select :genre, $revenue.percent
    """
    ).df()
    assert result.shape == (24, 2)
    assert result["revenue_percent_of_genre"].sum() == 1


def test_percent_of(project: Project):
    result = project.ql(
        """
    from example:chinook
    select :genre, :artist,
        $revenue.percent of (artist)
    """
    ).df()
    unique = (
        result.groupby("genre")["revenue_percent_of_artist_within_genre"]
        .sum()
        .round(4)
        .unique()
    )
    assert unique.size == 1
    assert unique[0] == 1.0


def test_percent_within(project: Project):
    result = project.ql(
        """
    from example:chinook
    select :genre, :artist, $revenue.percent within (genre)
    """
    ).df()
    unique = (
        result.groupby("genre")["revenue_percent_of_artist_within_genre"]
        .sum()
        .round(4)
        .unique()
    )
    assert unique.size == 1
    assert unique[0] == 1.0


def test_percent_of_within(project: Project):
    result = project.ql(
        """
    from example:chinook
    select :genre, :artist, :album,
        $revenue, $revenue.percent of (artist) within (genre)
    """
    ).df()
    values = result.query("genre == 'TV Shows' and artist == 'The Office'")[
        "revenue_percent_of_artist_within_genre"
    ].unique()
    assert len(values) == 1
    assert values.round(4)[0] == 0.3404


def test_percent_with_top(project: Project):
    """Percent should be calculated after top"""
    result = project.ql(
        """
    from example:chinook
    filter genre where revenue.top(5)
    select :genre, $revenue.percent
    """
    ).df()
    assert result.shape == (5, 2)
    assert round(result["revenue_percent_of_genre"].sum(), 4) == 1


def test_declare_alias(project: Project):
    result = project.ql(
        """
    from example:chinook
    declare tmp = $revenue
    select :genre, test = $tmp
    """
    ).df()
    assert tuple(result.columns) == ("genre", "test")


def test_declare_scalar_filter(project: Project):
    result = project.ql(
        """
    from example:chinook
    declare tmp = $revenue
    filter music
    select
        a = $revenue,
        b = $tmp,
    """
    ).df()
    pd.testing.assert_frame_equal(result, pd.DataFrame({"a": [2107.71], "b": [2328.6]}))


def test_declare_table_filter(project: Project):
    result = project.ql(
        """
    from example:chinook
    declare tmp = $revenue
    filter genre where revenue.top(5)
    select a = $revenue, b = $tmp
    """
    ).df()
    pd.testing.assert_frame_equal(result, pd.DataFrame({"a": [1805.24], "b": [2328.6]}))


def test_percent_with_top_within(project: Project):
    result = project.ql(
        """
    from example:chinook
    filter :artist where $revenue.top(1) within (:genre)
    select :genre, :artist,
        artist_revenue_within_genre = $revenue.percent within (:genre)
    """
    ).df()
    assert result.shape == (24, 3)
    assert result.columns[-1] == "artist_revenue_within_genre"  # alias works


def test_percent_integer(project: Project):
    """Check that the output type is handled correctly (float, not the original int)
    and that the percentages for unique values do not add up to 100%"""
    result = project.ql(
        """
    from example:chinook
    select :genre, $unique_paying_customers.percent
    """
    ).df()
    assert round(result["unique_paying_customers_percent_of_genre"].sum(), 2) == 7.46


def test_tops_with_matching_total_and_percent(project: Project):
    result = project.ql(
        """
    from example:chinook
    filter
        customer_country where revenue.top(5),
        customer_city where revenue.top(1) within (customer_country),
    select :customer_country, :customer_city,
        $revenue,
        "% of City Revenue" = $revenue.percent of (customer_city),
        "Revenue Total: Country" = $revenue.total within (customer_country),

    """
    ).df()
    assert result.shape == (5, 5)


def test_format_metric(project: Project):
    result = project.ql("from example:chinook select $revenue").df(format=True)
    assert result.columns[0] == "Revenue"  # column name comes from the metric name
    assert result.iloc[0, 0] == "$2,328.60"  # the value is formatted


def test_format_dimension_name(project: Project):
    result = project.ql("from example:chinook select :genre, $revenue").df(format=True)
    assert list(result.columns) == ["Genre", "Revenue"]


def test_format_transform(project: Project):
    result = project.ql("from example:chinook select :invoice_date.year, $revenue").df(
        format=True
    )
    assert list(result.columns) == ["Invoice Date (year)", "Revenue"]
    assert result.iloc[0, 0] == "2009"


def test_format_dimension_transform_alias(project: Project):
    result = project.ql(
        """
    from example:chinook
    select Year = :invoice_date.year,
        "Percent of Revenue" = $revenue.percent,
    """
    ).df(format=True)
    assert list(result.columns) == ["Year", "Percent of Revenue"]


def test_filtered_table(project: Project):
    result = project.ql("from example:chinook select $rock_revenue").df()
    assert result.iloc[0, 0] == 826.65


def test_filtered_measure(project: Project):
    result = project.ql("from example:chinook select $music_revenue").df()
    assert result.iloc[0, 0] == 2107.71


def test_filtered_and_unfiltered_measures_together(project: Project):
    result = project.ql("from example:chinook select $revenue, $music_revenue").df()
    assert next(result.itertuples(index=False)) == (2328.6, 2107.71)


def test_generic_time(project: Project):
    result = project.ql(
        "from example:chinook select :Time.datetrunc('year'), $revenue"
    ).df()
    assert result.shape == (5, 2)


def test_generic_time_alias_display_name(project: Project):
    result = project.ql("from example:chinook select test = :Year, $revenue").df(
        format=True
    )
    assert result.columns[0] == "test"


def test_generic_time_format(project: Project):
    result = project.ql("from example:chinook select :Year, $revenue").df(format=True)
    assert result.iloc[0]["Year"] == "2009"


@pytest.mark.parametrize(
    "dimension,n",
    [
        ("Year", 5),
        ("Quarter", 20),
        ("Month", 60),
        ("Week", 202),
        ("Day", 354),
        ("Date", 354),
    ],
)
def test_generic_time_trunc(project: Project, dimension: str, n: int):
    result = project.ql(f"from example:chinook select :{dimension}, $revenue").df()
    assert result.shape == (n, 2)


def test_generic_time_trunc_transform(project: Project):
    result = project.ql(
        "from example:chinook select :Month.datetrunc('year'), $revenue"
    ).df()
    assert result.shape == (5, 2)


# TODO: decide what to do with this, as total works too
@pytest.mark.xfail
def test_sum_table_transform(project: Project):
    result = project.select("revenue.sum").by("genre").df()
    assert result["revenue__sum"].unique().size == 1
    assert result["revenue__sum"].unique()[0] == 2328.6


@pytest.mark.xfail
def test_sum_table_transform_within(project: Project):
    result = (
        project.select("revenue", "revenue.sum() within (genre)")
        .by("genre", "artist")
        .df()
    )
    gb = result.groupby("genre").aggregate(
        {"revenue": "sum", "revenue__sum_within_genre": "max"}
    )
    assert (gb["revenue"].round(2) == gb["revenue__sum_within_genre"]).all()


def test_measure_with_related_column(project: Project):
    """Test that related columns are supported in measures"""
    project.ql("from example:chinook select $unique_paying_customers").df()  # no error


def test_default_time_format(project: Project):
    result = project.ql("from example:chinook select :invoice_date, $revenue").df(
        format=True
    )
    assert result.iloc[0]["Invoice Date"] == "1/1/2009"


def test_percents_without_alias(project: Project):
    """Bug found writing the docs. Something is wrong when running this query, fails
    in the pandas merge Visitor with KeyError, column not found.
    """
    project.ql(
        """
    from example:chinook
    select :invoice_date.year,
        :invoice_date.quarter,
        :invoice_date.month,
        $revenue.percent within (invoice_date.year),
        $revenue.percent of (invoice_date.quarter) within (invoice_date.year),
    """
    ).df()


def test_total_of_within_keyerror(project: Project):
    """Bug found writing the docs, similar to the above"""
    project.ql(
        """
    from example:chinook
    select :Year, :Quarter,
        $revenue.total within (Year),
        $revenue.total of (Year),
    """
    ).df()


@pytest.mark.xfail
def test_total_with_sum(project: Project):
    """Bug: When two metrics are requested, on SQLite PandasCompiler is used
    to calculate the window, and it was implemented incorrectly, resulting in an error.
    """
    project.select("revenue.total", "revenue.sum").by("genre").df()


@pytest.mark.xfail
def test_running_sum_single(project: Project):
    df = project.select("revenue.running_sum within (Year)").by("Year", "Month").df()
    assert df["revenue__running_sum_within_Year"].max() == 481.45


@pytest.mark.xfail
def test_running_sum_multiple(project: Project):
    df = (
        project.select(
            "items_sold",  # select multiple measures
            "revenue.sum within (Year)",
            "revenue.running_sum within (Year)",
        )
        .by("Year", "Month")
        .df()
    )
    assert df["revenue__running_sum_within_Year"].max() == 481.45


def test_literal_limit(project: Project):
    df = project.ql(
        """
    from example:chinook
    select :genre, $revenue
    limit 5
    """
    ).df()
    assert df.shape == (5, 2)


def test_select_union(project: Project):
    df = project.ql("from example:chinook select :country, $n_customers").df()
    assert df.shape == (24, 2)


def test_select_having_metric(project: Project):
    df = project.ql(
        """
    from example:chinook
    filter genre where revenue > 100
    select :genre, $revenue
    """
    ).df()
    assert df.shape == (4, 2)

    df = project.ql(
        """
    from example:chinook
    filter genre where revenue > 100
    select :genre, $revenue, $revenue.total
    """
    ).df()
    assert df.shape == (4, 3)


def test_select_having_table_transform(project: Project):
    df = project.ql(
        """
    from example:chinook
    filter genre where revenue.percent >= 1%
    select :genre, $revenue, $revenue.percent
    """
    ).df()
    assert df.shape == (13, 3)
