from itertools import chain

from dictum_core.exceptions import (
    DuplicateCalculationError,
    InvalidIDError,
    MissingCalculationError,
    MissingMetricTableError,
    MissingRelatedKeyError,
    MissingRelatedTableError,
)
from dictum_core.schema.model.model import Model


def _check_id_valid(id: str):
    if not isinstance(id, str):
        raise InvalidIDError(f"Invalid ID: {id}, ID must be a string")
    if not id.isidentifier():
        raise InvalidIDError(f"Invalid ID: {id}, ID must be a valid Python identifier")
    if id.startswith("__"):
        raise InvalidIDError(
            f"Invalid ID {id}, IDs starting from "
            "double underscore '__' are reserved for internal use"
        )
    if id[0].lower() != id[0]:
        raise InvalidIDError(
            f"Invalid ID: {id}, IDs starting with an uppercase letter "
            "are reserved for built-in objects"
        )


def _check_model_ids(model: Model):
    """Check IDs (keys) in the model:
    - tables
    - related tables
    - dimensions
    - metrics
    - unions
    """
    for id_ in model.tables:
        _check_id_valid(id_)
        table = model.tables[id_]
        for id_ in chain(table.related, table.dimensions):
            _check_id_valid(id_)
    for id_ in chain(model.metrics, model.unions):
        _check_id_valid(id_)


def _check_related_tables(model: Model):
    for table_id, table in model.tables.items():
        for alias, related in table.related.items():
            if related.str_table not in model.tables:
                raise MissingRelatedTableError(
                    f"Related table {related.str_table} on table {table_id} "
                    "doesn't exist"
                )
            if (
                related.str_related_key is None
                and model.tables[related.str_table].primary_key is None
            ):
                raise MissingRelatedKeyError(
                    f"Related table {alias} on table {table_id} doesn't "
                    "specify a related key and there no primary key specified for "
                    f"{related.str_table}. "
                    "Either one or another must be specified."
                )


def _check_duplicate_dimensions(model: Model):
    """Check that there are no duplicate dimension names.
    There are no such checks for metrics, tables and unions, because they are defined
    as either dict keys or files, with the file loader checking for duplicates.
    """
    dimensions = {}
    for table_id, table in model.tables.items():
        for key in table.dimensions:
            if key in dimensions:
                raise DuplicateCalculationError(
                    f"Duplicate dimension: {key} on tables "
                    f"{dimensions[key]} and {table_id}"
                )
            dimensions[key] = table_id


def _check_missing_unions(model: Model):
    for table in model.tables.values():
        for dimension_id, dimension in table.dimensions.items():
            if dimension.union is not None:
                # TODO: check that union type and dimension type match
                if dimension.union not in model.unions:
                    raise MissingCalculationError(
                        f"Missing union {dimension.union} is specified "
                        f"on dimension {dimension_id}"
                    )


def _check_metric_table_consistency(model: Model):
    for id_, metric in model.metrics.items():
        if metric.str_time is not None and metric.table is None:
            raise MissingMetricTableError(
                f"Metric {id_} specifies time, but doesn't specify table. "
                "If time is present, table must also be specified."
            )
        if metric.str_filter is not None and metric.table is None:
            raise MissingMetricTableError(
                f"Metric {id_} specifies filter, but doesn't specify table. "
                "If filter is present, table must also be specified."
            )


def _check_missing_metric_time_dimensions(model: Model):
    dimensions = [
        (id_, d) for t in model.tables.values() for id_, d in t.dimensions.items()
    ]
    valid_time_dimension_ids = set(id_ for id_, d in dimensions if "date" in d.type)
    for id_, metric in model.metrics.items():
        if metric.str_time is not None:
            if metric.str_time not in valid_time_dimension_ids:
                raise MissingCalculationError(
                    f"Time dimension {metric.str_time} is specified "
                    f"on metric {id_}, but there's no such dimension in the model."
                )


def check_model_config(model: Model):
    """Check that the model data is valid and internally consistent."""
    _check_model_ids(model)
    _check_related_tables(model)
    _check_duplicate_dimensions(model)
    _check_missing_unions(model)
    _check_metric_table_consistency(model)
    _check_missing_metric_time_dimensions(model)
