import pytest

from dictum_core.exceptions import (
    DuplicateCalculationError,
    InvalidIDError,
    MissingCalculationError,
    MissingMetricTableError,
    MissingRelatedKeyError,
    MissingRelatedTableError,
)
from dictum_core.schema.model.checks import (
    _check_duplicate_dimensions,
    _check_id_valid,
    _check_metric_table_consistency,
    _check_missing_metric_time_dimensions,
    _check_missing_unions,
    _check_model_ids,
    _check_related_tables,
)
from dictum_core.schema.model.model import Model


def test_check_duplicate_dimensions():
    model = Model.parse_obj(
        {
            "name": "test",
            "tables": {
                "t1": {
                    "source": "t1",
                    "dimensions": {
                        "dupe": {"name": "dupe", "expr": "x", "type": "int"}
                    },
                },
                "t2": {
                    "source": "t2",
                    "dimensions": {
                        "dupe": {"name": "dupe", "expr": "x", "type": "int"}
                    },
                },
            },
        }
    )
    with pytest.raises(DuplicateCalculationError):
        _check_duplicate_dimensions(model)


def test_check_model_ids_checks_tables():
    model = Model.parse_obj(
        {
            "name": "test",
            "tables": {"__t1": {"source": "t1"}},
        }
    )
    with pytest.raises(InvalidIDError):
        _check_model_ids(model)


def test_check_model_ids_checks_metrics():
    model = Model.parse_obj(
        {"name": "test", "metrics": {"__m1": {"name": "m1", "expr": "m1"}}}
    )
    with pytest.raises(InvalidIDError):
        _check_model_ids(model)


def test_check_model_ids_checks_unions():
    model = Model.parse_obj(
        {"name": "test", "unions": {"__id": {"name": "test", "type": "test"}}}
    )
    with pytest.raises(InvalidIDError):
        _check_model_ids(model)


def test_check_model_ids_checks_related():
    model = Model.parse_obj(
        {
            "name": "test",
            "tables": {
                "test": {
                    "source": "test",
                    "related": {"__id": {"table": "test", "foreign_key": "key"}},
                }
            },
        }
    )
    with pytest.raises(InvalidIDError):
        _check_model_ids(model)


def test_check_model_ids_checks_dimensions():
    model = Model.parse_obj(
        {
            "name": "test",
            "tables": {
                "t1": {
                    "source": "src",
                    "dimensions": {
                        "__id": {
                            "name": "test",
                            "expr": "test",
                            "type": "test",
                        }
                    },
                }
            },
        }
    )
    with pytest.raises(InvalidIDError):
        _check_model_ids(model)


def test_check_id_valid_checks_dunder():
    with pytest.raises(InvalidIDError):
        _check_id_valid("__id")


def test_check_id_valid_checks_uppercase_first_character():
    with pytest.raises(InvalidIDError):
        _check_id_valid("Test")


def test_check_id_valid_checks_identifier():
    with pytest.raises(InvalidIDError):
        _check_id_valid("not id")


def test_check_id_valid_ok():
    _check_id_valid("va1id_id")


def test_check_metric_table_consistency_checks_time():
    model = Model.parse_obj(
        {
            "name": "test",
            "metrics": {
                "test": {
                    "name": "test",
                    "expr": "test",
                    "time": "test",
                }
            },
        }
    )
    with pytest.raises(MissingMetricTableError):
        _check_metric_table_consistency(model)


def test_check_metric_table_consistency_checks_filter():
    model = Model.parse_obj(
        {
            "name": "test",
            "metrics": {
                "test": {
                    "name": "test",
                    "expr": "test",
                    "filter": "test",
                }
            },
        }
    )
    with pytest.raises(MissingMetricTableError):
        _check_metric_table_consistency(model)


def test_check_missing_time_dimensions_checks_metrics():
    model = Model.parse_obj(
        {
            "name": "test",
            "metrics": {
                "test": {
                    "name": "test",
                    "expr": "test",
                    "table": "test",
                    "time": "test",
                }
            },
        }
    )
    with pytest.raises(MissingCalculationError):
        _check_missing_metric_time_dimensions(model)


def test_check_missing_unions():
    model = Model.parse_obj(
        {
            "name": "test",
            "tables": {
                "t1": {
                    "source": "test",
                    "dimensions": {
                        "test": {
                            "name": "test",
                            "type": "test",
                            "expr": "test",
                            "union": "missing",
                        }
                    },
                }
            },
        }
    )
    with pytest.raises(MissingCalculationError):
        _check_missing_unions(model)


def test_check_related_tables_checks_table():
    model = Model.parse_obj(
        {
            "name": "test",
            "tables": {
                "t1": {
                    "source": "test",
                    "related": {"rel": {"table": "missing", "foreign_key": "fk"}},
                }
            },
        }
    )
    with pytest.raises(MissingRelatedTableError):
        _check_related_tables(model)


def check_test_related_tables_checks_related_key():
    model = Model.parse_obj(
        {
            "name": "test",
            "tables": {
                "t1": {
                    "source": "test",
                    "related": {"t2": {"table": "t2", "foreign_key": "fk"}},
                },
                "t2": {"source": "test"},
            },
        }
    )
    with pytest.raises(MissingRelatedKeyError):
        _check_related_tables(model)
