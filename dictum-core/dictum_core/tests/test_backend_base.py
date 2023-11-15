from unittest import mock

import pytest

from dictum_core.backends.base import ExpressionTransformer
from dictum_core.backends.secret import Secret
from dictum_core.backends.sql_alchemy import (
    ColumnTransformer,
    SQLAlchemyBackend,
    SQLAlchemyCompiler,
)
from dictum_core.project.actions import _get_backend_parameters


@pytest.fixture(scope="module", autouse=True)
def patch_sqlalchemy_url():
    with mock.patch(
        "dictum_core.backends.sql_alchemy.SQLAlchemyBackend.url", "sqlite://"
    ):
        yield


def test_compiler_case():
    compiler = mock.MagicMock()
    transformer = ExpressionTransformer(compiler)
    transformer.case([1, 2, 3, 4])
    compiler.case.assert_called_once_with([[1, 2], [3, 4]], else_=None)

    transformer.case([1, 2, 3, 4, 5])
    compiler.case.assert_called_with([[1, 2], [3, 4]], else_=5)


def test_sqlalchemy_no_default_schema():
    backend = SQLAlchemyBackend()
    with mock.patch("dictum_core.backends.sql_alchemy.Table") as Table:
        backend.table("table")
        assert Table.call_args[1]["schema"] is None


def test_sqlalchemy_default_schema():
    backend = SQLAlchemyBackend(default_schema="test")
    with mock.patch("dictum_core.backends.sql_alchemy.Table") as Table:
        backend.table("table")
        assert Table.call_args[1]["schema"] == "test"


def test_sqlalchemy_override_default_schema():
    backend = SQLAlchemyBackend(default_schema="test")
    with mock.patch("dictum_core.backends.sql_alchemy.Table") as Table:
        backend.table("table", schema="override")
        assert Table.call_args[1]["schema"] == "override"


def test_secret_parameters():
    class SecretBackend(SQLAlchemyBackend):
        type = "secret"

        def __init__(self, secret: Secret = "secret", x: int = 1):
            super().__init__(secret=secret, x=x)

    backend = SecretBackend()
    parameters = _get_backend_parameters(backend)
    assert parameters["x"] == 1
    assert parameters["secret"] == "{{ env.DICTUM_SECRET_SECRET_SECRET }}"


def test_sqlalchemy_case():
    import sqlalchemy as sa

    from dictum_core.model.expr import parse_expr

    parsed = parse_expr("count(case when tbl.type = 1 then 1 end) / count()")
    backend = SQLAlchemyBackend()
    compiler = SQLAlchemyCompiler(backend)
    ct = ColumnTransformer(
        {
            "tbl": sa.Table("tbl", sa.MetaData(), sa.Column("type", sa.Integer)),
        }
    )
    columned = ct.transform(parsed)
    compiler.transformer.transform(columned)
