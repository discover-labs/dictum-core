from dictum_core.interactive_model.expression import (
    AbstractInteractiveExpression,
    InteractiveColumn,
)
from dictum_core.interactive_model.table import InteractiveTable
from dictum_core.model import RelatedTable


def test_table_column(Model):
    assert isinstance(Model.col, AbstractInteractiveExpression)
    assert isinstance(Model.col, InteractiveColumn)
    assert isinstance(Model.col.children[1], InteractiveTable)
    assert Model.col.children[1].__id == "tbl"
    assert Model.col.children[2] == "col"


def test_column_is_expr(Model):
    expr = Model.col / 2
    assert expr.data == "expr"
    assert expr.children[1].data == "div"


def test_foreign_key(Model):
    assert "tbl" in Model._model.tables.get("other").related

    rel = Model._model.tables.get("other").related["tbl"]
    assert isinstance(rel, RelatedTable)
    assert rel.str_table == "tbl"
    assert rel.str_related_key == "id"
    assert rel.foreign_key == "tbl_id"
    assert rel.alias == "tbl"
    assert rel.parent == Model.other._InteractiveTable__table
    assert rel.tables == Model._model.tables


def test_set_name(Model):
    assert Model.a.children[0] == "a"


def test_specify_table(Model):
    assert "d" in Model._model.tables["other"].measures
