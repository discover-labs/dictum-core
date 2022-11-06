import pytest

from dictum_core.interactive_model.model import InteractiveModel
from dictum_core.interactive_model.table import InteractiveTable


@pytest.fixture(scope="module")
def Model():
    class Model(InteractiveModel):
        tbl = InteractiveTable()
        col = tbl.col.type("int")

        other = InteractiveTable()
        other.tbl_id >> tbl.id

        a = (tbl.a * tbl.b).sum()
        b = other.x.countd()
        c = a / b

    return Model
