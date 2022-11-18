from dictum_core.schema.model.calculations import Calculation


def test_expr_alias():
    calc = Calculation.parse_obj(
        {"id": "test", "name": "test", "type": "int", "expr": "str_expr"}
    )
    assert calc.str_expr == "str_expr"
