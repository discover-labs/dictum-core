from itertools import chain

from lark.exceptions import UnexpectedInput

from dictum_core.exceptions import (
    AggregateDimensionError,
    CircularReferenceError,
    ExpressionSyntaxError,
    FilteredMeasureReferenceError,
    InvalidMeasureError,
    InvalidReferenceError,
    MissingAggregationError,
    MissingDimensionError,
    MissingMeasureError,
    MissingPrimaryKeyError,
    NonAggregateMeasureError,
    UnknownFunctionError,
)
from dictum_core.model.calculations import Expression, Measure
from dictum_core.model.expr.introspection import (
    ExprKind,
    aggregate_functions,
    get_expr_kind,
)
from dictum_core.model.model import Model
from dictum_core.ordered_check_caller import OrderedCheckCaller

check_model = OrderedCheckCaller()


def _check_expr_syntax(expression: Expression):
    try:
        expression.parsed_expr
    except UnexpectedInput as e:
        raise ExpressionSyntaxError(e, expression)


def _check_expr_parse(model: Model):
    for expression in model.expressions:
        _check_expr_syntax(expression)


@check_model.depends_on(_check_expr_parse)
def _check_metrics_dont_use_dimension_refs(model: Model):
    for metric in model.metrics.values():
        for ref in metric.parsed_expr.find_data("dimension"):
            id_ = ref.children[0]
            raise InvalidReferenceError(
                f"{metric} references :{id_}, "
                "dimension references in metrics are not allowed. "
                "Consider specifying table."
            )


@check_model.depends_on(_check_expr_parse)
def _check_metrics_dont_use_column_refs(model: Model):
    for metric in model.metrics.values():
        for ref in metric.parsed_expr.find_data("column"):
            id_ = ".".join(ref.children)
            raise InvalidReferenceError(
                f"{metric} references column {id_}, "
                "column references in metrics are not allowed. "
                "Consider specifying table."
            )


known_functions = {
    "sum",
    "count",
    "countd",
    "min",
    "max",
    "avg",
    "abs",
    "floor",
    "ceil",
    "coalesce",
    "tointeger",
    "tofloat",
    "todate",
    "todatetime",
    "now",
    "today",
    "datediff",
    "datepart",
    "datetrunc",
    "IF",
}


def _check_expression_functions(expression: Expression):
    for ref in expression.parsed_expr.find_data("call"):
        fn, *_ = ref.children
        if fn not in known_functions:
            raise UnknownFunctionError(f"unknown function: {fn}")


@check_model.depends_on(_check_expr_parse)
def _check_known_functions(model: Model):
    for expression in model.expressions:
        _check_expression_functions(expression)


def _check_expression_known_calculations(model: Model, expression: Expression):
    for ref in expression.parsed_expr.find_data("dimension"):
        id_ = ref.children[0]
        if id_ not in model.dimensions:
            raise MissingDimensionError(
                f"{expression} references an unknown dimension: {id_}"
            )
    for ref in expression.parsed_expr.find_data("measure"):
        id_ = ref.children[0]
        if id_ not in set(chain(model.measures, model.metrics)):
            raise MissingAggregationError(
                f"{expression} references an unknown metric or measure: {id_}"
            )


@check_model.depends_on(_check_expr_parse)
def _check_known_calculations(model: Model):
    for expression in model.expressions:
        _check_expression_known_calculations(model, expression)


@check_model.depends_on(_check_expr_parse, _check_known_calculations)
def _check_measure_references_same_table(model: Model):
    for measure in model.measures.values():
        for ref in measure.parsed_expr.find_data("measure"):
            id_ = ref.children[0]
            if id_ not in measure.table.measures:
                raise MissingMeasureError(
                    f"{measure} references {id_}, "
                    f"but it's not present on {measure.table}"
                )


def _check_no_aggregate_functions(expression: Expression):
    for ref in expression.parsed_expr.find_data("call"):
        fn, *_ = ref.children
        if fn in aggregate_functions:
            raise AggregateDimensionError(
                f"Aggregate function {fn} in {expression} "
                f"expression: '{expression.str_expr}'"
            )


@check_model.depends_on(_check_expr_parse, _check_known_functions)
def _check_calculation_kinds(model: Model):
    # dimensions can be aggregate, so no direct check
    # validate against using aggregate functions in dimensions
    for dimension in model.dimensions.values():
        if not isinstance(dimension, Expression):
            continue
        _check_no_aggregate_functions(dimension)
        get_expr_kind(dimension.parsed_expr)

    for measure in model.measures.values():
        kind = get_expr_kind(measure.parsed_expr)
        if kind != ExprKind.aggregate:
            raise NonAggregateMeasureError(
                f"{measure} expression '{measure.str_expr}' is not aggregate"
            )
        if measure.filter is not None:
            _check_no_aggregate_functions(measure.filter)
            get_expr_kind(measure.filter.parsed_expr)  # check for mixed

    for table in model.tables.values():
        for table_filter in table.filters:
            _check_no_aggregate_functions(table_filter)
            get_expr_kind(table_filter.parsed_expr)  # check for mixed


def _check_expression_circular_refs(model: Model, expression: Expression, path=None):
    if path is None:
        path = ()
    for ref in expression.parsed_expr.find_data("measure"):
        id_ = ref.children[0]

        # metrics can self-reference by referencing a same-named measure
        if id_ in model.measures:
            continue

        if id_ in path:
            path = " -> ".join(path + (id_,))
            raise CircularReferenceError(f"Circular reference in calculations: {path}")

        # measures can't reference stuff, so we don't check for that
        # metric refs existence is checked before, so we don't have to
        _check_expression_circular_refs(model, model.metrics[id_], path=path + (id_,))
    for ref in expression.parsed_expr.find_data("dimension"):
        id_ = ref.children[0]
        if id_ in path:
            path = " -> ".join(path + (id_,))
            raise CircularReferenceError(f"Circular reference in calculations: {path}")
        _check_expression_circular_refs(
            model, model.dimensions[id_], path=path + (id_,)
        )


@check_model.depends_on(_check_expr_parse, _check_known_calculations)
def _check_circular_refs(model: Model):
    for expression in model.expressions:
        _check_expression_circular_refs(model, expression)


def _check_measure_uses_aggregate_function(measure: Measure):
    if measure.filter is not None:
        return
    for ref in measure.parsed_expr.find_data("call"):
        fn, *_ = ref.children
        if fn in aggregate_functions:
            return
    raise InvalidMeasureError(
        f"{measure} expression '{measure.str_expr}' only uses other measures "
        "and should be a metric. Remove table from the definition."
    )


@check_model.depends_on(_check_expr_parse, _check_known_functions)
def _check_measures_use_aggregate_functions(model: Model):
    for measure in model.measures.values():
        _check_measure_uses_aggregate_function(measure)


@check_model.depends_on(
    _check_expr_parse,
    _check_known_calculations,
    _check_measure_references_same_table,
    _check_measures_use_aggregate_functions,
)
def _check_measures_dont_reference_filtered_measures(model: Model):
    for measure in model.measures.values():
        for ref in measure.parsed_expr.find_data("measure"):
            id_ = ref.children[0]
            refd = measure.table.measures[id_]
            if refd.filter is not None:
                raise FilteredMeasureReferenceError(
                    f"{measure} references a filtered {refd}, "
                    "filtered measures are not allowed in measure references"
                )


@check_model.depends_on(
    _check_expr_parse, _check_known_calculations, _check_calculation_kinds
)
def _check_aggregate_dimension_table_has_pk(model: Model):
    for dimension in model.dimensions.values():
        if not isinstance(dimension, Expression):
            continue
        for _ in dimension.parsed_expr.find_data("measure"):
            if dimension.table.primary_key is None:
                raise MissingPrimaryKeyError(
                    f"{dimension} is aggregate, so its' parent {dimension.table} "
                    "requires a primary_key"
                )
