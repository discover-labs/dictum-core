import textwrap


class DictumError(Exception):
    """Root exception class"""


class LoadingError(DictumError):
    """Problems occurring when project YAML files are being loaded"""


class MissingPathError(LoadingError):
    """When any of the project paths don't exist"""

    def __init__(self, path, *args) -> None:
        super().__init__(f"Path {path} does not exist", *args)


class DuplicateFileError(LoadingError):
    """When file-based objects stored in nested folders have
    duplicate IDs (filenames)
    """


class ModelError(DictumError):
    """Problems with the model"""


class InvalidIDError(ModelError):
    """When an object's ID is invalid:
    - not a valid Python ID
    - starts with __ (reserved for internal use)
    - starts with an uppercase letter (reserved for internal use)
    """


class CalculationError(ModelError):
    """Errors with calculations (except expressions)"""


class DuplicateCalculationError(CalculationError):
    """When calculations are duplicate, only possible with dimensions"""


class MissingCalculationError(CalculationError):
    """When a calculation referenced outside an expression doesn't exist:
    - union
    - time
    """


class MissingMetricTableError(CalculationError):
    """When a metric must specify a table, but doesn't"""


class TableError(ModelError):
    """Problems with tables"""


class RelatedTableError(TableError):
    """Problems with related tables"""


class MissingRelatedTableError(RelatedTableError):
    """When the specified related table is not present in the model"""


class MissingRelatedKeyError(RelatedTableError):
    """When related_key is not specified and the related table doesn't
    have a primary_key specified
    """


class MissingPrimaryKeyError(TableError):
    """When a table requires a primary_key"""


class ExpressionError(ModelError):
    """Problems with expressions"""


class ExpressionSyntaxError(ExpressionError):
    """Error when parsing with lark"""

    def __init__(self, lark_exc, calc, *args):
        base_text = f"\n{lark_exc.__class__.__name__} error in {calc}: "
        context = textwrap.indent(
            lark_exc.get_context(calc.str_expr),
            prefix=" " * (len(base_text) - 1),
            predicate=lambda x: "^" in x,
        )
        self.lark_exc = lark_exc
        super().__init__(f"{base_text}{context}", *args)


class UnknownFunctionError(ExpressionError):
    """When a function used in the expression doesn't exist"""


class MixedExpressionError(ExpressionError):
    """When an expression mixes aggregate and non-aggregate parts,
    e.g. sum(amount) + tax
    """


class AggregateFunctionArgumentError(ExpressionError):
    """When an aggregate function's argument is aggregate itself"""


class AggregateDimensionError(ExpressionError):
    """When a dimension's expression is aggregate, e.g. sum(amount), except when it's
    a metric reference.
    """


class NonAggregateMeasureError(ExpressionError):
    """When a measure's expression is not aggregate, e.g. `amount * 1000`"""


class ReferenceError(ExpressionError):
    """Problems with expressions referencing other expressions"""


class InvalidReferenceError(ReferenceError):
    """When a certain kind of expression can't reference another kind of expression,
    e.g.
    - measures -> only columns and dimensions
    - metrics -> only other metrics and measures
    - dimensions -> only other dimensions, columns and metrics
    """


class AggregateDimensionPrimaryKeyError(ReferenceError):
    """When there's no primary key on an aggregate dimension's parent table"""


class MissingReferenceError(ReferenceError):
    """Problems with calculations referencing calculations that are not available
    with known joins or don't exist
    """


class MissingDimensionError(MissingReferenceError):
    """When referenced dimension is missing"""


class MissingAggregationError(MissingReferenceError):
    """When referenced metric or measure is missing"""


class MissingMeasureError(MissingReferenceError):
    """When referenced measure on the same table isn't found"""


class FilteredMeasureReferenceError(ReferenceError):
    """When a measure references a filtered measure"""


class InvalidMeasureError(ReferenceError):
    """When a measure doesn't reference any aggregate functions
    and should be a metric
    """


class CircularReferenceError(ReferenceError):
    """When calculations reference each other, and so can't be resolved"""


class ShorthandError(ModelError):
    """Problems with the interactive shorthand syntax for Jupyter"""


class ShorthandSyntaxError(ShorthandError):
    """Nicer syntax errors for shorthands"""

    def __init__(self, lark_exc, definition, *args):
        base_text = f"\n{lark_exc.__class__.__name__} error in expression: "
        context = textwrap.indent(
            lark_exc.get_context(definition),
            prefix=" " * (len(base_text) - 1),
            predicate=lambda x: "^" in x,
        )
        self.lark_exc = lark_exc
        super().__init__(f"{base_text}{context}", *args)


class MissingShorthandTableError(ShorthandError):
    """When a shorthand requires a table, but none is provided"""


class QueryError(DictumError):
    """Problems with the Query object"""


class MissingQueryMetricError(QueryError):
    """When requested metric doesn't exist"""


class MissingQueryDimensionError(QueryError):
    """When requested dimension doesn't exist"""


class MissingScalarTransformError(QueryError):
    """When requested scalar transform doesn't exist"""


class MissingTableTransformError(QueryError):
    """When requested table transform doesn't exist"""


class MissingTableTransformDimensionError(QueryError):
    """When transforms listed in of/within are not present in the query"""


class MisusedTableTransformError(QueryError):
    """Subj. e.g. when TOP is used in the select list or with scalars"""


class ScalarTransformTypeError(QueryError):
    """When there's funky business with scalar transform input types"""


class DimensionUnavailableError(QueryError):
    """When dimension exists, but can't be joined"""


class DuplicateColumnError(QueryError):
    """When a query is requesting duplicate columns in the output"""
