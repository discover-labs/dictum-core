from dictum_core.backends.base import Backend
from dictum_core.interactive_model.expression import AbstractInteractiveExpression
from dictum_core.interactive_model.table import InteractiveTable
from dictum_core.model import model
from dictum_core.project.project import Project


class InteractiveModelMeta(type):
    def __new__(cls, name, bases, attrs):
        attrs["_model"] = model.Model(name=name)
        return super().__new__(cls, name, bases, attrs)


class InteractiveModel(metaclass=InteractiveModelMeta):
    _model: model.Model

    def __init_subclass__(cls):
        # runs after __set_name__ is called on all child descriptors,
        # so table IDs are set
        # create related tables
        attrs = cls.__dict__.values()
        for attr in attrs:
            if isinstance(attr, InteractiveTable):
                for src, tgt in attr._InteractiveTable__foreign_keys:
                    _, _, src_column = src.children
                    _, tgt_table, tgt_column = tgt.children
                    attr._InteractiveTable__table.add_related(
                        str_table=tgt_table._InteractiveTable__id,
                        related_key=tgt_column,
                        foreign_key=src_column,
                        alias=tgt_table._InteractiveTable__id,
                        tables=cls._model.tables,
                    )

        for attr in attrs:
            if isinstance(attr, AbstractInteractiveExpression):
                str_expr = attr.get_str_expr()
                kind = AbstractInteractiveExpression.get_kind(str_expr)
                table = None
                id_ = attr.children[0]
                name = id_.replace("_", " ").title()
                if kind in {"measure", "dimension"}:
                    table: model.Table = (
                        next(attr.find_data("interactive_column"))
                        .children[1]
                        ._InteractiveTable__table
                    )

                if kind == "dimension":
                    if attr.type_ is None:
                        raise ValueError(f"Missing type for dimension {name}")
                    cls._model.add_dimension(
                        table=table,
                        id=id_,
                        name=name,
                        str_expr=str_expr,
                        type=attr.type_,
                    )
                    continue

                if attr.type_ is None:
                    attr.type("float")

                if kind == "measure":
                    cls._model.add_measure(
                        table=table,
                        id=id_,
                        name=name,
                        str_expr=str_expr,
                        type=attr.type_,
                        metric=True,
                    )
                    continue

                cls._model.add_metric(
                    id=id_, name=name, str_expr=str_expr, type=attr.type_
                )

    @classmethod
    def create_project(cls, backend: Backend) -> Project:
        return Project(cls._model, backend)


"""
__set_name__
- set table IDs
- set name attr on calculations

__init_subclass__
- add related tables
- for each calc:
    - set first expr child to name
- for each calc:
    - resolve calc expression without name
    - create model calculation
"""
