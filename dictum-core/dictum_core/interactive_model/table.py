from typing import List, Optional, Tuple, Union

from dictum_core import model
from dictum_core.interactive_model.expression import InteractiveColumn
from dictum_core.interactive_model.model import InteractiveModel


class InteractiveTable:
    def __init__(
        self,
        id: Optional[str] = None,
        source: Optional[Union[str, dict]] = None,
        primary_key: Optional[str] = None,
    ):
        self.__id = id
        self.__source = source
        self.__primary_key = primary_key
        self.__foreign_keys: List[Tuple[InteractiveColumn, InteractiveColumn]] = []
        self.__table = None

    def __get__(self, obj, objtype=None):
        return self

    def __set_name__(self, owner: InteractiveModel, name: str):
        if self.__id is None:
            self.__id = name
        if self.__source is None:
            self.__source = self.__id

        self.__table = owner._model.add_table(
            id=self.__id, source=self.__source, primary_key=self.__primary_key
        )
        for src, tgt in self.__foreign_keys:
            _, src_table, src_column = src.children
            _, tgt_table, tgt_column = tgt.children
            self.__table.related[tgt_table.__id] = model.RelatedTable(
                str_table=tgt_table.__id,
                str_related_key=tgt_column,
                foreign_key=src_column,
                alias=tgt_table.__id,
                parent=src_table.__table,
                tables=owner._model.tables,
            )

    def __getattr__(self, attr: str) -> InteractiveColumn:
        return InteractiveColumn(self, attr)
