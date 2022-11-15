from typing import List, Optional, Tuple, Union

from dictum_core.interactive_model.expression import InteractiveColumn


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

    def __set_name__(self, owner, name: str):
        if self.__id is None:
            self.__id = name
        if self.__source is None:
            self.__source = self.__id

        self.__table = owner._model.add_table(
            id=self.__id, source=self.__source, primary_key=self.__primary_key
        )

    def __getitem__(self, key: str) -> InteractiveColumn:
        return InteractiveColumn(self, key)

    def __getattr__(self, attr: str) -> InteractiveColumn:
        return self[attr]
