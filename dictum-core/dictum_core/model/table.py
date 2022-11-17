from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

from lark import Tree

from dictum_core.model.calculations import Dimension, Measure, TableFilter


@dataclass
class RelatedTable:
    str_table: str
    str_related_key: str
    foreign_key: str
    alias: str

    parent: "Table"
    tables: Dict[str, "Table"]

    @property
    def table(self) -> "Table":
        if self.str_table not in self.tables:
            raise KeyError(
                f"Table {self.str_table} is referenced as related to {self.parent.id} "
                f"with alias {self.alias}, but it doesn't exist in the config"
            )
        return self.tables[self.str_table]

    @property
    def related_key(self) -> str:
        if self.str_related_key is not None:
            return self.str_related_key
        if self.table.primary_key is None:
            raise ValueError(
                f"Table {self.table.id} with alias {self.alias} "
                f"is referenced as related to {self.parent.id}, "
                f"but neither primary_key on {self.table.id} nor "
                "related_key on the relationship are specified."
            )
        return self.table.primary_key

    @property
    def join_expr(self) -> Tree:
        return Tree(
            "expr",
            [
                Tree(
                    "eq",
                    [
                        Tree("column", [self.parent.id, self.foreign_key]),
                        Tree("column", [self.parent.id, self.alias, self.related_key]),
                    ],
                )
            ],
        )


@dataclass(repr=False)
class Table:
    """Represents a relational data table"""

    id: str
    source: Union[str, Dict[str, str]]
    description: Optional[str] = None
    primary_key: Optional[str] = None
    filters: List[TableFilter] = field(default_factory=list)
    related: Dict[str, RelatedTable] = field(default_factory=dict)
    measures: Dict[str, Measure] = field(default_factory=dict)
    dimensions: Dict[str, Dimension] = field(default_factory=dict)
    measure_backlinks: Dict[str, "Table"] = field(
        default_factory=dict
    )  # measure_id -> table

    def add_related(
        self, str_table: str, related_key: str, foreign_key: str, alias: str, tables
    ):
        self.related[alias] = RelatedTable(
            str_table=str_table,
            str_related_key=related_key,
            foreign_key=foreign_key,
            alias=alias,
            parent=self,
            tables=tables,
        )

    def find_all_paths(
        self, traversed_tables: Tuple[str] = ()
    ) -> List[Tuple["Table", List["str"]]]:
        """Find all join paths from this table to other tables, avoiding cycles"""
        traversed_tables += (self.id,)
        for rel in self.related.values():
            if isinstance(rel.table, Table) and rel.table.id not in traversed_tables:
                yield rel.table, [rel.alias]
                for target, path in rel.table.find_all_paths(traversed_tables):
                    yield target, [rel.alias, *path]

    @property
    def allowed_join_paths(self) -> Dict["Table", List[str]]:
        """A dict of table id -> tuple list of related table aliases. Join targets for
        which there exists only a single join path.
        """
        paths = defaultdict(lambda: [])
        for target, path in self.find_all_paths():
            paths[target].append(path)
        result = {t: v[0] for t, v in paths.items() if len(v) == 1}
        for rel in self.related.values():
            result[rel.table] = [rel.alias]
        return result

    @property
    def dimension_join_paths(self) -> Dict[str, List[str]]:
        result = {}
        for target, path in self.allowed_join_paths.items():
            for key, dim in target.dimensions.items():
                if not dim.is_union:
                    result[key] = [self.id] + path
        for dimension_id in self.dimensions:
            result[dimension_id] = [self.id]
        return result

    @property
    def allowed_dimensions(self) -> Dict[str, "Dimension"]:
        """Which dimensions are allowed to be used with this table as anchor.
        Only those to which there's a single direct join path. If there isn't,
        dimensions must be declared directly on a table that's available for join.
        Unions are not allowed for joins.
        """
        result = {}
        for target in self.allowed_join_paths.keys():
            result.update(
                {d.id: d for d in target.dimensions.values() if not d.is_union}
            )
        result.update(self.dimensions)
        return result

    def __eq__(self, other: "Table"):
        return isinstance(other, Table) and self.id == other.id

    def __hash__(self):
        return hash(self.id)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"Table({self.id})"
