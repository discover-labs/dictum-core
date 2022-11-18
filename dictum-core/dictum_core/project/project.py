import importlib
from pathlib import Path
from typing import Optional, Union

import altair as alt
import pandas as pd
from lark import Tree

from dictum_core import schema
from dictum_core.backends.base import Backend
from dictum_core.engine import Engine, Result
from dictum_core.model import Model
from dictum_core.project import actions, analyses
from dictum_core.project.calculations import ProjectDimensions, ProjectMetrics
from dictum_core.project.chart import ProjectChart
from dictum_core.project.magics import ProjectMagics
from dictum_core.project.magics.parser import (
    parse_shorthand_calculation,
    parse_shorthand_format,
    parse_shorthand_related,
    parse_shorthand_table,
)
from dictum_core.project.templates import environment
from dictum_core.project.yaml_mapped_dict import YAMLMappedDict
from dictum_core.schema import Query


def _get_subtree_str(text: str, tree: Tree):
    s, e = tree.meta.start_pos, tree.meta.end_pos
    return text[s:e]


def _get_calculation_kwargs(definition: str) -> dict:
    result = {}
    tree = parse_shorthand_calculation(definition)

    expr = next(tree.find_data("expr"))
    result["expr"] = _get_subtree_str(definition, expr)

    for ref in tree.find_data("table"):
        result["table"] = ref.children[0]
    for ref in tree.find_data("type"):
        result["type"] = ref.children[0]
    id_ = next(tree.find_data("id")).children[0]
    result["name"] = id_.replace("_", " ").title()

    for ref in tree.find_data("alias"):
        result["name"] = ref.children[0]

    return id_, result


def _update_nested(d: dict, u: dict):
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = _update_nested(d.get(k, {}), v)
        else:
            d[k] = v
    return d


class Project:
    def __init__(
        self,
        model_data: YAMLMappedDict,
        backend: Backend,
        project_config: Optional[schema.Project] = None,
    ):
        self.project_config = project_config
        self.backend = backend

        self.model_data = model_data
        model_config = schema.Model.parse_obj(model_data)

        self.model = Model.from_config(model_config)

        self.engine = Engine(self.model)
        self.metrics = ProjectMetrics(self)
        self.dimensions = ProjectDimensions(self)
        self.m, self.d = self.metrics, self.dimensions

        self.latest_calc = None

        if self.model.theme is not None:
            alt.themes.register("dictum_theme", lambda: self.model.theme)
            alt.themes.enable("dictum_theme")

        self.magic()

    @classmethod
    def new(
        cls,
        backend: Backend,
        path: Optional[Path] = None,
        name: str = "Untitled",
        locale: str = "en_US",
        currency: str = "USD",
    ):
        """Create a new project with an empty model. Useful for experimenting
        in Jupyter. If path is provided, creates a new project at that path.
        """
        if path is not None:
            path = Path(path)
            if path.exists() and path.is_dir() and (path / "project.yml").exists():
                print(f"Project already exists, loading project from {path}")
                return Project.from_path(path=path)
            actions.create_new_project(
                path=path, backend=backend, name=name, currency=currency, locale=locale
            )
            print(f"Created a new project at {path}")
            return Project.from_path(path=path)
        model_data = schema.Model(name=name, locale=locale, currency=currency).dict()
        model_data = YAMLMappedDict(model_data)
        project_config = schema.Project(name=name, locale=locale, currency=currency)
        return cls(
            model_data=model_data, backend=backend, project_config=project_config
        )

    @classmethod
    def from_path(
        cls, path: Optional[Union[str, Path]] = None, profile: Optional[str] = None
    ) -> "Project":
        if path is None:
            path = Path.cwd()
        if isinstance(path, str):
            path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Path {path} does not exist")

        project_config = schema.Project.load(path)

        model_data = YAMLMappedDict()
        model_data["name"] = project_config.name
        model_data["description"] = project_config.description
        model_data["locale"] = project_config.locale
        model_data["currency"] = project_config.currency
        model_data["tables"] = YAMLMappedDict.from_path(
            path / project_config.tables_path
        )
        model_data["metrics"] = YAMLMappedDict.from_path(
            path / project_config.metrics_path
        )
        model_data["unions"] = YAMLMappedDict.from_path(
            path / project_config.unions_path
        )

        profile = project_config.get_profile(profile)
        backend = Backend.create(profile.type, profile.parameters)

        return cls(
            model_data=model_data, backend=backend, project_config=project_config
        )

    def execute(self, query: Query) -> Result:
        computation = self.engine.get_computation(query)
        return computation.execute(self.backend)

    def query_graph(self, query: Query):
        computation = self.engine.get_computation(query)
        return computation.graph()

    def ql(self, query: str):
        return analyses.QlQuery(self, query)

    def select(self, *metrics: str) -> "analyses.Select":
        """
        Select metrics from the project.

        Arguments:
            *metrics: Metric IDs to select.

        Returns:
            A ``Select`` object that can be further modified by chain-calling it's
            methods.
        """
        return analyses.Select(self, *metrics)

    def pivot(self, *metrics: str) -> "analyses.Pivot":
        """Select metrics from the project and construct a pivot table.

        Arguments:
            *metrics: Metric IDs to select.

        Returns:
            A ``Select`` object that can be further modified by chain-calling it's
            methods.
        """
        return analyses.Pivot(self, *metrics)

    @property
    def chart(self) -> ProjectChart:
        return ProjectChart(self)

    @classmethod
    def example(cls, name: str) -> "Project":
        """Load an example project.

        Arguments:
            name (str):
                Name of the example project. Valid values: ``chinook``,
                ``empty``.

        Returns:
            CachedProject: same as ``Project``, but won't read the model config at each
            method invocation.
        """
        example = importlib.import_module(f"dictum_core.examples.{name}.generate")
        return example.generate()

    def describe(self) -> pd.DataFrame:
        """Show project's metrics and dimensions and their compatibility. If a metric
        can be used with a dimension, there will be a ``+`` sign at the intersection of
        their respective row and column.

        Returns:
            pandas.DataFrame: metric-dimension compatibility matrix
        """
        print(
            f"Project '{self.model.name}', {len(self.model.metrics)} metrics, "
            f"{len(self.model.dimensions)} dimensions. "
            f"Connected to {self.backend}."
        )
        data = []
        for metric in self.model.metrics.values():
            for dimension in metric.dimensions:
                data.append((metric.id, dimension.id, "✚"))
        return (
            pd.DataFrame(data=data, columns=["metric", "dimension", "check"])
            .pivot(index="dimension", columns="metric", values="check")
            .fillna("")
        )

    def magic(self):
        from IPython import get_ipython  # so that linters don't whine

        ip = get_ipython()
        if ip is not None:
            ip.register_magics(ProjectMagics(project=self, shell=ip))

    def _repr_html_(self):
        template = environment.get_template("project.html.j2")
        return template.render(project=self)

    def update_shorthand_table(self, definition: str):
        tree = parse_shorthand_table(definition)

        table_def, *items = tree.children
        table = table_def.children[0].children[0]
        source = next(table_def.find_data("source"), None)
        if source is not None:
            source = source.children[0]
        if source is None:
            source = table
        data = {"id": table, "source": source}

        pk = next(table_def.find_data("pk"), None)
        if pk is not None:
            data["primary_key"] = pk.children[0]
        self.update_model({"tables": {table: data}})

        # add items
        for item in items:
            if item.data == "related":
                str_shorthand = _get_subtree_str(definition, item)
                self.update_shorthand_related(f"{table} {str_shorthand}")
            elif item.data == "dimension":
                self.update_shorthand_dimension(
                    _get_subtree_str(definition, item.children[0]), table
                )
            elif item.data == "metric":
                self.update_shorthand_metric(
                    _get_subtree_str(definition, item.children[0]), table
                )
            elif item.data == "table_format":
                self.update_shorthand_format(
                    _get_subtree_str(definition, item.children[0])
                )

    def update_shorthand_related(self, definition: str):
        tree = parse_shorthand_related(definition)
        target, parent = list(t.children[0] for t in tree.find_data("table"))
        alias = next(tree.find_data("alias")).children[0]
        columns = list(c.children[0] for c in tree.find_data("column"))
        foreign_key = columns[0]
        related_key = None
        if len(columns) == 2:
            related_key = columns[1]
        update = {
            "tables": {
                parent: {
                    "related": {
                        alias: {
                            "table": target,
                            "related_key": related_key,
                            "foreign_key": foreign_key,
                        }
                    }
                }
            }
        }
        self.update_model(update)

    def update_shorthand_metric(self, definition: str, table: Optional[str] = None):
        id_, calc = _get_calculation_kwargs(definition)
        calc["table"] = calc.get("table", table)
        update = {"metrics": {id_: calc}}
        self.update_model(update)
        self.latest_calc = self.model_data["metrics"][id_]

    def update_shorthand_dimension(self, definition: str, table: Optional[str] = None):
        id_, calc = _get_calculation_kwargs(definition)
        schema.Dimension.parse_obj(calc)  # validate before updating
        if table is None:
            table = calc.pop("table", None)
        if table is None:
            raise ValueError("Table is required, please specify with '@ table'")
        update = {"tables": {table: {"dimensions": {id_: calc}}}}
        self.update_model(update)

        self.latest_calc = self.model_data["tables"][table]["dimensions"][id_]

    def update_shorthand_format(self, definition: str):
        format = parse_shorthand_format(definition)
        if isinstance(format.children[0], str):
            format = format.children[0]
        else:
            format = dict(format.children)
        update = {"format": format}
        self.latest_calc.update_recursive(update)
        self.update_model({})  # trigger model update

    def update_model(self, update: dict):
        self.model_data.update_recursive(update)
        model_config = schema.Model.parse_obj(self.model_data)
        self.model = Model.from_config(model_config)
        self.engine = Engine(self.model)
        self.metrics = ProjectMetrics(self)
        self.dimensions = ProjectDimensions(self)
        self.m, self.d = self.metrics, self.dimensions

    def write(self, path: Optional[Union[str, Path]] = None):
        if isinstance(path, str):
            path = Path(path)

        if path is None and self.project_config.root is None:
            raise ValueError(
                "Project doesn't have a path, please specify as an argument to write()"
            )
        if self.project_config.root is None:
            self.project_config.root = path
        path = self.project_config.root

        if not path.exists():
            actions.create_new_project(
                path,
                backend=self.backend,
                name=self.project_config.name,
                currency=self.project_config.currency,
                locale=self.project_config.locale,
            )

        # assign paths where they're not present
        self.model_data["tables"].assign_paths(
            self.project_config.root / self.project_config.tables_path
        )
        self.model_data["metrics"].assign_paths(
            self.project_config.root / self.project_config.metrics_path
        )
        self.model_data["unions"].assign_paths(
            self.project_config.root / self.project_config.unions_path
        )

        self.model_data.flush()
