from pathlib import Path
from typing import Optional, Union

import altair as alt
import pandas as pd
from lark import Tree

from dictum_core import examples, schema
from dictum_core.backends.base import Backend
from dictum_core.engine import Engine, Result
from dictum_core.exceptions import MissingPathError, MissingShorthandTableError
from dictum_core.model import Model
from dictum_core.project import actions, analyses
from dictum_core.project.calculations import ProjectDimensions, ProjectMetrics
from dictum_core.project.chart import ProjectChart
from dictum_core.project.magics import ProjectMagics
from dictum_core.project.magics.parser import (
    parse_shorthand_dimension,
    parse_shorthand_metric,
    parse_shorthand_related,
    parse_shorthand_table,
)
from dictum_core.project.templates import environment
from dictum_core.project.yaml_mapped_dict import YAMLMappedDict
from dictum_core.schema import Query


def _get_subtree_str(text: str, tree: Tree):
    s, e = tree.meta.start_pos, tree.meta.end_pos
    return text[s:e]


def _get_calculation_kwargs(definition: str, tree: Tree) -> dict:
    result = {}

    id_ = next(tree.find_data("id")).children[0]
    result["name"] = id_.replace("_", " ").title()

    expr = next(tree.find_data("expr"))
    result["expr"] = _get_subtree_str(definition, expr)

    for ref in tree.find_data("type"):
        result["type"] = ref.children[0]

    for ref in tree.find_data("table"):
        result["table"] = ref.children[0]

    for ref in tree.find_data("filter"):
        result["filter"] = _get_subtree_str(definition, ref)

    for ref in tree.find_data("alias"):
        result["name"] = ref.children[0]

    for ref in tree.find_data("properties"):
        result.update(ref.children[0])

    return id_, result


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
        self.staged_model_data = model_data
        model_config = schema.Model.parse_obj(model_data)

        self.model = Model.from_config(model_config)

        self.engine = Engine(self.model)
        self.metrics = ProjectMetrics(self)
        self.dimensions = ProjectDimensions(self)
        self.m, self.d = self.metrics, self.dimensions

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
            raise MissingPathError(path)

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
        return computation.graph(self.backend)

    def ql(self, query: str) -> analyses.QlQuery:
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
    def example(cls, name: str, empty: bool = False) -> "Project":
        """Load an example project.

        Arguments:
            name (str):
                Name of the example project. Valid values: ``chinook``,
                ``empty``.
            empty (bool):
                If True, the model will be empty.
                Defaults to False.


        Returns:
            CachedProject: same as ``Project``, but won't read the model config at each
            method invocation.
        """
        path = Path(examples.__file__).parent / name
        if not path.is_dir():
            raise FileNotFoundError(f"Example project {name} does not exist")
        result = Project.from_path(path)
        # prevent users from changing examples
        result.project_config.root = None
        if empty:  # initialize an empty model
            model_data = schema.Model(
                name=result.project_config.name,
                locale=result.project_config.locale,
                currency=result.project_config.currency,
            ).dict()
            result.model_data = YAMLMappedDict(model_data)
        else:  # remove all paths to avoid overwriting by user
            result.model_data = YAMLMappedDict(result.model_data.dict())
        return result

    def describe(self) -> alt.Chart:
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
                row = dict(
                    metric=metric.id,
                    metric_name=metric.name,
                    dimension=dimension.id,
                    dimension_name=dimension.name,
                )
                row["dimension_table"] = ""
                if hasattr(dimension, "table"):
                    row["dimension_table"] = dimension.table.id
                data.append(row)
        df = pd.DataFrame(data)
        return (
            alt.Chart(df)
            .mark_rect(stroke="white")
            .encode(
                x=alt.X(
                    "metric:N",
                    axis=alt.Axis(title="Metrics", orient="top", labelAngle=-45),
                ),
                y=alt.Y("dimension:N", axis=alt.Axis(title="Dimensions")),
                tooltip=[
                    {"title": "Metric ID", "field": "metric", "type": "nominal"},
                    {"title": "Metric Name", "field": "metric_name", "type": "nominal"},
                    {"title": "Dimension ID", "field": "dimension", "type": "nominal"},
                    {
                        "title": "Dimension Name",
                        "field": "dimension_name",
                        "type": "nominal",
                    },
                    {
                        "title": "Dimension Table",
                        "field": "dimension_table",
                        "type": "nominal",
                    },
                ],
            )
            .properties(width=alt.Step(15), height=alt.Step(15))
        )

    def magic(self):
        from IPython import get_ipython  # so that linters don't whine

        ip = get_ipython()
        if ip is not None:
            ip.register_magics(ProjectMagics(project=self, shell=ip))
            print(
                r"Magics %ql, %table, %metric, %dimension are registered and bound "
                f"to project {self.project_config.name}"
            )

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
        data = {"source": source}

        pk = next(table_def.find_data("pk"), None)
        if pk is not None:
            data["primary_key"] = pk.children[0]
        self.stage_model_data({"tables": {table: data}})

        # add items
        for item in items:
            if item.data == "related":
                str_shorthand = _get_subtree_str(definition, item)
                self.update_shorthand_related(f"{table}.{str_shorthand}", commit=False)
            elif item.data == "dimension":
                self.update_shorthand_dimension(
                    _get_subtree_str(definition, item), table, commit=False
                )
            elif item.data == "metric":
                self.update_shorthand_metric(
                    _get_subtree_str(definition, item), table, commit=False
                )
            elif item.data == "table_format":
                self.update_shorthand_format(
                    _get_subtree_str(definition, item.children[0])
                )

        self.commit_model_data()

    def update_shorthand_related(self, definition: str, commit: bool = True):
        tree = parse_shorthand_related(definition)
        tables = tree.find_data("table")
        parent = next(tables).children[0]
        target = next(tables, None)
        alias = next(tree.find_data("alias")).children[0]
        if target is None:
            raise MissingShorthandTableError(
                f"\nTable is required for standalone related shorthand: table.{alias}\n"
                "                                                    ^^^^^"
            )
        target = target.children[0]
        # parent, target = list(t.children[0] for t in tree.find_data("table"))
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
        self.stage_model_data(update)
        if commit:
            self.commit_model_data()

    def update_shorthand_metric(
        self, definition: str, table: Optional[str] = None, commit: bool = True
    ):
        tree = parse_shorthand_metric(definition)
        id_, kwargs = _get_calculation_kwargs(definition, tree)
        kwargs["table"] = kwargs.get("table", table)
        update = {"metrics": {id_: kwargs}}
        self.stage_model_data(update)
        if commit:
            self.commit_model_data()

    def update_shorthand_dimension(
        self, definition: str, table: Optional[str] = None, commit: bool = True
    ):
        tree = parse_shorthand_dimension(definition)
        id_, kwargs = _get_calculation_kwargs(definition, tree)
        schema.Dimension.parse_obj(kwargs)  # validate before updating
        if table is None:
            table = kwargs.pop("table", None)
        if table is None:
            raise ValueError("Table is required, please specify with '@ table'")
        update = {"tables": {table: {"dimensions": {id_: kwargs}}}}
        self.stage_model_data(update)
        if commit:
            self.commit_model_data()

    def stage_model_data(self, update: dict):
        # avoid updating model_data until the staged model is checked
        self.staged_model_data.update_recursive(update)

    def commit_model_data(self):
        model_config = schema.Model.parse_obj(self.staged_model_data)
        try:
            self.model = Model.from_config(model_config)  # model checks happen here
        except Exception:
            # staged model data is invalid, rollback an re-raise
            self.staged_model_data = self.model_data
            raise

        # we're ok, can update model data
        self.model_data = self.staged_model_data

        # do the rest of the stuff
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

        if not path.exists() or not (path / "project.yml").exists():
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
