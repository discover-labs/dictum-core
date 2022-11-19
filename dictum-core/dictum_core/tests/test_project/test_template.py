from pathlib import Path

import yaml
from jinja2 import Template

from dictum_core import project

templates_path = Path(project.__file__).parent / "project_template"


def test_profiles_correct_parameters_indent():
    """There was a bug where only the first parameter was indented correctly with the rest
    indented one level below, e.g.:

    default_profile: ...
    profiles:
      ...:
        type: ...
        parameters:
          param1: ...
        param2: ... <------- incorrect indentation
        param3: ...
    """
    template = Template((templates_path / "profiles.yml").read_text())
    result = template.render(
        profile="test",
        backend="test",
        backend_parameters=yaml.safe_dump({"a": 1, "b": 2, "c": 3}),
    )
    result_data = yaml.safe_load(result)
    assert len(result_data["profiles"]["test"]["parameters"]) == 3
