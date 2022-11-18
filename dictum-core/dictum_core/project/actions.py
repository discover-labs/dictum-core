import inspect
import shutil
from pathlib import Path

import yaml
from jinja2 import Template

from dictum_core.backends.base import Backend
from dictum_core.backends.secret import Secret

template_path = Path(__file__).parent / "project_template"


def _get_backend_parameters(backend: Backend) -> dict:
    result = {}
    for name, par in inspect.signature(backend.__init__).parameters.items():
        if name == "self":
            continue
        if name not in backend.parameters:
            raise KeyError(
                f"Missing parameter {name} in {backend.type} backend parameters"
            )
        if par.annotation is Secret:
            var_name = f"DICTUM_SECRET_{backend.type}_{name}".upper()
            result[name] = "{{ env.%s }}" % var_name
        else:
            result[name] = backend.parameters[name]
    return result


def copy_project_template(target_path: Path, template_vars: dict):
    for path in template_path.iterdir():
        if path.name in {"__init__.py", ".gitkeep"}:
            continue
        new_path = target_path / path.relative_to(template_path)
        if new_path.exists():
            raise FileExistsError(f"{new_path} already exists")
        if path.is_file():
            template = Template(path.read_text())
            rendered = template.render(**template_vars)
            new_path.write_text(rendered)
        elif path.is_dir():
            shutil.copytree(str(path), str(new_path))


def create_new_project(
    path: Path,
    backend: Backend,
    name: str,
    currency: str = "USD",
    locale: str = "en_US",
):
    if not path.parent.exists():
        raise FileNotFoundError(f"{path.parent} directory doesn't exist")
    if path.is_file():
        raise FileExistsError(f"{path} is file")
    if not path.exists():
        path.mkdir()
    if path.is_dir():
        template_vars = {
            "project_name": name,
            "profile": "default",
            "backend": backend.type,
            "backend_parameters": yaml.safe_dump(_get_backend_parameters(backend)),
            "currency": currency,
            "locale": locale,
        }
        copy_project_template(path, template_vars)
        return
    raise Exception("This shouldn't happen")
