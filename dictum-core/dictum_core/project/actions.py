import shutil
from pathlib import Path

import yaml
from jinja2 import Template

from dictum_core.backends.base import Backend

template_path = Path(__file__).parent / "project_template"


def copy_project_template(target_path: Path, template_vars: dict):
    for path in template_path.iterdir():
        if path.name in {"__init__.py", ".gitkeep"}:
            continue
        new_path = target_path / path.relative_to(template_path)
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
        raise FileExistsError(f"{path} is file, must be an empty directory")
    if not path.exists():
        path.mkdir()
    if path.is_dir():
        if next(path.iterdir(), None) is not None:
            raise FileExistsError(f"{path} directory is not empty")
        template_vars = {
            "project_name": name,
            "profile": "default",
            "backend": backend.type,
            "backend_parameters": yaml.safe_dump(backend.parameters),
            "currency": currency,
            "locale": locale,
        }
        copy_project_template(path, template_vars)
        return
    raise Exception("This shouldn't happen")
