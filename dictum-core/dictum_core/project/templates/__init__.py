import json
from pathlib import Path

from jinja2.environment import Environment
from jinja2.loaders import FileSystemLoader

path = Path(__file__).parent
environment = Environment(loader=FileSystemLoader(searchpath=path))

lineage_spec = json.loads((path / "lineage.json").read_text())
