import os
from getpass import getpass
from pathlib import Path
from typing import Dict, Optional, Union

import yaml
from jinja2 import Template
from pydantic import BaseModel

from dictum_core.schema.id import ID


class Env:
    def __init__(self):
        self.data = os.environ.copy()

    def __getattr__(self, key: str):
        try:
            return self.data[key]
        except KeyError:
            val = getpass(key)
            self.data[key] = val
            return val


env = Env()


def _load_yaml_template(path: Path):
    return yaml.load(Template(path.read_text()).render(env=env), Loader=yaml.SafeLoader)


class Profile(BaseModel):
    type: str
    parameters: dict = {}


class Profiles(BaseModel):
    default_profile: str
    profiles: Dict[ID, Profile]


class Project(BaseModel):
    name: str
    description: Optional[str]
    locale: str = "en_US"
    currency: str = "USD"

    tables_path: str = "tables"
    metrics_path: str = "metrics"
    unions_path: str = "unions.yml"
    profiles_path: str = "profiles.yml"

    root: Optional[Path]

    @classmethod
    def load(cls, path: Union[str, Path]):
        path = Path(path)
        if not path.is_dir():
            raise FileNotFoundError("Project path must be a directory")
        project_yml = path / "project.yml"
        if not project_yml.is_file():
            raise FileNotFoundError(
                "Project must contain a project.yml file at the root"
            )
        data = _load_yaml_template(project_yml)
        data["root"] = str(path)
        return cls.parse_obj(data)

    def get_profile(self, profile=None) -> dict:
        data = _load_yaml_template(self.root / self.profiles_path)
        profiles = Profiles.parse_obj(data)
        profile = profiles.default_profile if profile is None else profile
        try:
            return profiles.profiles[profile]
        except KeyError:
            raise KeyError(f"Profile {profile} is not present in profiles")
