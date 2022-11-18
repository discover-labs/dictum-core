from collections import UserDict
from itertools import chain
from pathlib import Path

import yaml


def _update_recursive(d, u):
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = _update_recursive(d.get(k, {}), v)
        else:
            d[k] = v
    return d


class YAMLMappedDict(UserDict):

    file_extensions = {".yaml", ".yml"}

    def __init__(self, data=None, /, **kwargs):
        if data is None:
            data = {}
        self.path = None
        self.parent = None

        for k, v in data.items():
            if isinstance(v, dict):
                data[k] = YAMLMappedDict(v)
                data[k].parent = self
        return super().__init__(data, **kwargs)

    @classmethod
    def from_path(cls, path: Path):
        if path.is_file() and path.suffix in cls.file_extensions:
            value = yaml.safe_load(path.read_text())
            result = cls(value)
            result.path = path
            return result
        if path.is_dir():
            items = {}
            for subpath in chain(
                *(path.glob(f"**/*{ext}") for ext in cls.file_extensions)
            ):
                key = subpath.stem
                items[key] = cls.from_path(subpath)
            return cls(items)

    def flush(self):
        if self.path is not None:
            self.path.write_text(yaml.safe_dump(self.dict(), sort_keys=False))
        for v in self.data.values():
            if isinstance(v, YAMLMappedDict):
                v.flush()

    def __setitem__(self, key: str, item):
        if isinstance(item, dict):
            item = YAMLMappedDict(item)
            item.parent = self
        self.data[key] = item

    def update_recursive(self, update: dict):
        _update_recursive(self, update)

    def dict(self) -> dict:
        result = {}
        for k, v in self.data.items():
            if isinstance(v, YAMLMappedDict):
                result[k] = v.dict()
            else:
                result[k] = v
        return result

    def assign_paths(self, base_path: Path):
        if base_path.is_dir():
            for k, v in self.items():
                if v.path is None:
                    v.path = base_path / f"{k}.yml"
        elif base_path.is_file():
            if self.path is None:
                self.path = base_path
        else:
            raise FileNotFoundError(f"{base_path} does not exist")
