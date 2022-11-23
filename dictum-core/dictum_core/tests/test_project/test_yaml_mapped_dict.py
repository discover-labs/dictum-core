from pathlib import Path

import yaml

from dictum_core.examples import chinook
from dictum_core.project.yaml_mapped_dict import YAMLMappedDict


def test_from_path():
    path = Path(chinook.__file__).parent / "metrics"
    result = YAMLMappedDict.from_path(path)
    assert "revenue" in result
    assert isinstance(result["revenue"], YAMLMappedDict)
    assert isinstance(result["revenue"].path, Path)


def test_update_recursive_setitem(tmp_path: Path):
    data = {"a": {"b": {"c": 1, "d": 2}}}
    path = tmp_path / "test.yml"
    path.write_text(yaml.safe_dump(data))
    d = YAMLMappedDict.from_path(path)
    d.update_recursive({"a": {"b": {"c": "test", "e": 3}}})
    d.flush()
    new_data = yaml.safe_load(path.read_text())
    assert new_data == {"a": {"b": {"c": "test", "d": 2, "e": 3}}}

    d["a"]["b"]["e"] = "42"
    d.flush()
    new_data = yaml.safe_load(path.read_text())
    assert new_data == {"a": {"b": {"c": "test", "d": 2, "e": "42"}}}


def test_set_str_to_dict():
    """There was a bug where the recursive update couldn't set a value that
    was previously a str to a dict
    """
    d = YAMLMappedDict({"a": "x"})
    d.update_recursive({"a": {"b": 1}})

    assert d["a"] == {"b": 1}


def test_allow_unicode(tmp_path: Path):
    """Make sure that the YAML produced by dumper has human-readable
    non-ASCII strings
    """
    value = "тест"
    path = tmp_path / "test.yml"
    d = YAMLMappedDict({"a": {"b": value}})
    d.path = path
    d.flush()
    assert value in path.read_text()
