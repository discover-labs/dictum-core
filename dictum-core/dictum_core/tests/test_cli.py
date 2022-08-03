from pathlib import Path
import yaml
from dictum_core import Project

from typer.testing import CliRunner

from dictum_core.cli import app

runner = CliRunner()


def test_new(tmp_path: Path):
    test = tmp_path / "test"
    result = runner.invoke(app, args=["new"], input=f"{test}\nTest project\n1\n\n")
    assert result.exit_code == 0
    assert test.is_dir()
    assert (test / "project.yml").exists()

    assert yaml.safe_load((test / "profiles.yml").read_text()) == {
        "default_profile": "sqlite",
        "profiles": {"sqlite": {"parameters": {"database": None}, "type": "sqlite"}},
    }
    assert (test / "metrics").exists() and (test / "metrics").is_dir()
    assert (test / "tables").exists() and (test / "tables").is_dir()

    Project(test)  # generated project is at least valid
