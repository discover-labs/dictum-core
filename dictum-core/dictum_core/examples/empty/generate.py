from dictum_core import Project
from dictum_core.project.yaml_mapped_dict import YAMLMappedDict


def generate():
    project = Project.example("chinook")
    project.model_data = YAMLMappedDict(
        {"name": "Chinook", "tables": {}, "metrics": {}, "unions": {}}
    )
    project.project_config.root = None
    project.update_model({})
    return project
