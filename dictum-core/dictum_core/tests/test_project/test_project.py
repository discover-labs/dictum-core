from dictum_core import Project


def test_pivot_in(project: Project):
    project.pivot("revenue").where(project.d.genre.isin("Rock"))


def test_example_empty():
    p = Project.example("chinook", empty=True)
    assert len(p.model.metrics) == 0


def test_staged_model_data_deepcopy():
    p = Project.example("chinook")
    assert p.staged_model_data is not p.model_data

    p.update_shorthand_metric("arppu = $revenue / $unique_paying_customers")
    assert p.staged_model_data is not p.model_data
