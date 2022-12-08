from dictum_core import Project
from dictum_core.engine.computation import RelationalQuery


def test_duplicate_subquery(project: Project):
    """There was a bug where RelationalQuery.add_join_path added the same aggregate
    subquery twice.
    """
    rq = RelationalQuery(source=project.model.tables["invoice_items"], join_tree=[])
    rq.add_join_path(["track", "__subquery__revenue"])
    rq.add_join_path(["track", "__subquery__revenue"])
    assert len(rq.join_tree[0].join_tree) == 1
