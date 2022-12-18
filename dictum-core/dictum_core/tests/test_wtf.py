from lark.exceptions import VisitError

from dictum_core import Project
from dictum_core.ql.v2 import compile_query


def test_wtf(project: Project):
    query = compile_query(
        """
    from example:test
    filter music
    filter genre where revenue.top(5)
    select :genre, $revenue, $arppu
    """
    )

    res = project.engine.get_computation(query).get_result(project.backend)
    breakpoint()
