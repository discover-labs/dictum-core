from graphlib import TopologicalSorter
from typing import List


class OrderedCheckCaller:
    """Allows the user to specify dependencies for each function
    and then calls all functions with the same arguments, but in the topological
    order.
    """

    def __init__(self) -> None:
        self.graph = {}
        self.fns = {}

    def register(self, fn: callable):
        self.fns[fn.__name__] = fn
        self.graph.setdefault(fn.__name__, set())

    def depends_on(self, *dependencies):
        def decorator(fn):
            self.register(fn)
            for dep in dependencies:
                self.register(dep)
                self.graph[fn.__name__].add(dep.__name__)
            return fn

        return decorator

    def static_order(self) -> List[str]:
        return TopologicalSorter(self.graph).static_order()

    def __call__(self, *args, **kwargs):
        for k in self.static_order():
            self.fns[k](*args, **kwargs)
