import inspect
from collections import UserDict
from functools import cached_property
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pkg_resources
from lark import Token, Transformer
from pandas import DataFrame

from dictum_core.engine import Column, LiteralOrderItem, RelationalQuery


@dataclass
class BackendResult:
    data: List[dict]
    duration: float  # ms
    raw_query: str


class CallOwnerOp:
    def __get__(self, owner, ownertype=None):
        self._owner = owner
        return self

    def __set_name__(self, owner: "Transformer", name: str):
        self._owner = owner
        self._name = name

    def __call__(self, children: list):
        raise NotImplementedError


class PassChildrenToCompiler(CallOwnerOp):
    def __call__(self, children: list):
        return getattr(self._owner.compiler, self._name)(*children)


class PassTokenValueToCompiler(CallOwnerOp):
    def __call__(self, token: Token):
        return getattr(self._owner.compiler, self._name)(token.value)


def _get_whens_else(args: list):
    whens = [args[i : i + 2] for i in range(0, len(args), 2)]  # noqa: E203
    else_ = None if len(whens[-1]) == 2 else whens.pop(-1)[0]
    return whens, else_


class ExpressionTransformer(Transformer):
    def __init__(self, compiler: "Compiler", visit_tokens: bool = True) -> None:
        self.compiler = compiler
        super().__init__(visit_tokens=visit_tokens)

    def ARG(self, _):
        raise ValueError(
            "'@' is not allowed in calculation expressions —"
            " only in user-defined filters and transforms"
        )

    FLOAT = PassTokenValueToCompiler()
    INTEGER = PassTokenValueToCompiler()
    STRING = PassTokenValueToCompiler()

    def TRUE(self, _):
        return self.compiler.TRUE()

    def FALSE(self, _):
        return self.compiler.FALSE()

    column = PassChildrenToCompiler()

    exp = PassChildrenToCompiler()
    neg = PassChildrenToCompiler()
    fdiv = PassChildrenToCompiler()
    div = PassChildrenToCompiler()
    mul = PassChildrenToCompiler()
    mod = PassChildrenToCompiler()
    add = PassChildrenToCompiler()
    sub = PassChildrenToCompiler()
    gt = PassChildrenToCompiler()
    ge = PassChildrenToCompiler()
    lt = PassChildrenToCompiler()
    le = PassChildrenToCompiler()
    eq = PassChildrenToCompiler()
    ne = PassChildrenToCompiler()

    isnull = PassChildrenToCompiler()

    def IN(self, children: list):
        value, *values = children
        return self.compiler.IN(value, values)

    NOT = PassChildrenToCompiler()
    AND = PassChildrenToCompiler()
    OR = PassChildrenToCompiler()

    def case(self, children: list):
        whens, else_ = _get_whens_else(children)
        return self.compiler.case(whens, else_=else_)

    def expr(self, children: list):
        return children[0]

    def call(self, children: list):
        fn, *args = children
        return self.compiler.call(fn, args)

    def order_by(self, children: list):
        return children

    def call_window(self, children: list):
        fn, *args, partition, order, rows = children
        return self.compiler.call_window(fn, args, partition, order, rows)

    def partition_by(self, children: list):
        return children


class Compiler(ABC):
    """Takes in a computation, returns an object that a connection will understand."""

    def __init__(self):
        self.transformer = ExpressionTransformer(self)

    # expression language elements

    @abstractmethod
    def FLOAT(self, value: str):
        """Float literal"""

    @abstractmethod
    def INTEGER(self, value: str):
        """Integer literal"""

    @abstractmethod
    def STRING(self, value: str):
        """String literal"""

    @abstractmethod
    def TRUE(self):
        """True boolean literal"""

    @abstractmethod
    def FALSE(self):
        """False boolean literal"""

    @abstractmethod
    def isnull(self, value):
        """Missing value check"""

    @abstractmethod
    def column(self, table: str, name: str):
        """Column reference.
        table is a dot-delimited sequence of identifiers
        name is the column name
        """

    def call(self, fn: str, args: list):
        """Function call. First element of children is function name, rest are
        the arguments. Calls the method on self.
        """
        call = getattr(self, fn)
        return call(*args)

    def call_window(
        self, fn: str, args: list, partition: list, order: list, rows: list
    ):
        fn = getattr(self, f"window_{fn}")
        return fn(args, partition, order, rows)

    @abstractmethod
    def exp(self, a, b):
        """Exponentiation — "power" operator, a to the power of b"""

    @abstractmethod
    def neg(self, value):
        """Unary number negation, e.g. -1"""

    @abstractmethod
    def div(self, a, b):
        """Normal division. Semantics depend on the underlying backend."""

    @abstractmethod
    def mul(sef, a, b):
        """Arithmetic multiplication"""

    @abstractmethod
    def mod(self, a, b):
        """Modulo, arithmetic remainder, e.g. 7 % 2 == 1"""

    @abstractmethod
    def add(self, a, b):
        """Arithmetic addition"""

    @abstractmethod
    def sub(self, a, b):
        """Arithmetic subtraction"""

    @abstractmethod
    def gt(self, a, b):
        """Greater than, a > b"""

    @abstractmethod
    def ge(self, a, b):
        """Greater than or equal, a >= b"""

    @abstractmethod
    def lt(self, a, b):
        """Less than, a < b"""

    @abstractmethod
    def le(self, a, b):
        """Less than or equal, a <= b"""

    @abstractmethod
    def eq(self, a, b):
        """Equality, a equals b"""

    @abstractmethod
    def ne(self, a, b):
        """Non-equality, a not equals b"""

    @abstractmethod
    def IN(self, value, values):
        """Value is in a tuple of values"""

    @abstractmethod
    def NOT(self, value):
        """Boolean negation, NOT x"""

    @abstractmethod
    def AND(self, a, b):
        """Logical conjunction"""

    @abstractmethod
    def OR(self, a, b):
        """Logical disjunction"""

    @abstractmethod
    def case(self, whens, else_=None):
        """whens: tuples of (condition, value)
        else: else value (optional)
        """

    def IF(self, *args):
        whens, else_ = _get_whens_else(args)
        return self.case(whens, else_=else_)

    # built-in functions
    # aggregate

    @abstractmethod
    def sum(self, arg):
        """Aggregate sum"""

    @abstractmethod
    def avg(self, arg):
        """Aggregate average"""

    @abstractmethod
    def min(self, arg):
        """Aggregate minimum"""

    @abstractmethod
    def max(self, arg):
        """Aggregate maximum"""

    @abstractmethod
    def count(self, arg=None):  # arg can be missing
        """Aggregate count, with optional argument"""

    @abstractmethod
    def countd(self, arg):
        """Aggregate distinct count"""

    # window functions

    @abstractmethod
    def window_sum(self, arg, partition_by, order_by, rows):
        """A windowed version of aggregate sum function"""

    @abstractmethod
    def window_row_number(self, arg, partition_by, order_by, rows):
        """Same as SQL row_number"""

    def window_avg(self, arg, partition, order, rows):
        """TODO"""

    def window_min(self, arg, partition, order, rows):
        """TODO"""

    def window_max(self, arg, partition, order, rows):
        """TODO"""

    def window_count(self, arg, partition, order, rows):
        """TODO"""

    # scalar functions

    @abstractmethod
    def abs(self, arg):
        """Absolute numeric value"""

    @abstractmethod
    def floor(self, arg):
        """Numeric floor"""

    @abstractmethod
    def ceil(self, arg):
        """Numeric ceiling"""

    @abstractmethod
    def coalesce(self, *args):
        """NULL-coalescing"""

    # type casting

    @abstractmethod
    def tointeger(self, arg):
        """cast as int"""

    @abstractmethod
    def tofloat(self, arg):
        """cast as float"""

    @abstractmethod
    def todate(self, arg):
        """cast as date"""

    @abstractmethod
    def todatetime(self, arg):
        """cast as datetime/timestamp"""

    # dates

    @abstractmethod
    def datepart(self, part, arg):
        """Part of a date as an integer. First arg is part as a string, e.g. 'month',
        second is date/datetime.
        """

    @abstractmethod
    def datetrunc(self, part, arg):
        """Date truncated to a given part. Args same as datepart."""

    @abstractmethod
    def datediff(self, part, start, end):
        """Difference between two dates, given as number of times there's a change of
        date at the given level.
        """

    @abstractmethod
    def now(self):
        """Current timestamp"""

    @abstractmethod
    def today(self):
        """Today's date"""

    # compilation

    @abstractmethod
    def compile_query(self, query: RelationalQuery):
        """Compile a single relational query into connection query."""

    @abstractmethod
    def merge_queries(self, queries: List, merge_on: List[str]):
        """Merge a list of relational queries on the relevant level of detail. Can be
        unsupported by a backend, raise NotImplementedError in this case.
        """

    @abstractmethod
    def calculate(self, query, columns: List[Column]):
        """Calculate expressions from the fields of a query"""

    @abstractmethod
    def filter(self, query, conditions: Dict[str, Any]):
        """Filter a query on a list of conditions"""

    @abstractmethod
    def filter_with_records(self, query, records: List[List[Dict[str, Any]]]):
        """Use a list of dicts as a filter for a query. Each dict is a
        combination of valid field values, the rest are filtered out.
        """

    @abstractmethod
    def inner_join(self, query, to_join, join_on: List[str]):
        """Use another query as an inner join for a query, join condition is equality
        on a list of columns (same names in both queries).
        """

    @abstractmethod
    def limit(self, query, limit: int):
        """Limit a query (mostly user with order)."""

    @abstractmethod
    def order(self, query, items: List[LiteralOrderItem]):
        """Add ordering to a query."""


class BackendRegistry(UserDict):
    def __init__(self, dict=None, /, **kwargs):
        super().__init__(dict, **kwargs)

    @cached_property
    def registry(self):
        Backend.discover_plugins()
        return self.data

    def __getitem__(self, key: str) -> "Backend":
        if key not in self.registry:
            raise ImportError(
                f"Backend {type} was not found. Try installing dictum[{type}] "
                "package."
            )
        return self.data[key]

    def __contains__(self, key: object) -> bool:
        return key in self.registry

    def __str__(self) -> str:
        return str(self.registry)


class Backend(ABC):
    """User-facing. Gets connection details, knows about it's compiler. Compiles
    the incoming computation, executes on the client.
    """

    type: str
    compiler_cls = Compiler

    registry: BackendRegistry = BackendRegistry()

    def __init__(self):
        self.compiler = self.compiler_cls(self)

    def __init_subclass__(cls):
        if hasattr(cls, "type"):
            cls.registry[cls.type] = cls

    @classmethod
    def create(cls, type: str, parameters: Optional[dict] = None):
        if parameters is None:
            parameters = {}
        return cls.registry[type](**parameters)

    @classmethod
    def parameters(cls) -> dict:
        result = {}
        for name, par in inspect.signature(cls.__init__).parameters.items():
            if name == "self":
                continue
            result[name] = par.default if par.default != inspect._empty else None
        return result

    @classmethod
    def discover_plugins(cls):
        for entry_point in pkg_resources.iter_entry_points("dictum.backends"):
            cls.registry[entry_point.name] = entry_point.load()

    def display_query(self, query):
        return str(query)

    def compile_query(self, query):
        return self.compiler.compile_query(query)

    def merge_queries(self, queries: List, merge_on: List[str]):
        return self.compiler.merge_queries(queries, merge_on)

    def calculate(self, query, columns: List[Column]):
        return self.compiler.calculate(query, columns)

    def filter(self, query, conditions: Dict[str, Any]):
        return self.compiler.filter(query, conditions)

    def filter_with_records(self, query, records: List[List[Dict[str, Any]]]):
        return self.compiler.filter_with_records(query, records)

    def inner_join(self, query, to_join, join_on: List[str]):
        return self.compiler.inner_join(query, to_join, join_on)

    def limit(self, query, limit: int):
        return self.compiler.limit(query, limit)

    def order(self, query, items: List[LiteralOrderItem]):
        return self.compiler.order(query, items)

    @abstractmethod
    def execute(self, query) -> DataFrame:
        """Execute query, return results"""
