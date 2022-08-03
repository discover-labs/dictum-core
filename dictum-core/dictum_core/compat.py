# Compatibility for earlier Python versions
try:
    from functools import cached_property
except ImportError:
    from cached_property import cached_property  # noqa: F401, for python < 3.8
