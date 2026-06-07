"""Shared test support — registered-family auto-discovery (GitHub #88, #89).

Registration = subclassing ``BaseFamily`` (its ``__init_subclass__`` enforces
the contract), so the recursive subclass walk finds every concrete family the
moment it is defined — gate tests parametrized over it need no hand-edit when
a family is added.  Coverage of the walk itself is asserted once, in
``tests/families/test_link_floor_gate.py``.
"""

from dune_bayes.families import BaseFamily


def concrete_families() -> list[type[BaseFamily]]:
    """All concrete BaseFamily subclasses, found by recursive subclass walk."""
    found: list[type[BaseFamily]] = []
    stack = list(BaseFamily.__subclasses__())
    while stack:
        cls = stack.pop()
        stack.extend(cls.__subclasses__())
        if not getattr(cls, "__abstractmethods__", None):
            found.append(cls)
    return sorted(found, key=lambda c: c.__name__)
