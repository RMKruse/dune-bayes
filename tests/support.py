"""Shared test support — registered-family auto-discovery (GitHub #88, #89).

Registration = subclassing ``BaseFamily`` (its ``__init_subclass__`` enforces
the contract), so the recursive subclass walk finds every concrete family the
moment it is defined — gate tests parametrized over it need no hand-edit when
a family is added.  Coverage of the walk itself is asserted once, in
``tests/families/test_link_floor_gate.py``.
"""

from dune_bayes.families import BaseFamily


def concrete_families() -> list[type[BaseFamily]]:
    """All concrete SHIPPED BaseFamily subclasses, by recursive subclass walk.

    Filtered to the ``dune_bayes`` namespace: the gates exist for families the
    package ships. Test-local fixture families (e.g. the minimal Poisson in
    ``tests/metrics/test_pit.py``, issue #93) also subclass BaseFamily and
    would otherwise leak into the walk depending on collection ORDER — the
    parametrization snapshots ``__subclasses__`` at import time, so an
    unfiltered walk passes or fails based on which test module imports first.
    """
    found: list[type[BaseFamily]] = []
    stack = list(BaseFamily.__subclasses__())
    while stack:
        cls = stack.pop()
        stack.extend(cls.__subclasses__())
        if not getattr(cls, "__abstractmethods__", None) and cls.__module__.startswith(
            "dune_bayes."
        ):
            found.append(cls)
    return sorted(found, key=lambda c: c.__name__)
