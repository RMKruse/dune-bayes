"""Pytest collection policy for repository-level test tiers."""

from __future__ import annotations

from pathlib import Path

import pytest

_EXPERIMENT_TESTS = Path(__file__).parent / "experiments"


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Mark experiment-tier tests so the default correctness gate stays fast."""
    for item in items:
        if _EXPERIMENT_TESTS in item.path.parents:
            item.add_marker(pytest.mark.experiment)
