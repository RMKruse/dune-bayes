"""Reader tutorial smoke checks (publication slice, GitHub #137)."""

from __future__ import annotations

import re
from pathlib import Path

TUTORIAL = Path(__file__).parents[1] / "docs" / "tutorials" / "reader_workflow.md"


def _python_blocks(markdown: str) -> list[str]:
    """Return fenced Python snippets from a Markdown document."""
    return re.findall(r"```python\n(.*?)\n```", markdown, flags=re.DOTALL)


def test_reader_tutorial_uses_project_terminology() -> None:
    """Tutorial text names the reader-facing uncertainty concepts from #137."""
    text = TUTORIAL.read_text(encoding="utf-8")
    lowered = text.lower()

    for term in (
        "shape function",
        "epistemic uncertainty",
        "aleatoric uncertainty",
        "posterior predictive",
        "variance decomposition",
    ):
        assert term in lowered


def test_reader_tutorial_avoids_private_or_unsupported_paths() -> None:
    """Tutorial should stay on public APIs and supported runtime dependencies."""
    text = TUTORIAL.read_text(encoding="utf-8")

    forbidden = (
        "._",
        "NAMpy",
        "tensorflow",
        "tensorflow_probability",
        "numpyro",
    )
    for token in forbidden:
        assert token not in text


def test_reader_tutorial_smoke_block_runs_public_api() -> None:
    """The marked tutorial snippet executes as a bounded documentation example."""
    text = TUTORIAL.read_text(encoding="utf-8")
    smoke_blocks = [
        block
        for block in _python_blocks(text)
        if "# docs-smoke: reader-workflow" in block
    ]
    assert len(smoke_blocks) == 1

    namespace: dict[str, object] = {"__name__": "__reader_tutorial_smoke__"}
    exec(smoke_blocks[0], namespace)

    assert namespace["effect_draws"].shape == (6, 24, 2)
    assert namespace["predictive_band"]["lo"].shape == (24,)
    variance = namespace["variance"]
    assert variance.aleatoric.shape == (24,)
    assert variance.epistemic.shape == (24,)


def test_reader_tutorial_snippets_are_smoke_or_illustrative() -> None:
    """Every Python snippet is either executed here or marked as illustrative."""
    text = TUTORIAL.read_text(encoding="utf-8")

    for block in _python_blocks(text):
        assert "# docs-smoke: reader-workflow" in block or "# illustrative:" in block, (
            block
        )
