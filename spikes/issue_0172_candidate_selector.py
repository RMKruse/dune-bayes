"""Throwaway decision prototype for GitHub issue 172.

Question: is the smallest evidence-backed branch the shared benchmark response
transform plus family-aware intercept initialization, or does the current
evidence justify broader model/inference work? Run with:

    .venv-torch/bin/python spikes/issue_0172_candidate_selector.py

Sources: issues 163, 168-171, and 174. This prototype selects development work;
it cannot promote a benchmark configuration or support a paper claim.
"""

from __future__ import annotations

import sys

ROOT_CAUSE_BRANCH = {"response transform", "intercept initialization"}
CANDIDATES = (
    ("response transform", "verified root cause; 6/8 Normal datasets improved"),
    ("intercept initialization", "verified Bike root cause; all 3 scores improved"),
    ("prior selection", "no challenger passed the uncertainty gate"),
    ("architecture expansion", "no controlled evidence yet"),
    ("richer posterior", "does not address the verified wrong-location fits"),
)


def verdict(selected: set[str]) -> str:
    """Return the decision state for a proposed intervention branch."""
    unsupported = selected - ROOT_CAUSE_BRANCH
    missing = ROOT_CAUSE_BRANCH - selected
    if unsupported:
        return "OVER-BROAD — remove unsupported additions before implementation."
    if missing:
        return "PARTIAL — continue only after both verified root causes are covered."
    return (
        "IMPLEMENT — evaluate this smallest branch on the development panel.\n"
        "PAPER CLAIM — undecided until confirmation and every hard guardrail pass."
    )


def render(selected: set[str]) -> None:
    """Render the complete decision state."""
    print("\033[2J\033[H", end="")
    print("\033[1mIssue 172 — smallest predictive-competitiveness branch\033[0m\n")
    print("Frozen benchmark controls (not intervention scope):")
    print("  12 candidates/dataset; validation-NLL stopping; family-matched models")
    print(
        "  fixed prior_scale=1.0; locked confirmation; uncertainty and "
        "interpretability gates\n"
    )
    print("Candidate branch:")
    for index, (name, evidence) in enumerate(CANDIDATES, 1):
        mark = "x" if name in selected else " "
        print(f"  [{mark}] {index}. {name} — \033[2m{evidence}\033[0m")
    print(f"\n\033[1mVerdict\033[0m\n{verdict(selected)}")
    print("\n[1-5] toggle  [r] recommended branch  [q] quit")


def main() -> None:
    """Run the interactive prototype or its smallest automated check."""
    if sys.argv[1:] == ["--check"]:
        assert verdict(ROOT_CAUSE_BRANCH).startswith("IMPLEMENT")
        assert verdict({"response transform"}).startswith("PARTIAL")
        assert verdict(ROOT_CAUSE_BRANCH | {"richer posterior"}).startswith(
            "OVER-BROAD"
        )
        print("3 decision checks passed")
        return

    selected = set(ROOT_CAUSE_BRANCH)
    while True:
        render(selected)
        key = input("> ").strip().lower()
        if key == "q":
            return
        if key == "r":
            selected = set(ROOT_CAUSE_BRANCH)
        elif key.isdigit() and 1 <= int(key) <= len(CANDIDATES):
            name = CANDIDATES[int(key) - 1][0]
            selected.symmetric_difference_update({name})


if __name__ == "__main__":
    main()
