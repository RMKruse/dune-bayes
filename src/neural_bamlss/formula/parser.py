"""Formula-string parser — additive terms (issue 0016 / GitHub #37).

Turns ``"y ~ BayesianMLP(x1) + NeuralLinearMLP(x2)"`` into the
``dict[str, nn.Module]`` formula that ``BayesianNAMLSS`` consumes, resolving
shape-function names via ``ShapeFunctionRegistry``. Reimplemented from
scratch per ADR-0006 (amended) — not ported from NAMpy's TF ``FormulaHandler``.

Scope: additive ``+``-separated single-input terms; interactions (``:``) are
issue 0017.
"""

from __future__ import annotations

import ast
import inspect
from dataclasses import dataclass, field
from typing import Any

import torch.nn as nn

from neural_bamlss.shapes import ShapeFunctionRegistry


@dataclass(frozen=True)
class Term:
    """One additive formula term, e.g. ``BayesianMLP(x1)``.

    Args:
        shape_name: Registry key of the shape function.
        feature: Input feature name (the X-dict key).
        kwargs: Per-term constructor keyword arguments.
    """

    shape_name: str
    feature: str
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ParsedFormula:
    """Result of :func:`parse_formula`.

    Args:
        response: Response name left of ``~``.
        terms: Parsed additive terms, in formula order.
    """

    response: str
    terms: tuple[Term, ...]


def _flatten_additive(node: ast.expr) -> list[ast.Call]:
    """Flatten a left-nested ``a + b + c`` BinOp tree into call nodes in order."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _flatten_additive(node.left) + _flatten_additive(node.right)
    if isinstance(node, ast.Call):
        return [node]
    raise ValueError(
        f"Cannot parse formula term {ast.unparse(node)!r}: expected "
        "'+'-separated ShapeName(feature, ...) calls."
    )


def _parse_term(call: ast.Call) -> Term:
    """Turn one ``ShapeName(feature, key=value, ...)`` call node into a Term."""
    if not isinstance(call.func, ast.Name):
        raise ValueError(
            f"Cannot parse formula term {ast.unparse(call)!r}: term must be "
            "a plain ShapeName(feature, ...) call."
        )
    if len(call.args) != 1 or not isinstance(call.args[0], ast.Name):
        raise ValueError(
            f"Term {ast.unparse(call)!r} must have exactly one feature name "
            "as its positional argument (interactions are issue 0017)."
        )
    kwargs: dict[str, Any] = {}
    for kw in call.keywords:
        if kw.arg is None:  # **something — not a formula construct
            raise ValueError(
                f"Term {ast.unparse(call)!r}: '**' unpacking is not supported."
            )
        # literal_eval restricts values to safe Python literals (numbers,
        # strings, tuples, ...) — formula strings never execute code.
        try:
            kwargs[kw.arg] = ast.literal_eval(kw.value)
        except ValueError:
            raise ValueError(
                f"Term {ast.unparse(call)!r}: keyword {kw.arg!r} must be a "
                "literal value (number, string, tuple, bool)."
            ) from None
    return Term(shape_name=call.func.id, feature=call.args[0].id, kwargs=kwargs)


def parse_formula(formula: str) -> ParsedFormula:
    """Parse an additive formula string.

    Args:
        formula: e.g. ``"y ~ BayesianMLP(x1, prior_scale=0.5) + MLP(x2)"``.

    Returns:
        The parsed response name and terms.

    Raises:
        ValueError: If the formula is not ``response ~ term + term + ...``
            with each term a ``ShapeName(feature, key=literal, ...)`` call.
    """
    if formula.count("~") != 1:
        raise ValueError(
            f"Formula {formula!r} must contain exactly one '~' separating "
            "the response from the additive terms."
        )
    lhs, rhs = formula.split("~")
    response = lhs.strip()
    if not response.isidentifier():
        raise ValueError(
            f"Response name {response!r} (left of '~') must be a valid identifier."
        )
    # The RHS grammar is a subset of Python expressions: '+'-chained calls
    # with literal keywords. Parsing it as Python handles nesting (tuples in
    # kwargs) that naive string-splitting gets wrong.
    try:
        tree = ast.parse(rhs.strip(), mode="eval")
    except SyntaxError as err:
        raise ValueError(f"Cannot parse formula RHS {rhs.strip()!r}: {err}") from None
    terms = tuple(_parse_term(call) for call in _flatten_additive(tree.body))
    seen: set[str] = set()
    for term in terms:
        # The formula dict is keyed by feature name — a duplicate would
        # silently drop an earlier term, so reject it outright.
        if term.feature in seen:
            raise ValueError(
                f"Feature {term.feature!r} appears in more than one term; "
                "each feature may appear once in an additive formula."
            )
        seen.add(term.feature)
    return ParsedFormula(response=response, terms=terms)


def build_formula(
    parsed: ParsedFormula, family: Any, n_obs: int | None = None
) -> dict[str, nn.Module]:
    """Instantiate shape functions for each parsed term.

    Args:
        parsed: Output of :func:`parse_formula`.
        family: Family object supplying ``param_count`` (output width).
        n_obs: Training-set size N. When given, terms whose constructor
            accepts ``kl_divisor`` (and whose kwargs don't set it) get
            ``kl_divisor=n_obs`` — the ELBO objective is mean-NLL + KL/N,
            and that /N lives inside each VariationalDense.

    Returns:
        Mapping of feature name → shape-function instance, the formula dict
        ``BayesianNAMLSS`` consumes.

    Raises:
        ValueError: If a term names an unregistered shape function.
    """
    formula: dict[str, nn.Module] = {}
    for term in parsed.terms:
        shape_cls = ShapeFunctionRegistry.get(term.shape_name)
        if shape_cls is None:
            raise ValueError(
                f"Unknown shape function {term.shape_name!r} in term "
                f"{term.shape_name}({term.feature}). Registered names in "
                f"ShapeFunctionRegistry: {ShapeFunctionRegistry.names()}."
            )
        kwargs = dict(term.kwargs)
        if n_obs is not None and "kl_divisor" not in kwargs:
            # Signature check, not a Bayesian-ness check: deterministic shape
            # functions simply have no kl_divisor parameter to wire.
            if "kl_divisor" in inspect.signature(shape_cls.__init__).parameters:
                kwargs["kl_divisor"] = float(n_obs)
        formula[term.feature] = shape_cls(
            in_features=1, param_count=family.param_count, **kwargs
        )
    return formula
