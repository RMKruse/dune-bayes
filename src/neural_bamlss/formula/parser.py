"""Formula-string parser — additive and interaction terms (issue 0016–0017).

Turns ``"y ~ BayesianMLP(x1) + BayesianMLP(x2):BayesianMLP(x3)"`` into the
``dict[str, nn.Module]`` formula that ``BayesianNAMLSS`` consumes, resolving
shape-function names via ``ShapeFunctionRegistry``. Reimplemented from
scratch per ADR-0006 (amended) — not ported from NAMpy's TF ``FormulaHandler``.

Interaction syntax: ``Net(x1):Net(x2)`` creates a single joint net over both
inputs, keyed ``"x1:x2"`` in the formula dict, with net type and kwargs taken
from the first factor (ADR-0005, issue 0017 / GitHub #41).
"""

from __future__ import annotations

import ast
import inspect
import re
from dataclasses import dataclass, field
from typing import Any

import torch.nn as nn

from neural_bamlss.shapes import ShapeFunctionRegistry


@dataclass(frozen=True)
class Term:
    """One formula term — additive (``BayesianMLP(x1)``, one feature) or a
    ``:``-joined interaction (``BayesianMLP(x1):BayesianMLP(x2)``, two or more).

    An interaction's joint net uses the shape name and kwargs from the first
    factor (ADR-0005); remaining factors contribute only their feature name.
    The formula dict key is the colon-joined feature names.

    Args:
        shape_name: Registry key of the shape function (first factor for
            interactions).
        features: Feature names, in formula order. One entry for an additive
            term, two or more for an interaction.
        kwargs: Per-term constructor keyword arguments (first factor for
            interactions).
    """

    shape_name: str
    features: tuple[str, ...]
    kwargs: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        """Formula dict key: colon-joined feature names."""
        return ":".join(self.features)


@dataclass(frozen=True)
class ParsedFormula:
    """Result of :func:`parse_formula`.

    Args:
        response: Response name left of ``~``.
        terms: Parsed terms (additive and/or interaction), in formula order.
    """

    response: str
    terms: tuple[Term, ...]


def _transform_interactions(rhs: str) -> str:
    # Replace the formula `:` between calls with `*` so ast.parse can handle
    # it: `Net(x1):Net(x2)` → `Net(x1)*Net(x2)`. The regex matches `)` followed
    # by optional whitespace, `:`, optional whitespace, and an identifier start
    # (uppercase or lowercase letter / underscore), which unambiguously identifies
    # a term boundary — colon inside kwargs (dict literals) never follows `)`.
    return re.sub(r"\)\s*:\s*(?=[A-Za-z_])", ")*", rhs)


def _flatten_additive(node: ast.expr) -> list[ast.expr]:
    """Flatten a ``+``-tree into top-level term nodes (Call or Mult BinOp)."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _flatten_additive(node.left) + _flatten_additive(node.right)
    if isinstance(node, ast.Call):
        return [node]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        # Interaction term (`:` rewritten to `*` by _transform_interactions).
        return [node]
    raise ValueError(
        f"Cannot parse formula term {ast.unparse(node)!r}: expected "
        "'+'-separated ShapeName(feature, ...) calls."
    )


def _flatten_interaction(node: ast.expr) -> list[ast.Call]:
    """Flatten a ``*``-tree (rewritten ``:`` interaction) into factor calls."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        return _flatten_interaction(node.left) + _flatten_interaction(node.right)
    if isinstance(node, ast.Call):
        return [node]
    raise ValueError(
        f"Cannot parse interaction factor {ast.unparse(node)!r}: "
        "each factor must be a ShapeName(feature, ...) call."
    )


def _parse_kwargs(call: ast.Call) -> dict[str, Any]:
    """Extract and literal-eval keyword arguments from a call node."""
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
    return kwargs


def _parse_factor(call: ast.Call, what: str) -> tuple[str, str]:
    """Validate one ``ShapeName(feature, ...)`` call, return (shape_name, feature)."""
    if not isinstance(call.func, ast.Name):
        raise ValueError(
            f"{what} {ast.unparse(call)!r} must be a plain "
            "ShapeName(feature, ...) call."
        )
    if len(call.args) != 1 or not isinstance(call.args[0], ast.Name):
        raise ValueError(
            f"{what} {ast.unparse(call)!r} must have exactly one feature name "
            "as its positional argument."
        )
    return call.func.id, call.args[0].id


def _parse_term(call: ast.Call) -> Term:
    """Turn one ``ShapeName(feature, key=value, ...)`` call node into a Term."""
    shape_name, feature = _parse_factor(call, "Term")
    return Term(
        shape_name=shape_name,
        features=(feature,),
        kwargs=_parse_kwargs(call),
    )


def _parse_interaction(factors: list[ast.Call]) -> Term:
    """Build the Term for the factor call nodes of a ``:``-joined chain.

    Shape name and kwargs come from the first factor (ADR-0005); remaining
    factors contribute only their feature name.
    """
    if len(factors) < 2:
        raise ValueError("An interaction term requires at least two factors.")
    parsed = [_parse_factor(call, "Interaction factor") for call in factors]
    return Term(
        shape_name=parsed[0][0],
        features=tuple(feature for _, feature in parsed),
        kwargs=_parse_kwargs(factors[0]),
    )


def parse_formula(formula: str) -> ParsedFormula:
    """Parse an additive/interaction formula string.

    Args:
        formula: e.g.
            ``"y ~ BayesianMLP(x1) + BayesianMLP(x2):BayesianMLP(x3)"``.
            Interaction terms use ``:`` to join two or more factor calls into a
            single joint net (ADR-0005, issue 0017).

    Returns:
        The parsed response name and terms.

    Raises:
        ValueError: If the formula is not ``response ~ term [+ term ...]``
            with each term a ``ShapeName(feature, ...)`` call or a
            ``:``-joined chain of such calls.
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
    # `:` is not a valid binary operator in Python expressions.  Rewrite
    # `Net(x1):Net(x2)` → `Net(x1)*Net(x2)` so ast.parse can handle it; the
    # resulting Mult BinOps are then distinguished from Add BinOps in the tree.
    rhs_transformed = _transform_interactions(rhs.strip())
    try:
        tree = ast.parse(rhs_transformed, mode="eval")
    except SyntaxError as err:
        raise ValueError(f"Cannot parse formula RHS {rhs.strip()!r}: {err}") from None

    parsed_terms: list[Term] = []
    for node in _flatten_additive(tree.body):
        if isinstance(node, ast.Call):
            parsed_terms.append(_parse_term(node))
        else:
            # Mult BinOp — an interaction chain.
            parsed_terms.append(_parse_interaction(_flatten_interaction(node)))

    seen: set[str] = set()
    for term in parsed_terms:
        for feat in term.features:
            if feat in seen:
                raise ValueError(
                    f"Feature {feat!r} appears in more than one term; "
                    "each feature may appear once in a formula."
                )
            seen.add(feat)
    return ParsedFormula(response=response, terms=tuple(parsed_terms))


def _instantiate(
    shape_name: str,
    in_features: int,
    kwargs: dict[str, Any],
    family: Any,
    n_obs: int | None,
    term_repr: str,
) -> nn.Module:
    """Resolve a shape name, wire kl_divisor, and instantiate."""
    shape_cls = ShapeFunctionRegistry.get(shape_name)
    if shape_cls is None:
        raise ValueError(
            f"Unknown shape function {shape_name!r} in term {term_repr!r}. "
            f"Registered names in ShapeFunctionRegistry: "
            f"{ShapeFunctionRegistry.names()}."
        )
    kwargs = dict(kwargs)
    # Signature check, not a Bayesian-ness check: deterministic shape
    # functions simply have no kl_divisor parameter to wire.
    if (
        n_obs is not None
        and "kl_divisor" not in kwargs
        and "kl_divisor" in inspect.signature(shape_cls.__init__).parameters
    ):
        kwargs["kl_divisor"] = float(n_obs)
    return shape_cls(in_features=in_features, param_count=family.param_count, **kwargs)


def build_formula(
    parsed: ParsedFormula, family: Any, n_obs: int | None = None
) -> dict[str, nn.Module]:
    """Instantiate shape functions for each parsed term.

    Additive terms produce ``{feature: Net(in_features=1, ...)}``.
    Interaction terms produce ``{"x1:x2": Net(in_features=2, ...)}`` — a
    single joint net over the concatenated inputs (ADR-0005, issue 0017).

    Args:
        parsed: Output of :func:`parse_formula`.
        family: Family object supplying ``param_count`` (output width).
        n_obs: Training-set size N. When given, terms whose constructor
            accepts ``kl_divisor`` (and whose kwargs don't set it) get
            ``kl_divisor=n_obs`` — the ELBO objective is mean-NLL + KL/N,
            and that /N lives inside each VariationalDense.

    Returns:
        Mapping of key → shape-function instance, the formula dict
        ``BayesianNAMLSS`` consumes.

    Raises:
        ValueError: If a term names an unregistered shape function.
    """
    formula: dict[str, nn.Module] = {}
    for term in parsed.terms:
        formula[term.key] = _instantiate(
            shape_name=term.shape_name,
            in_features=len(term.features),
            kwargs=term.kwargs,
            family=family,
            n_obs=n_obs,
            term_repr=term.key,
        )
    return formula
