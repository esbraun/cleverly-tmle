"""The declared bases of effect modifiers in :mod:`cleverly.bases`."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

import cleverly.bases as bases_module
from cleverly.bases import (
    INTERCEPT,
    Additive,
    BSpline,
    Constant,
    Cosine,
    Indicators,
    Linear,
    Modifiers,
    Sieve,
    Tensor,
    UserBasis,
    check_basis,
    interactions,
    resolve,
)
from cleverly.exceptions import CapabilityError, DataError, RefusalReason

MODIFIERS = Modifiers(["x", "g"], domains={"x": (0.0, 2.0)}, levels={"g": ("a", "b", "c")})
FRAME = {"x": np.array([0.0, 0.5, 1.0, 2.0]), "g": np.array(["a", "b", "c", "a"])}


def test_module_is_a_leaf_that_imports_numpy_and_exceptions_only() -> None:
    tree = ast.parse(Path(bases_module.__file__).read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add("." * node.level + (node.module or ""))
    standard = {"__future__", "collections.abc", "dataclasses", "math", "typing", "itertools"}
    assert imported - standard <= {"numpy", ".exceptions"}


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"columns": "x"}, "sequence of column names"),
        ({"columns": []}, "at least one column"),
        ({"columns": ["x", "x"]}, "distinct"),
        ({"columns": ["x"], "domains": {"z": (0, 1)}}, "not modifier columns"),
        ({"columns": ["x"], "domains": {"x": (0, 1)}, "levels": {"x": (0, 1)}}, "both"),
        ({"columns": ["x"], "outside": "drop"}, "outside"),
    ],
)
def test_modifiers_refuse_a_malformed_declaration(arguments: dict, message: str) -> None:
    with pytest.raises(DataError, match=message):
        Modifiers(**arguments)


def test_clipping_without_a_domain_is_the_sample_basis_refusal() -> None:
    """R3: the ends ``clip`` maps to would be read off the sample."""
    with pytest.raises(CapabilityError, match="declare no domain") as caught:
        Modifiers(["x"], outside="clip")
    assert caught.value.refusal.reason is RefusalReason.INVALID_DECLARATION


def test_modifiers_split_continuous_and_categorical_columns() -> None:
    assert MODIFIERS.continuous == ("x",)
    assert MODIFIERS.categorical == ("g",)
    assert MODIFIERS.domain("x") == (0.0, 2.0)
    assert MODIFIERS.levels_of("g") == ("a", "b", "c")
    assert MODIFIERS.domain("g") is None


def test_a_value_outside_its_domain_is_refused_or_clipped() -> None:
    frame = {"x": np.array([-1.0, 1.0, 3.0])}
    with pytest.raises(DataError, match="domain"):
        Modifiers(["x"], domains={"x": (0, 2)}).values(frame)
    clipped = Modifiers(["x"], domains={"x": (0, 2)}, outside="clip").values(frame)
    np.testing.assert_array_equal(clipped["x"], [0.0, 1.0, 2.0])


def test_an_undeclared_level_is_refused_even_under_clip() -> None:
    schema = Modifiers(["g"], levels={"g": ("a", "b")}, outside="clip")
    with pytest.raises(DataError, match="level"):
        schema.values({"g": np.array(["a", "z"])})


@pytest.mark.parametrize(
    ("basis", "terms"),
    [
        (Constant(), (INTERCEPT,)),
        (Linear(["x"]), (INTERCEPT, "x")),
        (Linear(["x"], intercept=False), ("x",)),
        (Cosine(["x"], 2), (INTERCEPT, "cos1(x)", "cos2(x)")),
        (Indicators("g"), (INTERCEPT, "g[b]", "g[c]")),
        (
            Additive(Cosine(["x"], 1), Indicators("g")),
            (INTERCEPT, "cos1(x)", "g[b]", "g[c]"),
        ),
        (
            Tensor(Linear(["x"]), Indicators("g")),
            (INTERCEPT, "g[b]", "g[c]", "x", "x:g[b]", "x:g[c]"),
        ),
    ],
)
def test_each_basis_names_one_term_per_design_column(basis, terms) -> None:
    resolved = resolve(basis, MODIFIERS)
    assert resolved.terms == terms
    assert resolved.n_terms == len(terms)
    assert resolved.basis_kind == "known"
    design = resolved.evaluate(FRAME, MODIFIERS)
    assert design.shape == (4, len(terms))
    if terms[0] == INTERCEPT:
        np.testing.assert_array_equal(design[:, 0], 1.0)


def test_cosine_is_orthonormal_on_its_domain_and_truncates_nested() -> None:
    grid = {"x": (np.arange(20000) + 0.5) / 20000 * 2.0}
    design = Cosine(["x"], 4).evaluate(grid, MODIFIERS)
    np.testing.assert_allclose(design.T @ design / design.shape[0], np.eye(5), atol=1e-6)
    small = Cosine(["x"], 4).truncate(2)
    assert small.terms == (INTERCEPT, "cos1(x)", "cos2(x)")
    np.testing.assert_allclose(small.evaluate(grid, MODIFIERS), design[:, :3])


def test_cosine_without_a_domain_is_refused_as_read_off_the_sample() -> None:
    with pytest.raises(CapabilityError, match="known function of the modifiers"):
        Cosine(["x"], 2).evaluate({"x": np.array([0.0, 1.0])}, Modifiers(["x"]))


def test_bspline_with_its_dropped_function_restored_sums_to_one() -> None:
    grid = {"x": np.linspace(0.0, 2.0, 101)}
    design = BSpline(["x"], intercept=False).evaluate(grid, MODIFIERS)
    full = BSpline(["x"]).evaluate(grid, MODIFIERS)
    # Under an intercept the first function is dropped, so the rest sum to at most one.
    assert design.shape[1] == full.shape[1]
    np.testing.assert_allclose(design.sum(axis=1), 1.0, atol=1e-12)
    assert np.all(full[:, 1:].sum(axis=1) <= 1.0 + 1e-12)


def test_linear_needs_no_domain() -> None:
    design = Linear(["x"]).evaluate({"x": np.array([5.0, -3.0])}, Modifiers(["x"]))
    np.testing.assert_array_equal(design, [[1.0, 5.0], [1.0, -3.0]])


def test_indicators_report_the_reference_level_as_the_intercept() -> None:
    design = resolve(Indicators("g"), MODIFIERS).evaluate(FRAME, MODIFIERS)
    np.testing.assert_array_equal(design[:, 1:], [[0, 0], [1, 0], [0, 1], [0, 0]])


def test_interactions_of_order_one_return_the_basis() -> None:
    additive = Additive(Linear(["x"]), Indicators("g"))
    assert interactions(additive, 1) is additive
    terms = resolve(interactions(additive, 2), MODIFIERS).terms
    assert "x:g[b]" in terms and "x:g[c]" in terms


def test_check_basis_refuses_an_undeclared_or_foreign_basis() -> None:
    check_basis(Linear(["x"]))
    known = UserBasis(lambda f: np.c_[np.ones(len(f["x"])), f["x"]], ("one", "x"), ("x",), "known")
    check_basis(known)
    for basis in (
        UserBasis(lambda f: f["x"], ("x",), ("x",)),
        UserBasis(lambda f: f["x"], ("x",), ("x",), "estimated"),
        object(),
    ):
        with pytest.raises(CapabilityError, match="known function of the modifiers"):
            check_basis(basis)


def test_user_basis_checks_the_shape_it_returns() -> None:
    basis = UserBasis(lambda f: np.ones((2, 3)), ("a", "b"), ("x",), "known")
    with pytest.raises(DataError, match="shape"):
        basis.evaluate({"x": np.array([0.0, 1.0])})


def test_sieve_resolves_a_declared_size_and_flags_cross_validation() -> None:
    sieve = Sieve(Cosine(["x"], 6), size={0: 1, 100: 3})
    assert sieve.size_at(50) == 1
    assert sieve.at(500).terms == (INTERCEPT, "cos1(x)", "cos2(x)", "cos3(x)")
    assert not sieve.chosen_by_cv
    assert Sieve(Cosine(["x"], 6), size=lambda n: 2).at(10).n_terms == 3
    assert Sieve(Cosine(["x"], 6), size=0).at(10).terms == (INTERCEPT,)
    cv = Sieve(Cosine(["x"], 6))
    assert cv.chosen_by_cv
    with pytest.raises(DataError, match="no declared size"):
        cv.size_at(10)


def test_default_sieve_is_additive_cosines_and_indicators() -> None:
    terms = Sieve.default(MODIFIERS, max_size=2).basis.terms
    assert terms == (INTERCEPT, "cos1(x)", "cos2(x)", "g[b]", "g[c]")
