"""Exact fitted score-load diagnostics, from the shared refusal table to the reports.

Three layers are covered here, and they need different instruments.

:mod:`cleverly.data.weighting` owns the guards and the ten concentration values. Every
refusal branch is reached by calling it with a hand-built array, because a fit that
produces a malformed artifact is not a case the fast tier can reach at all: the refusals
exist for artifacts that arrive from a pickle, not from the fit in the same process.

The three intervention axes own one thing each -- binding a validated column to the policy
declared in the same position. A fitted equation name is positional (``h_regime0``) and so
cannot witness that binding by itself, which is why the regime witness below uses two
policies whose structural zeros fall on complementary rows.

The reports and the assessment row own the rendering. Their column names, header order and
draw counts are pinned as literals, because a deleted column and a reordered header are
both silent otherwise.
"""

from __future__ import annotations

import dataclasses
import math
import pickle
import warnings
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pytest

from cleverly import load
from cleverly.data import CausalData
from cleverly.data.weighting import (
    REPORTED_DRAW,
    SCORE_LOAD_EMPTY_MASK,
    SCORE_LOAD_MASK_TOO_LARGE,
    SCORE_LOAD_MISSING,
    SCORE_LOAD_NO_EQUATION,
    SCORE_LOAD_NOT_FINITE,
    SCORE_LOAD_PREDATES,
    SCORE_LOAD_SHAPE_MISMATCH,
    format_score_load,
    score_load_row,
    validate_score_loads,
)
from cleverly.datasets import make_missing_outcome, make_nonlinear_ate, make_shift_dose
from cleverly.exceptions import DataError
from cleverly.interventions import (
    Incremental,
    IPSISet,
    RegimeSet,
    Rule,
    Shift,
    ShiftSet,
    Static,
    check_incremental_support,
    check_shift_support,
    check_support,
)
from cleverly.interventions.incremental import IncrementalSupport
from cleverly.interventions.shift import ShiftSupport
from cleverly.interventions.support import RegimeSupport, _intervention_loads
from cleverly.learners.density import ConditionalDensity
from cleverly.sensitivity import positivity_report
from cleverly.sensitivity.positivity import PositivityReport
from tests.conftest import fast_tmle
from tests.pickles import legacy_without as _legacy

#: The three intervention axes, in the order their reports declare them.
GROUPS = ("regime", "mtp", "ipsi")

#: Rows in the shared fitted results.  A multiple of one hundred, so that the top-share
#: row counts are ``n / 100`` and ``n / 20`` exactly and a test that hand-counts them does
#: not have to restate the rounding rule the small-column test pins on its own.
N = 500

#: Rows in the paired backend fits, also a multiple of one hundred.  Smaller because the
#: pair only has to agree with itself.
N_BACKEND = 300

#: Draws in the repeated fit, and the count every ``draw 01 of 03`` below refers to.
REPEATS = 3


def _screen_on_risk(frame: Any) -> np.ndarray:
    """A pickle-compatible dynamic rule for persistence coverage."""
    return (np.asarray(frame["W1"]) > 0.0).astype(float)


#: One declaration of each axis's policies and its dataset, read by every fit in this
#: module.  Two fixtures used to declare these separately and had already drifted -- one
#: named a dynamic rule where the other named a second static arm, and one weighted its
#: rows where the other did not -- so a test that ran over both was running over two
#: different questions under one name.
SPECS: dict[str, tuple[Callable[..., Any], int, dict[str, Any]]] = {
    # ``none`` and ``all`` are the semantic witness: their structural zeros fall on
    # complementary rows, so the artifact's columns are distinguishable without their
    # names.  ``screen`` keeps a dynamic rule in the persistence coverage, and makes the
    # reported minimum a choice among three rows rather than two.
    "regime": (
        make_nonlinear_ate,
        31,
        {
            "interventions": (
                Static(0, name="none"),
                Static(1, name="all"),
                Rule(_screen_on_risk, name="screen"),
            )
        },
    ),
    "mtp": (
        make_shift_dose,
        32,
        {
            "shifts": (Shift(0.0, cap=None, name="current"), Shift(0.5, cap=5.0, name="up")),
            "density_bins": 20,
        },
    ),
    "ipsi": (
        make_nonlinear_ate,
        31,
        {"incremental": (Incremental(0.5, name="down"), Incremental(2.0, name="up"))},
    ),
}

#: Every column :meth:`cleverly.interventions.SupportReport.to_frame` emits, in order.
#: Pinned as a literal tuple because a deleted column is otherwise invisible: the frame
#: still builds, and every other assertion in this file reads a column by name.
SUPPORT_FRAME_COLUMNS = (
    "regime",
    "min_propensity",
    "max_ratio",
    "effective_n",
    "unsupported",
    "score_equation",
    "score_effective_n",
    "score_load_ratio",
    "score_top_5pct",
    "score_load_omission",
)

#: Every heading of the :meth:`cleverly.interventions.SupportReport.summary` table, in
#: order.  The score column sits between the mechanism effective sample size and the
#: unsupported count, so a reader meets the two counts that are not each other side by
#: side.
SUPPORT_SUMMARY_HEADINGS = (
    "regime",
    "min g",
    "max ratio",
    "ratio effective n",
    "score load",
    "unsupported",
)

#: The caveat the regime table closes with.  It is the only place the table says what the
#: score column is not, so deleting it removes the whole distinction.
SUPPORT_SUMMARY_CAVEAT = (
    "score load is Kish-equivalent mask rows from abs(w_i * H_ij), not estimator ESS."
)


def _weighted(frame: Any) -> Any:
    """Add a two-valued weight column, in whichever backend the frame arrived in.

    The profile is deliberately far from constant. Equal weights make the fitted score load
    collapse onto the mechanism effective-sample-size ratio the report already carried, so
    an unweighted fixture cannot tell the new quantity from the old one.
    """
    weights = np.where(np.asarray(frame["W1"]) > 0.0, 4.0, 0.25)
    if hasattr(frame, "assign"):
        return frame.assign(w=weights)
    import polars as pl

    return frame.with_columns(pl.Series("w", weights))


def _fit(frame: Any, *, weighted: bool = True, fit_kwargs: Any = None, **settings: Any) -> Any:
    """Fit one small TMLE on the fast tier's estimator settings.

    ``fit_kwargs`` carries the roles that belong to :meth:`~cleverly.estimators.TMLE.fit`
    rather than to the constructor -- ``delta`` for a missing outcome, for instance -- so
    that a test needing one more column does not have to restate the whole estimator.
    """
    if weighted:
        frame = _weighted(frame)
    return (
        fast_tmle(**settings)
        .fit(
            frame,
            outcome="Y",
            treatment="A",
            weights="w" if weighted else None,
            **(fit_kwargs or {}),
        )
        .single()
    )


def _fit_group(group: str, *, n: int = N, backend: str = "pandas", **settings: Any) -> Any:
    """Fit the declared policies of one axis on that axis's dataset."""
    make, seed, spec = SPECS[group]
    frame, _ = make(n=n, seed=seed, backend=backend)
    return _fit(frame, random_state=seed, **spec, **settings)


@pytest.fixture(scope="module")
def intervention_results() -> dict[str, Any]:
    """One weighted fit per axis, shared by every test that only reads one."""
    return {group: _fit_group(group) for group in GROUPS}


@pytest.fixture(scope="module")
def backend_results() -> dict[str, tuple[Any, Any]]:
    """The same weighted fit per axis in both dataframe backends."""
    return {
        group: (
            _fit_group(group, n=N_BACKEND, backend="pandas"),
            _fit_group(group, n=N_BACKEND, backend="polars"),
        )
        for group in GROUPS
    }


@pytest.fixture(scope="module")
def repeated_regime_result() -> Any:
    """One repeated cross-fitted regime fit, for the draw count the reports print."""
    return _fit_group("regime", n=N_BACKEND, cross_fit=True, repeats=REPEATS)


def _rows(result: Any, group: str) -> Mapping[str, Any]:
    report = result.diagnostics.support()
    return report.regimes if group == "regime" else report


def _artifact(result: Any, group: str) -> np.ndarray:
    artifact = result.fluctuations[group].absolute_score_weights
    assert artifact is not None
    return np.asarray(artifact, dtype=float)


def _without_load(item: Any) -> dict[str, Any]:
    return {
        field.name: getattr(item, field.name)
        for field in dataclasses.fields(item)
        if field.name not in {"score_load", "score_load_omission"}
    }


def _with_artifact(result: Any, group: str, artifact: Any) -> Any:
    repeat = result.repeats[0]
    fluctuations = dict(repeat.fluctuations)
    fluctuations[group] = dataclasses.replace(fluctuations[group], absolute_score_weights=artifact)
    return dataclasses.replace(
        result,
        repeats=(dataclasses.replace(repeat, fluctuations=fluctuations),),
    )


def _kish(column: np.ndarray) -> float:
    return float(np.square(column.sum()) / np.square(column).sum())


def _support_detail(result: Any) -> str:
    """The assessment row for support, which is where the load fact is rendered."""
    return str(result.diagnostics.run_all()["support"].detail)


# --------------------------------------------------------------------------------------
# The shared refusal table
# --------------------------------------------------------------------------------------

#: Every way a fitted score artifact can be refused, and the reason each one earns.  The
#: guards short-circuit in the order they are written, so a case is described by the first
#: condition it meets rather than by every condition it happens to satisfy.
#:
#: The first three rows pin the *precedence* of the two states a caller can be in at once.
#: An absent artifact is refused before an absent equation list, because every public entry
#: point defaults ``equations=()`` and so a caller who passes no score weights passes no
#: equation names either.  That caller is in the absent-artifact state and reads
#: :data:`SCORE_LOAD_MISSING`.  ``SCORE_LOAD_NO_EQUATION`` is left to the state it was
#: introduced for: an artifact arrived and the fit recorded no equation to bind it to.
REFUSALS: tuple[tuple[str, Any, int, int, str], ...] = (
    ("no equation was recorded", np.ones((4, 1)), 0, 4, SCORE_LOAD_NO_EQUATION),
    ("no columns and no equations", np.ones((4, 0)), 0, 4, SCORE_LOAD_NO_EQUATION),
    ("the artifact is absent", None, 1, 4, SCORE_LOAD_MISSING),
    ("neither an artifact nor an equation", None, 0, 4, SCORE_LOAD_MISSING),
    ("one dimension", np.ones(4), 1, 4, SCORE_LOAD_SHAPE_MISMATCH),
    ("three dimensions", np.ones((4, 1, 1)), 1, 4, SCORE_LOAD_SHAPE_MISMATCH),
    ("no columns for one equation", np.ones((4, 0)), 1, 4, SCORE_LOAD_SHAPE_MISMATCH),
    ("fewer columns than equations", np.ones((4, 1)), 2, 4, SCORE_LOAD_SHAPE_MISMATCH),
    ("more columns than equations", np.ones((4, 3)), 2, 4, SCORE_LOAD_SHAPE_MISMATCH),
    ("a missing load", np.asarray([[1.0], [np.nan]]), 1, 4, SCORE_LOAD_NOT_FINITE),
    ("an infinite load", np.asarray([[1.0], [np.inf]]), 1, 4, SCORE_LOAD_NOT_FINITE),
    ("a negative load", np.asarray([[1.0], [-1e-12]]), 1, 4, SCORE_LOAD_NOT_FINITE),
    ("an empty mask", np.ones((0, 1)), 1, 4, SCORE_LOAD_EMPTY_MASK),
    ("a mask larger than the fit", np.ones((6, 1)), 1, 4, SCORE_LOAD_MASK_TOO_LARGE),
)

#: The refusal table as ``pytest.param``, since two entry points run the same cases.
REFUSAL_CASES = [pytest.param(*case[1:], id=case[0]) for case in REFUSALS]


@pytest.mark.parametrize(("artifact", "n_equations", "n_total", "reason"), REFUSAL_CASES)
def test_every_refused_score_artifact_names_the_one_condition_it_failed(
    artifact: Any, n_equations: int, n_total: int, reason: str
) -> None:
    """Each refusal is reached directly, because no fit in the fast tier produces one.

    A malformed artifact arrives from a stored result and not from the estimator that wrote
    it, so a fitted fixture can only reach the two shapes a fit can store at all. Calling
    the validator with a hand-built array reaches every branch, and the parametrisation is
    exhaustive over the branches rather than over the ones a fit happens to hit.

    The reason is asserted as the named constant. A reason that changes wording but not
    meaning should not fail here. A branch that starts answering a different reason must.
    """
    loads, refused = validate_score_loads(artifact, n_equations, n_total)
    assert loads is None
    assert refused == reason


def test_a_well_formed_score_artifact_is_accepted_unchanged() -> None:
    """The control the refusal table needs: the same call shape, and no refusal.

    Without it every guard could be replaced by an unconditional refusal and the table
    above would still pass.
    """
    artifact = np.asarray([[1.0, 2.0], [3.0, 0.0], [0.5, 4.0]])
    loads, refused = validate_score_loads(artifact, 2, 3)
    assert refused is None
    assert loads is not None
    np.testing.assert_array_equal(loads, artifact)


@pytest.mark.parametrize(("artifact", "n_equations", "n_total", "reason"), REFUSAL_CASES)
def test_the_intervention_path_refuses_for_the_same_reasons_as_the_validator(
    artifact: Any, n_equations: int, n_total: int, reason: str
) -> None:
    """The two callers of these guards had drifted into four disagreements.

    They are one implementation now, so the intervention path has to answer the same
    reason for the same artifact rather than a reason of its own.
    """
    equations = tuple(f"h{index}" for index in range(n_equations))
    rows, refused = _intervention_loads(equations, artifact, equations, n_total)
    assert rows == {}
    assert refused == reason


def test_a_label_count_that_disagrees_with_the_accepted_columns_is_refused() -> None:
    """The one refusal the intervention path owns: a validated block it cannot bind.

    The validator answers to the *equation* count, which is the fit's. The labels are the
    report's, and a block with one column per equation but not one per label leaves the
    extra column with no binding to take.
    """
    rows, refused = _intervention_loads(("only",), np.ones((4, 2)), ("h0", "h1"), 4)
    assert rows == {}
    assert refused == SCORE_LOAD_SHAPE_MISMATCH


def _bare_support_reports() -> dict[str, Mapping[str, Any]]:
    """Each public entry point called the way its signature lets a caller call it."""
    rng = np.random.default_rng(0)
    n = 40

    import pandas as pd

    arms = CausalData.from_frame(
        pd.DataFrame(
            {
                "Y": rng.binomial(1, 0.4, n).astype(float),
                "A": np.tile([0.0, 1.0], n // 2),
                "W1": rng.normal(size=n),
            }
        ),
        outcome="Y",
        treatment="A",
        covariates=["W1"],
    )
    g1 = np.full(n, 0.5)
    propensity = np.column_stack([1.0 - g1, g1])

    edges = np.array([-0.5, 0.5, 1.5, 2.5, 3.5])
    doses = np.tile(np.arange(4.0), n // 4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        continuous = CausalData.from_arrays(
            rng.binomial(1, 0.5, n).astype(float),
            doses,
            rng.normal(size=(n, 1)),
            treatment_kind="continuous",
        )
        density = ConditionalDensity(np.tile(np.array([0.4, 0.3, 0.2, 0.1]), (n, 1)), edges)
        shifts = ShiftSet.evaluate((Shift(1.0, cap=3.0, name="up"),), continuous, density)

    return {
        "check_support": check_support(
            RegimeSet.evaluate([Static(1)], arms), arms.treatment, propensity
        ).regimes,
        "check_shift_support": check_shift_support(shifts, density, continuous.treatment),
        "check_incremental_support": check_incremental_support(
            IPSISet.evaluate((Incremental(2.0),), arms, propensity), arms.treatment
        ),
    }


@pytest.mark.parametrize(
    "entry_point", ["check_support", "check_shift_support", "check_incremental_support"]
)
def test_a_direct_caller_who_passes_no_score_weights_reads_the_absent_artifact_reason(
    entry_point: str,
) -> None:
    """The precedence the three public entry points actually reach.

    All three default ``absolute_score_weights=None`` *and* ``equations=()``, so the caller
    who supplies neither is in both refusable states at once and the order of the two guards
    decides which reason is published. These strings are documented as machine-readable, so
    the answer is the one this API has always given: the artifact is absent.
    """
    rows = _bare_support_reports()[entry_point]
    assert rows
    assert all(row.score_load is None for row in rows.values())
    assert {row.score_load_omission for row in rows.values()} == {SCORE_LOAD_MISSING}


def test_an_empty_score_column_has_no_load_to_describe() -> None:
    """The row builder refuses what the validator already ruled out.

    The guard is unreachable through the report path, and is here so that a caller that
    reaches the row builder directly gets a named error rather than a division by zero.
    """
    with pytest.raises(DataError, match="empty score column"):
        score_load_row(np.asarray([], dtype=float), "h0", 4)


#: A hand-built row whose reported draw is neither ``1`` nor its own total.  Every fit
#: this package can produce retains draw one, so a row taken from a fit cannot tell a
#: renderer that *reads* ``reported_repeat`` from one that prints the literal ``01``.
#: Two and three are both wrong for either literal, which is the whole point of it.
SECOND_DRAW_ROW: dict[str, Any] = {
    "equation": "h0",
    "n_total": 400.0,
    "n_targeted": 300.0,
    "effective": 120.0,
    "targeted_ratio": 0.4,
    "total_ratio": 0.3,
    "top_1pct": 0.2,
    "top_5pct": 0.5,
    "max_load": 9.0,
    "zero_load": 0.0,
    "reported_repeat": 2,
    "n_repeats": 3,
}


@pytest.mark.parametrize(
    ("style", "expected"),
    [("cell", "draw 02/03"), ("inline", "draw 02 of 03"), ("detail", "draw 02 of 03)")],
)
def test_every_style_prints_the_draw_the_row_records(style: str, expected: str) -> None:
    """Each rendered style reads ``reported_repeat`` instead of repeating a literal.

    Parameters
    ----------
    style : str
        The rendering style to check.
    expected : str
        The draw text that style has to produce.
    """
    rendered = format_score_load(SECOND_DRAW_ROW, style=style)  # type: ignore[arg-type]
    assert expected in rendered
    assert "draw 01" not in rendered


#: A column whose top shares can be counted by hand.  Thirty rows makes the rounding rule
#: visible in both answers: one percent of thirty rows rounds *up* to one row and five
#: percent to two, so a rule that truncated instead would report no top row at all and a
#: smaller five-percent share.
KNOWN_COLUMN = np.asarray([10.0, 8.0] + [1.0] * 28)


def test_the_top_shares_of_a_hand_counted_column_are_the_documented_rounding() -> None:
    """Literal expected values, because the rounding rule is a choice and not a definition.

    This assertion used to recompute ``max(1, ceil(fraction * size))`` beside the
    implementation and compare the two, which agrees with the implementation by
    construction: if the rule were wrong, so was the test. The values below are counted off
    :data:`KNOWN_COLUMN` instead. The largest single entry is ``10`` out of a total of
    ``46``, and the largest two are ``18`` out of ``46``.
    """
    row = score_load_row(KNOWN_COLUMN, "h0", 30)

    assert row["equation"] == "h0"
    assert row["n_targeted"] == 30.0
    assert row["n_total"] == 30.0
    assert row["max_load"] == 10.0
    assert row["zero_load"] == 0.0
    assert row["top_1pct"] == pytest.approx(10.0 / 46.0, abs=0)
    assert row["top_5pct"] == pytest.approx(18.0 / 46.0, abs=0)
    assert row["effective"] == pytest.approx(46.0**2 / 192.0, abs=0)
    assert row["targeted_ratio"] == pytest.approx(46.0**2 / 192.0 / 30.0, abs=0)


# --------------------------------------------------------------------------------------
# One fitted column per declared policy
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("group", GROUPS)
def test_each_intervention_row_reads_its_exact_fitted_score_column(
    intervention_results: dict[str, Any], group: str
) -> None:
    """The hand calculation binds declaration order, masks, weights, and zeros at once.

    The two top shares are counted rather than recomputed from the rounding rule. The
    fixture size is a multiple of one hundred, so one and five percent of it are whole row
    counts and the rule is inert. Which rule applies away from a whole count is pinned by
    :func:`test_the_top_shares_of_a_hand_counted_column_are_the_documented_rounding`.
    """
    result = intervention_results[group]
    fluctuation = result.fluctuations[group]
    artifact = _artifact(result, group)
    rows = _rows(result, group)
    assert len(rows) == artifact.shape[1] == len(fluctuation.names)

    for index, (label, support) in enumerate(rows.items()):
        column = artifact[:, index]
        assert column.size % 100 == 0, "the fixture size has to make the rounding rule inert"
        effective = _kish(column)
        ordered = np.sort(column)
        reported = support.score_load
        assert reported is not None, support.score_load_omission
        assert reported == {
            "equation": fluctuation.names[index],
            "reported_repeat": 1,
            "n_repeats": 1,
            "n_total": float(result.data.n),
            "n_targeted": float(column.size),
            "effective": effective,
            "targeted_ratio": effective / float(column.size),
            "total_ratio": effective / float(result.data.n),
            "top_1pct": float(ordered[-(column.size // 100) :].sum() / column.sum()),
            "top_5pct": float(ordered[-(column.size // 20) :].sum() / column.sum()),
            "max_load": float(column.max()),
            "zero_load": float(np.count_nonzero(column == 0.0)),
        }, label


@pytest.mark.parametrize("group", GROUPS)
def test_the_old_intervention_ratio_cannot_stand_in_for_the_fitted_score_load(
    intervention_results: dict[str, Any], group: str
) -> None:
    """Unequal observation weights make the old ratio-only summary a nonzero mutation."""
    rows = _rows(intervention_results[group], group)
    assert any(
        support.score_load is not None
        and not np.isclose(support.score_load["effective"], support.effective_sample_size)
        for support in rows.values()
    )


@pytest.mark.parametrize("group", GROUPS)
def test_each_axis_reads_the_matching_column_and_keeps_old_support_fields(
    intervention_results: dict[str, Any], group: str
) -> None:
    result = intervention_results[group]
    moved = _artifact(result, group).copy()
    moved[:, 0] *= np.linspace(0.25, 3.0, moved.shape[0])
    mutated = _with_artifact(result, group, moved)
    before = _rows(result, group)
    after = _rows(mutated, group)
    labels = list(before)

    assert after[labels[0]].score_load != before[labels[0]].score_load
    assert after[labels[1]].score_load == before[labels[1]].score_load
    assert {name: _without_load(item) for name, item in after.items()} == {
        name: _without_load(item) for name, item in before.items()
    }


def test_the_column_a_regime_row_reports_is_the_one_that_policy_solved(
    intervention_results: dict[str, Any],
) -> None:
    """A witness where the columns are told apart by the policy and not by their position.

    A fitted equation name is positional -- ``h_regime0``, ``h_regime1`` -- so it carries no
    policy identity, and an assertion that reads the label and the column from the same
    index pins only that the helper reads column *j* for label *j*. Such an assertion
    passes on a permuted artifact.

    ``Static(0)`` and ``Static(1)`` are distinguishable by construction. The clever
    covariate is ``I(A = a) / g(a | W)``, so ``none`` carries a structural zero on exactly
    the rows where ``A == 1`` and ``all`` on exactly the rows where ``A == 0``. The column
    each row reports is located by that pattern, and the two patterns are complementary and
    both nonempty, so swapping the artifact's first two columns fails here.
    """
    result = intervention_results["regime"]
    treatment = np.asarray(result.data.treatment, dtype=float).reshape(-1)
    artifact = _artifact(result, "regime")
    rows = _rows(result, "regime")
    untouched = {"none": treatment == 1.0, "all": treatment == 0.0}

    assert np.array_equal(untouched["none"], ~untouched["all"])
    assert untouched["none"].any()
    assert untouched["all"].any()

    for name, zeros in untouched.items():
        found = [
            index
            for index in range(artifact.shape[1])
            if np.array_equal(artifact[:, index] == 0.0, zeros)
        ]
        assert len(found) == 1, f"{name} has no uniquely identifiable fitted column"
        column = artifact[:, found[0]]
        reported = rows[name].score_load
        assert reported is not None
        assert reported["zero_load"] == float(np.count_nonzero(zeros))
        assert reported["max_load"] == float(column.max())
        assert reported["effective"] == pytest.approx(_kish(column), abs=0)


def test_a_regime_structural_zero_stays_in_the_score_mask_denominator(
    intervention_results: dict[str, Any],
) -> None:
    result = intervention_results["regime"]
    screen = _rows(result, "regime")["screen"].score_load
    assert screen is not None
    assert screen["zero_load"] > 0.0
    assert screen["n_targeted"] == float(result.data.n)


@pytest.mark.parametrize("group", GROUPS)
def test_complete_data_uses_every_fitted_score_mask_row(
    intervention_results: dict[str, Any], group: str
) -> None:
    result = intervention_results[group]
    for row in _rows(result, group).values():
        assert row.score_load is not None
        assert row.score_load["n_targeted"] == row.score_load["n_total"] == float(result.data.n)


def test_missing_outcomes_use_the_fitted_score_mask_denominator() -> None:
    frame, _ = make_missing_outcome(n=400, seed=61)
    result = _fit(
        frame,
        weighted=False,
        random_state=61,
        interventions=(Static(0, name="none"), Static(1, name="all")),
        fit_kwargs={"delta": "Delta"},
    )
    reported = _rows(result, "regime")["all"].score_load
    assert reported is not None
    assert reported["n_targeted"] == float(result.data.observed.sum())
    assert reported["n_targeted"] < reported["n_total"]
    assert reported["total_ratio"] < reported["targeted_ratio"]


def test_a_score_column_of_only_zeros_reports_no_load_and_no_share(
    intervention_results: dict[str, Any],
) -> None:
    """The degenerate column, which is a real state and not a malformed artifact.

    A policy whose clever covariate vanishes on every row has a defined mask and no load in
    it. The Kish count is zero by the degenerate convention, and the two top shares are
    ``nan`` rather than zero, because a share of a total that is not positive is not a
    number that exists. ``to_frame`` carries that ``nan`` through, so the frame column means
    "no share" in the same way the row does.
    """
    result = intervention_results["regime"]
    artifact = _artifact(result, "regime").copy()
    artifact[:, 0] = 0.0
    report = _with_artifact(result, "regime", artifact).diagnostics.support()
    reported = report.regimes["none"].score_load
    assert reported is not None

    assert reported["effective"] == 0.0
    assert reported["targeted_ratio"] == 0.0
    assert reported["total_ratio"] == 0.0
    assert reported["max_load"] == 0.0
    assert reported["zero_load"] == float(result.data.n)
    assert math.isnan(reported["top_1pct"])
    assert math.isnan(reported["top_5pct"])

    frame = report.to_frame()
    assert next(iter(frame["score_equation"])) is not None
    assert next(iter(frame["score_load_omission"])) is None
    assert float(np.asarray(frame["score_load_ratio"])[0]) == 0.0
    assert math.isnan(float(np.asarray(frame["score_top_5pct"])[0]))


# --------------------------------------------------------------------------------------
# What the reports render
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("backend", [0, 1], ids=["pandas", "polars"])
def test_the_regime_frame_carries_every_score_column_in_order(
    backend_results: dict[str, tuple[Any, Any]], backend: int
) -> None:
    """Both backends, because ``to_frame`` builds one payload and emits it twice."""
    result = backend_results["regime"][backend]
    frame = result.diagnostics.support().to_frame()

    assert tuple(frame.columns) == SUPPORT_FRAME_COLUMNS
    assert list(frame["regime"]) == ["none", "all", "screen"]
    assert list(frame["score_equation"]) == list(result.fluctuations["regime"].names)
    assert np.asarray(frame["score_effective_n"]).tolist() == [
        row.score_load["effective"] for row in _rows(result, "regime").values()
    ]


def test_the_regime_table_puts_the_score_column_between_the_two_counts_it_is_not(
    intervention_results: dict[str, Any],
) -> None:
    """Header order, and the caveat the table ends on.

    Neither was asserted, so deleting the score column or the closing sentence left a table
    that still printed and still passed.
    """
    summary = intervention_results["regime"].diagnostics.support().summary()
    lines = summary.splitlines()
    # The title also starts with "regime", so the table header is picked out by a heading
    # only it carries.
    header = next(line for line in lines if line.startswith("regime") and "min g" in line)
    positions = [header.index(heading) for heading in SUPPORT_SUMMARY_HEADINGS]

    assert lines[0] == f"regime support (n = {N})"
    assert positions == sorted(positions)
    assert header.index("ratio effective n") < header.index("score load")
    assert header.index("score load") < header.index("unsupported")
    assert lines[-1] == SUPPORT_SUMMARY_CAVEAT


def test_the_regime_table_says_a_missing_score_load_is_unavailable(
    intervention_results: dict[str, Any],
) -> None:
    """The omission wording, in the table cell and in the frame's own column."""
    result = intervention_results["regime"]
    report = _with_artifact(result, "regime", None).diagnostics.support()
    summary = report.summary()
    frame = report.to_frame()

    assert summary.count("unavailable") == len(report.regimes)
    assert "draw" not in summary
    assert summary.splitlines()[-1] == SUPPORT_SUMMARY_CAVEAT
    assert list(frame["score_equation"]) == [None] * len(report.regimes)
    assert list(frame["score_load_omission"]) == [SCORE_LOAD_MISSING] * len(report.regimes)
    assert all(math.isnan(value) for value in np.asarray(frame["score_effective_n"]))


@pytest.mark.parametrize("group", ["mtp", "ipsi"])
def test_the_shift_and_tilt_summaries_carry_the_fitted_score_fragment(
    intervention_results: dict[str, Any], group: str
) -> None:
    """The one-line reports gained a clause that nothing pinned.

    The clause is written out here rather than rebuilt from the formatter, so a change to
    the wording, the precision or the draw count fails rather than following along.
    """
    for row in _rows(intervention_results[group], group).values():
        reported = row.score_load
        assert reported is not None
        assert (
            f"score load={reported['effective']:.1f}/{reported['n_targeted']:.0f} "
            "Kish-equivalent mask rows (draw 01 of 01)"
        ) in row.summary()
        assert "score load unavailable" in dataclasses.replace(row, score_load=None).summary()


def test_the_combined_row_reports_load_without_grading_or_pooling_it(
    intervention_results: dict[str, Any],
) -> None:
    result = intervention_results["ipsi"]
    direct = result.diagnostics.support()
    combined = result.diagnostics.run_all()["support"]
    assert combined.report is direct
    assert combined.status.value == "completed"
    assert "intervention load:" in combined.detail
    assert "not estimator ESS" in combined.detail
    assert "minimum effective-sample-size ratio" in combined.detail


def test_the_support_row_renders_the_whole_regime_load_fact(
    intervention_results: dict[str, Any],
) -> None:
    """The regime source of the assessment row, which nothing reached.

    Only the shift and incremental reports are mappings, so only their rows were exercised.
    A regime report keeps its rows under ``regimes`` instead, and deleting that source left
    every other assertion in this file passing.
    """
    result = intervention_results["regime"]
    loads = {name: row.score_load for name, row in _rows(result, "regime").items()}
    assert all(row is not None for row in loads.values())
    worst = min(loads, key=lambda name: loads[name]["targeted_ratio"])
    values = loads[worst]

    assert (
        f"intervention load: {worst}:{values['equation']} "
        f"{values['effective']:.1f}/{values['n_targeted']:.0f} Kish-equivalent mask rows "
        f"({values['targeted_ratio']:.1%}; {values['total_ratio']:.1%} all; "
        "draw 01 of 01); not estimator ESS"
    ) in _support_detail(result)


def test_the_assessment_row_keeps_a_digit_where_the_load_is_most_concentrated(
    intervention_results: dict[str, Any],
) -> None:
    """The precision the detail row needs, at the value that needs it.

    One row carrying the whole column is the smallest Kish ratio a column can have, and on
    this fixture it is ``1 / 500 = 0.2%``. That is exactly where a reader most needs the
    number, and it is where whole-percent precision destroys it: ``0.2%``, ``0.04%`` and a
    ratio of literally zero all render as ``0%``. The overlap verdict states the same
    quantity at whole percent on purpose, which
    ``test_post_fit_assessment_battery.py`` pins, so the two surfaces are checked apart.
    """
    result = intervention_results["regime"]
    artifact = _artifact(result, "regime").copy()
    artifact[:, 0] = 0.0
    artifact[0, 0] = 1.0

    detail = _support_detail(_with_artifact(result, "regime", artifact))
    assert f"{1.0 / N:.1%}" == "0.2%"
    assert f"1.0/{N} Kish-equivalent mask rows (0.2%; " in detail
    assert "(0%; " not in detail


@pytest.mark.parametrize("position", [0, 1, 2])
def test_the_support_row_reports_the_least_concentrated_of_several_rows(
    intervention_results: dict[str, Any], position: int
) -> None:
    """Three rows, and the minimum is moved to each position in turn.

    One row selected out of three cannot tell "the minimum" from "the first" or "the last"
    while the fitted ordering happens to agree with one of them. Concentrating a different
    column each time makes the answer follow the ratio and nothing else.
    """
    result = intervention_results["regime"]
    labels = list(_rows(result, "regime"))
    artifact = _artifact(result, "regime").copy()
    # One row carrying the whole column is the smallest Kish ratio a column can have.
    artifact[:, position] = 0.0
    artifact[0, position] = 1.0

    detail = _support_detail(_with_artifact(result, "regime", artifact))
    assert f"intervention load: {labels[position]}:" in detail
    assert f"1.0/{N} Kish-equivalent mask rows" in detail


# --------------------------------------------------------------------------------------
# Repeated draws
# --------------------------------------------------------------------------------------


def test_a_repeated_intervention_fit_prints_the_draw_count_it_stored(
    repeated_regime_result: Any,
) -> None:
    """Every rendering of a repeated fit's load names the same draw out of the same total.

    Four formatters wrote ``draw 01`` by hand and read the total from wherever they
    happened to hold it, which left one of them printing a total of one for a fit with
    three draws. They are one formatter now, so the table cell, the assessment row and the
    stored row have to agree.
    """
    result = repeated_regime_result
    assert result.n_repeats == REPEATS

    for row in _rows(result, "regime").values():
        assert row.score_load is not None
        assert row.score_load["reported_repeat"] == REPORTED_DRAW
        assert row.score_load["n_repeats"] == REPEATS

    assert f"draw {REPORTED_DRAW:02d}/{REPEATS:02d}" in result.diagnostics.support().summary()
    assert f"draw {REPORTED_DRAW:02d} of {REPEATS:02d}" in _support_detail(result)


def test_a_repeated_fit_reports_the_first_draws_matching_support_artifact(
    intervention_results: dict[str, Any],
) -> None:
    original = intervention_results["ipsi"]
    second = _artifact(original, "ipsi").copy()
    second[:, 0] *= np.linspace(0.25, 3.0, second.shape[0])
    changed_second = _with_artifact(original, "ipsi", second).repeats[0]
    repeated = dataclasses.replace(original, repeats=(original.repeats[0], changed_second))
    reported = {name: row.score_load for name, row in repeated.diagnostics.support().items()}
    first = {name: row.score_load for name, row in original.diagnostics.support().items()}
    second_rows = {
        name: row.score_load
        for name, row in _with_artifact(original, "ipsi", second).diagnostics.support().items()
    }
    assert reported == {
        name: {**values, "n_repeats": 2} for name, values in first.items() if values is not None
    }
    assert reported != second_rows


# --------------------------------------------------------------------------------------
# Refusals reached through a fit, and persistence
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("group", GROUPS)
def test_an_older_fluctuation_keeps_support_and_records_the_load_omission(
    intervention_results: dict[str, Any], group: str
) -> None:
    rows = _rows(_with_artifact(intervention_results[group], group, None), group)
    assert rows
    assert all(support.score_load is None for support in rows.values())
    assert {support.score_load_omission for support in rows.values()} == {SCORE_LOAD_MISSING}


@pytest.mark.parametrize("group", GROUPS)
def test_an_impossible_score_mask_size_records_an_omission(
    intervention_results: dict[str, Any], group: str
) -> None:
    artifact = _artifact(intervention_results[group], group)
    malformed = np.vstack([artifact, artifact[:1]])
    rows = _rows(_with_artifact(intervention_results[group], group, malformed), group)
    assert all(row.score_load is None for row in rows.values())
    assert {row.score_load_omission for row in rows.values()} == {SCORE_LOAD_MASK_TOO_LARGE}


@pytest.mark.parametrize("group", GROUPS)
def test_cached_intervention_loads_replay_after_persistence(
    intervention_results: dict[str, Any], tmp_path: Any, group: str
) -> None:
    result = intervention_results[group]
    result.diagnostics.support()
    detail = _support_detail(result)
    restored = load(result.save(tmp_path / f"{group}-loads.joblib"))
    assert {name: row.score_load for name, row in _rows(restored, group).items()} == {
        name: row.score_load for name, row in _rows(result, group).items()
    }
    assert _support_detail(restored) == detail
    restored.assessment_cache.clear()
    assert {name: row.score_load for name, row in _rows(restored, group).items()} == {
        name: row.score_load for name, row in _rows(result, group).items()
    }


@pytest.fixture(scope="module")
def score_load_records(intervention_results: dict[str, Any]) -> dict[str, Any]:
    """One instance of each record class the defaulting unpickle now serves."""
    records: dict[str, Any] = {
        group: next(iter(_rows(intervention_results[group], group).values())) for group in GROUPS
    }
    records["positivity"] = positivity_report(intervention_results["regime"])
    return records


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("regime", RegimeSupport),
        ("mtp", ShiftSupport),
        ("ipsi", IncrementalSupport),
        ("positivity", PositivityReport),
    ],
)
def test_every_score_load_record_survives_a_real_pickle_round_trip(
    score_load_records: dict[str, Any], key: str, expected: type
) -> None:
    """The four classes that share one restore, taken through ``dumps`` and ``loads``.

    The shared restore adds a ``__setstate__`` to classes that had none, which is the kind
    of change that can break an ordinary round trip while every hand-called restore keeps
    passing.
    """
    record = score_load_records[key]
    assert isinstance(record, expected)
    restored = pickle.loads(pickle.dumps(record))
    assert type(restored) is expected
    assert restored == record


def test_an_incremental_row_stays_hashable_with_a_fitted_score_load(
    score_load_records: dict[str, Any],
) -> None:
    """The tilt row is the one of the three that has ever been hashable, deliberately.

    Its ``score_load`` is a ``dict``, so the field is kept out of the generated hash and
    left in the generated equality: two tilts that differ only in their fitted load are
    still different rows.
    """
    row = score_load_records["ipsi"]
    assert row.score_load is not None
    assert len({row, dataclasses.replace(row)}) == 1
    without = dataclasses.replace(row, score_load=None)
    assert hash(without) == hash(row)
    assert without != row


@pytest.mark.parametrize("group", GROUPS)
def test_a_record_pickled_before_the_score_load_fields_says_the_report_predates_them(
    score_load_records: dict[str, Any], group: str
) -> None:
    """Only the sentinel is load-bearing here, and that is the whole point.

    ``score_load`` defaults to a plain ``None``, so ``dataclasses`` leaves a class attribute
    behind it and ``restored.score_load is None`` holds even with the restore deleted
    outright. ``score_load_omission`` defaults to ``None`` too, so the *sentinel* is the
    only value that cannot arrive by accident. A report that predates the diagnostic says
    so, where a fresh report with nothing to omit says ``None``.
    """
    record = score_load_records[group]
    assert record.score_load is not None
    assert record.score_load_omission is None

    restored = _legacy(record, "score_load", "score_load_omission")
    assert restored.score_load is None
    assert restored.score_load_omission == SCORE_LOAD_PREDATES
    assert _without_load(restored) == _without_load(record)


def test_a_positivity_report_pickled_before_group_leverage_restores_empty_mappings(
    score_load_records: dict[str, Any],
) -> None:
    """The factory-defaulted fields, which are the ones an absent restore cannot survive.

    ``dataclasses`` deletes the class attribute for a ``default_factory`` field, so these
    two have no fallback behind them. Without the restore this raises ``AttributeError``
    rather than answering a wrong value, which is what makes them the discriminating case
    that the plainly-defaulted fields are not.
    """
    report = score_load_records["positivity"]
    assert report.group_leverage

    restored = _legacy(report, "group_leverage", "group_leverage_omissions")
    assert restored.group_leverage == {}
    assert restored.group_leverage_omissions == {}
    assert "Absolute-load concentration is greatest" not in restored.verdict()


# --------------------------------------------------------------------------------------
# Backend parity
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("group", GROUPS)
def test_intervention_loads_are_identical_for_pandas_and_polars(
    backend_results: dict[str, tuple[Any, Any]], group: str
) -> None:
    """The paired fits are weighted, which is what makes the comparison distinctive.

    Under equal weights the fitted score load collapses onto the ratio effective sample
    size the report already carried, so an unweighted pair agrees on the new columns for a
    reason that has nothing to do with them.
    """
    from_pandas, from_polars = backend_results[group]
    pandas_rows = _rows(from_pandas, group)
    polars_rows = _rows(from_polars, group)
    assert {name: row.score_load for name, row in pandas_rows.items()} == {
        name: row.score_load for name, row in polars_rows.items()
    }
    assert any(
        row.score_load is not None
        and not np.isclose(row.score_load["effective"], row.effective_sample_size)
        for row in pandas_rows.values()
    )
