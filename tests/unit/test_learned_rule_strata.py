r"""The learned-rule value inside baseline strata (``strata=``).

One rule is learned per fold and read stratum by stratum. The strata are adjustment columns,
so the plug-in rule of each fold already reads them, and a rule learned per stratum would be a
different learner. Stratum :math:`s` of fold :math:`v` evaluates the fold's frozen rule
:math:`d_v` on the fold's validation rows in the stratum, with the fold-evaluated stratum
construction of X28 (``test_fold_evaluated_strata_exact.py``) and the fold-local one-hot rule
density. Conditional on the training rows, fold :math:`v` evaluates

.. math::

    \Psi_{s,d_v}(P) = \frac{E_P[I(S = s)\,\bar Q(d_v(W), W)]}{P(S = s)}

at the law :math:`T_v` whose covariate shares are the fold's own, whose mechanism is the
oracle's and whose regression is the fold's targeted :math:`\bar Q^*_v`.

:class:`TestTheExactLaw` runs on a discrete law that the sample realises exactly. The outcome
learner is the saturated cell mean of each training fold, shrunk towards one half, so each
fold's plug-in rule is learned from its own training rows. The blip at ``W = 1`` is ``1/60``,
so the folds learn different rules there. The module checks:

* every fold's stored rule is the sign of its training cell means, and two folds disagree;
* each reported stratum value is the fold average of the stratum plug-ins of the fold rules;
* each fold's stratum curve is the Gateaux derivative of :math:`\Psi_{s,d_v}` at :math:`T_v`,
  unweighted and under a covariate weight;
* each reported stratum score is solved exactly, while each fold's own stratum score is not;
* the marginal value is the share mixture of the stratum values.

Controls that must fail. One mutates the fit: the full-sample rule in place of the fold rules.
Three perturb the reported curve after the fit, and show that the checks have power: the global
stratum share in place of the fold's (the nonzero witness: it leaves a nonzero curve mean,
because the folds' stratum shares differ), another fold's rule in the derivative, and a curve
without the stratum share.

The covariate weight is constant inside each value of ``W``, so the weighted cell means, and
the weighted fold rules, equal the unweighted ones. The weighted fit witnesses the weighted
curves and scores, since the weight varies inside stratum 0. That the rule reads the weights is
the witness of the known-weights composition (``test_learned_rule_compositions.py``).

The remaining classes run on the continuous support law: the admitted compositions (known
weights, a rule learner, the mean over splits with its band), the refusals (clusters, fold
targeting, the omitted-variable bound, an empty stratum in a validation fold), and the
surfaces that read the fit (persistence, the frozen-rule replay, simulated confounding, and
the placebo, whose null inside a stratum is the stratum's outcome mean).
"""

from __future__ import annotations

import functools
import importlib
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LinearRegression

from cleverly import CausalStudy, LearnedRuleValue, PointTreatment, load
from cleverly.estimators import TMLE
from cleverly.estimators.serialize import dumps, loads
from cleverly.exceptions import CapabilityError, DataError, RefusalReason
from cleverly.interventions import LearnedRule
from cleverly.interventions.learned import fold_rules, freeze_learned_rule, frozen_rule_replay
from cleverly.sensitivity import (
    ConfounderStrengthGrid,
    omitted_variable_bounds,
    simulated_confounding,
)
from cleverly.validation.refute import refute
from tests import discrete_law as law
from tests.conftest import OracleTreatment
from tests.unit import _learned_rule_support as support
from tests.unit._declaration_support import tmle_module
from tests.unit._natural_course_support import NeverFit, never_fit_learners

pytestmark = pytest.mark.xdist_group("learned_rule_strata")

# ``cleverly.validation.refute`` names the function, so read the module by its path.
refute_module = importlib.import_module("cleverly.validation.refute")

NAME = support.NAME

# ----------------------------------------------------------------------------- exact law

#: The regression of the law, ``[w, a]``. The blip is ``+0.3``, ``+1/60`` and ``-0.2``, so the
#: rule treats at ``W = 0``, does not at ``W = 2``, and is learned with noise at ``W = 1``.
EXACT_Q = np.array([[0.40, 0.70], [0.50, 0.50 + 1.0 / 60.0], [0.80, 0.60]])
COUNTS = law.cell_counts(q=EXACT_Q)
PROBS = COUNTS / law.N
#: ``g(w)`` of the realised sample, which the treatment oracle returns.
G_EXACT = PROBS.sum(axis=2)[:, 1] / PROBS.sum(axis=(1, 2))
#: Three folds of 1,000 rows hold 334, 333 and 333 rows, so ``n / (V n_v)`` is not 1. At this
#: seed fold 2 learns not to treat at ``W = 1`` and folds 0 and 1 learn to treat there.
N_FOLDS = 3
SEED = 9
#: ``S = 1{W = 2}`` puts two covariate values in stratum 0, so the plug-in term of its curve
#: does not vanish, and the rules disagree inside stratum 0.
STRATUM_OF_W = np.array([0, 0, 1])
STRATA = (0, 1)
#: The learned cell means are shrunk towards one half, so the fit is wrong in every cell and
#: every block of the fluctuation moves. Shrinking keeps the sign of each blip.
SHRINK = 0.8
#: A weight on ``W`` alone. It tilts the covariate shares and leaves ``g`` and ``Q`` as they are.
COVARIATE_WEIGHT = law.cell_weights(lambda w, a, y: 1.0 + 0.6 * w)
FLOOR = 1e-4
EXACT = 1e-9


class _Law(law.DiscreteLaw):
    """:class:`tests.discrete_law.DiscreteLaw` on a design that appends ``S`` after ``W``."""

    @staticmethod
    def _index(covariates: Any) -> np.ndarray:
        values = np.asarray(covariates, dtype=float)
        return np.rint(values.reshape(values.shape[0], -1)[:, 0]).astype(int)


def _cell_means(w: np.ndarray, a: np.ndarray, y: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """The ``(3, 2)`` table of shrunk weighted cell means of ``y``."""
    table = np.empty((3, 2))
    for cell in range(3):
        for arm in (0, 1):
            inside = (w == cell) & (a == arm)
            table[cell, arm] = np.average(y[inside], weights=weight[inside])
    return (1.0 - SHRINK) / 2.0 + SHRINK * table


class CellMeans(ClassifierMixin, BaseEstimator):
    """The saturated outcome learner on the pooled design ``(A, W, S)``, shrunk to one half."""

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> CellMeans:
        design = np.asarray(X, dtype=float)
        outcome = np.asarray(y, dtype=float)
        weight = np.ones_like(outcome) if sample_weight is None else np.asarray(sample_weight)
        self.table_ = _cell_means(
            np.rint(design[:, 1]).astype(int), design[:, 0].astype(int), outcome, weight
        )
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        design = np.asarray(X, dtype=float)
        p = self.table_[np.rint(design[:, 1]).astype(int), design[:, 0].astype(int)]
        return np.column_stack([1.0 - p, p])


def _cells() -> np.ndarray:
    """The support point of each row of the sample."""
    return np.repeat(np.arange(len(law.SUPPORT)), [COUNTS[w, a, y] for w, a, y in law.SUPPORT])


def _frame(weighted: bool = False) -> pd.DataFrame:
    frame = law.frame(COUNTS).assign(S=lambda f: STRATUM_OF_W[f["W"].to_numpy(dtype=int)])
    if weighted:
        frame = frame.assign(weight=COVARIATE_WEIGHT[_cells()])
    return frame


def _exact_fit(*, weighted: bool = False) -> Any:
    estimator = TMLE(
        learned_rule=LearnedRule(),
        outcome_learner=CellMeans(),
        treatment_learner=OracleTreatment(_Law(PROBS)),
        cross_fit=True,
        cv_evaluation=True,
        n_folds=N_FOLDS,
        simultaneous=False,
        random_state=SEED,
        tol=1e-12,
        max_iter=200,
    )
    columns: dict[str, Any] = {"weights": "weight"} if weighted else {}
    return estimator.fit(
        _frame(weighted), outcome="Y", treatment="A", covariates=["W", "S"], strata=["S"], **columns
    ).single()


@pytest.fixture(scope="module")
def exact_fits() -> dict[str, Any]:
    return {"rows": _exact_fit(), "weighted": _exact_fit(weighted=True)}


def _stratum(code: int) -> str:
    return f"{NAME}[S={code}]"


def _rows(fit: Any) -> dict[str, np.ndarray]:
    data = fit.data
    w = np.rint(data.covariates[:, 0]).astype(int)
    return {
        "w": w,
        "s": STRATUM_OF_W[w],
        "a": np.asarray(data.treatment, dtype=float),
        "y": np.asarray(data.outcome, dtype=float),
        "weight": np.asarray(data.weights, dtype=float),
        "cell": _cells(),
    }


def _folds(fit: Any) -> list[tuple[np.ndarray, np.ndarray]]:
    return [(np.asarray(train), np.asarray(test)) for train, test in fit.nuisance.folds]


def _fold_rule(fit: Any, train: np.ndarray) -> np.ndarray:
    """``d_v(w)``, recomputed from the training rows: the sign of the cell-mean blip."""
    rows = _rows(fit)
    table = _cell_means(
        rows["w"][train], rows["a"][train].astype(int), rows["y"][train], rows["weight"][train]
    )
    return (table[:, 1] > table[:, 0]).astype(float)


def _pooled_rule(fit: Any) -> np.ndarray:
    """The rule of the full-sample cell means, which the mutation control uses."""
    return _fold_rule(fit, np.arange(fit.data.n))


def _density(rule: np.ndarray) -> np.ndarray:
    return np.column_stack([1.0 - rule, rule])


def _q_star(fit: Any, test: np.ndarray) -> np.ndarray:
    """The fold's targeted regression as a ``(3, 2)`` table, one value per ``(w, a)``."""
    targeted = fit.fluctuations["regime"].targeted
    w = _rows(fit)["w"][test]
    table = np.empty((3, 2))
    for cell in range(3):
        for arm in (0, 1):
            values = np.asarray(targeted.arms[float(arm)])[test][w == cell]
            assert np.ptp(values) == 0.0
            table[cell, arm] = values[0]
    return table


def _share(rows: dict[str, np.ndarray], index: np.ndarray, code: int) -> float:
    weight = rows["weight"][index]
    return float(np.sum(weight[rows["s"][index] == code]) / np.sum(weight))


def _plug_in(fit: Any, code: int, test: np.ndarray, rule: np.ndarray) -> float:
    """``psi_{s,v}``: the weighted mean of ``Q*_v(d_v(w), w)`` over the fold's stratum rows."""
    rows = _rows(fit)
    inside = test[rows["s"][test] == code]
    mean = (_density(rule) * _q_star(fit, test)).sum(axis=1)[rows["w"][inside]]
    return float(np.average(mean, weights=rows["weight"][inside]))


def _functional(rule: np.ndarray, code: int, weights: np.ndarray | None) -> Any:
    """``Psi_{s,d}`` longhand, of the tilted law when weighted."""
    inside = (code == STRATUM_OF_W).astype(float)
    density = _density(rule)

    def psi(probs: Any) -> Any:
        p = probs if weights is None else law.tilt(probs, weights)
        p_w = p.sum(axis=(1, 2))
        q = p[:, :, 1] / p.sum(axis=2)
        return (inside * p_w * (density * q).sum(axis=1)).sum() / (inside * p_w).sum()

    return psi


def _derivative(fit: Any, code: int, test: np.ndarray, rule: np.ndarray) -> np.ndarray:
    """The Gateaux derivative of ``Psi_{s,d}`` at the fold's law ``T_v``, by complex step."""
    weights = COVARIATE_WEIGHT if fit.data.is_weighted else None
    shares = np.bincount(_rows(fit)["w"][test], minlength=3) / test.size
    q = _q_star(fit, test)
    arm = np.stack([1.0 - G_EXACT, G_EXACT], axis=1)
    probs = shares[:, None, None] * arm[:, :, None] * np.stack([1.0 - q, q], axis=2)
    return law.contamination_eif(_functional(rule, code, weights), probs, law.SUPPORT)


def _raw(fit: Any, name: str) -> list[np.ndarray]:
    """Each fold's stratum curve, undoing the ``n / (V n_v)`` factor of the stored report."""
    curve = np.asarray(fit.estimates[name].influence_curve, dtype=float)
    return [curve[test] * N_FOLDS * test.size / fit.data.n for _, test in _folds(fit)]


def _score(fit: Any, code: int, test: np.ndarray, rule: np.ndarray) -> float:
    """The fold's stratum score ``sum w H (Y - Q*) / W_{v,s}`` with ``H = 1{A = d_v} / g``."""
    rows = _rows(fit)
    inside = test[rows["s"][test] == code]
    a = rows["a"][inside].astype(int)
    w = rows["w"][inside]
    g = np.where(a == 1, G_EXACT[w], 1.0 - G_EXACT[w])
    clever = _density(rule)[w, a] / g
    residual = rows["y"][inside] - _q_star(fit, test)[w, a]
    weight = rows["weight"][inside]
    return float(np.sum(weight * clever * residual) / np.sum(weight))


class TestTheExactLaw:
    """The fold-evaluated stratum construction at each fold's own learned rule."""

    def test_premise_the_stored_rules_are_the_training_fold_rules(
        self, exact_fits: dict[str, Any]
    ) -> None:
        for fit in exact_fits.values():
            stored = fold_rules(fit.nuisance)
            assert stored is not None
            w = _rows(fit)["w"]
            learned = []
            for fold, (train, _) in enumerate(_folds(fit)):
                rule = _fold_rule(fit, train)
                np.testing.assert_array_equal(stored[:, fold], rule[w] > 0.5)
                learned.append(tuple(rule))
            # Nonzero witness: the folds learn different rules, at W = 1 inside stratum 0.
            assert len(set(learned)) > 1, learned
            assert len({rule[1] for rule in learned}) == 2

    def test_premise_every_block_moves_and_the_shares_differ(
        self, exact_fits: dict[str, Any]
    ) -> None:
        for fit in exact_fits.values():
            epsilon = np.abs(np.asarray(fit.fluctuations["regime"].epsilon))
            assert epsilon.size == len(STRATA)
            assert float(np.min(epsilon)) > 1e-3, epsilon
            rows = _rows(fit)
            for code in STRATA:
                shares = [_share(rows, test, code) for _, test in _folds(fit)]
                assert max(shares) - min(shares) > 1e-2, shares
            assert len({test.size for _, test in _folds(fit)}) > 1
            assert {NAME, *(_stratum(code) for code in STRATA)} == set(fit.estimates)

    @pytest.mark.parametrize("which", ["rows", "weighted"])
    def test_each_stratum_value_is_the_fold_average_of_the_fold_rule_plug_ins(
        self, exact_fits: dict[str, Any], which: str
    ) -> None:
        fit = exact_fits[which]
        for code in STRATA:
            plug_ins = [
                _plug_in(fit, code, test, _fold_rule(fit, train)) for train, test in _folds(fit)
            ]
            name = _stratum(code)
            np.testing.assert_allclose(
                fit.cv_targeting.fold_estimates[name], plug_ins, rtol=0, atol=1e-12
            )
            assert fit.estimates[name].psi == pytest.approx(float(np.mean(plug_ins)), abs=1e-12)

    @pytest.mark.parametrize("which", ["rows", "weighted"])
    def test_each_fold_curve_is_the_gateaux_derivative_at_the_fold_rule(
        self, exact_fits: dict[str, Any], which: str
    ) -> None:
        fit = exact_fits[which]
        cells = _rows(fit)["cell"]
        for code in STRATA:
            for (train, test), raw in zip(_folds(fit), _raw(fit, _stratum(code)), strict=True):
                derivative = _derivative(fit, code, test, _fold_rule(fit, train))
                for point in np.unique(cells[test]):
                    np.testing.assert_allclose(
                        raw[cells[test] == point], derivative[point], rtol=0, atol=1e-10
                    )

    def test_control_another_folds_rule_is_not_the_derivative(
        self, exact_fits: dict[str, Any]
    ) -> None:
        fit = exact_fits["rows"]
        cells = _rows(fit)["cell"]
        folds = _folds(fit)
        rules = [_fold_rule(fit, train) for train, _ in folds]
        gaps = []
        for fold, ((_, test), raw) in enumerate(zip(folds, _raw(fit, _stratum(0)), strict=True)):
            for other, rule in enumerate(rules):
                if other == fold or np.array_equal(rule, rules[fold]):
                    continue
                derivative = _derivative(fit, 0, test, rule)
                gaps.append(float(np.max(np.abs(raw - derivative[cells[test]]))))
        assert gaps and min(gaps) > 1e-2, gaps

    @pytest.mark.parametrize("which", ["rows", "weighted"])
    def test_each_reported_stratum_score_is_solved(
        self, exact_fits: dict[str, Any], which: str
    ) -> None:
        fit = exact_fits[which]
        for code in STRATA:
            per_fold = [
                _score(fit, code, test, _fold_rule(fit, train)) for train, test in _folds(fit)
            ]
            assert abs(float(np.mean(per_fold))) < EXACT, per_fold
            # Nonzero witness: the pooled update leaves each fold's own score unsolved.
            assert max(abs(value) for value in per_fold) > 1e-3, per_fold
            assert abs(float(np.mean(fit.estimates[_stratum(code)].influence_curve))) < EXACT

    def test_control_the_global_share_leaves_a_nonzero_curve_mean(
        self, exact_fits: dict[str, Any]
    ) -> None:
        """The plan's unit witness: ``1 / P_n(S = s)`` in place of ``1 / P_{n,v}(S = s)``."""
        fit = exact_fits["rows"]
        rows = _rows(fit)
        everything = np.arange(fit.data.n)
        means = []
        for code in STRATA:
            curve = np.asarray(fit.estimates[_stratum(code)].influence_curve, dtype=float).copy()
            for _, test in _folds(fit):
                curve[test] *= _share(rows, test, code) / _share(rows, everything, code)
            means.append(abs(float(np.mean(curve))))
        assert max(means) > FLOOR, means

    def test_control_a_curve_without_the_stratum_share_is_caught(
        self, exact_fits: dict[str, Any]
    ) -> None:
        """The invalid-inference control of the planned study, at unit level."""
        fit = exact_fits["rows"]
        rows = _rows(fit)
        gaps = []
        for code in STRATA:
            for (train, test), raw in zip(_folds(fit), _raw(fit, _stratum(code)), strict=True):
                derivative = _derivative(fit, code, test, _fold_rule(fit, train))
                unnormalised = raw * _share(rows, test, code)
                gaps.append(float(np.max(np.abs(unnormalised - derivative[rows["cell"][test]]))))
        assert max(gaps) > 1e-2, gaps

    def test_control_the_full_sample_rule_is_caught(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Deliberate mutation: every fold evaluates the rule learned on every row."""
        original = tmle_module.learned_rule_treat

        def pooled(data: Any, nuisance: Any, rule: Any, **kwargs: Any) -> Any:
            treat, _, rules = original(data, nuisance, rule, **kwargs)
            w = np.rint(np.asarray(data.covariates)[:, 0]).astype(int)
            weight = np.asarray(data.weights, dtype=float)
            table = _cell_means(
                w, np.asarray(data.treatment).astype(int), np.asarray(data.outcome), weight
            )
            everywhere = (table[:, 1] > table[:, 0])[w]
            assert not np.array_equal(everywhere, treat)
            return everywhere, None, np.repeat(everywhere[:, None], rules.shape[1], axis=1)

        monkeypatch.setattr(tmle_module, "learned_rule_treat", pooled)
        fit = _exact_fit()
        gaps = []
        for code in STRATA:
            plug_ins = [
                _plug_in(fit, code, test, _fold_rule(fit, train)) for train, test in _folds(fit)
            ]
            gaps.append(abs(fit.estimates[_stratum(code)].psi - float(np.mean(plug_ins))))
        assert max(gaps) > FLOOR, gaps

    @pytest.mark.parametrize("which", ["rows", "weighted"])
    def test_the_marginal_value_is_the_share_mixture_of_the_strata(
        self, exact_fits: dict[str, Any], which: str
    ) -> None:
        fit = exact_fits[which]
        rows = _rows(fit)
        everything = np.arange(fit.data.n)
        shares = np.array([_share(rows, everything, code) for code in STRATA])
        parts = [fit.estimates[_stratum(code)] for code in STRATA]
        stratum_psi = np.array([part.psi for part in parts])
        psi = float(shares @ stratum_psi)
        estimate = fit.estimates[NAME]
        assert estimate.psi == pytest.approx(psi, abs=1e-12)
        curve = sum(
            share * np.asarray(part.influence_curve)
            for share, part in zip(shares, parts, strict=True)
        ) + rows["weight"] / rows["weight"].mean() * (stratum_psi[rows["s"]] - psi)
        np.testing.assert_allclose(estimate.influence_curve, curve, rtol=0, atol=1e-12)
        # The record describes the marginal value, whose fold values are the share mixtures.
        (split,) = fit.extra["learned_rule"].splits
        assert split.estimate == estimate.psi
        assert split.fold_estimates == fit.cv_targeting.fold_estimates[NAME]
        assert float(np.mean(split.fold_estimates)) == pytest.approx(psi, abs=1e-12)


# ----------------------------------------------------------------------- support law

#: The strata cut from ``W1``, at shares of about 0.3, 0.35 and 0.35.
CUTS = (-0.4, 0.3)
FIT = {"outcome": "Y", "treatment": "A", "covariates": (*support.COVARIATES, "S")}


def _strata_frame(n: int = 400, seed: int = 0) -> pd.DataFrame:
    frame = support.law_frame(n, seed)
    return frame.assign(
        S=np.digitize(frame["W1"], CUTS), w=1.0 + 1.5 * (frame["W2"] > 0), c=np.arange(n) // 4
    )


def _fit(frame: pd.DataFrame | None = None, keywords: Any = None, **settings: Any) -> Any:
    frame = _strata_frame() if frame is None else frame
    return (
        support.learned_tmle(**settings)
        .fit(frame, strata=("S",), **FIT, **(keywords or {}))
        .single()
    )


def _names(result: Any) -> list[str]:
    labels = [result.data.stratum_label(code) for code in range(result.data.n_strata)]
    return [NAME, *(f"{NAME}[{label}]" for label in labels)]


@pytest.fixture(scope="module")
def result() -> Any:
    return _fit()


class TestTheAdmittedCompositions:
    """Known weights, a rule learner and the mean over splits each report every stratum."""

    CASES: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {  # noqa: RUF012
        "rows": ({}, {}),
        "known weights": ({"weights": "w"}, {}),
        "blip learner": (
            {},
            {"learned_rule": LearnedRule(loss="blip", learner=LinearRegression())},
        ),
    }
    #: The support row of each case: ``learned-rule-strata`` measures the plug-in rule on rows
    #: and with a known weight (T41). A rule-learner loss with strata matches no row.
    ROWS: dict[str, str | None] = {  # noqa: RUF012
        "rows": "T41.binary",
        "known weights": "T41.binary_weighted",
        "blip learner": None,
    }

    @pytest.mark.parametrize("case", list(CASES))
    def test_each_stratum_is_the_fold_average_and_reads_its_row(self, case: str) -> None:
        keywords, settings = self.CASES[case]
        fit = _fit(keywords=keywords, **settings)
        assert set(fit.estimates) == set(_names(fit))
        for name in _names(fit):
            estimate = fit.estimates[name]
            folds = fit.cv_targeting.fold_estimates[name]
            assert estimate.psi == pytest.approx(float(np.mean(folds)), abs=1e-12)
            assert np.isfinite(estimate.std_error) and estimate.std_error > 0.0
            row = self.ROWS[case]
            assert estimate.support.label == ("experimental" if row is None else "supported")
            assert estimate.support.row == row

    def test_the_weights_move_the_stratum_values(self, result: Any) -> None:
        weighted = _fit(keywords={"weights": "w"})
        gaps = [abs(weighted[name].psi - result[name].psi) for name in _names(result)[1:]]
        assert min(gaps) > 1e-4, gaps

    def test_the_mean_over_splits_is_the_mean_of_single_split_fits(self) -> None:
        """W-i1 per stratum. The mean commutes with the share mixture, unlike the median."""
        repeated = _fit(repeats=3)
        singles = [_fit(random_state=repeat.seed) for repeat in repeated.repeats]
        for name in _names(repeated):
            parts = [single[name] for single in singles]
            reported = repeated[name]
            assert reported.psi == float(np.mean([part.psi for part in parts]))
            curve = np.mean([part.influence_curve for part in parts], axis=0)
            np.testing.assert_array_equal(reported.influence_curve, curve)
            assert reported.variance == pytest.approx(
                float(np.sum(curve**2)) / repeated.data.n**2, rel=1e-12
            )
            # Nonzero: the splits differ, so the mean is not any one of them.
            assert len({part.psi for part in parts}) == len(parts)
        shares = [
            float(np.average(repeated.data.strata == code, weights=repeated.data.weights))
            for code in range(repeated.data.n_strata)
        ]
        mixture = sum(
            share * repeated[name].psi
            for share, name in zip(shares, _names(repeated)[1:], strict=True)
        )
        assert repeated[NAME].psi == pytest.approx(mixture, abs=1e-12)

    def test_the_band_of_the_mean_over_splits_is_built(self) -> None:
        """The band reads the mean curves. The median's refusal no longer meets this fit."""
        fit = _fit(repeats=2, simultaneous=True)
        band = fit.simultaneous
        assert band is not None and band.support.label == "experimental"
        assert 1.959 < float(band.critical_value) < 10.0
        for name, estimate in fit.estimates.items():
            lower, upper = band.bands[name]
            pointwise = 1.959 * estimate.std_error
            assert lower < estimate.psi - pointwise and estimate.psi + pointwise < upper

    def test_a_study_fit_reports_the_engine_estimate(self, result: Any) -> None:
        study = CausalStudy(
            _strata_frame(),
            design=PointTreatment(
                outcome="Y", treatment="A", adjustment=FIT["covariates"], strata=("S",)
            ),
        )
        fitted = study.identify(LearnedRuleValue()).estimate(
            outcome_learner=support.outcome_learner(),
            treatment_learner=support.treatment_learner(),
            n_folds=5,
            cv_evaluation=True,
            simultaneous=False,
            random_state=0,
        )
        for name in _names(result):
            assert fitted[name].psi == pytest.approx(result[name].psi, abs=1e-12)
        key = fitted.parameter_keys[_names(result)[1]]
        assert (key.estimand, key.axis, key.value) == (
            "ey_learned_rule",
            "learned_rule",
            "learned rule",
        )
        assert key.stratum is not None


def _refused(call: Any, *fragments: str, error: type = CapabilityError) -> Any:
    NeverFit.calls = 0
    with pytest.raises(error) as caught:
        call()
    assert NeverFit.calls == 0
    for fragment in fragments:
        assert fragment in str(caught.value), str(caught.value)
    return caught.value


class TestTheRefusals:
    """Each unsupported composition refuses, before any learner where the data suffice."""

    def test_clusters_refuse_before_any_learner(self) -> None:
        error = _refused(
            lambda: _fit(keywords={"id": "c"}, **never_fit_learners()),
            "baseline strata (strata=) and clusters (id=)",
            "X28",
            "without strata=",
        )
        assert error.refusal.reason is RefusalReason.IMPLEMENTATION_GAP

    def test_clusters_refuse_at_identify(self) -> None:
        study = CausalStudy(
            _strata_frame(),
            design=PointTreatment(
                outcome="Y",
                treatment="A",
                adjustment=FIT["covariates"],
                strata=("S",),
                cluster="c",
            ),
        )
        _refused(lambda: study.identify(LearnedRuleValue()), "X28")

    def test_the_cluster_remedies_fit_as_written(self) -> None:
        clustered = support.learned_tmle().fit(_strata_frame(), id="c", **FIT).single()
        assert np.isfinite(clustered[NAME].psi)
        assert np.isfinite(_fit(n_folds=5)[NAME].psi)

    def test_fold_targeting_keeps_its_x61_refusal(self) -> None:
        _refused(
            lambda: _fit(targeting_scheme="fold", **never_fit_learners()),
            "targeting_scheme='fold'",
            "X61",
        )

    def test_the_omitted_variable_bound_refuses(self, result: Any) -> None:
        for name in _names(result):
            with pytest.raises(CapabilityError) as caught:
                omitted_variable_bounds(result, name, cf_y=0.05, cf_d=0.05)
            assert "baseline strata" in str(caught.value) and "X11 (n2)" in str(caught.value)
            assert caught.value.refusal.reason is RefusalReason.IMPLEMENTATION_GAP
        assert not result.sensitivity.capability("omitted_confounding").available

    def test_the_bound_remedy_runs(self) -> None:
        unstratified = support.learned_tmle().fit(_strata_frame(), **FIT).single()
        bound = omitted_variable_bounds(unstratified, NAME, cf_y=0.05, cf_d=0.05)
        assert bound.lower < bound.psi < bound.upper

    def test_an_empty_stratum_in_a_validation_fold_refuses_without_the_stacked_remedy(
        self,
    ) -> None:
        frame = _strata_frame()
        rare = frame.index[frame["S"] == 0][:3]
        frame.loc[frame.index.difference(rare), "S"] = np.clip(
            frame.loc[frame.index.difference(rare), "S"], 1, None
        )
        error = _refused(
            lambda: _fit(frame, **never_fit_learners()),
            "validation fold",
            "without strata=",
            error=DataError,
        )
        assert "cv_evaluation=False" not in str(error)


class TestTheSurfaces:
    """The surfaces that read a stratified learned-rule fit."""

    @pytest.mark.parametrize("route", ["save", "dumps"])
    def test_a_round_trip_keeps_every_stratum(self, result: Any, route: str, tmp_path: Any) -> None:
        restored = (
            load(result.save(tmp_path / "strata.joblib"))
            if route == "save"
            else loads(dumps(result))
        )
        assert restored.extra["learned_rule"] == result.extra["learned_rule"]
        np.testing.assert_array_equal(restored.data.strata, result.data.strata)
        for name in _names(result):
            saved, estimate = restored.estimates[name], result.estimates[name]
            assert saved.psi == estimate.psi and saved.variance == estimate.variance
            np.testing.assert_array_equal(saved.influence_curve, estimate.influence_curve)

    def test_the_frozen_rule_replay_reproduces_every_stratum(self, result: Any) -> None:
        replay = frozen_rule_replay(result.estimator, freeze_learned_rule(result))
        refit = replay.fit(_strata_frame(), strata=("S",), **FIT).single()
        for name in _names(result):
            assert refit[name].psi == result[name].psi

    def test_a_rule_holding_refutation_runs_on_a_stratum(self, result: Any) -> None:
        name = _names(result)[1]
        report = refute(result, estimand=name, tests=("random_common_cause",), n_replicates=2)
        (test,) = report.tests
        assert len(test.values) == 2 and np.all(np.isfinite(test.values))

    def test_simulated_confounding_holds_each_stratum_at_zero_strength(self) -> None:
        """The stratum alias passes the replay checks, and so does the marginal alias."""
        fitted = _study_fit()
        grid = ConfounderStrengthGrid(treatment=(0.0, 0.2), outcome=(0.0, 0.2))
        for name in (_names(fitted)[0], _names(fitted)[1]):
            surface = simulated_confounding(fitted, estimand=name, grid=grid, random_state=0)
            assert surface.original_estimate == fitted[name].psi
            assert len(surface.cells) == 4 and all(cell.failure is None for cell in surface.cells)
            (zero,) = [
                cell
                for cell in surface.cells
                if cell.treatment_strength == 0.0 and cell.outcome_strength == 0.0
            ]
            # The frozen-rule replay of an unperturbed draw is the fit itself.
            assert zero.estimate == fitted[name].psi

    def test_the_placebo_of_a_stratum_reads_its_stratum_outcome_mean(self) -> None:
        """Under the placebo law every rule's value inside stratum ``s`` is ``E[Y | S = s]``."""
        fitted = _study_fit()
        t = _placebo_t(fitted)
        assert max(abs(value) for value in t.values()) < 4.0, t

    def test_control_the_marginal_outcome_mean_fails_a_stratum_placebo(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mutation: every alias reads ``E[Y]``. The strata's outcome means differ, so it fails."""
        fitted = _study_fit()
        data = fitted.data
        means = [
            float(
                np.average(
                    data.outcome[data.strata == code], weights=data.weights[data.strata == code]
                )
            )
            for code in range(data.n_strata)
        ]
        # Nonzero witness: the stratum outcome means differ from each other.
        assert max(means) - min(means) > 0.2, means
        monkeypatch.setattr(refute_module, "_learned_rule_stratum", lambda *_: None)
        t = _placebo_t(fitted)
        assert abs(t[NAME]) < 4.0
        assert max(abs(value) for name, value in t.items() if name != NAME) > 8.0, t


@functools.cache
def _study_fit() -> Any:
    study = CausalStudy(
        _strata_frame(),
        design=PointTreatment(
            outcome="Y", treatment="A", adjustment=FIT["covariates"], strata=("S",)
        ),
    )
    return study.identify(LearnedRuleValue()).estimate(
        outcome_learner=support.outcome_learner(),
        treatment_learner=support.treatment_learner(),
        n_folds=5,
        cv_evaluation=True,
        simultaneous=False,
        random_state=0,
    )


def _placebo_t(fitted: Any) -> dict[str, float]:
    """The ``t`` statistic of 30 placebo draws of each name, against the reported null."""
    out = {}
    for name in _names(fitted):
        (test,) = refute(fitted, estimand=name, tests=("placebo",), n_replicates=30).tests
        values = np.asarray(test.values, dtype=float)
        null = float(test.detail.split("against the exact null ")[1].split(";")[0])
        out[name] = float((values.mean() - null) / (values.std(ddof=1) / np.sqrt(values.size)))
    return out
