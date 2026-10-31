r"""The fold-evaluated stratified fit (X28) on a law the sample realises exactly.

``TMLE(cv_evaluation=True)`` with ``strata=`` reports, for each stratum :math:`s`, the ``1/V``
average of the fold plug-ins :math:`\psi_{s,v}` over the rows of validation fold :math:`v` in
the stratum. Conditional on the training rows, fold :math:`v` evaluates the functional

.. math::

    \Psi_s(P) = \frac{E_P[I(S = s)\,\textstyle\sum_a g^\star(a \mid W)\,\bar Q(a, W)]}{P(S = s)}

at the law :math:`T_v` whose covariate shares are the fold's own, whose mechanism is the
treatment oracle's and whose regression is the targeted :math:`\bar Q^*`. Its Gateaux
derivative at :math:`T_v` is the fold's stratum curve, with the fold's share
:math:`T_v(S = s) = P_{n,v}(S = s)` in the denominator. The derivative is taken by complex step
from a longhand statement of :math:`\Psi_s`, as :mod:`tests.discrete_law` does for every other
exact-law test, and no library code enters it.

The stratum ``S = 1{W = 2}`` puts two covariate values in stratum 0, so the plug-in term of
its curve does not vanish. The outcome oracle is wrong, so every block of the fluctuation
moves and the residual term does not vanish either. The random folds give unequal stratum
shares, so the fold-local share and the global share differ in every fold, and unequal fold
sizes, so the ``n / (V n_v)`` factor of the stored curves differs from 1.

The module checks:

* each reported stratum estimate equals the fold average of the stratum plug-ins of the
  targeted regression, recomputed here from the targeted predictions;
* each fold's stratum curve is the Gateaux derivative at :math:`T_v`, for the arm means, the
  ATE and three regimes, unweighted and under a covariate weight;
* each reported stratum score is solved exactly, while each fold's own stratum score is not;
* the marginal estimate is the share mixture of the stratum estimates, with the
  delta-method curve, whose share term is the Gateaux derivative of
  :math:`P \mapsto \sum_s P_w(S = s)\,c_s` at the sample's law with the reported stratum
  estimates held fixed as :math:`c_s`.

Controls that must fail: the global share :math:`P_n(S = s)` in place of the fold's (it leaves
a nonzero curve mean, the unit witness of the plan), the unstratified fold weights in the
update (the reported stratum score is then not solved), and a curve without the stratum share.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from cleverly.estimators import TMLE
from tests import discrete_law as law
from tests import regimes as reg
from tests.conftest import OracleOutcome, OracleTreatment
from tests.unit._declaration_support import tmle_module

#: Three folds of the 1,000 rows hold 334, 333 and 333 rows, so the ``n / (V n_v)`` factor of
#: the stored stratum curves is not 1.
N_FOLDS = 3
SEED = 3
#: The stratum of each covariate value: ``S = 1{W = 2}``.
STRATUM_OF_W = np.array([0, 0, 1])
STRATA = (0, 1)
#: A wrong outcome regression, so every fluctuation block moves.
WRONG_Q = np.array([[0.45, 0.60], [0.35, 0.65], [0.50, 0.70]])
#: A weight on ``W`` alone. It tilts the covariate shares and leaves ``g`` and ``Q`` as they are.
COVARIATE_WEIGHT = law.cell_weights(lambda w, a, y: 1.0 + 0.6 * w)
#: The densities ``g*(a | w)`` of the arm means, keyed by the parameter's stem.
ARMS: dict[str, np.ndarray] = {
    "ey[0]": np.array([[1.0, 0.0]] * 3),
    "ey[1]": np.array([[0.0, 1.0]] * 3),
}
#: Floor for a quantity a correct construction leaves nonzero.
FLOOR = 1e-4
#: Bar for a quantity a correct construction leaves at zero.
EXACT = 1e-9


class Law(law.DiscreteLaw):
    """:class:`tests.discrete_law.DiscreteLaw` on a design that appends ``S`` after ``W``."""

    @staticmethod
    def _index(covariates: Any) -> np.ndarray:
        values = np.asarray(covariates, dtype=float)
        return np.rint(values.reshape(values.shape[0], -1)[:, 0]).astype(int)


def _cells_law(q: np.ndarray, g: np.ndarray, p_w: np.ndarray = law.P_W) -> np.ndarray:
    arm = np.stack([1.0 - g, g], axis=1)
    outcome = np.stack([1.0 - q, q], axis=2)
    return p_w[:, None, None] * arm[:, :, None] * outcome


def _frame(weighted: bool = False) -> Any:
    frame = law.frame().assign(S=lambda f: STRATUM_OF_W[f["W"].to_numpy(dtype=int)])
    if weighted:
        frame = frame.assign(weight=law.row_weights(COVARIATE_WEIGHT))
    return frame


def _fit(*, weighted: bool = False, regimes: bool = False, **extra: Any) -> Any:
    estimator = TMLE(
        outcome_learner=OracleOutcome(Law(_cells_law(WRONG_Q, law.G_EXACT))),
        treatment_learner=OracleTreatment(Law()),
        cross_fit=True,
        cv_evaluation=True,
        n_folds=N_FOLDS,
        estimands=("ey_regime", "ate_regime") if regimes else ("ey", "ate"),
        interventions=reg.interventions() if regimes else None,
        simultaneous=False,
        random_state=SEED,
        tol=1e-12,
        max_iter=200,
        **extra,
    )
    columns: dict[str, Any] = {"weights": "weight"} if weighted else {}
    return estimator.fit(
        _frame(weighted), outcome="Y", treatment="A", covariates=["W", "S"], strata=["S"], **columns
    ).single()


@pytest.fixture(scope="module")
def fits() -> dict[str, Any]:
    return {
        "arms": _fit(),
        "weighted": _fit(weighted=True),
        "regimes": _fit(regimes=True),
    }


def _group(fit: Any) -> str:
    return "regime" if "regime" in fit.fluctuations else "mean"


def _densities(fit: Any) -> dict[str, np.ndarray]:
    if _group(fit) == "mean":
        return ARMS
    return {f"ey_regime[{name}]": density for name, density in law.REGIMES.items()}


def _contrasts(fit: Any) -> dict[str, tuple[str, str]]:
    if _group(fit) == "mean":
        return {"ate": ("ey[1]", "ey[0]")}
    reference = law.REGIME_REFERENCE
    return {
        f"ate_regime[{name} vs {reference}]": (
            f"ey_regime[{name}]",
            f"ey_regime[{reference}]",
        )
        for name in law.REGIMES
        if name != reference
    }


def _stratum(stem: str, code: int) -> str:
    return f"{stem}[S={code}]"


def _rows(fit: Any) -> dict[str, np.ndarray]:
    frame = fit.data
    w = np.rint(frame.covariates[:, 0]).astype(int)
    return {
        "w": w,
        "s": STRATUM_OF_W[w],
        "a": np.asarray(frame.treatment, dtype=float),
        "y": np.asarray(frame.outcome, dtype=float),
        "weight": np.asarray(frame.weights, dtype=float),
        "cell": law.cell_of_row(),
    }


def _q_star(fit: Any) -> np.ndarray:
    """The targeted regression as a ``(3, 2)`` table; the oracles make it one per ``(w, a)``."""
    targeted = fit.fluctuations[_group(fit)].targeted
    w = _rows(fit)["w"]
    table = np.empty((3, 2))
    for cell in range(3):
        for a in (0, 1):
            values = np.asarray(targeted.arms[float(a)])[w == cell]
            assert np.ptp(values) == 0.0
            table[cell, a] = values[0]
    return table


def _folds(fit: Any) -> list[np.ndarray]:
    return [np.asarray(test) for _, test in fit.nuisance.folds]


def _share(rows: dict[str, np.ndarray], index: np.ndarray, code: int) -> float:
    weight = rows["weight"][index]
    return float(np.sum(weight[rows["s"][index] == code]) / np.sum(weight))


def _stratum_functional(density: np.ndarray, code: int, weights: np.ndarray | None) -> Callable:
    """``Psi_s`` of the regime ``density``, longhand, optionally of the tilted law."""
    inside = (code == STRATUM_OF_W).astype(float)

    def psi(probs: Any) -> Any:
        p = probs if weights is None else law.tilt(probs, weights)
        p_w = p.sum(axis=(1, 2))
        q = p[:, :, 1] / p.sum(axis=2)
        return (inside * p_w * (density * q).sum(axis=1)).sum() / (inside * p_w).sum()

    return psi


def _fold_law(fit: Any, index: np.ndarray, weights: np.ndarray | None) -> np.ndarray:
    """The law the fold's rows come from: the fold's raw covariate shares, ``g`` and ``Q*``."""
    rows = _rows(fit)
    shares = np.bincount(rows["w"][index], minlength=3) / index.size
    return _cells_law(_q_star(fit), law.G_EXACT, shares)


def _raw(fit: Any, name: str) -> list[np.ndarray]:
    """Each fold's stratum curve, undoing the ``n / (V n_v)`` factor of the stored report."""
    curve = np.asarray(fit.estimates[name].influence_curve, dtype=float)
    folds = _folds(fit)
    return [curve[index] * len(folds) * index.size / fit.data.n for index in folds]


def _derivative(fit: Any, stem: str, code: int, index: np.ndarray) -> np.ndarray:
    weights = COVARIATE_WEIGHT if fit.data.is_weighted else None
    densities = _densities(fit)
    contrasts = _contrasts(fit)
    probs = _fold_law(fit, index, weights)
    if stem in contrasts:
        left, right = contrasts[stem]
        return _derivative(fit, left, code, index) - _derivative(fit, right, code, index)
    return law.contamination_eif(
        _stratum_functional(densities[stem], code, weights), probs, law.SUPPORT
    )


def _stems(fit: Any) -> list[str]:
    return [*_densities(fit), *_contrasts(fit)]


def _plug_in(fit: Any, stem: str, code: int, index: np.ndarray) -> float:
    """``psi_{s,v}`` from the targeted table and the fold's weighted stratum rows."""
    contrasts = _contrasts(fit)
    if stem in contrasts:
        left, right = contrasts[stem]
        return _plug_in(fit, left, code, index) - _plug_in(fit, right, code, index)
    rows = _rows(fit)
    inside = index[rows["s"][index] == code]
    mean = (_densities(fit)[stem] * _q_star(fit)).sum(axis=1)[rows["w"][inside]]
    return float(np.average(mean, weights=rows["weight"][inside]))


def _score(fit: Any, stem: str, code: int, index: np.ndarray) -> float:
    """The fold's stratum score ``sum w H (Y - Q*) / W_{v,s}`` of a regime mean."""
    rows = _rows(fit)
    inside = index[rows["s"][index] == code]
    a = rows["a"][inside].astype(int)
    w = rows["w"][inside]
    g = np.where(a == 1, law.G_EXACT[w], 1.0 - law.G_EXACT[w])
    clever = _densities(fit)[stem][w, a] / g
    residual = rows["y"][inside] - _q_star(fit)[w, a]
    weight = rows["weight"][inside]
    return float(np.sum(weight * clever * residual) / np.sum(weight))


class TestThePremises:
    def test_every_block_of_the_fluctuation_moves(self, fits: dict[str, Any]) -> None:
        for fit in fits.values():
            epsilon = np.abs(np.asarray(fit.fluctuations[_group(fit)].epsilon))
            assert epsilon.size == len(_densities(fit)) * len(STRATA)
            assert float(np.min(epsilon)) > 1e-3, epsilon

    def test_the_folds_have_unequal_stratum_shares(self, fits: dict[str, Any]) -> None:
        for fit in fits.values():
            rows = _rows(fit)
            for code in STRATA:
                shares = [_share(rows, index, code) for index in _folds(fit)]
                assert max(shares) - min(shares) > 1e-2, shares

    def test_the_folds_have_unequal_sizes(self, fits: dict[str, Any]) -> None:
        for fit in fits.values():
            sizes = {index.size for index in _folds(fit)}
            assert len(sizes) > 1, sizes

    def test_every_stratum_and_contrast_is_reported(self, fits: dict[str, Any]) -> None:
        for fit in fits.values():
            for stem in _stems(fit):
                assert stem in fit.estimates
                for code in STRATA:
                    assert _stratum(stem, code) in fit.estimates


class TestTheStratumEstimate:
    def test_it_is_the_fold_average_of_the_stratum_plug_ins(self, fits: dict[str, Any]) -> None:
        for fit in fits.values():
            for stem in _stems(fit):
                for code in STRATA:
                    name = _stratum(stem, code)
                    plug_ins = [_plug_in(fit, stem, code, index) for index in _folds(fit)]
                    np.testing.assert_allclose(
                        fit.cv_targeting.fold_estimates[name], plug_ins, rtol=0, atol=1e-12
                    )
                    assert fit.estimates[name].psi == pytest.approx(
                        float(np.mean(plug_ins)), abs=1e-12
                    )

    @pytest.mark.parametrize("which", ["arms", "weighted", "regimes"])
    def test_each_fold_curve_is_the_gateaux_derivative(
        self, fits: dict[str, Any], which: str
    ) -> None:
        fit = fits[which]
        cells = _rows(fit)["cell"]
        for stem in _stems(fit):
            for code in STRATA:
                raws = _raw(fit, _stratum(stem, code))
                for index, raw in zip(_folds(fit), raws, strict=True):
                    derivative = _derivative(fit, stem, code, index)
                    for point in np.unique(cells[index]):
                        np.testing.assert_allclose(
                            raw[cells[index] == point], derivative[point], rtol=0, atol=1e-10
                        )

    def test_its_variance_is_the_fold_second_moment(self, fits: dict[str, Any]) -> None:
        fit = fits["arms"]
        for stem in _stems(fit):
            for code in STRATA:
                name = _stratum(stem, code)
                raws = _raw(fit, name)
                expected = (
                    sum(
                        float(np.sum(raw**2)) / index.size**2
                        for raw, index in zip(raws, _folds(fit), strict=True)
                    )
                    / N_FOLDS**2
                )
                assert fit.estimates[name].variance == pytest.approx(expected, rel=1e-12)


class TestTheScores:
    @pytest.mark.parametrize("which", ["arms", "weighted", "regimes"])
    def test_each_reported_stratum_score_is_solved(self, fits: dict[str, Any], which: str) -> None:
        fit = fits[which]
        for stem in _densities(fit):
            for code in STRATA:
                per_fold = [_score(fit, stem, code, index) for index in _folds(fit)]
                assert abs(float(np.mean(per_fold))) < EXACT, per_fold
                # Nonzero witness: the pooled update leaves each fold's own score unsolved.
                assert max(abs(value) for value in per_fold) > 1e-3, per_fold
                name = _stratum(stem, code)
                assert abs(float(np.mean(fit.estimates[name].influence_curve))) < EXACT

    def test_control_the_unstratified_fold_weights_leave_it_unsolved(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fold_weights(data: Any, validation: Any) -> np.ndarray:
            weights = np.array(data.weights, dtype=float, copy=True)
            for test in validation:
                weights[test] *= data.n / (len(validation) * float(np.sum(weights[test])))
            return weights

        monkeypatch.setattr(tmle_module, "fold_evaluated_stratum_weights", fold_weights)
        fit = _fit()
        residuals = [
            abs(float(np.mean([_score(fit, stem, code, index) for index in _folds(fit)])))
            for stem in ARMS
            for code in STRATA
        ]
        assert max(residuals) > FLOOR, residuals


class TestTheShareControls:
    def test_control_the_global_share_leaves_a_nonzero_curve_mean(
        self, fits: dict[str, Any]
    ) -> None:
        """The plan's unit witness: ``1 / P_n(S = s)`` in place of ``1 / P_{n,v}(S = s)``."""
        fit = fits["arms"]
        rows = _rows(fit)
        everything = np.arange(fit.data.n)
        means = []
        for stem in _stems(fit):
            for code in STRATA:
                name = _stratum(stem, code)
                curve = np.asarray(fit.estimates[name].influence_curve, dtype=float).copy()
                for index in _folds(fit):
                    curve[index] *= _share(rows, index, code) / _share(rows, everything, code)
                assert abs(float(np.mean(fit.estimates[name].influence_curve))) < EXACT
                means.append(abs(float(np.mean(curve))))
        assert max(means) > FLOOR, means

    def test_control_the_global_share_is_not_the_derivative(self, fits: dict[str, Any]) -> None:
        fit = fits["arms"]
        rows = _rows(fit)
        everything = np.arange(fit.data.n)
        gaps = []
        for stem in ARMS:
            for code in STRATA:
                raws = _raw(fit, _stratum(stem, code))
                for index, raw in zip(_folds(fit), raws, strict=True):
                    ratio = _share(rows, index, code) / _share(rows, everything, code)
                    derivative = _derivative(fit, stem, code, index)
                    gaps.append(
                        float(np.max(np.abs(raw * ratio - derivative[rows["cell"][index]])))
                    )
        assert max(gaps) > 1e-3, gaps

    def test_control_a_curve_without_the_stratum_share_is_caught(
        self, fits: dict[str, Any]
    ) -> None:
        """The invalid-inference control of the planned study, at unit level."""
        fit = fits["arms"]
        rows = _rows(fit)
        gaps = []
        for stem in ARMS:
            for code in STRATA:
                raws = _raw(fit, _stratum(stem, code))
                for index, raw in zip(_folds(fit), raws, strict=True):
                    derivative = _derivative(fit, stem, code, index)
                    unnormalised = raw * _share(rows, index, code)
                    gaps.append(
                        float(np.max(np.abs(unnormalised - derivative[rows["cell"][index]])))
                    )
        assert max(gaps) > 1e-2, gaps


class TestTheMarginalEstimate:
    @pytest.mark.parametrize("which", ["arms", "weighted", "regimes"])
    def test_it_is_the_share_mixture_of_the_strata(self, fits: dict[str, Any], which: str) -> None:
        fit = fits[which]
        rows = _rows(fit)
        everything = np.arange(fit.data.n)
        shares = np.array([_share(rows, everything, code) for code in STRATA])
        weight = rows["weight"] / rows["weight"].mean()
        for stem in _stems(fit):
            parts = [fit.estimates[_stratum(stem, code)] for code in STRATA]
            stratum_psi = np.array([part.psi for part in parts])
            psi = float(shares @ stratum_psi)
            estimate = fit.estimates[stem]
            assert estimate.psi == pytest.approx(psi, abs=1e-12)
            curve = sum(
                share * np.asarray(part.influence_curve)
                for share, part in zip(shares, parts, strict=True)
            ) + weight * (stratum_psi[rows["s"]] - psi)
            np.testing.assert_allclose(estimate.influence_curve, curve, rtol=0, atol=1e-12)
            assert abs(float(np.mean(estimate.influence_curve))) < EXACT
            assert estimate.variance == pytest.approx(
                float(np.sum(curve**2)) / fit.data.n**2, rel=1e-12
            )
            assert float(np.mean(fit.cv_targeting.fold_estimates[stem])) == pytest.approx(
                psi, abs=1e-12
            )

    @pytest.mark.parametrize("which", ["arms", "weighted", "regimes"])
    def test_its_share_term_is_the_gateaux_derivative_of_the_shares(
        self, fits: dict[str, Any], which: str
    ) -> None:
        r"""An independent check of the share term, under the covariate tilt when weighted.

        With each reported stratum estimate held fixed as :math:`c_s`, the shares'
        contribution to the marginal is :math:`\Psi(P) = \sum_s P_w(S = s)\,c_s`. Its
        contamination derivative at the sample's own law, taken by complex step from that
        longhand statement, must equal the marginal curve less its stratum parts.
        """
        fit = fits[which]
        weights = COVARIATE_WEIGHT if fit.data.is_weighted else None
        rows = _rows(fit)
        cells = rows["cell"]
        everything = np.arange(fit.data.n)
        shares = np.array([_share(rows, everything, code) for code in STRATA])
        for stem in _stems(fit):
            parts = [fit.estimates[_stratum(stem, code)] for code in STRATA]
            fixed = np.array([part.psi for part in parts])

            def mixture(probs: Any, fixed: np.ndarray = fixed) -> Any:
                p = probs if weights is None else law.tilt(probs, weights)
                p_w = p.sum(axis=(1, 2))
                return sum(p_w[code == STRATUM_OF_W].sum() * fixed[code] for code in STRATA)

            derivative = law.contamination_eif(mixture, law.PROBS, law.SUPPORT)
            share_term = np.asarray(fit.estimates[stem].influence_curve) - sum(
                share * np.asarray(part.influence_curve)
                for share, part in zip(shares, parts, strict=True)
            )
            np.testing.assert_allclose(share_term, derivative[cells], rtol=0, atol=1e-10)
            # Nonzero witness: the stratum estimates differ, so the share term moves.
            assert float(np.max(np.abs(derivative))) > FLOOR, derivative

    def test_it_differs_from_the_fold_average_of_marginal_plug_ins(
        self, fits: dict[str, Any]
    ) -> None:
        """Nonzero witness for the reporting choice: fold shares differ from the global ones."""
        fit = fits["arms"]
        rows = _rows(fit)
        gaps = []
        for stem in _stems(fit):
            marginal = float(
                np.mean(
                    [
                        sum(
                            _share(rows, index, code) * _plug_in(fit, stem, code, index)
                            for code in STRATA
                        )
                        for index in _folds(fit)
                    ]
                )
            )
            gaps.append(abs(marginal - fit.estimates[stem].psi))
        assert max(gaps) > 1e-5, gaps
