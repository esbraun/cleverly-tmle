"""Incremental propensity-score interventions with baseline strata, against their exact form.

A stratum parameter is the incremental mean of the law given ``V = s``.  Kennedy (2019),
Theorem 2 of arXiv v3, gives its efficient influence function inside the stratum,

    D_s = h_delta(A, W) {Y - Q(A, W)} + delta {Q(1, W) - Q(0, W)} (A - g) / D^2 + m(W) - psi_s,

with ``D = delta g + 1 - g``, ``h_delta(1, W) = delta / D``, ``h_delta(0, W) = 1 / D`` and
``m = q Q(1, W) + (1 - q) Q(0, W)``.  Its full-law gradient is ``I(V = s) D_s / P(V = s)``.

The fit starts from a deliberately wrong outcome regression and a perturbed treatment
mechanism, both fixed functions, so both stratum score blocks move.  The tests read the
targeted predictions and the targeted mechanism the fit reports and check, stratum by
stratum and tilt by tilt:

* the outcome block ``P_n[I(V = s) h (Y - Q*)]`` and the mechanism block
  ``P_n[I(V = s) H_g (A - g*)]`` are zero;
* each stratum estimate and curve equal the analytic ones at ``(Q*, g*)``;
* the pooled fit equals three separate unstratified fits on the stratum subsets;
* the marginal estimate is the stratum mixture, and ``delta = 1`` gives the stratum mean.

Each mutation replaces one piece of the construction and must fail a witness: one marginal
mechanism column, one marginal outcome column, no ``P_n(V = s)`` in the un-scaling, and no
``n / n_s`` in the embedding.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from cleverly.estimators import TMLE
from cleverly.exceptions import DataError
from cleverly.interventions import Incremental
from tests.studies import stratified_law as law
from tests.unit._declaration_support import assert_every_witness_fails, tmle_module
from tests.unit._natural_course_support import Counting, CountingLogistic
from tests.unit._stratified_alternating_support import (
    PerturbedOutcome,
    PerturbedTreatment,
    marginal_blocks,
)

N = 3_000
SEED = 20261102
TILTS = (Incremental(1.0), Incremental(2.0), Incremental(0.5))
#: The names the fit gives the three tilts, in declaration order.
NAMES = ("natural course", "odds x2", "odds x0.5")
DELTAS = (1.0, 2.0, 0.5)
#: Score blocks are solved to ``tol = 1e-10`` relative.
SCORE_TOL = 1e-9


def estimator(tilts: tuple[Incremental, ...] = TILTS, stratum: int | None = None) -> TMLE:
    return TMLE(
        incremental=list(tilts),
        outcome_learner=PerturbedOutcome(stratum),
        treatment_learner=PerturbedTreatment(stratum),
        cross_fit=False,
        simultaneous=False,
        max_iter=100,
        tol=1e-10,
        random_state=0,
    )


def fit(frame: pd.DataFrame, *, strata: bool = True, subset: int | None = None) -> Any:
    """The stratified fit, or the unstratified fit of the stratum ``subset``'s rows."""
    roles: dict[str, Any] = {"strata": ["V"]} if strata else {}
    return (
        estimator(stratum=subset)
        .fit(frame, outcome="Y", treatment="A", covariates=["W", "V"], **roles)
        .single()
    )


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return law.sample(N, SEED)


@pytest.fixture(scope="module")
def fitted(frame: pd.DataFrame) -> Any:
    return fit(frame)


def targeted(result: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``Q*(1, W)``, ``Q*(0, W)`` and ``g*`` as the fit reports them."""
    fluctuation = result.fluctuations["ipsi"]
    arms = fluctuation.targeted.arms
    return (
        np.asarray(arms[1.0], dtype=float),
        np.asarray(arms[0.0], dtype=float),
        np.asarray(fluctuation.mechanism.propensity, dtype=float),
    )


def tilt_pieces(delta: float, q1: np.ndarray, q0: np.ndarray, g: np.ndarray, a: np.ndarray) -> Any:
    """``h_delta(A, W)``, ``H_g`` and ``m(W)`` at the given nuisances."""
    d = delta * g + 1.0 - g
    q = delta * g / d
    h = np.where(a == 1.0, delta / d, 1.0 / d)
    h_g = delta * (q1 - q0) / d**2
    m = q * q1 + (1.0 - q) * q0
    return h, h_g, m


def block_scores(result: Any, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """``(3 strata, 3 tilts)`` outcome and mechanism block scores at the reported fit."""
    q1, q0, g = targeted(result)
    a = frame["A"].to_numpy(dtype=float)
    y = frame["Y"].to_numpy(dtype=float)
    v = frame["V"].to_numpy()
    fitted_q = np.where(a == 1.0, q1, q0)
    outcome = np.zeros((len(law.STRATA), len(DELTAS)))
    mechanism = np.zeros_like(outcome)
    for s in law.STRATA:
        inside = (v == s).astype(float)
        for r, delta in enumerate(DELTAS):
            h, h_g, _ = tilt_pieces(delta, q1, q0, g, a)
            outcome[s, r] = np.mean(inside * h * (y - fitted_q))
            mechanism[s, r] = np.mean(inside * h_g * (a - g))
    return outcome, mechanism


def analytic(
    result: Any,
    frame: pd.DataFrame,
    delta: float,
    stratum: int,
    *,
    unscale: bool = True,
    embed: bool = True,
) -> tuple[float, np.ndarray]:
    """The stratum estimate and its full-sample curve at the reported ``(Q*, g*)``.

    ``unscale=False`` and ``embed=False`` are the two ``P_n(V = s)`` mutations.
    """
    q1, q0, g = targeted(result)
    a = frame["A"].to_numpy(dtype=float)
    y = frame["Y"].to_numpy(dtype=float)
    v = frame["V"].to_numpy()
    inside = v == stratum
    mass = float(inside.mean())
    h, h_g, m = tilt_pieces(delta, q1, q0, g, a)
    if not unscale:
        h = h / mass
    psi = float(m[inside].mean())
    curve = h * (y - np.where(a == 1.0, q1, q0)) + h_g * (a - g) + m - psi
    return psi, np.where(inside, curve / (mass if embed else 1.0), 0.0)


def check_blocks(result: Any, frame: pd.DataFrame) -> None:
    outcome, mechanism = block_scores(result, frame)
    assert np.max(np.abs(outcome)) < SCORE_TOL, outcome
    assert np.max(np.abs(mechanism)) < SCORE_TOL, mechanism


def check_curves(result: Any, frame: pd.DataFrame, **mutation: bool) -> None:
    for s in law.STRATA:
        for name, delta in zip(NAMES, DELTAS, strict=True):
            estimate = result[f"ey_ipsi[{name}][V={s}]"]
            psi, curve = analytic(result, frame, delta, s, **mutation)
            assert estimate.psi == pytest.approx(psi, abs=1e-12)
            np.testing.assert_allclose(estimate.influence_curve, curve, rtol=0.0, atol=1e-10)


class TestTheStratumEquationsAreSolved:
    def test_every_block_moved(self, fitted: Any) -> None:
        """The nonzero witness: each stratum's mechanism and outcome coefficients moved."""
        fluctuation = fitted.fluctuations["ipsi"]
        mechanism = np.asarray(fluctuation.mechanism.epsilon).reshape(len(law.STRATA), -1)
        # delta = 1 leaves the mechanism equation inert; the other two tilts move it.
        assert np.min(np.abs(mechanism[:, 1:])) > 1e-4, mechanism
        assert np.asarray(fluctuation.epsilon).shape == (len(law.STRATA) * len(DELTAS),)

    def test_each_outcome_and_mechanism_block_is_zero(
        self, fitted: Any, frame: pd.DataFrame
    ) -> None:
        check_blocks(fitted, frame)

    def test_each_stratum_curve_is_kennedys_curve(self, fitted: Any, frame: pd.DataFrame) -> None:
        check_curves(fitted, frame)

    @pytest.mark.parametrize("stratum", law.STRATA)
    def test_each_contrast_is_the_difference(self, fitted: Any, stratum: int) -> None:
        reference = fitted[f"ey_ipsi[natural course][V={stratum}]"]
        for name in NAMES[1:]:
            contrast = fitted[f"ate_ipsi[{name} vs natural course][V={stratum}]"]
            tilted = fitted[f"ey_ipsi[{name}][V={stratum}]"]
            assert contrast.psi == pytest.approx(tilted.psi - reference.psi, abs=1e-14)
            np.testing.assert_allclose(
                contrast.influence_curve,
                tilted.influence_curve - reference.influence_curve,
                rtol=0.0,
                atol=1e-12,
            )


class TestTheStratumIsTheSubsetFit:
    """W-ID: the pooled block solve is three separate solves on the stratum subsets."""

    @pytest.mark.parametrize("stratum", law.STRATA)
    def test_each_stratum_equals_its_subset_fit(
        self, fitted: Any, frame: pd.DataFrame, stratum: int
    ) -> None:
        subset = frame[frame["V"] == stratum].reset_index(drop=True)
        alone = fit(subset, strata=False, subset=stratum)
        for name in NAMES:
            pooled = fitted[f"ey_ipsi[{name}][V={stratum}]"]
            separate = alone[f"ey_ipsi[{name}]"]
            assert pooled.psi == pytest.approx(separate.psi, abs=1e-9)
            inside = frame["V"].to_numpy() == stratum
            share = float(inside.mean())
            np.testing.assert_allclose(
                pooled.influence_curve[inside] * share,
                separate.influence_curve,
                rtol=0.0,
                atol=1e-8,
            )


class TestTheMarginalAndTheNaturalCourse:
    def test_the_marginal_estimate_is_the_stratum_mixture(
        self, fitted: Any, frame: pd.DataFrame
    ) -> None:
        v = frame["V"].to_numpy()
        for name in NAMES:
            mixture = sum(
                float(np.mean(v == s)) * fitted[f"ey_ipsi[{name}][V={s}]"].psi for s in law.STRATA
            )
            assert fitted[f"ey_ipsi[{name}]"].psi == pytest.approx(mixture, abs=1e-12)

    def test_the_marginal_curve_is_the_linearity_curve(
        self, fitted: Any, frame: pd.DataFrame
    ) -> None:
        """``D = sum_s I(S = s)(D_s + psi_s) - psi``: the stratum curves, re-weighted."""
        v = frame["V"].to_numpy()
        for name in NAMES:
            marginal = fitted[f"ey_ipsi[{name}]"]
            linear = np.zeros(len(frame))
            for s in law.STRATA:
                stratum = fitted[f"ey_ipsi[{name}][V={s}]"]
                inside = v == s
                linear[inside] = (
                    stratum.influence_curve[inside] * float(inside.mean()) + stratum.psi
                )
            np.testing.assert_allclose(
                marginal.influence_curve, linear - marginal.psi, rtol=0.0, atol=1e-10
            )

    @pytest.mark.parametrize("stratum", law.STRATA)
    def test_delta_one_is_the_stratum_outcome_mean(
        self, fitted: Any, frame: pd.DataFrame, stratum: int
    ) -> None:
        inside = frame["V"].to_numpy() == stratum
        y = frame["Y"].to_numpy(dtype=float)
        estimate = fitted[f"ey_ipsi[natural course][V={stratum}]"]
        # Exact once the outcome block is solved, so to the solver's tolerance.
        assert estimate.psi == pytest.approx(float(y[inside].mean()), abs=1e-9)
        expected = np.where(inside, (y - estimate.psi) / float(inside.mean()), 0.0)
        np.testing.assert_allclose(estimate.influence_curve, expected, rtol=0.0, atol=1e-8)


class TestAnAbsentArm:
    def test_an_all_control_stratum_refuses_before_any_learner(self) -> None:
        frame = law.sample(600, SEED).assign(A=lambda rows: np.where(rows["V"] == 2, 0, rows["A"]))
        Counting.calls = 0
        with pytest.raises(DataError, match="treatment-mechanism score equation has no finite"):
            TMLE(
                incremental=[Incremental(2.0)],
                outcome_learner=CountingLogistic(max_iter=1000),
                treatment_learner=CountingLogistic(max_iter=1000),
                cross_fit=False,
                simultaneous=False,
            ).fit(frame, outcome="Y", treatment="A", covariates=["W", "V"], strata=["V"])
        assert Counting.calls == 0

    def test_a_stratum_with_one_treated_row_fits_and_solves_its_block(self) -> None:
        frame = law.sample(1_200, SEED)
        treated = np.flatnonzero((frame["V"] == 2).to_numpy() & (frame["A"] == 1).to_numpy())
        frame.loc[treated[1:], "A"] = 0
        result = fit(frame)
        check_blocks(result, frame)


def _marginal_mechanism(matrix: Any, strata: Any, probabilities: Any) -> np.ndarray:
    """Mutation: one marginal mechanism column per tilt, the other blocks zero."""
    values = np.asarray(matrix, dtype=float)
    padding = np.zeros((values.shape[0], values.shape[1] * (len(probabilities) - 1)))
    return np.hstack([values, padding])


class TestTheMutationsFail:
    def test_a_marginal_mechanism_column(
        self, frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(tmle_module, "stratify_columns", _marginal_mechanism)
        mutated = fit(frame)
        assert_every_witness_fails([lambda: check_blocks(mutated, frame)])

    def test_a_marginal_outcome_column(
        self, frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(tmle_module, "stratify", marginal_blocks)
        mutated = fit(frame)
        assert_every_witness_fails([lambda: check_blocks(mutated, frame)])

    def test_no_mass_in_the_unscaling(
        self, frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(tmle_module, "unscale_block", lambda values, probability: values)
        mutated = fit(frame)
        assert_every_witness_fails([lambda: check_curves(mutated, frame)])

    def test_no_mass_in_the_embedding(
        self, frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def unembedded(curve: Any, index: Any, n: int) -> np.ndarray:
            out = np.zeros(n)
            out[index] = curve
            return out

        monkeypatch.setattr(tmle_module, "embed_stratum_curve", unembedded)
        mutated = fit(frame)
        assert_every_witness_fails([lambda: check_curves(mutated, frame)])

    def test_the_analytic_curve_sees_both_masses(self, fitted: Any, frame: pd.DataFrame) -> None:
        """The comparison itself is sensitive to each ``P_n(V = s)``."""
        assert_every_witness_fails(
            [
                lambda: check_curves(fitted, frame, unscale=False),
                lambda: check_curves(fitted, frame, embed=False),
            ]
        )


class TestAWeightedStratumIsTheWeightedSubsetFit:
    """W-ID under fixed analysis weights: ``P_n(S = s)`` is the weighted share.

    The un-scaling multiplies a block by the weighted share, and the embedding multiplies the
    curve by the row count ``n / n_s``.  The blocks and the un-scaling read one
    ``tmle.stratum_probabilities`` call, and a block's root is invariant to its scale, so a
    wrong share there cancels.  ``tests/unit/test_stratified_drtmle_exact.py`` carries the
    mutation control, where the blocks read the share in another module.
    """

    @staticmethod
    def weighted(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.assign(wt=np.random.default_rng(5).uniform(0.3, 2.5, len(frame)))

    @staticmethod
    def fit_weighted(frame: pd.DataFrame, *, subset: int | None = None) -> Any:
        roles: dict[str, Any] = {} if subset is not None else {"strata": ["V"]}
        return (
            estimator(stratum=subset)
            .fit(frame, outcome="Y", treatment="A", covariates=["W", "V"], weights="wt", **roles)
            .single()
        )

    def check(self, result: Any, frame: pd.DataFrame) -> None:
        v = frame["V"].to_numpy()
        for stratum in law.STRATA:
            inside = v == stratum
            alone = self.fit_weighted(frame[inside].reset_index(drop=True), subset=stratum)
            for name in NAMES:
                pooled = result[f"ey_ipsi[{name}][V={stratum}]"]
                separate = alone[f"ey_ipsi[{name}]"]
                assert pooled.psi == pytest.approx(separate.psi, abs=1e-9)
                np.testing.assert_allclose(
                    pooled.influence_curve[inside] * float(inside.mean()),
                    separate.influence_curve,
                    rtol=0.0,
                    atol=1e-8,
                )

    def test_each_weighted_stratum_equals_its_weighted_subset_fit(
        self, frame: pd.DataFrame
    ) -> None:
        weighted = self.weighted(frame)
        self.check(self.fit_weighted(weighted), weighted)
