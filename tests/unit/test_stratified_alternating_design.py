"""The declared design of the stratified incremental and MSM study, recomputed before any run.

Each condition below was declared with the study and is checked here from the law itself:

* the incremental stratum means, and the dose slopes, are resolvably apart (at least 0.03);
* the treatment mechanism of L1 overlaps between 0.25 and 0.75;
* the dose grid of L2 lies inside the 1st to 99th percentile of the dose in every cell;
* each declared control is displaced from the truth by at least one standard deviation of the
  positive arm at n = 2,000, computed as the population limit of the control's own estimator
  on the law's 36 support points.  The both-wrong robustness control is the exception: it is
  displaced at least 0.45 positive SD in every stratum, and a 300-replication smoke run before
  the declaration measured 1.37, 0.83 and 0.53 of its own SD, so the 99% interval of its
  standardized bias clears the 0.25 margin at R = 1,200 by design;
* the single-correct robustness configurations have the truth as their population limit;
* the published cells are the declared cells, with the declared law's truths, before any run.

It also pins why three planned families are not declared: on L1 their controls stay near the
truth, so they could not fail.

The population limits solve each estimator's estimating equations with the law's probabilities
as weights, so no number here is a Monte Carlo estimate.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression

from tests.studies import canonical_stratified_incremental_msm as study
from tests.studies import stratified_alternating_law as law
from tests.studies import stratified_incremental_msm_properties as properties

N = 2_000
SUPPORT = np.asarray(law.SUPPORT, dtype=float)
V, W, A, Y = SUPPORT.T
PROBS = law.PROBS
G = np.asarray(law.base.propensity(W, V))
Q1 = np.asarray(law.base.outcome(1.0, W, V))
Q0 = np.asarray(law.base.outcome(0.0, W, V))
#: ``P(V = v, W = w)`` at each support point.
CELL = PROBS.reshape(3, 3, 2, 2).sum(axis=(2, 3)).ravel()[np.arange(PROBS.size) // 4]
#: One support point per ``(v, w)`` cell.
FIRST = (A == 0) & (Y == 0)


def _sd(name: str, link: str | None = None) -> float:
    return law.efficiency_sd(name, link) / np.sqrt(N)


def _projection_without_v() -> tuple[np.ndarray, np.ndarray]:
    """The logistic projection of ``Y`` on ``(A, W)``, at each support point and each arm."""
    model = LogisticRegression(C=1e12, max_iter=10_000, tol=1e-12).fit(
        np.column_stack([A, W]), Y, sample_weight=PROBS
    )

    def at(arm: float) -> np.ndarray:
        return model.predict_proba(np.column_stack([np.full_like(W, arm), W]))[:, 1]

    return at(1.0), at(0.0)


def _fluctuate(initial: np.ndarray, covariate: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """``epsilon`` solving ``sum p w H (Y - expit(logit Q + H epsilon)) = 0`` by Newton."""
    epsilon = np.zeros(covariate.shape[1])
    for _ in range(100):
        fitted = expit(logit(initial) + covariate @ epsilon)
        score = covariate.T @ (PROBS * weights * (Y - fitted))
        hessian = (covariate * (PROBS * weights * fitted * (1.0 - fitted))[:, None]).T @ covariate
        step = np.linalg.lstsq(hessian, score, rcond=None)[0]
        epsilon += step
        if np.max(np.abs(step)) < 1e-14:
            break
    return epsilon


def _stratum_mean(values: np.ndarray, stratum: int) -> float:
    """``E[values | V = s]``, where ``values`` is a function of ``(W, V)``."""
    inside = FIRST & (stratum == V)
    return float(np.sum(CELL[inside] * values[inside]) / np.sum(CELL[inside]))


def _tilt(delta: float, g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``q_delta(1 | W, V)`` and the observed-arm weight ``h_delta(A, W, V)``."""
    d = delta * g + 1.0 - g
    return delta * g / d, np.where(A == 1.0, delta / d, 1.0 / d)


def _arm_weight(delta: float, arm: float) -> np.ndarray:
    d = delta * G + 1.0 - G
    return delta / d if arm == 1.0 else 1.0 / d


def incremental_necessity_limit(stratum: int) -> float:
    """The marginal incremental fit's stratum mixture at ``delta = 2``, with ``Q`` on ``(A, W)``."""
    p1, p0 = _projection_without_v()
    deltas = list(law.DELTAS.values())
    covariate = np.column_stack([_tilt(delta, G)[1] for delta in deltas])
    epsilon = _fluctuate(np.where(A == 1.0, p1, p0), covariate, np.ones_like(Y))
    star1 = expit(logit(p1) + np.column_stack([_arm_weight(d, 1.0) for d in deltas]) @ epsilon)
    star0 = expit(logit(p0) + np.column_stack([_arm_weight(d, 0.0) for d in deltas]) @ epsilon)
    q = _tilt(2.0, G)[0]
    return _stratum_mean(q * star1 + (1.0 - q) * star0, stratum)


def mechanism_wrong_limit(stratum: int) -> float:
    """The stratified incremental contrast at ``delta = 2`` with ``Q`` right and ``g`` on ``V``."""
    cells = PROBS.reshape(3, 3, 2, 2)
    by_v = cells[:, :, 1, :].sum(axis=(1, 2)) / cells.sum(axis=(1, 2, 3))
    g = by_v[V.astype(int)]
    inside = (stratum == V).astype(float)
    deltas = list(law.DELTAS.values())
    for _ in range(200):
        covariate = (
            np.column_stack([d * (Q1 - Q0) / (d * g + 1.0 - g) ** 2 for d in deltas])
            * inside[:, None]
        )
        epsilon = np.zeros(len(deltas))
        for _ in range(100):
            tilted = expit(logit(g) + covariate @ epsilon)
            score = covariate.T @ (PROBS * (A - tilted))
            hessian = (covariate * (PROBS * tilted * (1.0 - tilted))[:, None]).T @ covariate
            step = np.linalg.lstsq(hessian, score, rcond=None)[0]
            epsilon += step
            if np.max(np.abs(step)) < 1e-14:
                break
        updated = expit(logit(g) + covariate @ epsilon)
        moved = float(np.max(np.abs(updated - g)))
        g = updated
        if moved < 1e-13:
            break

    def mean(delta: float) -> float:
        q = _tilt(delta, g)[0]
        return _stratum_mean(q * Q1 + (1.0 - q) * Q0, stratum)

    return mean(2.0) - mean(1.0)


def _projection(
    q1: np.ndarray, q0: np.ndarray, mass: np.ndarray, *, with_w: bool = True
) -> np.ndarray:
    """The logit projection of ``(q1, q0)`` on ``(1, a, W)``, or ``(1, a)``, over ``mass``."""
    w = W[FIRST]
    cells = mass[FIRST]
    target = np.concatenate([q0[FIRST], q1[FIRST]])
    columns = [np.ones(2 * w.size), np.r_[np.zeros(w.size), np.ones(w.size)]]
    design = np.column_stack([*columns, np.r_[w, w]] if with_w else columns)
    weights = np.r_[cells, cells]
    beta = np.zeros(design.shape[1])
    for _ in range(100):
        m = expit(design @ beta)
        first = m * (1 - m)
        residual = target - m
        score = design.T @ (weights * first * residual)
        curvature = first**2 - residual * first * (1 - 2 * m)
        jacobian = (design * (weights * curvature)[:, None]).T @ design
        beta = beta + np.linalg.solve(jacobian, score)
    return beta


def _cell_mass(stratum: int | None) -> np.ndarray:
    return CELL if stratum is None else np.where(stratum == V, CELL, 0.0)


def _msm_covariate(
    beta: np.ndarray, arm: float, share: np.ndarray, inside: np.ndarray
) -> np.ndarray:
    """``I(S = s) m'(phi beta) phi(arm, W) / g(arm)`` at each support point."""
    columns = [np.ones_like(W), np.full_like(W, arm), W][: beta.size]
    m = expit(np.column_stack(columns) @ beta)
    phi = np.column_stack(columns)
    return phi * (m * (1 - m) / share * inside)[:, None]


def _linked_msm_limit(
    p1: np.ndarray,
    p0: np.ndarray,
    g: np.ndarray,
    scope: int | None,
    report: int | None,
    *,
    with_w: bool = True,
) -> float:
    """The logit-MSM slope of an alternation over ``scope`` (``None`` is marginal).

    The fluctuated regression is projected on the ``report`` stratum's cells.  ``with_w``
    selects the working model ``(1, a, W)`` or ``(1, a)``.
    """
    inside = np.ones_like(W) if scope is None else (scope == V).astype(float)
    beta = _projection(p1, p0, _cell_mass(scope), with_w=with_w)
    star1, star0 = p1, p0
    for _ in range(200):
        h1 = _msm_covariate(beta, 1.0, g, inside)
        h0 = _msm_covariate(beta, 0.0, 1.0 - g, inside)
        covariate = np.where((A == 1.0)[:, None], h1, h0)
        epsilon = _fluctuate(np.where(A == 1.0, p1, p0), covariate, np.ones_like(Y))
        star1 = expit(logit(p1) + h1 @ epsilon)
        star0 = expit(logit(p0) + h0 @ epsilon)
        updated = _projection(star1, star0, _cell_mass(scope), with_w=with_w)
        moved = float(np.max(np.abs(updated - beta)))
        beta = updated
        if moved < 1e-12:
            break
    return float(_projection(star1, star0, _cell_mass(report), with_w=with_w)[1])


def projection_necessity_limit(stratum: int) -> float:
    """The marginal logit-MSM fit with ``Q`` on ``(A, W)``, projected on one stratum."""
    p1, p0 = _projection_without_v()
    return _linked_msm_limit(p1, p0, G, None, stratum)


#: The intercept-only regressions of ``Y`` and ``A``: the wrong nuisances.
CONSTANT_Q = np.full_like(W, float(np.sum(PROBS * Y)))
CONSTANT_G = np.full_like(W, float(np.sum(PROBS * A)))


def both_wrong_limit(stratum: int, *, with_w: bool = False) -> float:
    """The stratified logit-MSM slope with intercept-only ``Q`` and ``g``.

    The declared robustness family fits ``(1, a)``; ``with_w=True`` is ``(1, a, W)``.
    """
    return _linked_msm_limit(CONSTANT_Q, CONSTANT_Q, CONSTANT_G, stratum, stratum, with_w=with_w)


def single_correct_limit(stratum: int, configuration: str) -> float:
    """The ``(1, a)`` slope limit with one nuisance right and the other intercept-only."""
    if configuration == "outcome_correct":
        return _linked_msm_limit(Q1, Q0, CONSTANT_G, stratum, stratum, with_w=False)
    return _linked_msm_limit(CONSTANT_Q, CONSTANT_Q, G, stratum, stratum, with_w=False)


def untargeted_limit(stratum: int) -> float:
    """The stratum projection of the logistic regression of ``Y`` on ``(A, W)``, untargeted."""
    p1, p0 = _projection_without_v()
    return float(_projection(p1, p0, _cell_mass(stratum))[1])


# ------------------------------------------------------------------------------ the tests


def test_the_incremental_stratum_means_are_resolvably_apart() -> None:
    for name in law.DELTAS:
        means = sorted(law.TRUTH_IPSI[f"ey_ipsi[{name}][V={s}]"] for s in law.STRATA)
        assert min(np.diff(means)) >= 0.03, name


def test_the_dose_slopes_are_resolvably_apart() -> None:
    for link in ("identity", "logit"):
        slopes = sorted(law.l2_truths(link)[f"msm[a][V={s}]"] for s in law.STRATA)
        assert min(np.diff(slopes)) >= 0.03, link


def test_the_mechanism_overlaps() -> None:
    assert G.min() >= 0.249 and G.max() <= 0.75


def test_the_grid_lies_inside_the_dose_percentiles_of_every_cell() -> None:
    for v in (0, 1, 2):
        for w in (0, 1, 2):
            mean = law.dose_mean(w, v)
            assert mean - 2.3263 < law.GRID[0] and law.GRID[-1] < mean + 2.3263, (v, w)


@pytest.mark.parametrize("stratum", study.NECESSITY_STRATA)
def test_the_incremental_necessity_control_is_displaced(stratum: int) -> None:
    name = f"ey_ipsi[odds x2][V={stratum}]"
    displacement = (incremental_necessity_limit(stratum) - law.TRUTH_IPSI[name]) / _sd(name)
    assert abs(displacement) >= 1.0, displacement


@pytest.mark.parametrize("stratum", [int(label[1]) for label in study.TARGETING_LABELS])
def test_the_untargeted_msm_control_is_displaced(stratum: int) -> None:
    name = f"msm[a][V={stratum}]"
    displacement = (untargeted_limit(stratum) - law.TRUTH_MSM["logit"][name]) / _sd(name, "logit")
    assert abs(displacement) >= 1.0, displacement


def _arm_sd(name: str) -> float:
    return law.efficiency_sd(name, "logit", law.ARM_TERMS) / np.sqrt(N)


@pytest.mark.parametrize("stratum", [int(label[1]) for label in study.BOTH_WRONG_LABELS])
def test_the_both_wrong_msm_control_is_displaced(stratum: int) -> None:
    name = f"msm[a][V={stratum}]"
    displacement = (both_wrong_limit(stratum) - law.TRUTH_ARM_MSM[name]) / _arm_sd(name)
    assert abs(displacement) >= 0.45, displacement


@pytest.mark.parametrize("configuration", ["outcome_correct", "treatment_correct"])
@pytest.mark.parametrize("stratum", [int(label[1]) for label in study.DOUBLE_ROBUST_LABELS])
def test_one_correct_nuisance_reaches_the_truth(stratum: int, configuration: str) -> None:
    name = f"msm[a][V={stratum}]"
    limit = single_correct_limit(stratum, configuration)
    assert limit == pytest.approx(law.TRUTH_ARM_MSM[name], abs=1e-8)


class TestTheUndeclaredControlsCouldNotFail:
    """Why three planned families are not declared: on L1 their controls stay near the truth."""

    def test_stratum_two_does_not_resolve_the_untargeted_control(self) -> None:
        """Why the MSM targeting family reads strata 0 and 1."""
        name = "msm[a][V=2]"
        displacement = (untargeted_limit(2) - law.TRUTH_MSM["logit"][name]) / _sd(name, "logit")
        assert abs(displacement) < 0.6

    @pytest.mark.parametrize("stratum", law.STRATA)
    def test_the_mechanism_tilt_repairs_a_mechanism_that_omits_w(self, stratum: int) -> None:
        name = f"ate_ipsi[odds x2 vs natural course][V={stratum}]"
        displacement = (mechanism_wrong_limit(stratum) - law.TRUTH_IPSI[name]) / _sd(name)
        assert abs(displacement) < 0.1

    @pytest.mark.parametrize("stratum", law.STRATA)
    def test_the_both_wrong_msm_fit_with_w_stays_near_the_truth(self, stratum: int) -> None:
        """Why the robustness family fits ``(1, a)`` and not the calibration model."""
        name = f"msm[a][V={stratum}]"
        truth = law.TRUTH_MSM["logit"][name]
        displacement = (both_wrong_limit(stratum, with_w=True) - truth) / _sd(name, "logit")
        assert abs(displacement) < 0.5

    @pytest.mark.parametrize("stratum", study.NECESSITY_STRATA)
    def test_the_marginal_msm_fluctuation_stays_near_the_truth(self, stratum: int) -> None:
        name = f"msm[a][V={stratum}]"
        truth = law.TRUTH_MSM["logit"][name]
        displacement = (projection_necessity_limit(stratum) - truth) / _sd(name, "logit")
        assert abs(displacement) < 0.6


def test_the_declared_cells_are_the_cells_a_run_publishes() -> None:
    """Pre-run guard for the truth-binding gate of ``tests/unit/test_method_evidence.py``."""
    declared = {(cell.property, cell.cell): cell for cell in properties.declared_cells()}
    rows = properties.generate_property_rows(n_jobs=1, budget=2)
    published = {tuple(key) for key in rows.groupby(["property", "cell"]).groups}
    assert published == set(declared)
    assert set(declared) == {
        (family, cell) for family, cells in study.STUDY.property_cells.items() for cell in cells
    }
    for (family, name), cell in declared.items():
        if family == "simultaneous_coverage":
            continue
        truth = rows.loc[(rows["property"] == family) & (rows["cell"] == name), "truth"]
        np.testing.assert_allclose(truth, cell.dgp.truth()[cell.estimand], rtol=1e-12, atol=0)


def test_no_two_families_share_a_declared_stream() -> None:
    streams: dict[tuple[str, int], set[str]] = {}
    for cell in properties.declared_cells():
        if cell.property == "simultaneous_coverage":
            continue
        streams.setdefault((cell.dgp.name, cell.seed), set()).add(cell.property)
    assert all(len(families) == 1 for families in streams.values())


# ------------------------------------------------- the identity-MSM and DR-TMLE studies


def test_the_identity_msm_declared_cells_are_the_published_cells() -> None:
    from tests.studies import canonical_stratified_msm_identity as identity
    from tests.studies import stratified_msm_identity_properties as identity_properties

    declared = {(cell.property, cell.cell): cell for cell in identity_properties.declared_cells()}
    rows = identity_properties.generate_property_rows(n_jobs=1, budget=2)
    published = {tuple(key) for key in rows.groupby(["property", "cell"]).groups}
    assert published == set(declared)
    assert set(declared) == {
        (family, cell) for family, cells in identity.STUDY.property_cells.items() for cell in cells
    }
    for (family, name), cell in declared.items():
        truth = rows.loc[(rows["property"] == family) & (rows["cell"] == name), "truth"]
        np.testing.assert_allclose(truth, cell.dgp.truth()[cell.estimand], rtol=1e-12, atol=0)


def test_the_natural_course_truths_are_l1_s_conditional_means() -> None:
    """Missingness leaves ``E[Y | V = s]`` alone: it is the incremental natural-course mean."""
    truths = law.natural_course_truths()
    for stratum in (None, *law.STRATA):
        suffix = "" if stratum is None else f"[V={stratum}]"
        assert truths[f"ey_obs{suffix}"] == pytest.approx(
            law.TRUTH_IPSI[f"ey_ipsi[natural course]{suffix}"], abs=1e-12
        )


def test_the_bounded_outcome_keeps_l1_s_mean() -> None:
    """The identity-MSM truths are L1's because the bounded outcome keeps ``E[Y | A, W, V]``."""
    frame = law.beta_sample(200_000, 7)
    expected = law.base.outcome(frame["A"], frame["W"], frame["V"])
    residual = frame["Y"].to_numpy() - np.asarray(expected)
    assert abs(float(np.mean(residual))) < 3e-3


def test_the_drtmle_declared_cells_are_the_published_cells(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With a stand-in fit: the names and truths a run publishes do not read the estimator.

    The stand-in is the in-sample stratified TMLE of the same estimands, which reports the same
    names at a fraction of the cross-fitted DR-TMLE fit's cost.
    """
    from cleverly.estimators import TMLE
    from tests.studies import canonical_stratified_drtmle as drtmle
    from tests.studies import stratified_drtmle_properties as drtmle_properties

    def stand_in(frame: object, configuration: str) -> object:
        del configuration
        return (
            TMLE(
                estimands=("ey", "ate"),
                outcome_learner=LogisticRegression(max_iter=1000),
                treatment_learner=LogisticRegression(max_iter=1000),
                cross_fit=False,
                simultaneous=False,
            )
            .fit(
                frame, outcome="Y", treatment="A", covariates=list(drtmle.COVARIATES), strata=["V"]
            )
            .single()
        )

    monkeypatch.setattr(drtmle_properties, "PROPERTY_N", 400)
    monkeypatch.setattr(drtmle_properties, "fit_cleverly", stand_in)
    declared = {(cell.property, cell.cell): cell for cell in drtmle_properties.declared_cells()}
    rows = drtmle_properties.generate_property_rows(n_jobs=1, budget=2)
    published = {tuple(key) for key in rows.groupby(["property", "cell"]).groups}
    assert published == set(declared)
    assert set(declared) == {
        (family, cell) for family, cells in drtmle.STUDY.property_cells.items() for cell in cells
    }
    for (family, name), cell in declared.items():
        truth = rows.loc[(rows["property"] == family) & (rows["cell"] == name), "truth"]
        np.testing.assert_allclose(truth, cell.dgp.truth()[cell.estimand], rtol=1e-12, atol=0)


def test_the_drtmle_stratum_effects_are_resolvably_apart() -> None:
    effects = sorted(law.paper_truths()[f"ate[V={s}]"] for s in law.STRATA)
    assert min(np.diff(effects)) >= 0.03


def test_the_drtmle_both_wrong_control_is_displaced() -> None:
    """Both GLMs without ``W12`` bias each stratum ATE, in units of its SD at n = 2,000.

    This reads one large-sample fit at n = 10,000, whose own sampling noise is 0.45 of those
    units.  Measured at n = 60,000 the displacements are 6.7, 5.8 and 3.6.
    """
    from tests.studies import canonical_stratified_drtmle as drtmle

    frame, truth = drtmle.draw_from_seed(drtmle.SCENARIO, 10_000, 11)
    result = drtmle.fit_cleverly(frame, "both_wrong")
    for label, sd in drtmle.EFFICIENCY_SD.items():
        name = drtmle.ate_name(label)
        displacement = (result[name].psi - truth[name]) / (sd / np.sqrt(2_000))
        assert abs(displacement) >= 1.0, (name, displacement)
