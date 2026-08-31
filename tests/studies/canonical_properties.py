"""Independent repeated-sampling properties for ordinary canonical TMLE.

The claims are van der Laan and Rubin's: double robustness, root-n behaviour, and calibrated
inference.  A study that passes ``efficiency_bounds`` adds an efficiency comparison against a
bound computed outside the estimator; without one, no cell here claims efficiency.  Each cell
is a law and a nuisance configuration chosen so that a claim can *fail*; the verdicts come from
:mod:`tests.studies.evidence.property_verdicts` and are bounded by the same margins the
implementation comparison declares, so the two halves of the study cannot drift apart.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.datasets import DGP, linear_dgp, nonlinear_dgp
from cleverly.estimators import TMLE
from cleverly.utils.bounds import expit
from tests.conftest import OracleOutcomeContinuous, OracleTreatment
from tests.parallel import STUDY_JOBS
from tests.studies import canonical_cvtmle
from tests.studies.canonical_tmle import G_BOUNDS, STUDY
from tests.studies.evidence.properties import PropertyCell, run_cells
from tests.studies.evidence.property_verdicts import apply_shared_verdicts, finish

#: Sized by what the claim needs, not by habit.  The 99% interval's half-width is about
#: ``2.6 / sqrt(m)`` empirical standard deviations, so ``m`` decides the largest bias the study
#: can still place inside the 0.25 margin: 300 replications resolve 0.10, and 1,200 resolve
#: about 0.175.  The treatment-correct arm uses ``n = 2,000`` because it alone relies on
#: inverse weighting when the outcome regression is wrong.  The other three arms use
#: ``n = 700``.  Raising the budget cannot buy a pass for a bad estimator: the interval
#: contracts on the truth, so a bias past the margin becomes *discriminated* instead.
DOUBLE_ROBUST_REPLICATES = 1_200
DOUBLE_ROBUST_N = 700
DOUBLE_ROBUST_TREATMENT_N = 2_000
DOUBLE_ROBUST_SEED = 17_100

#: The relative score norm a preflight fit has to reach before the study spends its budget.
#: Declared here rather than read from the estimator's ``tol``, and that separation is the
#: whole point: ``Fluctuation.converged`` reports the solver against whatever tolerance it
#: was configured with, so a study that loosened ``tol`` would still report a converged fit.
#: Checking this value as well holds every preflight to one standard.  Keep it equal to the
#: ``tol`` the estimators below pass, and change the two together deliberately.
SOLVED_SCORE_TOL = 1e-10

#: Half-width of the treatment mechanism's logit.  ``L * tanh(u / L)`` maps the unbounded
#: linear predictor onto the open interval ``(-L, L)``, so it leaves ``u`` almost unchanged
#: where the law is moderate and compresses only the tail that would otherwise be clipped.
#: Bounding the *inputs* instead compresses ``W2`` and ``W3`` exactly where the outcome
#: regression varies, which is where the both-wrong control gets its bias from.
DOUBLE_ROBUST_LOGIT_SCALE = 2.5

#: The bounded mechanism below has this open support.  ``u(W)`` reaches every real value,
#: ``tanh`` is a strictly increasing bijection onto ``(-1, 1)``, and ``expit`` is strictly
#: increasing, so the mechanism's range is exactly ``(expit(-L), expit(L))`` and attains
#: neither endpoint.  Both sit inside ``DOUBLE_ROBUST_G_BOUNDS`` with over one logit unit
#: of headroom, so no study estimator clips the oracle probabilities.
DOUBLE_ROBUST_LOGIT_RANGE = (-DOUBLE_ROBUST_LOGIT_SCALE, DOUBLE_ROBUST_LOGIT_SCALE)
DOUBLE_ROBUST_G_RANGE = tuple(float(expit(value)) for value in DOUBLE_ROBUST_LOGIT_RANGE)

#: The narrowest treatment bound any consumer of this law configures, taken as an envelope
#: rather than read from one study.  Ordinary TMLE runs the law at ``(0.01, 0.99)`` and the
#: four cross-fitted studies run it at ``(0.025, 0.975)``, so the design check below has to
#: clear the tighter pair.  Reading a single study's bound would let a study widen its own
#: clipping and silently weaken the check for every other consumer.
DOUBLE_ROBUST_G_BOUNDS = (
    max(G_BOUNDS[0], canonical_cvtmle.G_BOUNDS[0]),
    min(G_BOUNDS[1], canonical_cvtmle.G_BOUNDS[1]),
)

#: Sized by what the *coverage* gate in these cells needs, which is the binding one and was
#: the budget's real constraint.  A 99% Clopper-Pearson lower endpoint clearing 0.90 needs
#: 742 of 800 covered replications (92.75%); at 200 it needed 191 (95.5%), which a correctly
#: calibrated 95% interval misses 54.5% of the time.  Raising the budget does not buy a pass:
#: an estimator whose true coverage is 0.90 is refused at 800 with probability 0.996, the same
#: as at 200.  What it buys is that a *passing* cell means something -- the committed n = 500
#: cell sat at exactly 191 of 200, one replication from red for no reason a commit caused.
RATE_REPLICATES = 800
NULL_REPLICATES = 400

#: The calibration cell below.  The 99% resampling interval for the SE ratio has half-width
#: about ``2.58 / sqrt(2 * (m - 1))`` -- 3.7% at 2,400 -- so it leaves about half of the 7%
#: equivalence margin for a real departure rather than spending all of it on Monte Carlo error.
CALIBRATION_REPLICATES = 2_400
CALIBRATION_N = 2000

#: Three sizes, not two: a rate estimated from two points is a ratio with no residual, and
#: quadrupling twice separates a root-n contraction from a merely decreasing one.
RATE_SIZES = (500, 2000, 8000)

#: Power positive control: an effect this large is rejected essentially always, so a
#: rejection indicator that never fires cannot pass the type-I cell by being inert.
ALTERNATIVE_EFFECT = 0.5


def double_robustness_dgp() -> DGP:
    """Bounded nonlinear confounding with an exact ATE of 1.75.

    The outcome law is the established nonlinear point-treatment law.  Its conditional
    contrast varies with ``W1`` and ``I(W2 > 0)``, so a main-effects linear regression is
    genuinely wrong.  The treatment law keeps that law's linear predictor and bounds the
    *logit* rather than the inputs.

    ``u(W)`` reaches every real value: it goes to ``+inf`` along ``W1 -> +inf`` at
    ``W2 = W3 = 0``, and to ``-inf`` along ``|W2| -> inf`` at ``W3 = 0``.  ``tanh`` is a
    strictly increasing bijection from the reals onto ``(-1, 1)``, so ``L * tanh(u / L)``
    has range exactly ``(-L, L)`` and attains neither endpoint.  ``expit`` is strictly
    increasing, so the mechanism's range is exactly ``DOUBLE_ROBUST_G_RANGE`` and no study
    estimator clips the oracle probabilities.

    Bounding each input instead would compress ``W2`` and ``W3`` where the outcome
    regression varies, and the both-wrong control's bias is the correlation between the two
    misspecifications.  The truth does not move either way: the ATE is
    ``E[2.0 + 0.7 * W1 - 0.5 * I(W2 > 0)] = 1.75``, which the propensity does not enter.
    """

    def propensity(w: np.ndarray) -> np.ndarray:
        logit = 0.6 * w[:, 0] - 0.4 * w[:, 1] ** 2 + 0.5 * w[:, 1] * w[:, 2] + 0.3 * (w[:, 3] > 0)
        scale = DOUBLE_ROBUST_LOGIT_SCALE
        return expit(scale * np.tanh(logit / scale))

    # ``DGP`` is frozen, so the outcome law is the shipped one by construction rather than by
    # a copy that a reader has to diff.  Only the treatment mechanism differs.
    return replace(
        nonlinear_dgp(),
        name="canonical_double_robustness_bounded_nonlinear",
        propensity=propensity,
    )


def _assert_double_robustness_design(cells: tuple[PropertyCell, ...]) -> None:
    """Refuse a driver whose declared robustness law loses a design witness."""
    robust = tuple(cell for cell in cells if cell.property == "double_robustness")
    expected = {
        "both_correct": (DOUBLE_ROBUST_N, DOUBLE_ROBUST_REPLICATES, DOUBLE_ROBUST_SEED),
        "outcome_correct": (DOUBLE_ROBUST_N, DOUBLE_ROBUST_REPLICATES, DOUBLE_ROBUST_SEED + 1),
        "treatment_correct": (
            DOUBLE_ROBUST_TREATMENT_N,
            DOUBLE_ROBUST_REPLICATES,
            DOUBLE_ROBUST_SEED + 2,
        ),
        "both_wrong": (DOUBLE_ROBUST_N, DOUBLE_ROBUST_REPLICATES, DOUBLE_ROBUST_SEED + 3),
    }
    declared = {cell.cell: (cell.n, cell.replicates, cell.seed) for cell in robust}
    if declared != expected:
        raise RuntimeError(
            f"the predeclared double-robustness budgets or seeds changed: {declared!r}"
        )
    if len({id(cell.dgp) for cell in robust}) != 1:
        raise RuntimeError("all four double-robustness arms must use one shared law")
    lower, upper = DOUBLE_ROBUST_G_BOUNDS
    if not (lower < DOUBLE_ROBUST_G_RANGE[0] < DOUBLE_ROBUST_G_RANGE[1] < upper):
        raise RuntimeError(
            "the analytic treatment-mechanism range is not strictly inside "
            f"[{lower}, {upper}], the narrowest bound any consumer configures"
        )
    dgp = robust[0].dgp
    # Two points where ``tanh`` saturates to exactly +-1 in float64, so the mechanism
    # returns the analytic endpoints bit for bit rather than approaching them.
    range_witnesses = np.array([[0.0, 50.0, 0.0, 0.0], [100.0, 0.0, 0.0, 1.0]])
    if not np.allclose(
        dgp.propensity(range_witnesses),
        DOUBLE_ROBUST_G_RANGE,
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError("the treatment law no longer attains its analytic limiting range")

    # A main-effects regression has one treatment coefficient and therefore a constant
    # arm contrast.  These deterministic points make that restriction visible against the
    # law's varying contrast, without relying on a noisy realized sample.
    w = np.array(
        [
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, -1.0, 0.0, 0.0],
            [-1.0, 1.0, 0.0, 0.0],
        ]
    )
    true_contrast = dgp.outcome_mean(w, 1.0, None) - dgp.outcome_mean(w, 0.0, None)
    design = np.vstack(
        [
            np.column_stack([np.zeros(len(w)), w]),
            np.column_stack([np.ones(len(w)), w]),
        ]
    )
    means = np.concatenate([dgp.outcome_mean(w, 0.0, None), dgp.outcome_mean(w, 1.0, None)])
    wrong = LinearRegression().fit(design, means)
    wrong_contrast = wrong.predict(np.column_stack([np.ones(len(w)), w])) - wrong.predict(
        np.column_stack([np.zeros(len(w)), w])
    )
    if np.ptp(true_contrast) <= 0.5 or np.ptp(wrong_contrast) > 1e-12:
        raise RuntimeError("the main-effects wrong-Q contrast witness no longer discriminates")
    if not np.isclose(float(np.mean(true_contrast[[0, 1]])), 1.75):
        raise RuntimeError("the deterministic outcome contrast no longer witnesses ATE 1.75")


def assert_double_robustness_fit(
    cell: PropertyCell,
    result: Any,
    *,
    g_bounds: tuple[float, float],
) -> None:
    """Assert the oracle mechanism, wrong-Q witness, and solved targeting step."""
    w = np.asarray(result.data.covariates, dtype=float)
    oracle = np.asarray(cell.dgp.propensity(w), dtype=float)
    stored = result.nuisance.propensity.arm(1.0)
    bounded = result.nuisance.bounded_propensity(g_bounds)[
        :, result.nuisance.propensity.column_for(1.0)
    ]
    if not np.array_equal(stored, oracle):
        raise RuntimeError("the treatment-correct preflight did not store the oracle mechanism")
    if not np.array_equal(bounded, stored):
        raise RuntimeError("the bounded-overlap oracle was clipped before targeting")

    true_contrast = cell.dgp.outcome_mean(w, 1.0, None) - cell.dgp.outcome_mean(w, 0.0, None)
    # ``lower`` cancels in the difference, so this is the scaler's own method rather than an
    # open-coded ``range *`` that a reader has to check against it.
    scaler = result.nuisance.scaler
    arms = result.nuisance.outcome.arms
    wrong_contrast = scaler.unscale_levels(arms[1.0]) - scaler.unscale_levels(arms[0.0])
    wrong_q_error = float(np.sqrt(np.mean((true_contrast - wrong_contrast) ** 2)))
    if np.ptp(true_contrast) <= 0.5 or wrong_q_error <= 0.25:
        raise RuntimeError("the fitted main-effects wrong-Q contrast witness vanished")

    fluctuation = result.fluctuations["mean"]
    if np.max(np.abs(fluctuation.epsilon)) <= 1e-8:
        raise RuntimeError("the targeting preflight has no nonzero fluctuation witness")
    # Two separate questions. ``converged`` asks whether the solver met its own configured
    # tolerance, and ``SOLVED_SCORE_TOL`` asks whether the score reaches the standard this
    # module declares. The pooled preflight measures 1e-17 here, so the second leaves seven
    # orders of headroom and still bites on a fit that was configured to stop early.
    if not fluctuation.converged or fluctuation.relative_score_norm > SOLVED_SCORE_TOL:
        raise RuntimeError("the targeting preflight did not solve its score")


def null_dgp(effect: float = 0.0) -> DGP:
    """Confounded, with a treatment effect of ``effect``.

    The sharp null keeps the confounding: an unadjusted comparison of arms is biased here, so
    a calibrated rejection rate is evidence about the estimator rather than about a
    randomized experiment.
    """

    def propensity(w: np.ndarray) -> np.ndarray:
        return expit(0.5 * w[:, 0] - 0.3 * w[:, 1])

    def outcome_mean(w: np.ndarray, a: float, z: float | None) -> np.ndarray:
        del z
        return 1.0 + 0.9 * w[:, 0] + 0.6 * w[:, 1] - 0.4 * w[:, 2] + effect * a

    return DGP(
        name=f"canonical_tmle_null_{effect:g}",
        n_latent=3,
        covariate_names=("W1", "W2", "W3"),
        propensity=propensity,
        outcome_mean=outcome_mean,
    )


def cells() -> tuple[PropertyCell, ...]:
    """Every property cell, with the seed each is drawn with.

    Distinct seeds throughout.  The sizes in the rate study in particular are meant to be
    independent runs -- the slope's interval is computed as though they were -- and sharing a
    seed across them would make that assumption false for no gain.
    """
    nonlinear = double_robustness_dgp()
    linear = linear_dgp()
    nuisances: tuple[tuple[str, Callable[[], Any], Callable[[], Any]], ...] = (
        (
            "both_correct",
            lambda: OracleOutcomeContinuous(nonlinear),
            lambda: OracleTreatment(nonlinear),
        ),
        (
            "outcome_correct",
            lambda: OracleOutcomeContinuous(nonlinear),
            lambda: LogisticRegression(max_iter=1000),
        ),
        ("treatment_correct", lambda: LinearRegression(), lambda: OracleTreatment(nonlinear)),
        ("both_wrong", lambda: LinearRegression(), lambda: LogisticRegression(max_iter=1000)),
    )
    out: list[PropertyCell] = []
    for index, (name, outcome, treatment) in enumerate(nuisances):
        out.append(
            PropertyCell(
                property="double_robustness",
                cell=name,
                dgp=nonlinear,
                outcome_learner=outcome,
                treatment_learner=treatment,
                n=(DOUBLE_ROBUST_TREATMENT_N if name == "treatment_correct" else DOUBLE_ROBUST_N),
                replicates=DOUBLE_ROBUST_REPLICATES,
                seed=DOUBLE_ROBUST_SEED + index,
                role="control" if name == "both_wrong" else "positive",
            )
        )
    for index, size in enumerate(RATE_SIZES):
        out.append(
            PropertyCell(
                property="root_n_and_efficiency",
                cell=f"n_{size}",
                dgp=linear,
                outcome_learner=LinearRegression,
                treatment_learner=lambda: LogisticRegression(max_iter=1000),
                n=size,
                replicates=RATE_REPLICATES,
                seed=8100 + 100 * index,
            )
        )
    out.append(
        PropertyCell(
            property="interval_calibration",
            cell="correctly_specified",
            dgp=linear,
            outcome_learner=LinearRegression,
            treatment_learner=lambda: LogisticRegression(max_iter=1000),
            n=CALIBRATION_N,
            replicates=CALIBRATION_REPLICATES,
            seed=11_100,
        )
    )
    out.append(
        PropertyCell(
            property="type_i_error",
            cell="sharp_null",
            dgp=null_dgp(),
            outcome_learner=LinearRegression,
            treatment_learner=lambda: LogisticRegression(max_iter=1000),
            n=1000,
            replicates=NULL_REPLICATES,
            seed=9100,
        )
    )
    out.append(
        PropertyCell(
            property="power",
            cell="alternative",
            dgp=null_dgp(ALTERNATIVE_EFFECT),
            outcome_learner=LinearRegression,
            treatment_learner=lambda: LogisticRegression(max_iter=1000),
            n=1000,
            replicates=NULL_REPLICATES,
            seed=9200,
        )
    )
    result = tuple(out)
    _assert_double_robustness_design(result)
    return result


def _estimator(cell: PropertyCell) -> Callable[[], Any]:
    return lambda: TMLE(
        outcome_learner=cell.outcome_learner(),
        treatment_learner=cell.treatment_learner(),
        cross_fit=False,
        # A bare name is ``resolve_estimands``'s single-estimand form, and unlike a
        # one-tuple it does not have to claim a narrower element type than the cell has.
        estimands=cell.estimand,
        simultaneous=False,
        g_bounds=G_BOUNDS,
        max_iter=100,
        tol=1e-10,
        random_state=0,
    )


def run_double_robustness_preflight(
    declared: tuple[PropertyCell, ...],
    build: Callable[[PropertyCell], Callable[[], Any]],
    *,
    g_bounds: tuple[float, float],
    n: int = 400,
) -> Any:
    """Fit the treatment-correct arm once and assert its deterministic controls.

    The fitted result is returned so a cross-fitted caller can add its own fold and
    cross-validated evaluation checks to the same fit rather than paying for a second one.
    """
    cell = next(
        cell
        for cell in declared
        if cell.property == "double_robustness" and cell.cell == "treatment_correct"
    )
    frame, _ = cell.dgp.sample(n, seed=cell.seed)
    result = build(cell)().fit(frame, **cell.fit_kwargs).single()
    assert_double_robustness_fit(cell, result, g_bounds=g_bounds)
    return result


def generate_property_rows(*, n_jobs: int = STUDY_JOBS) -> pd.DataFrame:
    """Run every property cell and return the per-replication rows."""
    declared = cells()
    run_double_robustness_preflight(declared, _estimator, g_bounds=G_BOUNDS)
    return run_cells(declared, _estimator, n_jobs=n_jobs)


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """Per-cell descriptive summary, the two rate rows, and every verdict."""
    return finish(*apply_shared_verdicts(rows, STUDY))
