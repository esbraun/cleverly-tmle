"""Bounded twins of the laws the cross-fitted property cells sample from.

Cross-fitting a continuous outcome needs a known outcome support.  The estimator maps
``Y`` onto ``[0, 1]`` before it fits ``Qbar``, and without a declared ``q_bounds`` it
derives that map from the observed sample -- which is a function of the held-out rows, so
each fold's training predictions depend on the fold it predicts.  Every cross-fitted
property cell here therefore declares ``q_bounds=(0, 1)`` and samples from a law whose
outcome is a proportion, drawn as ``Beta(phi m, phi (1 - m))`` with ``m`` the conditional
mean.  The Gaussian laws stay where they are: the in-sample rows keep sampling from them,
and this module changes no cell they declare.

Each law is a :func:`dataclasses.replace` over the package law its Gaussian twin uses, so
the two differ in the outcome mean and the family and in nothing else.  Every propensity
is carried over unchanged, which is what lets a cell's treatment oracle, its ``g_bounds``
and the overlap the study documents stay the same statement about the same mechanism.
Truths come from :meth:`~cleverly.datasets.DGP.truth`, the same Sobol rule the Gaussian
truths come from.

Declared before the run
-----------------------

Every constant below was chosen from a **disposable pilot** run outside the repository, on
a separately labelled ``stream_seed(record, "pilot", ...)`` stream rather than on any
registered seed.  The measurements are recorded here so a reader can see what each
threshold was set against.

Most of the figures below are design quantities of a law, or a *control* cell's own
discrimination.  Two are not, and this section used to claim that none of them were.
The generated-design figures are the **control**'s paired SE-ratio deficit, which is the
statistic the rule at :data:`GENERATED_DESIGN_EFFECT` is allowed to read, and that rule
states which quantities it refuses to read.  The selector-necessity RMSE ratio enters the
joint verdict of a family that also contains a positive cell.  It is reported here as a
property of the law, and no constant in this module was selected against it.

**Bounded double-robustness law** (:func:`double_robustness_dgp`), ``kappa = 0.7``,
``centre = 2.8``, ``phi = 12``:

===================================================  ==========================
quantity                                             pilot measurement
===================================================  ==========================
analytic mean range                                  ``(0.0759, 0.9241)``
smallest beta shape (analytic)                       ``0.910``
``ptp`` true contrast at the witness rows            ``0.2859``
``ptp`` main-effects contrast at the witness rows    ``0.0``
``ptp`` true contrast, realized sample at n = 700    ``0.6450``
wrong-Q contrast RMS error at n = 700                ``0.1163``
both-wrong bias at n = 700, 400 replications         ``-0.0358``, ``2.296`` SD
both-wrong bias by quadrature                        ``-0.0344``
===================================================  ==========================

The last two rows are the same quantity by two routes.  The sampled figure is the control
the study runs; the quadrature figure is the population one-step bias against the
population projections the two misspecified learners converge to.  They agree to four per
cent, which is what says the deterministic check below reads the control the study runs.
The equivalence margin asks for ``2 x 0.25`` empirical standard deviations, which is
``0.0078`` at this law's spread, so the law delivers 4.4 times what the control needs.

**Bounded linear law** (:func:`linear_dgp`), ``phi = 12``: mean range ``(0.1058, 0.9019)``,
smallest beta shape ``1.178``, ATE ``0.1072047``.  ``QuasiBinomialGLM`` recovers all six
declared coefficients on the ``2**16`` quadrature grid to ``1.4e-15``, so the root-n and
calibration cells are correctly specified in both nuisances, which is what those cells
claim.

**Bounded null law** (:func:`null_dgp`), ``phi = 20``: mean range ``(0.0597, 0.9501)`` at
the sharp null, smallest beta shape ``0.998``.  The unadjusted arm difference sits
``3.59`` empirical standard deviations from the truth at ``n = 1,000``, so a calibrated
rejection rate here is evidence about the estimator rather than about a randomized
experiment.  The power control rejects ``1.0000`` of 400 replications at
:data:`ALTERNATIVE_EFFECT`.

:data:`GENERATED_DESIGN_EFFECT` and :data:`GENERATED_DESIGN_REPLICATES` follow the rule
stated at those two constants.  The rule reads the two laws' quadrature and the control's
paired SE-ratio deficit, and it names what it refuses to read.  The design floor selects
the coefficient ``0.15``, and the pilot measured the deficit there over the declared budget
ladder at ``n = 1,000``.

=========  ===========================  ============  ============  ==========
budget     paired SE-ratio deficit      half-width    clearance     resolved
=========  ===========================  ============  ============  ==========
1,200      ``[-0.055974, -0.015935]``   ``0.020020``  ``0.005935``  no
2,400      ``[-0.043563, -0.015273]``   ``0.014145``  ``0.005273``  no
4,800      ``[-0.043227, -0.024355]``   ``0.009436``  ``0.014355``  yes
=========  ===========================  ============  ============  ==========

The clearance is the distance from the interval's upper endpoint to the ``0.01``
threshold.  A budget resolves the deficit when the clearance exceeds the interval's own
half-width, so 4,800 is the first budget on the ladder the rule accepts.  Raising the
budget cannot buy a pass for a design with no deficit: the interval contracts on zero.

**Bounded nonlinear law** (:func:`nonlinear_dgp`) is the shipped
:func:`~cleverly.datasets.nonlinear_bounded_dgp`, mean range ``(0.1314, 0.9377)``,
smallest beta shape ``0.748``.  The in-sample tree control reports an SE ratio of
``0.4552`` and a coverage of ``0.4675`` at ``n = 500`` over 400 replications, against the
``0.75`` ceiling the control has to fall below.  The outcome-adaptive outcome-wrong
control sits ``1.493`` standard deviations from the truth at ``n = 700``, with an SE ratio
of ``0.892`` inside the union-model band.

**Bounded instrument law** (:func:`instrument_dgp`), ``phi = 12``: mean range
``(0.1043, 0.9233)``, smallest beta shape ``0.920``, ATE ``0.0826``.  The collaborative
arm's RMSE is ``0.2016`` of the empty-candidate control's over 400 replications, against
the ``0.25`` the selector-necessity design asks for, and the control sits ``5.610``
standard deviations from the truth.  On the bounded linear law the selector's both-Dummy
control sits ``0.621`` standard deviations from the truth at ``n = 700``, above the
``2 x 0.25`` the equivalence margin needs.

Where the checks run
--------------------

:func:`bounded_twin` and :func:`bounded_cells` run :func:`assert_bounded_law_design`,
which is deterministic and touches no quadrature grid, so a driver cannot build a cell
from a degenerate law.  The quadrature statements -- the mean stays inside ``(0, 1)``, the
quasibinomial recovery, the both-wrong population bias -- are cached measurement functions
that :mod:`tests.unit.test_bounded_cv_laws` asserts, because a Sobol grid at import time
would be paid by every study driver and every fast test that imports one.  Sampling a beta
law whose mean leaves ``(0, 1)`` is refused by the package itself, inside
:meth:`~cleverly.datasets.DGP.sample`.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Any

import numpy as np
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor

from cleverly.data import CausalData
from cleverly.datasets import DGP, binary_outcome_dgp, nonlinear_bounded_dgp
from cleverly.datasets import instrument_dgp as gaussian_instrument_dgp
from cleverly.datasets import linear_dgp as gaussian_linear_dgp
from cleverly.datasets import nonlinear_dgp as gaussian_nonlinear_dgp
from cleverly.estimators import TMLE
from cleverly.learners.crossfit import Folds, make_folds
from cleverly.utils.bounds import expit
from tests.conftest import OracleOutcomeContinuous, OracleOutcomeUnit, OracleTreatment
from tests.studies import canonical_properties
from tests.studies.canonical_properties import DoubleRobustnessDesign
from tests.studies.evidence.properties import PropertyCell
from tests.studies.evidence.property_verdicts import (
    DIAGNOSTIC_ROLE,
    FOLD_POLICY_FAMILY,
    FOLD_POLICY_REFERENCE_CELL,
)
from tests.studies.fractional_glm import QuasiBinomialGLM

#: The outcome bounds every cell here declares.  A proportion's support is the law's, not
#: the sample's, so it is the one case where a continuous outcome can be cross-fitted
#: without the scale depending on the held-out rows.  Never passed on a binary outcome:
#: :meth:`~cleverly.estimators.TMLE._scaler` refuses ``q_bounds`` there, because the scaler
#: is already the identity and a second declaration could only disagree with it.
Q_BOUNDS = (0.0, 1.0)

#: The beta precision most laws here declare.  ``Var(Y | A, W) = m (1 - m) / (1 + phi)``,
#: and the shapes ``phi m`` and ``phi (1 - m)`` must stay well away from zero or a draw
#: rounds to exactly 0 or 1 in double precision.  The shipped bounded law keeps its
#: smallest shape above 0.68 for that reason, and every law here clears 0.74.
CONCENTRATION = 12.0

#: The bounded null law's own precision.  Higher than :data:`CONCENTRATION` on purpose: the
#: type-I cell needs the law to be *confounded* enough that an unadjusted comparison is
#: visibly wrong, and the pilot reached 3.59 standard deviations only once the outcome
#: coefficients were large enough to need a tighter draw to keep the beta shapes usable.
NULL_CONCENTRATION = 20.0

#: How steeply the bounded double-robustness law's logit tracks the Gaussian law's mean,
#: and where it is centred.  ``kappa = 0`` collapses the law to a constant 0.5 at both
#: arms, which is the mutation :mod:`tests.unit.test_bounded_cv_laws` uses to show the
#: design check bites.
DOUBLE_ROBUST_LOGIT_SLOPE = 0.7
DOUBLE_ROBUST_CENTRE = 2.8

#: The bounded double-robustness law's analytic mean range, and the covariate rows that
#: attain it.  ``L * tanh(u / L)`` has range exactly ``(-L, L)``, so the mean's range is
#: exactly ``(expit(-L), expit(L))`` -- the same construction, and the same ``L``, the
#: treatment mechanism already uses.  ``tanh`` saturates to exactly ``-1`` and ``+1`` in
#: double precision at these rows, so the law returns the endpoints bit for bit rather
#: than approaching them.
DOUBLE_ROBUST_MEAN_RANGE = (
    float(expit(-canonical_properties.DOUBLE_ROBUST_LOGIT_SCALE)),
    float(expit(canonical_properties.DOUBLE_ROBUST_LOGIT_SCALE)),
)
#: The first row saturates the clamp downward through ``-0.5 W3 W4`` and the second upward
#: through ``0.6 W2 ** 2``, at both arms, so the pair reads the range in its declared order.
MEAN_RANGE_WITNESS_ROWS = np.array([[0.0, 0.0, 100.0, 100.0], [0.0, 50.0, 0.0, 0.0]])

#: What the deterministic contrast at the first two
#: :data:`~tests.studies.canonical_properties.CONTRAST_WITNESS_ROWS` averages to.  The
#: bounded counterpart of the Gaussian law's 1.75: there is no closed form for it, so it is
#: quoted from the law itself and compared at a relative tolerance rather than a bitwise
#: one, because ``tanh`` and ``expit`` are allowed to differ in the last place across
#: platforms.
DOUBLE_ROBUST_WITNESS_CONTRAST = 0.28531287511295667

#: The two thresholds the bounded double-robustness design is read against, and the
#: pilot figures behind them.  The contrast spread measured 0.2859 at the witness rows and
#: 0.6450 on a realized sample, and the wrong-Q contrast RMS error measured 0.1163, so each
#: threshold sits at roughly half of what the law delivers.  Halving is deliberate: the
#: check has to fail on a law that lost its discriminating property, not on one that moved.
DOUBLE_ROBUST_CONTRAST_SPREAD = 0.15
DOUBLE_ROBUST_WRONG_Q_ERROR = 0.05

#: What the population both-wrong bias has to reach.  The equivalence margin asks the
#: control to sit ``2 x 0.25`` empirical standard deviations from the truth, which is
#: ``0.0078`` at the spread the pilot measured for this law at ``n = 700``.  The law
#: delivers ``0.0344``.  The threshold is stated on the population quantity rather than on
#: the sampled one because it is then a statement about the *law*, checkable without
#: spending a replication budget.
BOTH_WRONG_LIMIT_BIAS = 0.010

#: The bounded linear law, on :func:`~cleverly.datasets.linear_dgp`'s covariates and
#: propensity.  Deliberately a plain logistic-linear mean and no ``tanh`` clamp, because
#: :class:`~tests.studies.fractional_glm.QuasiBinomialGLM` has to be *exactly* correct
#: here: the root-n ladder and the calibration cell both claim correctly specified
#: nuisances, and a clamp would make the outcome learner subtly wrong and the claim false.
LINEAR_INTERCEPT = -0.15
LINEAR_EFFECT = 0.45
LINEAR_SCALE = 0.30
LINEAR_WEIGHTS = (1.0, 0.5, -0.8, 0.4)

#: The largest coefficient error :func:`quasibinomial_recovery` accepts.  The pilot
#: measured ``1.4e-15``, so this leaves twelve orders of headroom and still fails a law
#: that stopped being logistic-linear.
LINEAR_RECOVERY_TOLERANCE = 1e-3

#: The bounded null law's outcome coefficients, on
#: :func:`~tests.studies.canonical_properties.null_dgp`'s propensity.  The treatment enters
#: the logit, so an effect of exactly zero makes the two arm means *identically* equal and
#: the ATE exactly zero rather than zero to quadrature error.
NULL_WEIGHTS = (0.50, 0.26, -0.16)

#: How far from the truth the unadjusted arm difference has to sit, in its own empirical
#: standard deviations, for the sharp null to be a confounded one.  Pilot: 3.59.
NULL_CONFOUNDING_DISPLACEMENT = 3.0

#: The power control's effect, on the logit.  Pilot rejection 1.0000 of 400 replications
#: at ``n = 1,000``; the shared floor is 0.80.
ALTERNATIVE_EFFECT = 0.2

#: The outcome-adaptive generated-design cells' effect, on the logit, and the rule that
#: sets it.
#:
#: The rule reads two kinds of quantity and no others: a design quantity of the two laws,
#: and the *control*'s discrimination.  The control is ``generated_design/estimated`` and
#: its statistic is the paired SE-ratio deficit against the oracle arm.  The rule does not
#: read the oracle arm's own SE ratio, its resampling interval, or the calibration band
#: those answer to, because that cell is ``role="positive"`` and the band is its verdict.
#:
#: The floor is the design quantity.  The coefficient is at least the value at which this
#: law's standardized treatment effect -- its ATE over the square root of the population
#: mean conditional outcome variance -- equals the retired Gaussian generated-design law's.
#: The Gaussian law reaches ``0.3000000`` at its own declared 0.3, and this law matches it
#: at ``0.1359433``, so the floor on the declared 0.05 grid is 0.15.  The coefficient is
#: not the Gaussian law's own number: that one is an additive shift of an unbounded mean
#: and this one is a shift of a logit, which is why the two need a common currency at all.
#:
#: The choice is the smallest coefficient on that grid, at the smallest budget on the
#: ladder :data:`GENERATED_DESIGN_REPLICATES` declares, whose control resolves the deficit.
#: Smallest rather than largest: a larger coefficient moves the law further from the null
#: and makes every cell in the family easier, so the rule states the weakest law whose
#: control still works.  The floor itself qualifies, at 4,800 replications.
GENERATED_DESIGN_EFFECT = 0.15

#: What the generated-design cells' replication budget has to be, and it is not the
#: Gaussian study's 1,200.
#:
#: The ladder is 1,200, then its doublings 2,400 and 4,800.  The budget is the smallest
#: rung at which the control's 99% deficit interval lies below the ``0.01`` threshold *and*
#: clears it by more than the interval's own half-width.  The second clause is what says
#: the pilot resolved the deficit instead of meeting the threshold at its own resolution.
#: The deficit is a *paired* difference of two SE ratios, and a bounded outcome's estimates
#: are an order of magnitude less spread than the Gaussian law's, so the same design
#: produces a smaller number against the same Monte Carlo error.  A rung's replication
#: seeds are a prefix of the next rung's, so one pilot run measured all three.  The margin
#: does not move; the budget does.  Both cells of the family carry it, because the deficit
#: is resampled on the two arms' shared replication indices.
GENERATED_DESIGN_REPLICATES = 4_800

#: The Gaussian generated-design effect, which names the law
#: :func:`~tests.studies.canonical_properties.null_dgp` builds for that family.  Mirrored
#: from ``ctmle_oat_properties.GENERATED_DESIGN_EFFECT`` rather than imported, because that
#: module imports this one and a law must not depend on the study that declares it.
#: ``tests/unit/test_bounded_cv_laws.py`` asserts the two still agree.
GAUSSIAN_GENERATED_DESIGN_EFFECT = 0.3

#: The bounded instrument law's outcome coefficients.  ``W2`` is absent on purpose -- it is
#: the instrument, and the selector-necessity cell exists because adjusting for it is
#: harmful.  The propensity stays :func:`~cleverly.datasets.instrument_dgp`'s, so the
#: overlap the cell is about is unchanged.
INSTRUMENT_INTERCEPT = -0.05
INSTRUMENT_EFFECT = 0.35
INSTRUMENT_CONFOUNDER = 0.40
INSTRUMENT_PREDICTOR = 0.25

#: The smallest ``phi min(m, 1 - m)`` any declared law may reach on the quadrature grid.
#:
#: Both beta shapes have to stay away from zero, or a draw rounds to exactly 0 or 1 in
#: double precision and the outcome leaves the open interval the law claims.  The shipped
#: bounded law states 0.68 as the figure that keeps a draw off the endpoints and measures
#: 0.748; the tightest law here measures 0.749, so this floor leaves each of them room to
#: move without leaving the claim unchecked.
MINIMUM_BETA_SHAPE = 0.70

#: The seed offset each consuming property module applies to the canonical cells it
#: inherits.  Declared here rather than spelled in each module, because
#: :func:`bounded_cells` is what reproduces them and a bounded twin that lost an offset
#: would silently share a sample stream with the study it inherited from.
INHERITED_SEED_OFFSETS: Mapping[str, int] = {
    "cvtmle_properties": 0,
    "ctmle_selector_properties": 4_000,
    "ctmle_oat_properties": 5_000,
}

#: The fold-policy diagnostic family, its three policies, and its shared budget.
#:
#: The diagnostic compares three outer-split policies on one binary law and one set of
#: draws.  All three cells are *diagnostic* rather than gated: the roadmap's claim is that
#: such a comparison can detect a failure, not that it supplies an inference result, and a
#: gate would publish a verdict the design cannot support.  Two of the three policies are
#: also no longer selectable through the public interface, which is why the study reaches
#: them through :class:`FoldPolicyTMLE` rather than through ``stratify_folds=``.
#:
#: 800 replications at ``n = 500``, matching
#: :data:`~tests.studies.canonical_properties.RATE_REPLICATES`: a 99% Clopper-Pearson lower
#: endpoint clearing the 0.90 floor needs 742 of 800, so the reported coverage intervals
#: are read on the same scale as every other coverage interval the study publishes.
#: The family name, the reference arm and the role come from the framework, which owns the
#: reporting rule; only the policies and the budget belong to the study.
FOLD_POLICIES = (
    FOLD_POLICY_REFERENCE_CELL,
    "treatment_stratified",
    "treatment_outcome_stratified",
)
FOLD_POLICY_N = 500
FOLD_POLICY_REPLICATES = 800
FOLD_POLICY_FOLDS = 10
FOLD_POLICY_SEED = 14_100


# --------------------------------------------------------------------------- laws


def double_robustness_dgp() -> DGP:
    """The bounded twin of the double-robustness law, with an exact ATE of 0.2368749.

    The mean is ``expit(L tanh(kappa (mu(a, W) - c) / L))`` with ``mu`` the shipped
    nonlinear law's outcome mean, so the contrast still varies with ``W1`` and
    ``I(W2 > 0)`` and a main-effects regression is still genuinely wrong.  The ``tanh``
    clamp is the construction the treatment mechanism already uses, and it earns the same
    thing here: an analytic range, :data:`DOUBLE_ROBUST_MEAN_RANGE`, which bounds the
    smallest beta shape below by ``phi expit(-L) = 0.91`` whatever ``kappa`` is.  Without
    it a steeper law would put means next to 0 and 1, where a shape near zero puts real
    probability on a draw that rounds to the endpoint.

    Returns
    -------
    DGP
        The law the four bounded double-robustness arms share.
    """
    gaussian = gaussian_nonlinear_dgp()
    scale = canonical_properties.DOUBLE_ROBUST_LOGIT_SCALE

    def outcome_mean(w: np.ndarray, a: float, z: float | None) -> np.ndarray:
        raw = np.asarray(gaussian.outcome_mean(w, a, z), dtype=float)
        logit = DOUBLE_ROBUST_LOGIT_SLOPE * (raw - DOUBLE_ROBUST_CENTRE)
        return expit(scale * np.tanh(logit / scale))

    return replace(
        canonical_properties.double_robustness_dgp(),
        name="bounded_double_robustness",
        outcome_mean=outcome_mean,
        family="beta",
        concentration=CONCENTRATION,
    )


def linear_dgp() -> DGP:
    """The bounded twin of the linear law: a logistic-linear mean on the same covariates.

    Returns
    -------
    DGP
        The law the root-n ladder, the calibration cell and the selector's
        double-robustness arms sample from.
    """
    weights = np.asarray(LINEAR_WEIGHTS, dtype=float)

    def outcome_mean(w: np.ndarray, a: float, z: float | None) -> np.ndarray:
        del z
        return expit(LINEAR_INTERCEPT + LINEAR_EFFECT * a + LINEAR_SCALE * (w[:, :4] @ weights))

    return replace(
        gaussian_linear_dgp(),
        name="bounded_linear",
        outcome_mean=outcome_mean,
        family="beta",
        concentration=CONCENTRATION,
    )


def null_dgp(effect: float = 0.0) -> DGP:
    """The bounded twin of the confounded null law, with a treatment effect of ``effect``.

    Parameters
    ----------
    effect : float
        The treatment's coefficient on the logit.  At zero the two arm means are
        identically equal, so the ATE is exactly zero rather than zero to quadrature error.

    Returns
    -------
    DGP
        The law the type-I, power and generated-design cells sample from.
    """
    weights = np.asarray(NULL_WEIGHTS, dtype=float)

    def outcome_mean(w: np.ndarray, a: float, z: float | None) -> np.ndarray:
        del z
        return expit(w[:, :3] @ weights + effect * a)

    return replace(
        canonical_properties.null_dgp(),
        name=f"bounded_null_{effect:g}",
        outcome_mean=outcome_mean,
        family="beta",
        concentration=NULL_CONCENTRATION,
    )


def nonlinear_dgp() -> DGP:
    """The bounded twin of the nonlinear law: the shipped bounded law itself.

    Returns
    -------
    DGP
        :func:`~cleverly.datasets.nonlinear_bounded_dgp` at this module's concentration.
    """
    return nonlinear_bounded_dgp(concentration=CONCENTRATION)


def instrument_dgp() -> DGP:
    """The bounded twin of the instrument law, with ``W2`` still absent from the outcome.

    Returns
    -------
    DGP
        The law the selector-necessity cells sample from.
    """

    def outcome_mean(w: np.ndarray, a: float, z: float | None) -> np.ndarray:
        del z
        return expit(
            INSTRUMENT_INTERCEPT
            + INSTRUMENT_EFFECT * a
            + INSTRUMENT_CONFOUNDER * w[:, 0]
            + INSTRUMENT_PREDICTOR * w[:, 2]
        )

    return replace(
        gaussian_instrument_dgp(),
        name="bounded_instrument",
        outcome_mean=outcome_mean,
        family="beta",
        concentration=CONCENTRATION,
    )


def declared_laws() -> tuple[DGP, ...]:
    """Every bounded law a cell here samples from.

    Returns
    -------
    tuple of DGP
        The beta laws, in declaration order.  The binary fold-policy law is not among them:
        it draws no beta variate and has no concentration to check.
    """
    return (
        double_robustness_dgp(),
        linear_dgp(),
        null_dgp(),
        null_dgp(ALTERNATIVE_EFFECT),
        null_dgp(GENERATED_DESIGN_EFFECT),
        nonlinear_dgp(),
        instrument_dgp(),
    )


def fold_policy_dgp() -> DGP:
    """The binary law the fold-policy diagnostic samples from.

    Returns
    -------
    DGP
        :func:`~cleverly.datasets.binary_outcome_dgp`, the stacked study's own binary law.
        Binary on purpose: ``treatment+outcome`` strata need a binary outcome to cross the
        arms with, and a binary outcome needs no ``q_bounds`` at all.
    """
    return binary_outcome_dgp()


# ------------------------------------------------------------------- law mapping


@dataclass(frozen=True)
class BoundedTwin:
    """One Gaussian law's bounded replacement, and what its linear outcome learner becomes.

    Parameters
    ----------
    dgp : DGP
        The bounded law.
    linear_outcome : Callable
        What a cell's ``LinearRegression`` outcome learner becomes on this law.  The
        mapping is the law's, not the learner's: on the linear and null laws the Gaussian
        cells use ``LinearRegression`` as a *correctly specified* outcome regression, and
        its bounded counterpart is the quasibinomial solver, because a logistic-linear mean
        is what a correct learner has to reach.  On the double-robustness and nonlinear
        laws the same class is the deliberately wrong learner and stays exactly itself.
    """

    dgp: DGP
    linear_outcome: Callable[[], Any]


@lru_cache(maxsize=1)
def twins() -> Mapping[str, BoundedTwin]:
    """Every Gaussian law a cross-fitted property cell declares, keyed by its name.

    Returns
    -------
    Mapping
        Gaussian law name to its bounded twin.  Cached, so one law object is shared by
        every cell that names it, which is what
        :func:`~tests.studies.canonical_properties.assert_double_robustness_design`'s
        one-law rule reads.
    """
    return {
        canonical_properties.double_robustness_dgp().name: BoundedTwin(
            double_robustness_dgp(), LinearRegression
        ),
        gaussian_linear_dgp().name: BoundedTwin(linear_dgp(), QuasiBinomialGLM),
        gaussian_nonlinear_dgp().name: BoundedTwin(nonlinear_dgp(), LinearRegression),
        gaussian_instrument_dgp().name: BoundedTwin(instrument_dgp(), LinearRegression),
        canonical_properties.null_dgp().name: BoundedTwin(null_dgp(), QuasiBinomialGLM),
        canonical_properties.null_dgp(canonical_properties.ALTERNATIVE_EFFECT).name: BoundedTwin(
            null_dgp(ALTERNATIVE_EFFECT), QuasiBinomialGLM
        ),
        canonical_properties.null_dgp(GAUSSIAN_GENERATED_DESIGN_EFFECT).name: BoundedTwin(
            null_dgp(GENERATED_DESIGN_EFFECT), QuasiBinomialGLM
        ),
    }


def _bounded_outcome(factory: Callable[[], Any], twin: BoundedTwin) -> Callable[[], Any]:
    """What a Gaussian cell's outcome learner becomes on the bounded law.

    Total by construction: an unrecognised learner raises rather than passing through.  A
    learner that reached a bounded cell unmapped would be the failure this module exists to
    prevent -- :class:`~tests.conftest.OracleOutcomeContinuous` on a law whose scaler is the
    identity fits a regression to recover a map that is already the identity, and reports a
    fitted line where the exact mean was available.
    """
    learner = factory()
    if isinstance(learner, OracleOutcomeContinuous):
        return lambda: OracleOutcomeUnit(twin.dgp)
    if isinstance(learner, LinearRegression):
        return twin.linear_outcome
    if isinstance(learner, DummyRegressor | DecisionTreeRegressor):
        return factory
    raise TypeError(
        f"{type(learner).__name__} has no declared bounded counterpart; add one rather "
        f"than letting a cross-fitted cell fit an unmapped learner on a bounded law"
    )


def _bounded_treatment(factory: Callable[[], Any], twin: BoundedTwin) -> Callable[[], Any]:
    """What a Gaussian cell's treatment learner becomes on the bounded law.

    Every bounded law carries its Gaussian twin's propensity unchanged, so the oracle is
    rebound to the bounded law for readability rather than for a different number.
    """
    learner = factory()
    if isinstance(learner, OracleTreatment):
        return lambda: OracleTreatment(twin.dgp)
    if isinstance(learner, LogisticRegression | DummyClassifier):
        return factory
    raise TypeError(
        f"{type(learner).__name__} has no declared bounded counterpart; add one rather "
        f"than letting a cross-fitted cell fit an unmapped learner on a bounded law"
    )


def bounded_twin(cell: PropertyCell) -> PropertyCell:
    """One Gaussian property cell, restated on its law's bounded twin.

    Everything the cell declares but its law and its two learners is carried over: the
    family, the cell name, the role, the size, the replication budget, the seed and the fit
    keywords.  That is the point of writing the twin as a transformation rather than as a
    second set of cells.  A budget or a seed retyped for a bounded copy is a budget that
    can be changed in one of them, and a seed that moved would publish a study whose
    committed rows no longer redraw.

    Parameters
    ----------
    cell : PropertyCell
        The Gaussian cell.

    Returns
    -------
    PropertyCell
        The bounded cell.

    See Also
    --------
    bounded_cells : The same transformation over a whole inherited block.
    assert_bounded_law_design : The design witnesses this runs before it returns.
    """
    assert_bounded_law_design()
    try:
        twin = twins()[cell.dgp.name]
    except KeyError:
        raise KeyError(
            f"{cell.property}/{cell.cell} samples from {cell.dgp.name!r}, which has no "
            f"declared bounded twin. A cross-fitted cell needs one: without it the fit "
            f"either derives its outcome scale from the held-out rows or declares "
            f"q_bounds it cannot support"
        ) from None
    return replace(
        cell,
        dgp=twin.dgp,
        outcome_learner=_bounded_outcome(cell.outcome_learner, twin),
        treatment_learner=_bounded_treatment(cell.treatment_learner, twin),
    )


def bounded_cells(
    consumer: str,
    *,
    exclude: Collection[str] = (),
    replicates: Mapping[str, int] | None = None,
) -> tuple[PropertyCell, ...]:
    """The canonical property cells a module inherits, restated on the bounded laws.

    Parameters
    ----------
    consumer : str
        The inheriting property module's name, which selects its seed offset from
        :data:`INHERITED_SEED_OFFSETS`.
    exclude : Collection of str
        Property families the consumer declares itself and does not inherit.
    replicates : Mapping or None
        Property family to the replication budget the consumer declares for it, where that
        differs from the canonical budget.

    Returns
    -------
    tuple of PropertyCell
        The inherited cells, in the canonical order.
    """
    try:
        offset = INHERITED_SEED_OFFSETS[consumer]
    except KeyError:
        raise KeyError(
            f"{consumer!r} declares no inherited seed offset. Add one to "
            f"INHERITED_SEED_OFFSETS rather than passing a literal, so a bounded twin "
            f"cannot silently share a sample stream with the study it inherited from"
        ) from None
    budgets = dict(replicates or {})
    out = tuple(
        replace(
            bounded_twin(cell),
            seed=cell.seed + offset,
            replicates=budgets.get(cell.property, cell.replicates),
        )
        for cell in canonical_properties.cells()
        if cell.property not in exclude
    )
    double_robustness = tuple(cell for cell in out if cell.property == "double_robustness")
    if double_robustness:
        canonical_properties.assert_double_robustness_design(out, design=DESIGN)
    return out


# ------------------------------------------------------------------ design checks


def _assert_bounded_contrast(dgp: DGP) -> None:
    """The bounded law's own witness: the analytic range and the deterministic contrast.

    Both clauses vanish together when the law degenerates.  At ``kappa = 0`` the mean is
    the constant 0.5 at both arms, so the range collapses onto a point and the contrast
    onto zero, and either clause alone would catch it.  Both are kept because they fail for
    different reasons: a law that lost its clamp keeps its contrast and loses its range,
    and a law re-centred away from the interesting region keeps its range and loses its
    contrast.
    """
    means = np.array(
        [dgp.outcome_mean(MEAN_RANGE_WITNESS_ROWS, a, None) for a in (0.0, 1.0)], dtype=float
    )
    if not np.allclose(means, np.array(DOUBLE_ROBUST_MEAN_RANGE), rtol=0.0, atol=1e-12):
        raise RuntimeError(
            f"the bounded double-robustness law no longer attains its analytic mean range "
            f"{DOUBLE_ROBUST_MEAN_RANGE} at the saturating witness rows"
        )
    contrast, _ = canonical_properties.main_effects_contrast(dgp)
    witnessed = float(np.mean(contrast[[0, 1]]))
    if not np.isclose(witnessed, DOUBLE_ROBUST_WITNESS_CONTRAST, rtol=1e-9, atol=0.0):
        raise RuntimeError(
            f"the deterministic outcome contrast witnesses {witnessed!r}, not the declared "
            f"{DOUBLE_ROBUST_WITNESS_CONTRAST!r}"
        )


#: The bounded law's half of the double-robustness design check.  The Gaussian law keeps
#: :data:`~tests.studies.canonical_properties.GAUSSIAN_DESIGN`, unchanged, so the ordinary
#: and in-sample rows are read against exactly the constants they always were.
DESIGN = DoubleRobustnessDesign(
    contrast_spread=DOUBLE_ROBUST_CONTRAST_SPREAD,
    wrong_q_error=DOUBLE_ROBUST_WRONG_Q_ERROR,
    witness=_assert_bounded_contrast,
)


@lru_cache(maxsize=1)
def assert_bounded_law_design() -> None:
    """Refuse every declared bounded law that lost a deterministic design witness.

    Deterministic and free of any quadrature grid, so :func:`bounded_twin` can run it on
    every call without a driver paying for a Sobol rule it does not need.  The quadrature
    statements are in :func:`mean_range`, :func:`quasibinomial_recovery` and
    :func:`both_wrong_limit_bias`, which :mod:`tests.unit.test_bounded_cv_laws` asserts.

    Cached, because the laws are fixed at import and the answer cannot change within a
    process.
    """
    _assert_bounded_contrast(double_robustness_dgp())

    null = null_dgp()
    rows = np.zeros((3, 3))
    rows[:, 0] = (-1.0, 0.0, 1.0)
    if not np.array_equal(null.outcome_mean(rows, 1.0, None), null.outcome_mean(rows, 0.0, None)):
        raise RuntimeError(
            "the bounded sharp null's two arm means are not identically equal, so its ATE "
            "is zero only to quadrature error and the type-I cell tests a near-null"
        )
    for effect in (ALTERNATIVE_EFFECT, GENERATED_DESIGN_EFFECT):
        law = null_dgp(effect)
        moved = np.asarray(
            law.outcome_mean(rows, 1.0, None) - law.outcome_mean(rows, 0.0, None), dtype=float
        )
        if not np.all(moved > 0.0):
            raise RuntimeError(
                f"the bounded null law at effect {effect} has no positive contrast, so its "
                f"control cannot discriminate"
            )

    instrument = instrument_dgp()
    shifted = np.zeros((2, 3))
    shifted[1, 1] = 5.0
    if not np.array_equal(
        instrument.outcome_mean(shifted, 1.0, None)[0],
        instrument.outcome_mean(shifted, 1.0, None)[1],
    ):
        raise RuntimeError(
            "W2 entered the bounded instrument law's outcome mean, so it is no longer an "
            "instrument and the selector-necessity control has nothing to be wrong about"
        )


def assert_unit_outcome_scaler(result: Any) -> None:
    """Refuse a fit whose outcome scaler is not the identity.

    The condition :class:`~tests.conftest.OracleOutcomeUnit` cannot check for itself.  A
    learner is handed the already-scaled outcome, and a scaler derived from the observed
    range maps it into ``[1/12, 11/12]``, which is a subset of what the identity produces:
    no input distinguishes the two.  Here the scaler is on the result, so the check is
    exact.

    Parameters
    ----------
    result : Any
        A fitted single-parameter result.

    Raises
    ------
    RuntimeError
        When the fit derived its outcome scale instead of taking the declared ``(0, 1)``.
    """
    scaler = result.nuisance.scaler
    if (float(scaler.lower), float(scaler.upper)) != Q_BOUNDS:
        raise RuntimeError(
            f"the bounded cells need the identity outcome scaler, and this fit derived "
            f"({scaler.lower}, {scaler.upper}) from the observed outcome. Pass "
            f"q_bounds={Q_BOUNDS} for a proportion, and pass nothing for a binary outcome"
        )


def assert_bounded_double_robustness_fit(
    cell: PropertyCell,
    result: Any,
    *,
    g_bounds: tuple[float, float],
) -> None:
    """The shared fitted double-robustness controls, plus the bounded scale check.

    Parameters
    ----------
    cell : PropertyCell
        The treatment-correct cell the preflight fitted.
    result : Any
        Its fitted result.
    g_bounds : tuple of float
        The treatment bounds the study configures.
    """
    assert_unit_outcome_scaler(result)
    canonical_properties.assert_double_robustness_fit(
        cell, result, g_bounds=g_bounds, design=DESIGN
    )


# ------------------------------------------------- quadrature design measurements


@lru_cache(maxsize=16)
def mean_range(dgp: DGP) -> tuple[float, float, float]:
    """A law's conditional-mean range on the quadrature grid, and its smallest beta shape.

    Parameters
    ----------
    dgp : DGP
        A beta law.

    Returns
    -------
    tuple of float
        The smallest mean, the largest mean, and ``phi min(m, 1 - m)`` over the grid.
    """
    latent = dgp.quadrature()
    means = np.concatenate(
        [np.asarray(dgp.outcome_mean(latent, a, None), dtype=float) for a in (0.0, 1.0)]
    )
    low, high = float(means.min()), float(means.max())
    return low, high, float(dgp.concentration or 0.0) * min(low, 1.0 - high)


@lru_cache(maxsize=1)
def quasibinomial_recovery(points: int = 2**16) -> float:
    """The largest coefficient error a quasibinomial fit of the bounded linear law leaves.

    The law's mean is exactly a logistic-linear function of ``[A, W]``, so the
    quasibinomial score vanishes at the declared coefficients and an exact solver returns
    them.  Measuring it is what turns "the learner is correctly specified" from a comment
    into a check: a law that acquired a nonlinear term would leave a residual here while
    every cell it feeds went on claiming correctly specified nuisances.

    Parameters
    ----------
    points : int
        Quadrature points, a power of two.

    Returns
    -------
    float
        ``max |recovered - declared|`` over the intercept, the treatment coefficient and
        the four covariate coefficients.
    """
    law = linear_dgp()
    latent = law.quadrature(points)
    design = np.vstack(
        [
            np.column_stack([np.zeros(points), latent]),
            np.column_stack([np.ones(points), latent]),
        ]
    )
    target = np.concatenate(
        [law.outcome_mean(latent, 0.0, None), law.outcome_mean(latent, 1.0, None)]
    )
    model = QuasiBinomialGLM(tol=1e-12).fit(design, target)
    recovered = np.concatenate([model.intercept_, model.coef_[0]])
    declared = np.concatenate(
        [[LINEAR_INTERCEPT, LINEAR_EFFECT], LINEAR_SCALE * np.asarray(LINEAR_WEIGHTS)]
    )
    return float(np.max(np.abs(recovered - declared)))


def _population_logistic(covariates: np.ndarray, target: np.ndarray) -> np.ndarray:
    """The coefficients an unpenalized logistic regression converges to under the law."""
    design = np.column_stack([np.ones(len(covariates)), covariates])
    beta = np.zeros(design.shape[1])
    for _ in range(200):
        fitted = expit(design @ beta)
        variance = np.clip(fitted * (1.0 - fitted), 1e-12, None)
        working = design @ beta + (target - fitted) / variance
        root = np.sqrt(variance)
        updated = np.linalg.lstsq(design * root[:, None], working * root, rcond=None)[0]
        if np.max(np.abs(updated - beta)) < 1e-12:
            return np.asarray(updated, dtype=float)
        beta = updated
    raise RuntimeError("the population logistic limit did not converge")


@lru_cache(maxsize=4)
def both_wrong_limit_bias(g_bounds: tuple[float, float]) -> float:
    """The population bias of the both-wrong arm on the bounded double-robustness law.

    Both misspecified learners have a population limit: a weighted main-effects least
    squares for the outcome regression, taken over both arms at their treatment
    probabilities, and an unpenalized logistic fit for the mechanism.  Substituting the two
    into the one-step estimating equation gives the classical double-robustness remainder,
    which is the control's bias with no replication budget spent on it.

    The pilot's sampled both-wrong bias at ``n = 700`` was ``-0.0358``, against ``-0.0344``
    here, so the two routes agree to four per cent and this is the same control.

    Parameters
    ----------
    g_bounds : tuple of float
        The treatment bounds the study clips the mechanism with, applied to the limit as
        the estimator applies them to the fit.

    Returns
    -------
    float
        The signed bias of the one-step limit against the law's ATE.
    """
    law = double_robustness_dgp()
    latent = law.quadrature()
    g = np.asarray(law.propensity(latent), dtype=float)
    q1 = np.asarray(law.outcome_mean(latent, 1.0, None), dtype=float)
    q0 = np.asarray(law.outcome_mean(latent, 0.0, None), dtype=float)

    stacked = np.vstack(
        [
            np.column_stack([np.zeros(len(latent)), latent]),
            np.column_stack([np.ones(len(latent)), latent]),
        ]
    )
    design = np.column_stack([np.ones(len(stacked)), stacked])
    root = np.sqrt(np.concatenate([1.0 - g, g]))
    coefficient = np.linalg.lstsq(
        design * root[:, None], np.concatenate([q0, q1]) * root, rcond=None
    )[0]
    fitted = design @ coefficient
    # The estimator clips its outcome predictions onto the unit interval before it
    # fluctuates them, so the limit is clipped too.
    q0_star = np.clip(fitted[: len(latent)], 0.0, 1.0)
    q1_star = np.clip(fitted[len(latent) :], 0.0, 1.0)

    g_star = np.clip(
        expit(np.column_stack([np.ones(len(latent)), latent]) @ _population_logistic(latent, g)),
        *g_bounds,
    )
    return float(
        np.mean((1.0 - (1.0 - g) / (1.0 - g_star)) * (q0 - q0_star))
        - np.mean((1.0 - g / g_star) * (q1 - q1_star))
    )


# -------------------------------------------------------------- fold-policy seam


class FoldPolicyTMLE(TMLE):
    """Study-only TMLE that draws its outer folds under a named policy.

    The seam is :class:`~tests.studies.fold_targeted_cvtmle.FixedFoldTMLE`'s: override
    :meth:`~cleverly.estimators.TMLE._folds` and build the partition directly, rather than
    configure the shipped estimator.  Two of the three policies this diagnostic reports are
    no longer reachable through ``stratify_folds=``, because a split that reads the
    treatment or the outcome is what the package refuses.  A diagnostic restricted to the
    policies
    the package still permits could report nothing about the ones it refuses, which is the
    one thing it exists to report.

    The strata are built here rather than through
    :meth:`~cleverly.estimators.TMLE._fold_strata` for the same reason, and the two are
    deliberately not shared: this class has to keep drawing a refused split after the
    estimator stops offering one.
    """

    def __init__(self, policy: str, **kwargs: Any) -> None:
        if policy not in FOLD_POLICIES:
            raise ValueError(f"policy must be one of {FOLD_POLICIES}; got {policy!r}")
        self._policy = policy
        super().__init__(**kwargs)

    def _strata(self, data: CausalData) -> np.ndarray | None:
        if self._policy == "unstratified":
            return None
        if self._policy == "treatment_stratified":
            return np.asarray(data.treatment, dtype=float)
        outcome = np.where(data.observed, data.outcome, -1.0)
        codes = np.unique(np.column_stack([data.treatment, outcome]), axis=0, return_inverse=True)[
            1
        ]
        return np.asarray(codes, dtype=float)

    def _folds(self, data: CausalData, seed: int | None = None) -> Folds:
        return make_folds(
            data.n,
            FOLD_POLICY_FOLDS,
            stratify=self._strata(data),
            cluster=data.cluster,
            random_state=self.random_state if seed is None else seed,
        )


def fold_policy_cells() -> tuple[PropertyCell, ...]:
    """The three fold-policy cells, on one binary law and one set of draws.

    Returns
    -------
    tuple of PropertyCell
        One cell per policy in :data:`FOLD_POLICIES`, all with role ``"diagnostic"`` and
        all on :data:`FOLD_POLICY_SEED`.  The shared seed is what makes the reported
        coverage difference a *paired* one: the three cells see identical samples and
        differ only in how the outer split was drawn.
    """
    law = fold_policy_dgp()
    return tuple(
        PropertyCell(
            property=FOLD_POLICY_FAMILY,
            cell=policy,
            dgp=law,
            outcome_learner=lambda: LogisticRegression(C=1e6, max_iter=2000, solver="lbfgs"),
            treatment_learner=lambda: LogisticRegression(C=1e6, max_iter=2000, solver="lbfgs"),
            n=FOLD_POLICY_N,
            replicates=FOLD_POLICY_REPLICATES,
            seed=FOLD_POLICY_SEED,
            role=DIAGNOSTIC_ROLE,
        )
        for policy in FOLD_POLICIES
    )


def fold_policy_estimator(
    *, g_bounds: tuple[float, float]
) -> Callable[[PropertyCell], Callable[[], Any]]:
    """The estimator factory the fold-policy cells are fitted with.

    Parameters
    ----------
    g_bounds : tuple of float
        The treatment bounds the study configures.

    Returns
    -------
    Callable
        A factory taking a cell and returning a no-argument estimator builder.  No
        ``q_bounds``: the law is binary, and
        :meth:`~cleverly.estimators.TMLE._scaler` refuses ``q_bounds`` there.
    """

    def factory(cell: PropertyCell) -> Callable[[], Any]:
        return lambda: FoldPolicyTMLE(
            cell.cell,
            outcome_learner=cell.outcome_learner(),
            treatment_learner=cell.treatment_learner(),
            cross_fit=True,
            n_folds=FOLD_POLICY_FOLDS,
            estimands=cell.estimand,
            simultaneous=False,
            g_bounds=g_bounds,
            max_iter=100,
            tol=1e-10,
            random_state=0,
        )

    return factory
