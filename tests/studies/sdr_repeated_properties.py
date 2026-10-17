r"""Independent repeated-sampling properties of repeated-outcome ``SDR`` (X32b, PR B).

Every binary family samples :func:`cleverly.datasets.make_longitudinal_repeated` on its own
stream ``stream_seed(STUDY, "property_sample", family, configuration, replicate)``.  The
continuous arm samples :data:`~tests.studies.ltmle_repeated_law.CONTINUOUS`.  Every truth and
every control limit comes from :mod:`tests.studies.ltmle_repeated_law`.  The labels are the X32a
labels: ``never_t3`` is the mean of ``Y_3`` under never treat, and ``rule_vs_never_t3`` is the
rule's contrast with ``never``.  The five ``*_t3`` labels are the gated labels.

The ``SDR`` limit
-----------------

With the mechanisms correct, ``SDR`` converges to the truth whatever the regression is (Díaz
et al. 2023, Lemma 4).  With the regression correct, it converges to the truth whatever the
mechanism is.  With both wrong, it converges to
:meth:`~tests.studies.ltmle_repeated_law.RepeatedOutcomeLaw.sdr_limit`.  An in-sample fit with
a saturated regression has a vanishing correction on seen cells, so the in-sample cells test
the plug-in and the mechanism terms.  The ``double_robustness`` and ``correction_necessity``
families use a regression on ``W`` alone, whose correction is nonzero.

The reversed-rule gaps
----------------------

The declared rule's efficient standard deviation is 0.747 at ``t = 2`` and 1.240 at ``t = 3``.
The reversed rule's is 0.708 and 1.130.  The ``reversed_rule`` control fits the reversed rule, so
its bias reads in its own spread: the gaps at ``n = 2,000`` are -3.58 and -2.35 EIF-SE, the
values of plan section 5.3 (``allocation.py`` divides by ``eif_sd("reversed", h)``).  In the
declared rule's spread, which is the unit of the paired displacement, the same gaps are -3.39 and
-2.14.  The plan is right for the discrimination check, and the first draft of the fast test used
the wrong unit.  :data:`REVERSED_EFFICIENCY_SD` carries the correct one.

Declared extension to plan section 5.7
--------------------------------------

The C6 ``n`` rule reads the share ``1 - d/5`` (:data:`C6_SHARE`), one of the five ``SDR``
calibration terms: C1, C3, C5, C6 and the C4 joint band.  Plan section 5.7 states the ``IPSI``
share ``1 - d/4``.  This is a declared extension for ``SDR``, and the plan is to be amended.

Declared before any run
-----------------------

==============================  =========================================================
family and cells                what is fitted, replications and size
==============================  =========================================================
C1 ``interval_calibration``     ``*__in_sample`` for all thirteen labels: one fold,
                                :class:`~tests.studies.ltmle_repeated_properties.
                                CellMeansBackoff` for the outcome and pseudo-outcome
                                regressions, GLM mechanisms, ``simultaneous=True``.
                                ``n =`` :data:`IN_SAMPLE_N`, :data:`CALIBRATION_REPLICATES`.  The
                                derived ``shrunken_se_control`` and ``noise_control`` arms
                                read these rows (twenty-six controls)
C3 ``interval_calibration``     ``*_t3__crossfit``: C1 at five folds, five labels,
                                ``n =`` :data:`CALIBRATION_N`
C6 ``interval_calibration``     ``*_t3__continuous``: the three-value law
                                :data:`~tests.studies.ltmle_repeated_law.CONTINUOUS`,
                                ``q_bounds = (0, 1)``, in sample, five labels, ``n =``
                                :data:`CONTINUOUS_N`
C5 ``interval_calibration``     ``*_t3__crossfit_boosted``: five folds, ``n =``
                                :data:`CALIBRATION_N`, a shallow
                                regularized booster for both regressions.  Gated one-sided
                                (plan 5.6): the SE ratio's 99% lower bound is at least the
                                lower band end, and so is the coverage interval's lower
                                end.  The two-sided result is the ``two_sided_passed``
                                column
C4 ``simultaneous_coverage``    ``all_reported``: the default band of the C1 fits over all
                                fifteen reported parameters, and its pointwise control.
                                It reads the C1 rows, so ``n =`` :data:`IN_SAMPLE_N`
``small_sample_calibration``    C1 and C6 refitted at ``n = 2,000``, five labels.  Role
                                ``diagnostic``.  2,000 replications
C2 ``double_robustness``        Five ``*_t3`` labels, ``n = 2,000``.  ``mechanism_correct``:
                                a ``W``-only regression, GLM mechanisms.
                                ``outcome_correct``: :class:`CellMeansBackoff` with
                                ``W``-only mechanisms.  ``both_wrong``: the ``W``-only
                                regression with intercept mechanisms, a control whose limit
                                is :data:`BOTH_WRONG_LIMIT`
``correction_necessity``        Five ``*_t3`` labels, ``n = 2,000``.  Positive: the
                                ``mechanism_correct`` fit.  Control: the same sequential
                                regression with no correction, written without cleverly
                                (:func:`uncorrected`).  Its limit is :data:`PLUG_IN_LIMIT`
C7 ``rule_necessity``           ``rule_t{2,3}``, ``n = 2,000``: the declared rule against
                                the reversed rule
N1 ``repeated_outcome_necessity``  ``{never,always,rule}_t{2,3}__repeated`` against
                                ``__absorbing_reading``, an end-of-study ``SDR`` fit of
                                :math:`\max_{s \le h} Y_s`.  ``n`` is :data:`ABSORBING_N`.
                                The ``n`` rule applies
N2 ``outcome_history_necessity``  ``{never,always,always_vs_never}_t3__repeated`` against
                                ``__history_free``, an end-of-study fit with every
                                ``Y_s`` left out of every history.  ``n`` is
                                :data:`HISTORY_FREE_N`
N3 ``full_history_necessity``   Five ``*_t3`` labels, ``n =`` :data:`FULL_HISTORY_N`.
                                Intercept mechanisms.  Positive: full-history
                                :class:`CellMeansBackoff`.  Control: the Markov history
                                ``[W, Y_{t-1}, L_t]``, whose limit is :data:`MARKOV_LIMIT`
==============================  =========================================================

There is no mixed sequential-DR family.  ``SDR`` and ``LTMLE`` differ by at most 0.34 EIF-SE at
``n = 2,000`` on this law in such patterns, so the family would measure nothing.  The exact
witness E2b records Lemma 4 at a mixed pattern.

Gating, declared before any run:

* the SE-ratio band and the coverage band gate every C1 cell at every horizon.  The C3 and C6
  arms gate the ``t = 3`` labels.  C5 gates one-sided;
* every other positive cell answers to its family's bias rule, and each control to the opposite
  direction of the same instrument.  A control also answers to its own limit;
* a red row is diagnosed.  A defect is fixed, and a finite-sample shortfall gets the ``n`` that a
  pilot shows removes it.  Either change is re-declared and re-run on fresh seeds.

Sizes
-----

Every size is a rule of :mod:`tests.studies.repeated_sizing` applied to a pilot table of this
module.  The pilots draw from ``stream_seed(STUDY, "pilot", ...)``, which no published cell
reads.

=========================  ===========  ====================================================
pilot                      draws        sets
=========================  ===========  ====================================================
P1 primary, two scenarios  2,000 each   the primary ``n`` rule and ``PRIMARY_REPLICATES``
P2 C1, C3, C5, C6 (and C4) 8,000 each   ``CALIBRATION_REPLICATES`` or the cap, the C6 rule.
                                         C1 and C6 re-run at 32,000 (see Pilot P2 finding)
P3 every property family   2,000 each   each family's size, and the N1 ``n`` rule
=========================  ===========  ====================================================

The union bound gives the discrimination checks 0.001 and each of the three sized instruments
:math:`d = 0.049 / 3`.  The calibration instrument has five terms: C1, C3, C5, C6 and C4.  The
property instrument has six families.  The study fails with probability at most 0.05.

Pilot P2 finding
----------------

Pilot P2 at ``n = 8,000`` gave in-sample SE ratios of 0.920 (C1 ``rule_t3``), 0.949 (C1
``rule_vs_never_t3``), 0.897 and 0.911 (C6 ``rule_t3`` and ``rule_vs_never_t3``).  The
cross-fit (C3) and booster (C5) arms sit at 0.98 to 1.00.  The cause is a finite-sample
shortfall of the in-sample variance estimate under saturated full-history
:class:`CellMeansBackoff` outcome regressions.  The in-sample residuals of a cell with ``k``
units lose variance by the factor ``(k - 1) / k``, and the rare high-weight cells of the rule at
node 3 dominate the estimate.  The deficit decays as ``1 / n``.  The in-sample standard error
times :math:`\sqrt{n}`, over the standard deviation of the efficient influence function, for
``rule_t3`` is 0.857, 0.907, 0.944, 0.966 and 0.984 at ``n`` = 2,000, 4,000, 8,000, 16,000 and
32,000.

This is not an estimator defect.  A read-only probe gave the same standard error as ``LTMLE``
with the same learners, to 1e-6.  A GLM outcome regression gives ratios of 0.98 to 1.00 at every
``n``.  The declared route for a finite-sample shortfall applies: the cell gets the ``n`` that a
pilot shows removes it, re-declared and re-run on fresh seeds.  A design change would move the
study off the saturated regression that makes the in-sample cells test the plug-in and the
mechanism terms.  So :data:`IN_SAMPLE_N` and :data:`CONTINUOUS_N` double twice, from 8,000 to
32,000.  :data:`CALIBRATION_N` stays at 8,000, because the C3 and C5 arms show no shortfall.
The C4 joint band reads the C1 rows, so it moves with :data:`IN_SAMPLE_N`.

The re-pilot at 32,000 ran.  The in-sample ``rule_t3`` ratio is 0.9707 and ``rule_vs_never_t3``
is 0.9796.  The continuous ``rule_t3`` ratio is 0.9691 and ``rule_vs_never_t3`` is 0.9689.  All
other in-sample and continuous ratios lie between 0.966 and 1.018.  The C4 joint coverage is
0.9486.  The C6 ``n`` rule does not fire, so both sizes stay at 32,000.  The calibration rule
gives :data:`CALIBRATION_REPLICATES` of 38,000 on the pinned values, under the cap of 50,000.

Pilot P3 shows ratios below 1 in two families that gate bias only.  The ``double_robustness``
cell ``rule_t3__outcome_correct`` has ratio 0.763 and ``full_history_necessity``
``rule_t3__full_history`` has 0.772.  Both fits use a wrong mechanism (``W`` only, or intercept
only).  Under a wrong mechanism the Wald standard error is inconsistent, and the study reports
it with no calibration claim.  The robustness and
necessity rules gate bias, never coverage or the SE ratio.  The ratios are therefore expected,
and they are not a shortfall.

Pilot state
-----------

Pilots P1, P2 and P3 are recorded, and the state is ``"recorded"``.  The pilot driver
(``tests/canonical/sdr_repeated/pilot.py``) writes the pilot rows in resumable parts.
:func:`summarize_primary_pilot`, :func:`summarize_calibration_pilot` and
:func:`summarize_property_pilot` turn them into these tables.
"""

from __future__ import annotations

import dataclasses
import functools
import time
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.dummy import DummyClassifier

from cleverly.datasets import make_longitudinal_repeated
from cleverly.longitudinal import SDR, DynamicRegimen
from tests.discrete_law_longitudinal import CellMeans
from tests.parallel import STUDY_JOBS
from tests.studies import ltmle_repeated_law as law
from tests.studies import repeated_sizing as sizing
from tests.studies.canonical_ltmle import G_BOUNDS
from tests.studies.canonical_ltmle_repeated import (
    COLUMNS,
    GATED_LABELS,
    RULE_LABEL,
    label_estimand,
    treat_after,
)
from tests.studies.canonical_sdr_repeated import (
    ABSORBING_ARMS,
    BOOSTER_ARM,
    CORRECTION_ARMS,
    DOUBLE_ROBUSTNESS_ARMS,
    FULL_HISTORY_ARMS,
    HISTORY_FREE_ARMS,
    JOINT_LABEL,
    LABELS,
    PRIMARY_N,
    RULE_ARMS,
    SCENARIO_FOLDS,
    SCENARIOS,
    SMALL_SAMPLE_ARMS,
    STUDY,
    fit_cleverly,
    require_recorded_pilots,
)
from tests.studies.evidence import property_failures as pf
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.properties import (
    REPLICATE_COLUMNS,
    PropertyCell,
    control_row,
    payload_cell,
    payload_replicate,
    replicate_row,
    se_ratio_interval,
    summary_interval,
)
from tests.studies.evidence.property_verdicts import (
    DIAGNOSTIC_ROLE,
    Necessity,
    alternative_target_necessity_verdicts,
    alternative_target_verdicts,
    apply_shared_verdicts,
    calibration_controls,
    calibration_derived,
    calibration_verdicts,
    finish,
    simultaneous_coverage_verdicts,
)
from tests.studies.evidence.replication import fit_properties, splat_draw, splat_drawn
from tests.studies.evidence.seeds import stream_seed
from tests.studies.evidence.simultaneous import FAMILY as JOINT_FAMILY
from tests.studies.evidence.simultaneous import (
    joint_cells,
    joint_coverage_rows,
    joint_property_cells,
)
from tests.studies.fractional_glm import QuasiBinomialGLM
from tests.studies.ltmle_repeated_properties import (
    ABSORBING_LABELS,
    ABSORBING_LIMIT,
    CONTINUOUS_TRUTH,
    HISTORY_FREE_LIMIT,
    HISTORY_LABELS,
    REVERSED,
    REVERSED_TRUTH,
    RULE_LABELS,
    TRUTH,
    BaselineOnly,
    CellMeansBackoff,
    CellMeansOn,
    booster,
    label_parts,
    plug_in,
    regimens,
)

MARGINS = STUDY.margins
N = 2_000
#: The sample size of the cross-fit and booster ``interval_calibration`` arms.
CALIBRATION_N = 8_000
#: The sample size of the C1 in-sample arm, which the C4 joint band and the two derived
#: calibration controls also read.  It is declared apart from :data:`CALIBRATION_N`, because
#: pilot P2 showed a finite-sample shortfall of the in-sample variance estimate (module
#: docstring, "Pilot P2 finding").  The declared route is a doubling of this size alone.
IN_SAMPLE_N_BASE = 8_000
#: How many times the in-sample ``n`` rule has doubled :data:`IN_SAMPLE_N` (0, 1 or 2).
IN_SAMPLE_N_DOUBLINGS = 2
IN_SAMPLE_N = IN_SAMPLE_N_BASE * 2**IN_SAMPLE_N_DOUBLINGS
#: The sample size of the C6 arm (``continuous``).  It is declared apart from
#: :data:`CALIBRATION_N`, so the C6 ``n`` rule doubles this size alone.
CONTINUOUS_N_BASE = 8_000
#: How many times the C6 ``n`` rule has doubled :data:`CONTINUOUS_N` (0, 1 or 2).
CONTINUOUS_N_DOUBLINGS = 2
CONTINUOUS_N = CONTINUOUS_N_BASE * 2**CONTINUOUS_N_DOUBLINGS
#: The sample size of N1, which the ``n`` rule may double, twice at most.
ABSORBING_N_BASE = 8_000
#: How many times the N1 ``n`` rule has doubled :data:`ABSORBING_N` (0, 1 or 2).
ABSORBING_N_DOUBLINGS = 0
ABSORBING_N = ABSORBING_N_BASE * 2**ABSORBING_N_DOUBLINGS
#: The sample size of N2.
HISTORY_FREE_N = 2_000
#: The sample size of N3.
FULL_HISTORY_N = 8_000
#: The reported arm draws this many replications.
SMALL_SAMPLE_REPLICATES = 2_000
FOLDS = 5
SHRUNKEN_SE_FACTOR = 0.70
#: The least paired displacement of a necessity family, in the positive arm's empirical
#: standard deviations: the standardized-bias margin.
NECESSITY_DISPLACEMENT = 0.25
#: The least move the correction must make, in the corrected arm's empirical standard
#: deviations, before ``correction_necessity`` calls the correction load bearing.
CORRECTION_DISPLACEMENT = 0.25
CRITICAL = float(norm.ppf(1.0 - MARGINS.alpha / 2.0))
#: The families that the property instrument sizes: its union-bound terms.
PROPERTY_FAMILIES = (
    "double_robustness",
    "correction_necessity",
    "rule_necessity",
    "repeated_outcome_necessity",
    "outcome_history_necessity",
    "full_history_necessity",
)
#: The share of the C6 ``n`` rule: one of the five ``SDR`` calibration terms.
C6_SHARE = 1.0 - sizing.SIZED_FAIL / sizing.SDR_CALIBRATION_TERMS
#: The configurations whose outcome regression reads ``W`` alone.
BASELINE_Q = frozenset({"mechanism_correct", "both_wrong"})
#: The configurations whose mechanisms are intercept only.
INTERCEPT_G = frozenset({"both_wrong", "full_history", "markov_history"})

# ---------------------------------------------------------------------------- declared sizes
#
# Each size is the rule applied to the recorded pilot tables of the section "pilot tables"
# below.  A fast test re-derives every one of them.

#: ``calibration_size`` on P2 (``CALIBRATION_BUDGET_MET`` is False when the rule reached the cap).
CALIBRATION_REPLICATES = 38_000
CALIBRATION_BUDGET_MET = True
#: ``property_replicates`` on P3 at the share ``family_share(6)``.
DOUBLE_ROBUST_REPLICATES = 1_000
CORRECTION_REPLICATES = 1_000
RULE_REPLICATES = 1_000
ABSORBING_REPLICATES = 1_000
HISTORY_FREE_REPLICATES = 1_000
FULL_HISTORY_REPLICATES = 1_000
BOOSTER_REPLICATES = CALIBRATION_REPLICATES

# ---------------------------------------------------------------------------- label limits


def _sdr_value(
    name: str, q: str, g: str, *, plug: bool = False, model: law.RepeatedOutcomeLaw = law.BINARY
) -> float:
    """A label's ``SDR`` limit on ``model``: a mean, or the contrast of two means.

    ``q`` is the history of every regression and ``g`` the mechanism, both at every node.  With
    ``plug`` the limit is that of the sequential regression with no correction.
    """
    plan, reference, horizon = label_parts(name)

    def mean(which: str) -> float:
        if plug:
            return float(model.sdr_plug_in_limit(which, horizon, q))
        return float(model.sdr_limit(which, horizon, [q] * horizon, [g] * horizon))

    value = mean(plan)
    if reference is not None:
        value -= mean(reference)
    return value


def efficiency_sd(model: law.RepeatedOutcomeLaw, name: str) -> float:
    """The standard deviation of the efficient influence curve of a label."""
    plan, reference, horizon = label_parts(name)
    return float(model.eif_sd(plan, horizon, reference=reference))


#: The limit of the single-fold fit with a ``W`` regression and intercept mechanisms: the
#: ``both_wrong`` control.
BOTH_WRONG_LIMIT = {name: _sdr_value(name, "baseline", "intercept") for name in GATED_LABELS}
#: The limit of the sequential regression on ``W`` with no correction: the ``uncorrected``
#: control of ``correction_necessity``.
PLUG_IN_LIMIT = {name: _sdr_value(name, "baseline", "true", plug=True) for name in GATED_LABELS}
#: The limit of the fit with a Markov regression and intercept mechanisms: the N3 control.
MARKOV_LIMIT = {name: _sdr_value(name, "markov", "intercept") for name in GATED_LABELS}
#: The efficient influence curve's standard deviation of each label.  It scales the noise
#: control.  The study claims no efficiency, so no efficiency band reads it.
EFFICIENCY_SD = {name: efficiency_sd(law.BINARY, name) for name in LABELS}
#: The efficient influence curve's standard deviation of the reversed rule, by rule label.  The
#: ``reversed_rule`` control is a fit of another parameter, so its bias reads in its own spread.
REVERSED_EFFICIENCY_SD = {
    name: float(law.BINARY.eif_sd(REVERSED, label_parts(name)[2])) for name in RULE_LABELS
}

# ---------------------------------------------------------------------------- learners


def _learners(configuration: str) -> tuple[Any, Any, Any, Any]:
    """``(outcome, pseudo, treatment, censoring)`` learners of a configuration."""
    if configuration == BOOSTER_ARM:
        q, pseudo = booster("classifier"), booster("regressor")
    elif configuration in BASELINE_Q:
        q, pseudo = CellMeansOn("baseline", CellMeans()), CellMeansOn("baseline", CellMeans())
    elif configuration == "markov_history":
        q = CellMeansOn("markov", CellMeansBackoff())
        pseudo = CellMeansOn("markov", CellMeansBackoff())
    else:
        q, pseudo = CellMeansBackoff(), CellMeansBackoff()
    if configuration in INTERCEPT_G:
        g, c = DummyClassifier(strategy="prior"), DummyClassifier(strategy="prior")
    elif configuration == "outcome_correct":
        g, c = BaselineOnly(QuasiBinomialGLM()), BaselineOnly(QuasiBinomialGLM())
    else:
        g, c = QuasiBinomialGLM(), QuasiBinomialGLM()
    return q, pseudo, g, c


def fit(
    frame: pd.DataFrame,
    configuration: str = "in_sample",
    *,
    simultaneous: bool = False,
    reversed_rule: bool = False,
) -> Any:
    """The repeated-outcome fit of one configuration."""
    q, pseudo, g, c = _learners(configuration)
    folds = FOLDS if configuration in {"crossfit", BOOSTER_ARM} else 1
    return SDR(
        regimens(reversed_rule=reversed_rule),
        reference="never",
        outcome_learner=q,
        pseudo_learner=pseudo,
        treatment_learner=g,
        censoring_learner=c,
        n_folds=folds,
        g_bounds=G_BOUNDS,
        q_bounds=(0.0, 1.0) if configuration == "continuous" else None,
        simultaneous=simultaneous,
        random_state=0,
    ).fit(frame, **COLUMNS)


def absorbing_fit(frame: pd.DataFrame, horizon: int) -> Any:
    r"""The N1 control: an end-of-study fit of :math:`\max_{s \le h} Y_s` at ``horizon``.

    The earlier outcomes stay in the history, so the mechanisms read what the repeated fit
    reads.  Only the outcome is the absorbing reading.  The GLM mechanisms are correct, so by
    Lemma 4 the limit is the absorbing g-formula whatever the regression is.
    """
    panel = frame.copy()
    outcomes = panel[[f"Y{s}" for s in range(1, horizon + 1)]]
    panel["M"] = outcomes.max(axis=1, skipna=False)
    q, pseudo, g, c = _learners("in_sample")
    plans = {
        "never": 0,
        "always": 1,
        RULE_LABEL: DynamicRegimen(
            RULE_LABEL,
            (1, *(treat_after(f"Y{s}") for s in range(1, horizon))),
            rule_kind="known",
        ),
    }
    blocks: list[list[str]] = [[], *([f"Y{s - 1}", f"L{s}"] for s in range(2, horizon + 1))]
    return SDR(
        plans,
        reference="never",
        outcome_learner=q,
        pseudo_learner=pseudo,
        treatment_learner=g,
        censoring_learner=c,
        n_folds=1,
        g_bounds=G_BOUNDS,
        simultaneous=False,
        random_state=0,
    ).fit(
        panel,
        outcome="M",
        treatment=[f"A{s}" for s in range(1, horizon + 1)],
        baseline=["W"],
        time_varying=blocks,
        censoring=[f"C{s}" for s in range(1, horizon + 1)],
    )


def history_free_fit(frame: pd.DataFrame, horizon: int = 3) -> Any:
    """The N2 control: an end-of-study fit of ``Y_h`` with no ``Y_s`` in any history.

    Saturated cell-mean regressions on ``[W, L_2, ..., L_t]`` are correct for that coarser
    history, so by Lemma 4 the fit converges to its coarse g-formula whatever the mechanisms.
    """
    _, _, g, c = _learners("in_sample")
    return SDR(
        {"never": 0, "always": 1},
        reference="never",
        outcome_learner=CellMeans(),
        pseudo_learner=CellMeans(),
        treatment_learner=g,
        censoring_learner=c,
        n_folds=1,
        g_bounds=G_BOUNDS,
        simultaneous=False,
        random_state=0,
    ).fit(
        frame,
        outcome=f"Y{horizon}",
        treatment=[f"A{s}" for s in range(1, horizon + 1)],
        baseline=["W"],
        time_varying=[[], *([f"L{s}"] for s in range(2, horizon + 1))],
        censoring=[f"C{s}" for s in range(1, horizon + 1)],
    )


def _end_of_study_name(label: str) -> str:
    """The name an end-of-study fit reports for a label: the horizon leaves the bracket."""
    return label_estimand(label).rsplit(" @ t=", 1)[0] + "]"


# ---------------------------------------------------------------------------- plug-in


def uncorrected(frame: pd.DataFrame, label: str) -> float:
    """The ``correction_necessity`` control: the plug-in of a label, or of its contrast.

    The sequential regression on ``W`` among the followers, carried back with no correction,
    written with ``numpy`` and ``pandas`` alone and not by calling the estimator with the
    correction disabled, so the control is not a statement about a branch of the code it
    audits.
    """
    plan, reference, horizon = label_parts(label)
    value = plug_in(frame, plan, horizon)
    if reference is not None:
        value -= plug_in(frame, reference, horizon)
    return value


# ---------------------------------------------------------------------------- rows


def sample(configuration: str, seed: int, n: int = N) -> pd.DataFrame:
    """One draw of size ``n`` of the law a configuration reads."""
    if configuration == "continuous":
        return law.CONTINUOUS.sample(n, seed)
    frame, _ = make_longitudinal_repeated(n=n, seed=seed, censoring=True, backend="pandas")
    return frame


def family_n(family: str) -> int:
    """The sample size a property family draws (the C1 and C6 arms: see :func:`fit_n`)."""
    if family in {"interval_calibration"}:
        return CALIBRATION_N
    if family == "repeated_outcome_necessity":
        return ABSORBING_N
    if family == "outcome_history_necessity":
        return HISTORY_FREE_N
    if family == "full_history_necessity":
        return FULL_HISTORY_N
    return N


def fit_n(family: str, configuration: str) -> int:
    """The sample size of one fit set: the C1 in-sample and C6 arms have their own sizes."""
    if family == "interval_calibration" and configuration == "continuous":
        return CONTINUOUS_N
    if family == "interval_calibration" and configuration == "in_sample":
        return IN_SAMPLE_N
    return family_n(family)


def _row(
    family: str,
    cell: str,
    role: str,
    replicate: int,
    requested: int,
    truth: float,
    estimate: Any,
    n: int,
) -> dict[str, Any]:
    return replicate_row(
        property_name=family,
        cell=cell,
        role=role,
        replicate=replicate,
        n=n,
        requested=requested,
        truth=truth,
        estimate=estimate,
        alpha=MARGINS.alpha,
    )


def _paired(
    family: str,
    labels: Sequence[str],
    arms: tuple[str, str],
    replicate: int,
    requested: int,
    n: int,
    positive: Any,
    control: Any,
) -> list[dict[str, Any]]:
    """The positive and control rows of a paired family, both read at the true value."""
    rows: list[dict[str, Any]] = []
    for label in labels:
        name = label_estimand(label)
        truth = TRUTH[name]
        rows.append(
            _row(
                family,
                f"{label}__{arms[0]}",
                "positive",
                replicate,
                requested,
                truth,
                positive[name],
                n,
            )
        )
        rows.append(
            _row(
                family,
                f"{label}__{arms[1]}",
                "control",
                replicate,
                requested,
                truth,
                control[name],
                n,
            )
        )
    return rows


_Payload = tuple[str, str, int, int, int, int]


def _draw(payload: _Payload) -> pd.DataFrame:
    """One replication's sample, drawn outside the fit."""
    _, configuration, _, _, seed, n = payload
    return sample(configuration, seed, n)


def _replication(payload: _Payload) -> list[dict[str, Any]]:
    """One replication's rows, drawn and fitted in one call: the pilots' and the timer's form."""
    return _fit_drawn(payload, _draw(payload))


def _fit_drawn(payload: _Payload, frame: pd.DataFrame) -> list[dict[str, Any]]:
    """One replication's rows from its drawn sample."""
    family, configuration, replicate, requested, _, n = payload
    rows: list[dict[str, Any]] = []
    if family in {"interval_calibration", "small_sample_calibration"}:
        calibration = family == "interval_calibration"
        joint = calibration and configuration == "in_sample"
        result = fit(frame, configuration, simultaneous=joint)
        continuous = configuration == "continuous"
        labels = LABELS if joint else GATED_LABELS
        role = "positive" if calibration else DIAGNOSTIC_ROLE
        for name in labels:
            truth = CONTINUOUS_TRUTH[name] if continuous else TRUTH[label_estimand(name)]
            rows.append(
                _row(
                    family,
                    f"{name}__{configuration}",
                    role,
                    replicate,
                    requested,
                    truth,
                    result[label_estimand(name)],
                    n,
                )
            )
        if joint:
            rows.extend(
                joint_coverage_rows(
                    result,
                    {name: TRUTH[name] for name in result.estimates},
                    tuple(result.estimates),
                    label=JOINT_LABEL,
                    replicate=replicate,
                    n=n,
                    requested=requested,
                    pointwise_critical=CRITICAL,
                )
            )
        return rows
    if family == "double_robustness":
        result = fit(frame, configuration)
        role = "control" if configuration == "both_wrong" else "positive"
        for name in GATED_LABELS:
            estimand = label_estimand(name)
            rows.append(
                _row(
                    family,
                    f"{name}__{configuration}",
                    role,
                    replicate,
                    requested,
                    TRUTH[estimand],
                    result[estimand],
                    n,
                )
            )
        return rows
    if family == "correction_necessity":
        result = fit(frame, "mechanism_correct")
        corrected, control_arm = CORRECTION_ARMS
        for name in GATED_LABELS:
            estimand = label_estimand(name)
            rows.append(
                _row(
                    family,
                    f"{name}__{corrected}",
                    "positive",
                    replicate,
                    requested,
                    TRUTH[estimand],
                    result[estimand],
                    n,
                )
            )
            # The plug-in has no influence curve of its own, so the row carries the corrected
            # fit's standard error and the family gates on bias alone.
            rows.append(
                control_row(
                    property_name=family,
                    cell=f"{name}__{control_arm}",
                    replicate=replicate,
                    n=n,
                    requested=requested,
                    truth=TRUTH[estimand],
                    estimate=uncorrected(frame, name),
                    standard_error=float(result[estimand].std_error),
                    critical=CRITICAL,
                )
            )
        return rows
    if family == "rule_necessity":
        result = fit(frame, reversed_rule=True)
        for name in RULE_LABELS:
            estimand = label_estimand(name)
            reversed_name = estimand.replace(RULE_LABEL, REVERSED)
            for arm, source, role in (
                (RULE_ARMS[0], result[estimand], "positive"),
                (RULE_ARMS[1], result[reversed_name], "control"),
            ):
                rows.append(
                    _row(
                        family,
                        f"{name}__{arm}",
                        role,
                        replicate,
                        requested,
                        TRUTH[estimand],
                        source,
                        n,
                    )
                )
        return rows
    if family == "repeated_outcome_necessity":
        result = fit(frame)
        controls = {horizon: absorbing_fit(frame, horizon) for horizon in (2, 3)}
        for name in ABSORBING_LABELS:
            estimand = label_estimand(name)
            horizon = label_parts(name)[2]
            for arm, source, role in (
                (ABSORBING_ARMS[0], result[estimand], "positive"),
                (ABSORBING_ARMS[1], controls[horizon][_end_of_study_name(name)], "control"),
            ):
                rows.append(
                    _row(
                        family,
                        f"{name}__{arm}",
                        role,
                        replicate,
                        requested,
                        TRUTH[estimand],
                        source,
                        n,
                    )
                )
        return rows
    if family == "outcome_history_necessity":
        result = fit(frame)
        control = history_free_fit(frame)
        for name in HISTORY_LABELS:
            estimand = label_estimand(name)
            for arm, source, role in (
                (HISTORY_FREE_ARMS[0], result[estimand], "positive"),
                (HISTORY_FREE_ARMS[1], control[_end_of_study_name(name)], "control"),
            ):
                rows.append(
                    _row(
                        family,
                        f"{name}__{arm}",
                        role,
                        replicate,
                        requested,
                        TRUTH[estimand],
                        source,
                        n,
                    )
                )
        return rows
    if family == "full_history_necessity":
        return _paired(
            family,
            GATED_LABELS,
            FULL_HISTORY_ARMS,
            replicate,
            requested,
            n,
            fit(frame, "full_history"),
            fit(frame, "markov_history"),
        )
    raise KeyError(family)


#: Every fit set: ``(family, configuration, replications)``.
FIT_SETS: tuple[tuple[str, str, int], ...] = (
    ("interval_calibration", "in_sample", CALIBRATION_REPLICATES),
    ("interval_calibration", "crossfit", CALIBRATION_REPLICATES),
    ("interval_calibration", "continuous", CALIBRATION_REPLICATES),
    ("interval_calibration", BOOSTER_ARM, BOOSTER_REPLICATES),
    *(("small_sample_calibration", arm, SMALL_SAMPLE_REPLICATES) for arm in SMALL_SAMPLE_ARMS),
    *(("double_robustness", arm, DOUBLE_ROBUST_REPLICATES) for arm in DOUBLE_ROBUSTNESS_ARMS),
    ("correction_necessity", "paired", CORRECTION_REPLICATES),
    ("rule_necessity", "paired", RULE_REPLICATES),
    ("repeated_outcome_necessity", "paired", ABSORBING_REPLICATES),
    ("outcome_history_necessity", "paired", HISTORY_FREE_REPLICATES),
    ("full_history_necessity", "paired", FULL_HISTORY_REPLICATES),
)


def _payloads(
    fit_sets: tuple[tuple[str, str, int], ...] = FIT_SETS,
) -> list[tuple[_Payload]]:
    payloads = []
    for family, configuration, replicates in fit_sets:
        for replicate in range(replicates):
            seed = stream_seed(STUDY, "property_sample", family, configuration, replicate)
            payloads.append(
                (
                    (
                        family,
                        configuration,
                        replicate,
                        replicates,
                        seed,
                        fit_n(family, configuration),
                    ),
                )
            )
    return payloads


def assemble_property_rows(outcomes: Sequence[Sequence[Mapping[str, Any]]]) -> pd.DataFrame:
    """The property rows of every fit set, with the derived calibration controls.

    ``outcomes`` is the list of :func:`_replication` results, in the order of :func:`_payloads`.
    A pure function of the rows, so the pilot and the smoke run assemble the same way.
    """
    rows = pd.DataFrame([row for result in outcomes for row in result])
    rows = pd.concat(
        [
            rows,
            calibration_controls(
                rows,
                STUDY,
                labels=LABELS,
                efficiency_bounds=EFFICIENCY_SD,
                calibration_n=IN_SAMPLE_N,
                shrunken_se_factor=SHRUNKEN_SE_FACTOR,
                critical=CRITICAL,
                positive_suffix="in_sample",
            ),
        ],
        ignore_index=True,
    )
    return rows.loc[:, list(REPLICATE_COLUMNS)].sort_values(
        ["property", "cell", "replicate"], ignore_index=True
    )


#: The calibration positive arm to the controls derived from it.
DERIVED_CELLS = calibration_derived(LABELS, positive_suffix="in_sample")

#: Each paired family's labels and arms, for :func:`_cells_of`.
_PAIRED_CELLS: dict[str, tuple[tuple[str, ...], tuple[str, str]]] = {
    "correction_necessity": (GATED_LABELS, CORRECTION_ARMS),
    "rule_necessity": (RULE_LABELS, RULE_ARMS),
    "repeated_outcome_necessity": (ABSORBING_LABELS, ABSORBING_ARMS),
    "outcome_history_necessity": (HISTORY_LABELS, HISTORY_FREE_ARMS),
    "full_history_necessity": (GATED_LABELS, FULL_HISTORY_ARMS),
}


def _cells_of(key: tuple[_Payload]) -> list[str]:
    """The ``property/cell`` names the key's rows and derived controls fill."""
    family, configuration = key[0][:2]
    if family in _PAIRED_CELLS:
        labels, arms = _PAIRED_CELLS[family]
        return [f"{family}/{label}__{arm}" for label in labels for arm in arms]
    joint = family == "interval_calibration" and configuration == "in_sample"
    names = [f"{family}/{label}__{configuration}" for label in (LABELS if joint else GATED_LABELS)]
    if joint:
        names += [f"{JOINT_FAMILY}/{name}" for name in joint_cells(JOINT_LABEL)]
    for name in list(names):
        names += DERIVED_CELLS.get(name, ())
    return names


@dataclass(frozen=True)
class DeclaredLaw:
    """The law one fit set draws, by its name, with the exact truth of each estimand it reports.

    Parameters
    ----------
    name : str
        The fit set, as ``family__configuration``.
    truths : Mapping[str, float]
        Each reported estimand's exact truth.
    """

    name: str
    truths: Mapping[str, float]

    def truth(self) -> dict[str, float]:
        return dict(self.truths)


def _role(family: str, cell: str, configuration: str) -> str:
    """The role of one sampled cell."""
    if family in _PAIRED_CELLS:
        return ("positive", "control")[_PAIRED_CELLS[family][1].index(cell.split("__", 1)[1])]
    if family == "double_robustness":
        return "control" if configuration == "both_wrong" else "positive"
    return "positive" if family == "interval_calibration" else DIAGNOSTIC_ROLE


def declared_cells() -> tuple[PropertyCell, ...]:
    """Every sampled cell with its law, role, size, replicates and seed.

    The calibration controls are derived from the positive arm's rows (:data:`DERIVED_CELLS`).
    They are listed here with the positive arm's law, so that a committed truth can be read back
    against the cell it came from.  The failure differential
    (``tests/canonical/property_differential.py``) reads this list to build synthetic rows
    while the study holds no committed rows, and it skips the derived controls.
    """
    cells: list[PropertyCell] = []
    controls = {name for names in DERIVED_CELLS.values() for name in names}
    for family, configuration, replicates in FIT_SETS:
        seed = stream_seed(STUDY, "property_sample", family, configuration, 0)
        n = fit_n(family, configuration)
        key = ((family, configuration, 0, replicates, seed, n),)
        for name in _cells_of(key):
            if name.startswith(f"{JOINT_FAMILY}/"):
                continue
            _, cell = name.split("/", 1)
            label = cell.split("__", 1)[0]
            estimand = label_estimand(label)
            truth = CONTINUOUS_TRUTH[label] if configuration == "continuous" else TRUTH[estimand]
            cells.append(
                PropertyCell(
                    property=family,
                    cell=cell,
                    dgp=DeclaredLaw(f"{family}__{configuration}", {estimand: truth}),
                    outcome_learner=lambda: None,
                    treatment_learner=lambda: None,
                    n=n,
                    replicates=replicates,
                    seed=seed,
                    role="control" if name in controls else _role(family, cell, configuration),
                    estimand=estimand,
                )
            )
    cells.extend(
        joint_property_cells(
            JOINT_LABEL,
            n=IN_SAMPLE_N,
            replicates=CALIBRATION_REPLICATES,
            seed=stream_seed(STUDY, "property_sample", "interval_calibration", "in_sample", 0),
        )
    )
    return tuple(cells)


#: The share of a cell's requested replications that the accounting lets a run lose, declared
#: for the study's next property run (RM39b2 rev 3 section 6).  Up to ``floor(0.01 R)`` lost
#: replicates are judged by the worst case of each clause; one more is ``bound exceeded``.
FAILURE_BOUND = 0.01

#: The draw-level statistic the accounting reports for each lost replicate (rev 3 D3'f).
PROPERTY_WITNESS = property_verdicts.SAMPLE_WITNESS


def generate_property_rows(*, n_jobs: int = STUDY_JOBS) -> pd.DataFrame:
    """The declared property study.  Refused while the pilot tables are placeholders."""
    require_recorded_pilots("the declared property run")
    outcomes = fit_properties(
        _payloads(),
        splat_drawn(_fit_drawn),
        draw=splat_draw(_draw),
        cells_of=_cells_of,
        cell_of=payload_cell,
        replicate_of=payload_replicate,
        n_jobs=n_jobs,
        derived=DERIVED_CELLS,
    )
    return assemble_property_rows(outcomes)


# ---------------------------------------------------------------------------- verdicts

#: The calibration family reported beside its SE-ratio interval and gated by nothing.
REPORTED_CALIBRATION = ("small_sample_calibration",)
#: The arms whose cells keep the two-sided rule ``apply_shared_verdicts`` wrote, apart from the
#: booster arm, which reads the one-sided rule.
TWO_SIDED_ARMS = ("crossfit", "continuous")


def _calibration_report(summary: pd.DataFrame, rows: pd.DataFrame, name: str) -> None:
    """Publish a reported family's SE-ratio interval beside its coverage, gated by nothing."""
    family = summary["property"] == name
    for index in summary.index[family]:
        cell = str(summary.loc[index, "cell"])
        group = rows.loc[(rows["property"] == name) & (rows["cell"] == cell)]
        interval = se_ratio_interval(
            group,
            replicates=MARGINS.bootstrap_replicates,
            confidence_level=MARGINS.confidence_level,
            seed=stream_seed(STUDY, name, cell),
        )
        summary.loc[index, "se_ratio_ci_lower"] = interval.low
        summary.loc[index, "se_ratio_ci_upper"] = interval.high
    summary.loc[family, "passed"] = True
    summary.loc[family, "property_passed"] = True


def one_sided_calibration_verdicts(summary: pd.DataFrame) -> None:
    """The C5 rule (plan 5.6): the lower ends of both bands, and the two-sided result beside it.

    With correct parametric mechanisms and a regression that converges to a wrong limit, the
    Wald variance reads :math:`\\mathrm{Var}\\,\\varphi(\\bar m, r)`, which is at least the true
    variance.  The interval is asymptotically conservative, so the study gates the lower half of
    the band only.  ``two_sided_passed`` keeps the verdict that the two-sided band gives.
    """
    boosted = (summary["property"] == "interval_calibration") & summary["cell"].str.endswith(
        f"__{BOOSTER_ARM}"
    )
    # An object column takes booleans beside the missing values of the other arms.
    summary["two_sided_passed"] = summary["two_sided_passed"].astype(object)
    for index in summary.index[boosted]:
        ratio = summary_interval(summary, index, "se_ratio")
        coverage = summary_interval(summary, index, "coverage")
        summary.loc[index, "two_sided_passed"] = bool(summary.loc[index, "passed"])
        summary.loc[index, "passed"] = bool(
            ratio.low >= MARGINS.calibration_se_ratio[0]
            and coverage.low >= MARGINS.calibration_coverage[0]
        )


def _necessity(
    family: str,
    labels: Sequence[str],
    arms: tuple[str, str],
    limits: Mapping[str, float],
    *,
    column: str = "necessity_displacement",
    threshold: float = NECESSITY_DISPLACEMENT,
) -> Necessity:
    return Necessity(
        family=family,
        labels=tuple(labels),
        arms=arms,
        column=column,
        threshold=threshold,
        alternative_truths=limits,
    )


#: Every necessity family's declaration, read by its verdict and its rule.  Each control reads
#: its exact limit.  ``correction_necessity`` publishes its own displacement column and floor.
NECESSITIES = (
    _necessity(
        "correction_necessity",
        GATED_LABELS,
        CORRECTION_ARMS,
        PLUG_IN_LIMIT,
        column="correction_displacement",
        threshold=CORRECTION_DISPLACEMENT,
    ),
    _necessity("rule_necessity", RULE_LABELS, RULE_ARMS, REVERSED_TRUTH),
    _necessity("repeated_outcome_necessity", ABSORBING_LABELS, ABSORBING_ARMS, ABSORBING_LIMIT),
    _necessity("outcome_history_necessity", HISTORY_LABELS, HISTORY_FREE_ARMS, HISTORY_FREE_LIMIT),
    _necessity("full_history_necessity", GATED_LABELS, FULL_HISTORY_ARMS, MARKOV_LIMIT),
)

#: The ``both_wrong`` limit of each gated label, by control cell, which
#: ``alternative_target_verdicts`` reads.
BOTH_WRONG_TARGETS = {f"{name}__both_wrong": BOTH_WRONG_LIMIT[name] for name in GATED_LABELS}


#: The centre of each control cell in the failure differential's synthetic rows: the exact
#: limit that the control must reach (``tests/canonical/property_differential.py``).
SYNTHETIC_CENTRES = {
    **{f"double_robustness/{cell}": limit for cell, limit in BOTH_WRONG_TARGETS.items()},
    **{
        f"{necessity.family}/{label}__{necessity.arms[1]}": float(
            necessity.alternative_truths[label]
        )
        for necessity in NECESSITIES
        if necessity.alternative_truths is not None
        for label in necessity.labels
    },
}


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    summary, rates = apply_shared_verdicts(
        rows,
        STUDY,
        extra_columns=("necessity_displacement", "correction_displacement", "two_sided_passed"),
        rate_labels=(),
    )
    # The C1 arm and its two derived controls read the instrument's rule for each kind.  The
    # C3 and C6 arms keep the two-sided rule ``apply_shared_verdicts`` wrote, which is the same
    # rule.  The booster arm reads the one-sided rule below.
    primary = (summary["property"] == "interval_calibration") & ~summary["cell"].str.endswith(
        tuple(f"__{arm}" for arm in (*TWO_SIDED_ARMS, BOOSTER_ARM))
    )
    subset = summary.loc[primary].copy()
    calibration_verdicts(subset, margins=MARGINS, positive_suffix="in_sample")
    summary.loc[subset.index, "passed"] = subset["passed"]
    one_sided_calibration_verdicts(summary)

    for name in REPORTED_CALIBRATION:
        _calibration_report(summary, rows, name)
    simultaneous_coverage_verdicts(summary, margins=MARGINS)

    # ``both_wrong`` reads its exact limit: the fit differs from the truth, and the limit says
    # by how much.  ``apply_shared_verdicts`` wrote the family's own verdicts.
    alternative_target_verdicts(
        summary,
        rows,
        STUDY,
        family="double_robustness",
        alternative_truths=BOTH_WRONG_TARGETS,
    )
    for necessity in NECESSITIES:
        alternative_target_necessity_verdicts(summary, rows, STUDY, necessity)
    return finish(summary, rates)


def _calibration_rule() -> pf.FamilyRule:
    """The ``interval_calibration`` rule: the shared one, with the booster arm one-sided.

    The shared rule reads every positive cell against the two-sided SE-ratio and coverage
    bands.  :func:`one_sided_calibration_verdicts` replaces that verdict on the booster arm
    with the lower end of both bands, so the booster cells read two floor clauses instead.
    """
    shared = property_verdicts.calibration_rule(efficiency_band=None)

    def booster(ctx: pf.Context, row: pd.Series) -> bool:
        return str(row["cell"]).endswith(f"__{BOOSTER_ARM}")

    def two_sided(clause: pf.Clause) -> pf.Clause:
        applies = clause.applies
        return dataclasses.replace(
            clause, applies=lambda ctx, row: applies(ctx, row) and not booster(ctx, row)
        )

    def seed(ctx: pf.Context, row: pd.Series) -> int:
        return stream_seed(ctx.record, "interval_calibration", str(row["cell"]))

    clauses = [
        two_sided(clause) if clause.name in TWO_SIDED_CLAUSES else clause
        for clause in shared.clauses
    ]
    clauses += [
        pf.se_interval_clause(
            "se ratio at calibration floor",
            which="se_ratio",
            judge=pf.floor_judge(lambda ctx: ctx.margins.calibration_se_ratio[0]),
            seed=seed,
            applies=booster,
        ),
        pf.proportion_clause(
            "coverage at calibration floor",
            statistic="S2",
            judge=pf.floor_judge(lambda ctx: ctx.margins.calibration_coverage[0]),
            applies=booster,
        ),
    ]
    return dataclasses.replace(shared, clauses=tuple(clauses))


#: The shared clauses that read a positive cell against a two-sided band.
TWO_SIDED_CLAUSES = ("se ratio in calibration band", "coverage in calibration band")

#: The failure accounting's declared rules (RM39b2 rev 3, D3'a).  ``double_robustness`` adds the
#: ``both_wrong`` limit to the union-model rule, ``interval_calibration`` reads the booster arm
#: one-sided, and each necessity family reads its declaration.  The simultaneous-coverage and
#: union-model clauses are the shared rules'.
PROPERTY_RULES = (
    property_verdicts.robustness_rule(alternative_truths=BOTH_WRONG_TARGETS),
    _calibration_rule(),
    *property_verdicts.necessity_rules(*NECESSITIES),
)


# ---------------------------------------------------------------------------- label correlations


@functools.cache
def _curve(model: str, plan: str, horizon: int, reference: str | None) -> np.ndarray:
    laws = {"binary": law.BINARY, "continuous": law.CONTINUOUS}
    return laws[model].eif(plan, horizon, reference=reference)


def _correlations(model: law.RepeatedOutcomeLaw) -> sizing.Correlations:
    def correlations(labels: Sequence[str]) -> np.ndarray:
        curves = []
        for name in labels:
            plan, reference, horizon = label_parts(name)
            curves.append(_curve(model.name, plan, horizon, reference))
        return sizing.influence_correlations(curves, model.paths["p"].to_numpy())

    return correlations


#: The correlation, under the binary law, of the efficient influence curves of labels.  Two
#: estimates of the labels from one sample have this correlation to first order.
label_correlations = _correlations(law.BINARY)
#: The same under the continuous law, which C6 reads.
continuous_label_correlations = _correlations(law.CONTINUOUS)
#: The correlation function of each calibration arm.
ARM_CORRELATIONS: dict[str, sizing.Correlations] = {
    "in_sample": label_correlations,
    "crossfit": label_correlations,
    "continuous": continuous_label_correlations,
    BOOSTER_ARM: label_correlations,
}
#: The arms whose cells read the lower end of the band only.
ONE_SIDED_ARMS = (BOOSTER_ARM,)


def effective_cells(labels: Sequence[str]) -> float:
    r"""How many independent cells ``labels`` are worth when their estimates are averaged."""
    return sizing.effective_cells(labels, label_correlations)


# ---------------------------------------------------------------------------- pilot tables
#
# These tables are pilot results.  ``summarize_*_pilot`` produces them, and the sizing rules
# read them.

#: Pilot P1 (primary): the covered count and the standardized bias of each ``scenario/label``
#: cell over :data:`PRIMARY_PILOT_REPLICATES` draws at ``n = PRIMARY_N`` from the pilot stream.
PRIMARY_PILOT_REPLICATES = 2_000
#: The sample size P1 ran at.  A recorded table must equal :data:`PRIMARY_N`.
PRIMARY_PILOT_N = 12_000
# Recorded from pilot P1: 2,000 draws per scenario at n = 12,000.  The primary ``n`` rule did
# not fire, and the rule gives 2,400 replications.
PRIMARY_PILOT_COVERED = {
    "repeated_folds_1/never_t1": 1901,
    "repeated_folds_1/never_t2": 1901,
    "repeated_folds_1/never_t3": 1909,
    "repeated_folds_1/always_t1": 1875,
    "repeated_folds_1/always_t2": 1897,
    "repeated_folds_1/always_t3": 1898,
    "repeated_folds_1/rule_t2": 1899,
    "repeated_folds_1/rule_t3": 1900,
    "repeated_folds_1/always_vs_never_t1": 1889,
    "repeated_folds_1/always_vs_never_t2": 1904,
    "repeated_folds_1/always_vs_never_t3": 1915,
    "repeated_folds_1/rule_vs_never_t2": 1913,
    "repeated_folds_1/rule_vs_never_t3": 1900,
    "repeated_folds_5/never_t1": 1907,
    "repeated_folds_5/never_t2": 1907,
    "repeated_folds_5/never_t3": 1899,
    "repeated_folds_5/always_t1": 1885,
    "repeated_folds_5/always_t2": 1898,
    "repeated_folds_5/always_t3": 1892,
    "repeated_folds_5/rule_t2": 1890,
    "repeated_folds_5/rule_t3": 1915,
    "repeated_folds_5/always_vs_never_t1": 1907,
    "repeated_folds_5/always_vs_never_t2": 1903,
    "repeated_folds_5/always_vs_never_t3": 1898,
    "repeated_folds_5/rule_vs_never_t2": 1903,
    "repeated_folds_5/rule_vs_never_t3": 1912,
}
#: Mean reported standard error over the empirical spread of the estimate, by cell (reported).
PRIMARY_PILOT_RATIOS = {
    "repeated_folds_1/never_t1": 0.9957,
    "repeated_folds_1/never_t2": 1.0087,
    "repeated_folds_1/never_t3": 1.0095,
    "repeated_folds_1/always_t1": 0.9712,
    "repeated_folds_1/always_t2": 1.0091,
    "repeated_folds_1/always_t3": 0.9978,
    "repeated_folds_1/rule_t2": 1.014,
    "repeated_folds_1/rule_t3": 1.0091,
    "repeated_folds_1/always_vs_never_t1": 0.9826,
    "repeated_folds_1/always_vs_never_t2": 1.0142,
    "repeated_folds_1/always_vs_never_t3": 1.0246,
    "repeated_folds_1/rule_vs_never_t2": 1.0157,
    "repeated_folds_1/rule_vs_never_t3": 1.0171,
    "repeated_folds_5/never_t1": 1.0104,
    "repeated_folds_5/never_t2": 1.0048,
    "repeated_folds_5/never_t3": 0.9885,
    "repeated_folds_5/always_t1": 0.9873,
    "repeated_folds_5/always_t2": 1.0068,
    "repeated_folds_5/always_t3": 0.9946,
    "repeated_folds_5/rule_t2": 0.9847,
    "repeated_folds_5/rule_t3": 1.036,
    "repeated_folds_5/always_vs_never_t1": 0.9992,
    "repeated_folds_5/always_vs_never_t2": 1.0289,
    "repeated_folds_5/always_vs_never_t3": 0.9976,
    "repeated_folds_5/rule_vs_never_t2": 1.0117,
    "repeated_folds_5/rule_vs_never_t3": 1.0305,
}
PRIMARY_PILOT_BIASES = {
    "repeated_folds_1/never_t1": -0.0064,
    "repeated_folds_1/never_t2": 0.0292,
    "repeated_folds_1/never_t3": -0.0047,
    "repeated_folds_1/always_t1": 0.004,
    "repeated_folds_1/always_t2": 0.0572,
    "repeated_folds_1/always_t3": 0.0208,
    "repeated_folds_1/rule_t2": -0.0084,
    "repeated_folds_1/rule_t3": 0.0058,
    "repeated_folds_1/always_vs_never_t1": 0.0075,
    "repeated_folds_1/always_vs_never_t2": 0.0103,
    "repeated_folds_1/always_vs_never_t3": 0.017,
    "repeated_folds_1/rule_vs_never_t2": -0.0281,
    "repeated_folds_1/rule_vs_never_t3": 0.0074,
    "repeated_folds_5/never_t1": -0.0407,
    "repeated_folds_5/never_t2": 0.0059,
    "repeated_folds_5/never_t3": -0.0379,
    "repeated_folds_5/always_t1": -0.021,
    "repeated_folds_5/always_t2": -0.0195,
    "repeated_folds_5/always_t3": -0.0101,
    "repeated_folds_5/rule_t2": 0.0193,
    "repeated_folds_5/rule_t3": -0.0106,
    "repeated_folds_5/always_vs_never_t1": 0.0222,
    "repeated_folds_5/always_vs_never_t2": -0.0166,
    "repeated_folds_5/always_vs_never_t3": 0.0237,
    "repeated_folds_5/rule_vs_never_t2": 0.008,
    "repeated_folds_5/rule_vs_never_t3": 0.0218,
}

#: Pilot P2 (calibration): the SE ratio and the standardized bias of each gated calibration
#: cell over :data:`CALIBRATION_PILOT_REPLICATES` draws per arm, and the joint coverage of the
#: C4 band over the C1 fits.
#:
#: The cross-fit and booster arms ran at ``n = 8,000``.  The in-sample and continuous arms ran
#: at the doubled ``n = 32,000`` (module docstring, "Pilot P2 finding").  The 8,000-draw rows of
#: those two arms are not pinned, because no table may describe a size the study no longer
#: declares.
CALIBRATION_PILOT_REPLICATES = 8_000
#: The sample size each P2 arm ran at.  A recorded table must equal the arm's declared size.
CALIBRATION_PILOT_N = {
    "crossfit": 8000,
    "crossfit_boosted": 8000,
    "in_sample": 32000,
    "continuous": 32000,
}
CALIBRATION_CELLS = (
    *(f"{name}__in_sample" for name in LABELS),
    *(f"{name}__crossfit" for name in GATED_LABELS),
    *(f"{name}__continuous" for name in GATED_LABELS),
    *(f"{name}__{BOOSTER_ARM}" for name in GATED_LABELS),
)
CALIBRATION_PILOT_RATIOS = {
    "never_t1__in_sample": 1.0155,
    "never_t2__in_sample": 0.9955,
    "never_t3__in_sample": 0.9858,
    "always_t1__in_sample": 1.0002,
    "always_t2__in_sample": 0.9833,
    "always_t3__in_sample": 1.0036,
    "rule_t2__in_sample": 0.9962,
    "rule_t3__in_sample": 0.9707,
    "always_vs_never_t1__in_sample": 1.0177,
    "always_vs_never_t2__in_sample": 0.9913,
    "always_vs_never_t3__in_sample": 0.995,
    "rule_vs_never_t2__in_sample": 0.9979,
    "rule_vs_never_t3__in_sample": 0.9796,
    "never_t3__crossfit": 1.0048,
    "always_t3__crossfit": 1.0058,
    "rule_t3__crossfit": 0.9815,
    "always_vs_never_t3__crossfit": 1.0102,
    "rule_vs_never_t3__crossfit": 0.9913,
    "never_t3__continuous": 0.9659,
    "always_t3__continuous": 0.9869,
    "rule_t3__continuous": 0.9691,
    "always_vs_never_t3__continuous": 0.97,
    "rule_vs_never_t3__continuous": 0.9689,
    "never_t3__crossfit_boosted": 1.0056,
    "always_t3__crossfit_boosted": 0.9964,
    "rule_t3__crossfit_boosted": 1.0036,
    "always_vs_never_t3__crossfit_boosted": 1.0042,
    "rule_vs_never_t3__crossfit_boosted": 1.0015,
}
CALIBRATION_PILOT_BIASES = {
    "never_t1__in_sample": 0.0191,
    "never_t2__in_sample": 0.0075,
    "never_t3__in_sample": 0.021,
    "always_t1__in_sample": -0.0014,
    "always_t2__in_sample": 0.0055,
    "always_t3__in_sample": 0.0094,
    "rule_t2__in_sample": 0.0093,
    "rule_t3__in_sample": -0.0022,
    "always_vs_never_t1__in_sample": -0.0168,
    "always_vs_never_t2__in_sample": -0.0027,
    "always_vs_never_t3__in_sample": -0.0107,
    "rule_vs_never_t2__in_sample": 0.0001,
    "rule_vs_never_t3__in_sample": -0.0169,
    "never_t3__crossfit": 0.0015,
    "always_t3__crossfit": -0.0064,
    "rule_t3__crossfit": 0.01,
    "always_vs_never_t3__crossfit": -0.0052,
    "rule_vs_never_t3__crossfit": 0.0059,
    "never_t3__continuous": 0.0098,
    "always_t3__continuous": 0.0141,
    "rule_t3__continuous": -0.0015,
    "always_vs_never_t3__continuous": 0.0008,
    "rule_vs_never_t3__continuous": -0.0084,
    "never_t3__crossfit_boosted": -0.003,
    "always_t3__crossfit_boosted": -0.0099,
    "rule_t3__crossfit_boosted": 0.0153,
    "always_vs_never_t3__crossfit_boosted": -0.0038,
    "rule_vs_never_t3__crossfit_boosted": 0.0127,
}
#: The covered count of each calibration cell over the pilot draws of its arm (reported).
CALIBRATION_PILOT_COVERED = {
    "never_t1__in_sample": 7621,
    "never_t2__in_sample": 7604,
    "never_t3__in_sample": 7551,
    "always_t1__in_sample": 7618,
    "always_t2__in_sample": 7570,
    "always_t3__in_sample": 7597,
    "rule_t2__in_sample": 7604,
    "rule_t3__in_sample": 7523,
    "always_vs_never_t1__in_sample": 7612,
    "always_vs_never_t2__in_sample": 7580,
    "always_vs_never_t3__in_sample": 7587,
    "rule_vs_never_t2__in_sample": 7601,
    "rule_vs_never_t3__in_sample": 7560,
    "never_t3__crossfit": 7593,
    "always_t3__crossfit": 7606,
    "rule_t3__crossfit": 7520,
    "always_vs_never_t3__crossfit": 7612,
    "rule_vs_never_t3__crossfit": 7585,
    "never_t3__continuous": 7532,
    "always_t3__continuous": 7566,
    "rule_t3__continuous": 7541,
    "always_vs_never_t3__continuous": 7548,
    "rule_vs_never_t3__continuous": 7531,
    "never_t3__crossfit_boosted": 7595,
    "always_t3__crossfit_boosted": 7578,
    "rule_t3__crossfit_boosted": 7579,
    "always_vs_never_t3__crossfit_boosted": 7609,
    "rule_vs_never_t3__crossfit_boosted": 7589,
}
#: The empirical spread of each calibration cell's estimate (reported).
CALIBRATION_PILOT_SPREADS = {
    "never_t1__in_sample": 0.00373758,
    "never_t2__in_sample": 0.00508928,
    "never_t3__in_sample": 0.0075317,
    "always_t1__in_sample": 0.00246345,
    "always_t2__in_sample": 0.00378451,
    "always_t3__in_sample": 0.00592582,
    "rule_t2__in_sample": 0.00418989,
    "rule_t3__in_sample": 0.00700133,
    "always_vs_never_t1__in_sample": 0.00443956,
    "always_vs_never_t2__in_sample": 0.00633498,
    "always_vs_never_t3__in_sample": 0.00955088,
    "rule_vs_never_t2__in_sample": 0.00656875,
    "rule_vs_never_t3__in_sample": 0.0102638,
    "never_t3__crossfit": 0.0154934,
    "always_t3__crossfit": 0.0120405,
    "rule_t3__crossfit": 0.0150922,
    "always_vs_never_t3__crossfit": 0.0195123,
    "rule_vs_never_t3__crossfit": 0.0216704,
    "never_t3__continuous": 0.00619167,
    "always_t3__continuous": 0.00476747,
    "rule_t3__continuous": 0.00547518,
    "always_vs_never_t3__continuous": 0.00783601,
    "rule_vs_never_t3__continuous": 0.00824153,
    "never_t3__crossfit_boosted": 0.0150606,
    "always_t3__crossfit_boosted": 0.0120505,
    "rule_t3__crossfit_boosted": 0.0141239,
    "always_vs_never_t3__crossfit_boosted": 0.0192321,
    "rule_vs_never_t3__crossfit_boosted": 0.0206962,
}
CALIBRATION_PILOT_JOINT_COVERAGE = 0.9486


def property_cells() -> dict[str, list[str]]:
    """The bias-checked cells of each property family: positive arm and control, by label."""
    pairs = (
        ("double_robustness", GATED_LABELS, DOUBLE_ROBUSTNESS_ARMS),
        ("correction_necessity", GATED_LABELS, CORRECTION_ARMS),
        ("rule_necessity", RULE_LABELS, RULE_ARMS),
        ("repeated_outcome_necessity", ABSORBING_LABELS, ABSORBING_ARMS),
        ("outcome_history_necessity", HISTORY_LABELS, HISTORY_FREE_ARMS),
        ("full_history_necessity", GATED_LABELS, FULL_HISTORY_ARMS),
    )
    return {
        family: [f"{family}/{name}__{arm}" for name in labels for arm in arms]
        for family, labels, arms in pairs
    }


#: Pilot P3 (property families): the standardized bias of each ``family/cell`` around its
#: target (the truth for a positive arm, the control's own limit for a control) over
#: :data:`PROPERTY_PILOT_REPLICATES` draws at the family's own size.
PROPERTY_PILOT_REPLICATES = 2_000
#: The sample size each P3 family ran at.  A recorded table must equal the family's declared size.
PROPERTY_PILOT_N = {
    "double_robustness": 2_000,
    "correction_necessity": 2_000,
    "rule_necessity": 2_000,
    "repeated_outcome_necessity": 8_000,
    "outcome_history_necessity": 2_000,
    "full_history_necessity": 8_000,
}
# Recorded from pilot P3: 2,000 draws per family at the family's declared size.  The N1 ``n``
# rule did not fire, and the rule gives 1,000 replications for every family.
PROPERTY_PILOT_BIASES = {
    "double_robustness/never_t3__mechanism_correct": -0.046,
    "double_robustness/never_t3__outcome_correct": -0.0282,
    "double_robustness/never_t3__both_wrong": -0.0077,
    "double_robustness/always_t3__mechanism_correct": 0.024,
    "double_robustness/always_t3__outcome_correct": -0.0038,
    "double_robustness/always_t3__both_wrong": 0.0129,
    "double_robustness/rule_t3__mechanism_correct": 0.0177,
    "double_robustness/rule_t3__outcome_correct": -0.0162,
    "double_robustness/rule_t3__both_wrong": 0.0058,
    "double_robustness/always_vs_never_t3__mechanism_correct": 0.0519,
    "double_robustness/always_vs_never_t3__outcome_correct": 0.0208,
    "double_robustness/always_vs_never_t3__both_wrong": 0.0145,
    "double_robustness/rule_vs_never_t3__mechanism_correct": 0.0443,
    "double_robustness/rule_vs_never_t3__outcome_correct": 0.0106,
    "double_robustness/rule_vs_never_t3__both_wrong": 0.0095,
    "correction_necessity/never_t3__corrected": 0.0503,
    "correction_necessity/never_t3__uncorrected": 0.0633,
    "correction_necessity/always_t3__corrected": 0.0101,
    "correction_necessity/always_t3__uncorrected": -0.005,
    "correction_necessity/rule_t3__corrected": -0.0259,
    "correction_necessity/rule_t3__uncorrected": -0.012,
    "correction_necessity/always_vs_never_t3__corrected": -0.0342,
    "correction_necessity/always_vs_never_t3__uncorrected": -0.0498,
    "correction_necessity/rule_vs_never_t3__corrected": -0.0541,
    "correction_necessity/rule_vs_never_t3__uncorrected": -0.0564,
    "rule_necessity/rule_t2__declared": 0.0097,
    "rule_necessity/rule_t2__reversed_rule": -0.0154,
    "rule_necessity/rule_t3__declared": -0.0131,
    "rule_necessity/rule_t3__reversed_rule": -0.0226,
    "repeated_outcome_necessity/never_t2__repeated": 0.0132,
    "repeated_outcome_necessity/never_t2__absorbing_reading": -0.0023,
    "repeated_outcome_necessity/never_t3__repeated": 0.0245,
    "repeated_outcome_necessity/never_t3__absorbing_reading": 0.0223,
    "repeated_outcome_necessity/always_t2__repeated": -0.0181,
    "repeated_outcome_necessity/always_t2__absorbing_reading": -0.038,
    "repeated_outcome_necessity/always_t3__repeated": 0.0217,
    "repeated_outcome_necessity/always_t3__absorbing_reading": 0.0117,
    "repeated_outcome_necessity/rule_t2__repeated": -0.0471,
    "repeated_outcome_necessity/rule_t2__absorbing_reading": -0.0642,
    "repeated_outcome_necessity/rule_t3__repeated": 0.0141,
    "repeated_outcome_necessity/rule_t3__absorbing_reading": -0.0443,
    "outcome_history_necessity/never_t3__repeated": 0.0226,
    "outcome_history_necessity/never_t3__history_free": 0.0217,
    "outcome_history_necessity/always_t3__repeated": 0.0226,
    "outcome_history_necessity/always_t3__history_free": 0.0264,
    "outcome_history_necessity/always_vs_never_t3__repeated": -0.0041,
    "outcome_history_necessity/always_vs_never_t3__history_free": 0.0008,
    "full_history_necessity/never_t3__full_history": -0.0627,
    "full_history_necessity/never_t3__markov_history": -0.0518,
    "full_history_necessity/always_t3__full_history": -0.0416,
    "full_history_necessity/always_t3__markov_history": -0.0383,
    "full_history_necessity/rule_t3__full_history": -0.0131,
    "full_history_necessity/rule_t3__markov_history": -0.0056,
    "full_history_necessity/always_vs_never_t3__full_history": 0.0227,
    "full_history_necessity/always_vs_never_t3__markov_history": 0.0118,
    "full_history_necessity/rule_vs_never_t3__full_history": 0.0371,
    "full_history_necessity/rule_vs_never_t3__markov_history": 0.0366,
}
PROPERTY_PILOT_RATIOS = {
    "double_robustness/never_t3__mechanism_correct": 1.0531,
    "double_robustness/never_t3__outcome_correct": 0.9676,
    "double_robustness/never_t3__both_wrong": 1.1691,
    "double_robustness/always_t3__mechanism_correct": 0.9996,
    "double_robustness/always_t3__outcome_correct": 1.2117,
    "double_robustness/always_t3__both_wrong": 1.2818,
    "double_robustness/rule_t3__mechanism_correct": 1.0238,
    "double_robustness/rule_t3__outcome_correct": 0.7629,
    "double_robustness/rule_t3__both_wrong": 0.9028,
    "double_robustness/always_vs_never_t3__mechanism_correct": 1.0504,
    "double_robustness/always_vs_never_t3__outcome_correct": 1.0745,
    "double_robustness/always_vs_never_t3__both_wrong": 1.2245,
    "double_robustness/rule_vs_never_t3__mechanism_correct": 1.0181,
    "double_robustness/rule_vs_never_t3__outcome_correct": 0.8746,
    "double_robustness/rule_vs_never_t3__both_wrong": 1.0504,
    "correction_necessity/never_t3__corrected": 1.0209,
    "correction_necessity/never_t3__uncorrected": 1.1502,
    "correction_necessity/always_t3__corrected": 0.9917,
    "correction_necessity/always_t3__uncorrected": 0.9186,
    "correction_necessity/rule_t3__corrected": 1.0208,
    "correction_necessity/rule_t3__uncorrected": 1.2952,
    "correction_necessity/always_vs_never_t3__corrected": 1.0127,
    "correction_necessity/always_vs_never_t3__uncorrected": 1.0445,
    "correction_necessity/rule_vs_never_t3__corrected": 1.0182,
    "correction_necessity/rule_vs_never_t3__uncorrected": 1.2158,
    "rule_necessity/rule_t2__declared": 0.9701,
    "rule_necessity/rule_t2__reversed_rule": 0.982,
    "rule_necessity/rule_t3__declared": 0.7981,
    "rule_necessity/rule_t3__reversed_rule": 0.9321,
    "repeated_outcome_necessity/never_t2__repeated": 1.0087,
    "repeated_outcome_necessity/never_t2__absorbing_reading": 0.9866,
    "repeated_outcome_necessity/never_t3__repeated": 0.9501,
    "repeated_outcome_necessity/never_t3__absorbing_reading": 0.9978,
    "repeated_outcome_necessity/always_t2__repeated": 1.0018,
    "repeated_outcome_necessity/always_t2__absorbing_reading": 1.0052,
    "repeated_outcome_necessity/always_t3__repeated": 0.9762,
    "repeated_outcome_necessity/always_t3__absorbing_reading": 0.998,
    "repeated_outcome_necessity/rule_t2__repeated": 0.9955,
    "repeated_outcome_necessity/rule_t2__absorbing_reading": 0.9942,
    "repeated_outcome_necessity/rule_t3__repeated": 0.9188,
    "repeated_outcome_necessity/rule_t3__absorbing_reading": 1.0071,
    "outcome_history_necessity/never_t3__repeated": 0.8583,
    "outcome_history_necessity/never_t3__history_free": 0.9452,
    "outcome_history_necessity/always_t3__repeated": 0.9333,
    "outcome_history_necessity/always_t3__history_free": 0.9789,
    "outcome_history_necessity/always_vs_never_t3__repeated": 0.8876,
    "outcome_history_necessity/always_vs_never_t3__history_free": 0.9577,
    "full_history_necessity/never_t3__full_history": 1.0638,
    "full_history_necessity/never_t3__markov_history": 1.1367,
    "full_history_necessity/always_t3__full_history": 1.3138,
    "full_history_necessity/always_t3__markov_history": 1.2589,
    "full_history_necessity/rule_t3__full_history": 0.7717,
    "full_history_necessity/rule_t3__markov_history": 0.9471,
    "full_history_necessity/always_vs_never_t3__full_history": 1.165,
    "full_history_necessity/always_vs_never_t3__markov_history": 1.1857,
    "full_history_necessity/rule_vs_never_t3__full_history": 0.9621,
    "full_history_necessity/rule_vs_never_t3__markov_history": 1.0777,
}
PROPERTY_PILOT_COVERED = {
    "double_robustness/never_t3__mechanism_correct": 1916,
    "double_robustness/never_t3__outcome_correct": 1874,
    "double_robustness/never_t3__both_wrong": 1242,
    "double_robustness/always_t3__mechanism_correct": 1889,
    "double_robustness/always_t3__outcome_correct": 1955,
    "double_robustness/always_t3__both_wrong": 1920,
    "double_robustness/rule_t3__mechanism_correct": 1879,
    "double_robustness/rule_t3__outcome_correct": 1740,
    "double_robustness/rule_t3__both_wrong": 1613,
    "double_robustness/always_vs_never_t3__mechanism_correct": 1920,
    "double_robustness/always_vs_never_t3__outcome_correct": 1936,
    "double_robustness/always_vs_never_t3__both_wrong": 1263,
    "double_robustness/rule_vs_never_t3__mechanism_correct": 1908,
    "double_robustness/rule_vs_never_t3__outcome_correct": 1840,
    "double_robustness/rule_vs_never_t3__both_wrong": 1760,
    "correction_necessity/never_t3__corrected": 1904,
    "correction_necessity/never_t3__uncorrected": 1348,
    "correction_necessity/always_t3__corrected": 1875,
    "correction_necessity/always_t3__uncorrected": 1680,
    "correction_necessity/rule_t3__corrected": 1890,
    "correction_necessity/rule_t3__uncorrected": 1865,
    "correction_necessity/always_vs_never_t3__corrected": 1908,
    "correction_necessity/always_vs_never_t3__uncorrected": 1101,
    "correction_necessity/rule_vs_never_t3__corrected": 1892,
    "correction_necessity/rule_vs_never_t3__uncorrected": 1889,
    "rule_necessity/rule_t2__declared": 1890,
    "rule_necessity/rule_t2__reversed_rule": 136,
    "rule_necessity/rule_t3__declared": 1756,
    "rule_necessity/rule_t3__reversed_rule": 646,
    "repeated_outcome_necessity/never_t2__repeated": 1899,
    "repeated_outcome_necessity/never_t2__absorbing_reading": 0,
    "repeated_outcome_necessity/never_t3__repeated": 1875,
    "repeated_outcome_necessity/never_t3__absorbing_reading": 0,
    "repeated_outcome_necessity/always_t2__repeated": 1897,
    "repeated_outcome_necessity/always_t2__absorbing_reading": 0,
    "repeated_outcome_necessity/always_t3__repeated": 1886,
    "repeated_outcome_necessity/always_t3__absorbing_reading": 0,
    "repeated_outcome_necessity/rule_t2__repeated": 1890,
    "repeated_outcome_necessity/rule_t2__absorbing_reading": 0,
    "repeated_outcome_necessity/rule_t3__repeated": 1853,
    "repeated_outcome_necessity/rule_t3__absorbing_reading": 0,
    "outcome_history_necessity/never_t3__repeated": 1810,
    "outcome_history_necessity/never_t3__history_free": 1372,
    "outcome_history_necessity/always_t3__repeated": 1864,
    "outcome_history_necessity/always_t3__history_free": 1819,
    "outcome_history_necessity/always_vs_never_t3__repeated": 1821,
    "outcome_history_necessity/always_vs_never_t3__history_free": 1309,
    "full_history_necessity/never_t3__full_history": 1924,
    "full_history_necessity/never_t3__markov_history": 221,
    "full_history_necessity/always_t3__full_history": 1976,
    "full_history_necessity/always_t3__markov_history": 1557,
    "full_history_necessity/rule_t3__full_history": 1738,
    "full_history_necessity/rule_t3__markov_history": 527,
    "full_history_necessity/always_vs_never_t3__full_history": 1950,
    "full_history_necessity/always_vs_never_t3__markov_history": 168,
    "full_history_necessity/rule_vs_never_t3__full_history": 1888,
    "full_history_necessity/rule_vs_never_t3__markov_history": 1705,
}
#: The empirical spread of each cell's estimate.  The discrimination test reads the gaps of the
#: controls in these spreads.
PROPERTY_PILOT_SPREADS = {
    "double_robustness/never_t3__mechanism_correct": 0.0320336,
    "double_robustness/never_t3__outcome_correct": 0.0329263,
    "double_robustness/never_t3__both_wrong": 0.0276271,
    "double_robustness/always_t3__mechanism_correct": 0.0245427,
    "double_robustness/always_t3__outcome_correct": 0.0241385,
    "double_robustness/always_t3__both_wrong": 0.0266817,
    "double_robustness/rule_t3__mechanism_correct": 0.0311515,
    "double_robustness/rule_t3__outcome_correct": 0.0284902,
    "double_robustness/rule_t3__both_wrong": 0.0252038,
    "double_robustness/always_vs_never_t3__mechanism_correct": 0.0397248,
    "double_robustness/always_vs_never_t3__outcome_correct": 0.0402684,
    "double_robustness/always_vs_never_t3__both_wrong": 0.0384181,
    "double_robustness/rule_vs_never_t3__mechanism_correct": 0.045669,
    "double_robustness/rule_vs_never_t3__outcome_correct": 0.0440528,
    "double_robustness/rule_vs_never_t3__both_wrong": 0.0375926,
    "correction_necessity/never_t3__corrected": 0.0329861,
    "correction_necessity/never_t3__uncorrected": 0.0292795,
    "correction_necessity/always_t3__corrected": 0.0247195,
    "correction_necessity/always_t3__uncorrected": 0.026686,
    "correction_necessity/rule_t3__corrected": 0.0310519,
    "correction_necessity/rule_t3__uncorrected": 0.0244743,
    "correction_necessity/always_vs_never_t3__corrected": 0.0411442,
    "correction_necessity/always_vs_never_t3__uncorrected": 0.0398937,
    "correction_necessity/rule_vs_never_t3__corrected": 0.0454851,
    "correction_necessity/rule_vs_never_t3__uncorrected": 0.0380934,
    "rule_necessity/rule_t2__declared": 0.0171295,
    "rule_necessity/rule_t2__reversed_rule": 0.0159923,
    "rule_necessity/rule_t3__declared": 0.0297429,
    "rule_necessity/rule_t3__reversed_rule": 0.025696,
    "repeated_outcome_necessity/never_t2__repeated": 0.0100373,
    "repeated_outcome_necessity/never_t2__absorbing_reading": 0.0105292,
    "repeated_outcome_necessity/never_t3__repeated": 0.0153341,
    "repeated_outcome_necessity/never_t3__absorbing_reading": 0.0121653,
    "repeated_outcome_necessity/always_t2__repeated": 0.00741439,
    "repeated_outcome_necessity/always_t2__absorbing_reading": 0.00829078,
    "repeated_outcome_necessity/always_t3__repeated": 0.0121135,
    "repeated_outcome_necessity/always_t3__absorbing_reading": 0.0126401,
    "repeated_outcome_necessity/rule_t2__repeated": 0.00837,
    "repeated_outcome_necessity/rule_t2__absorbing_reading": 0.00895662,
    "repeated_outcome_necessity/rule_t3__repeated": 0.0142244,
    "repeated_outcome_necessity/rule_t3__absorbing_reading": 0.011185,
    "outcome_history_necessity/never_t3__repeated": 0.0320719,
    "outcome_history_necessity/never_t3__history_free": 0.0306699,
    "outcome_history_necessity/always_t3__repeated": 0.0246629,
    "outcome_history_necessity/always_t3__history_free": 0.0263881,
    "outcome_history_necessity/always_vs_never_t3__repeated": 0.0404344,
    "outcome_history_necessity/always_vs_never_t3__history_free": 0.0405456,
    "full_history_necessity/never_t3__full_history": 0.0148269,
    "full_history_necessity/never_t3__markov_history": 0.0140667,
    "full_history_necessity/always_t3__full_history": 0.0119434,
    "full_history_necessity/always_t3__markov_history": 0.0130565,
    "full_history_necessity/rule_t3__full_history": 0.014324,
    "full_history_necessity/rule_t3__markov_history": 0.0118688,
    "full_history_necessity/always_vs_never_t3__full_history": 0.0190868,
    "full_history_necessity/always_vs_never_t3__markov_history": 0.0193299,
    "full_history_necessity/rule_vs_never_t3__full_history": 0.0199951,
    "full_history_necessity/rule_vs_never_t3__markov_history": 0.0181201,
}

#: The control arm of each property family, with the limit it reads its bias around.
CONTROL_LIMITS: dict[str, tuple[str, Mapping[str, float]]] = {
    "double_robustness": (DOUBLE_ROBUSTNESS_ARMS[2], BOTH_WRONG_LIMIT),
    "correction_necessity": (CORRECTION_ARMS[1], PLUG_IN_LIMIT),
    "rule_necessity": (RULE_ARMS[1], REVERSED_TRUTH),
    "repeated_outcome_necessity": (ABSORBING_ARMS[1], ABSORBING_LIMIT),
    "outcome_history_necessity": (HISTORY_FREE_ARMS[1], HISTORY_FREE_LIMIT),
    "full_history_necessity": (FULL_HISTORY_ARMS[1], MARKOV_LIMIT),
}


def bias_target(family: str, cell: str) -> float:
    """The value a family's cell reads its bias around: the truth, or a control's limit."""
    name, arm = cell.split("__", 1)
    control, limits = CONTROL_LIMITS[family]
    if arm == control:
        return limits[name]
    return TRUTH[label_estimand(name)]


# ---------------------------------------------------------------------------- sizing rules


def _bias_group(
    biases: Mapping[str, float], pilot_replicates: int, correlations: sizing.Correlations
) -> sizing.PilotGroup:
    names = list(biases)
    return sizing.PilotGroup(
        np.array([biases[name] for name in names]),
        correlations(sizing.cell_labels(names)) / pilot_replicates,
    )


def primary_groups(biases: Mapping[str, float], pilot_replicates: int) -> list[sizing.PilotGroup]:
    """One bias group for each primary scenario, from the ``scenario/label`` biases."""
    return list(sizing.property_bias_groups(biases, pilot_replicates, label_correlations).values())


def primary_replicates(
    covered: Mapping[str, int], biases: Mapping[str, float], pilot_replicates: int
) -> int:
    """The declared rule for ``PRIMARY_REPLICATES``: two implementations, clauses (a) to (d).

    The shares are :data:`~tests.studies.repeated_sizing.PRIMARY_CLAUSE_SHARE` for the coverage
    clause and for the bias clause, which reads the robust probability of the pilot biases.  The
    lmtp rows of the later change fit the same replications, so the coverage clause counts two
    implementations.
    """
    return sizing.primary_replicates(
        MARGINS,
        covered,
        pilot_replicates,
        implementations=2,
        coverage_pass=sizing.PRIMARY_CLAUSE_SHARE,
        bias_groups=primary_groups(biases, pilot_replicates),
        bias_pass=sizing.PRIMARY_CLAUSE_SHARE,
    )


def primary_n_rule_fires(biases: Mapping[str, float], pilot_replicates: int) -> bool:
    """Whether the primary ``n`` doubles: the robust bias statistic is below the clause share."""
    return sizing.n_rule_fires(
        MARGINS, primary_groups(biases, pilot_replicates), sizing.PRIMARY_CLAUSE_SHARE
    )


def calibration_size(
    ratios: Mapping[str, float],
    biases: Mapping[str, float],
    pilot_replicates: int,
    joint_coverage: float,
) -> sizing.CalibrationSize:
    """The declared rule for ``CALIBRATION_REPLICATES``, with its cap.

    Each arm is a term of the union bound, read with the pooled reading: C1, C3, C6 and C5, the
    last one-sided.  The C4 joint band is the fifth term.  The coverage half reads the pilot
    bias.
    """
    return sizing.calibration_size(
        MARGINS,
        ratios,
        pilot_replicates,
        ARM_CORRELATIONS,
        one_sided_arms=ONE_SIDED_ARMS,
        biases=biases,
        joint_coverage=joint_coverage,
    )


def calibration_bias_groups(
    biases: Mapping[str, float], pilot_replicates: int
) -> dict[str, sizing.PilotGroup]:
    """One bias group for each calibration arm, from the ``label__arm`` biases."""
    groups = {}
    for arm, correlations in ARM_CORRELATIONS.items():
        selected = {name: value for name, value in biases.items() if name.endswith(f"__{arm}")}
        groups[arm] = _bias_group(selected, pilot_replicates, correlations)
    return groups


def c6_rule_fires(biases: Mapping[str, float], pilot_replicates: int) -> bool:
    """Whether the C6 ``n`` doubles: the coverage half at the cap, read on the C6 biases.

    The other calibration arms feed the spread.  The share is :data:`C6_SHARE`, one of the five
    ``SDR`` calibration terms.
    """
    groups = calibration_bias_groups(biases, pilot_replicates)
    return sizing.c6_rule_fires(
        MARGINS,
        groups["continuous"],
        extra=[group for arm, group in groups.items() if arm != "continuous"],
        share=C6_SHARE,
    )


def property_groups(
    biases: Mapping[str, float], pilot_replicates: int
) -> dict[str, sizing.PilotGroup]:
    """One bias group for each property family."""
    return sizing.property_bias_groups(biases, pilot_replicates, label_correlations)


def property_replicates(biases: Mapping[str, float], pilot_replicates: int) -> dict[str, int]:
    """The declared rule for the replications of each property family.

    Each family takes the smallest multiple of 200 at which its pass probability, with
    :math:`\\tau` at its 80% bound read from every family together, reaches
    :func:`~tests.studies.repeated_sizing.family_share` for the six families.
    """
    return sizing.property_bias_replicates(
        MARGINS,
        biases,
        pilot_replicates,
        label_correlations,
        per_family=sizing.family_share(len(PROPERTY_FAMILIES)),
    )


def property_n_rule_fires(biases: Mapping[str, float], pilot_replicates: int, family: str) -> bool:
    """Whether the ``n`` of ``family`` doubles (N1).

    The family is the judged group.  The other families of the study feed the spread.
    """
    groups = property_groups(biases, pilot_replicates)
    return sizing.n_rule_fires(
        MARGINS,
        [groups[family]],
        sizing.family_share(len(groups)),
        extra=[group for name, group in groups.items() if name != family],
    )


def declared_sizes() -> dict[str, Any]:
    """Every size and rule verdict that the pilot tables give, for the fast test to compare."""
    sizes = property_replicates(PROPERTY_PILOT_BIASES, PROPERTY_PILOT_REPLICATES)
    calibration = calibration_size(
        CALIBRATION_PILOT_RATIOS,
        CALIBRATION_PILOT_BIASES,
        CALIBRATION_PILOT_REPLICATES,
        CALIBRATION_PILOT_JOINT_COVERAGE,
    )
    return {
        "primary_replicates": primary_replicates(
            PRIMARY_PILOT_COVERED, PRIMARY_PILOT_BIASES, PRIMARY_PILOT_REPLICATES
        ),
        "calibration_replicates": calibration.replicates,
        "calibration_budget_met": calibration.budget_met,
        **sizes,
        "primary_n_rule": primary_n_rule_fires(PRIMARY_PILOT_BIASES, PRIMARY_PILOT_REPLICATES),
        "c6_n_rule": c6_rule_fires(CALIBRATION_PILOT_BIASES, CALIBRATION_PILOT_REPLICATES),
        "n1_n_rule": property_n_rule_fires(
            PROPERTY_PILOT_BIASES, PROPERTY_PILOT_REPLICATES, "repeated_outcome_necessity"
        ),
    }


# ---------------------------------------------------------------------------- pilots
#
# Every pilot draws from ``stream_seed(STUDY, "pilot", ...)``, which no published cell reads.
# The sample size is part of the stream label, so a pilot repeated at a doubled ``n`` draws
# fresh seeds.  ``tests/canonical/sdr_repeated/pilot.py`` runs these tasks in resumable parts.


def primary_pilot_task(payload: tuple[str, int, int]) -> list[dict[str, Any]]:
    """P1: one primary fit on the pilot stream, one row for each label."""
    scenario, replicate, n = payload
    frame, _ = make_longitudinal_repeated(
        n=n,
        seed=stream_seed(STUDY, "pilot", "primary", scenario, n, replicate),
        censoring=True,
        backend="pandas",
    )
    result = fit_cleverly(frame, folds=SCENARIO_FOLDS[scenario])
    rows = []
    for name in LABELS:
        estimand = label_estimand(name)
        estimate = result[estimand]
        low, high = estimate.ci
        rows.append(
            {
                "scenario": scenario,
                "replicate": replicate,
                "label": name,
                "n": n,
                "estimate": float(estimate.psi),
                "std_error": float(estimate.std_error),
                "covered": int(low <= TRUTH[estimand] <= high),
            }
        )
    return rows


def calibration_pilot_task(payload: tuple[str, int, int]) -> list[dict[str, Any]]:
    """P2: one calibration fit on the pilot stream: the property rows of the arm."""
    arm, replicate, n = payload
    seed = stream_seed(STUDY, "pilot", "interval_calibration", arm, n, replicate)
    return _replication(("interval_calibration", arm, replicate, 0, seed, n))


def family_configurations(family: str) -> tuple[str, ...]:
    """The configurations a P3 task of ``family`` runs.

    The double-robustness family draws each of its three arms from its own stream.  The other
    families are one paired fit.
    """
    return DOUBLE_ROBUSTNESS_ARMS if family == "double_robustness" else ("paired",)


def property_pilot_task(payload: tuple[str, int, int]) -> list[dict[str, Any]]:
    """P3: one replication of a property family on the pilot stream."""
    family, replicate, n = payload
    rows: list[dict[str, Any]] = []
    for configuration in family_configurations(family):
        seed = stream_seed(STUDY, "pilot", family, configuration, n, replicate)
        rows.extend(_replication((family, configuration, replicate, 0, seed, n)))
    return rows


#: The units of each pilot.  A unit is one fit set the driver can run, re-run or override alone.
PILOT_UNITS: dict[str, tuple[str, ...]] = {
    "P1": SCENARIOS,
    "P2": ("in_sample", "crossfit", "continuous", BOOSTER_ARM),
    "P3": PROPERTY_FAMILIES,
}


def pilot_declared_n(pilot: str, unit: str) -> int:
    """The sample size the study declares now for a pilot unit.

    A pilot row is current only when it was drawn at this size.  The ``n`` rules change the
    constants this reads, and a re-pilot then draws fresh seeds because ``n`` is in the stream.
    """
    if pilot == "P1":
        return PRIMARY_N
    if pilot == "P2":
        return fit_n("interval_calibration", unit)
    return family_n(unit)


def pilot_recorded_n(pilot: str, unit: str) -> int:
    """The sample size the pinned pilot table of a unit was drawn at."""
    if pilot == "P1":
        return PRIMARY_PILOT_N
    if pilot == "P2":
        return CALIBRATION_PILOT_N[unit]
    return PROPERTY_PILOT_N[unit]


def _cell_stats(group: pd.DataFrame, target: float) -> dict[str, float]:
    """Everything plan 5.7 reads of one pilot cell: count, covered, SE ratio, bias, spread."""
    estimates = group["estimate"].to_numpy(dtype=float)
    spread = float(estimates.std(ddof=1))
    return {
        "draws": float(len(group)),
        "covered": float(group["covered"].sum()),
        "ratio": float(group["std_error"].mean() / spread),
        "bias": float((estimates.mean() - target) / spread),
        "spread": spread,
    }


def primary_cell_table(rows: pd.DataFrame) -> pd.DataFrame:
    """P1: the statistics of each ``scenario/label`` cell."""
    table = {}
    for scenario in SCENARIOS:
        for name in LABELS:
            group = rows.loc[(rows["scenario"] == scenario) & (rows["label"] == name)]
            table[f"{scenario}/{name}"] = _cell_stats(group, TRUTH[label_estimand(name)])
    return pd.DataFrame.from_dict(table, orient="index")


def calibration_cell_table(rows: pd.DataFrame) -> pd.DataFrame:
    """P2: the statistics of each calibration cell."""
    calibration = rows.loc[rows["property"] == "interval_calibration"]
    table = {}
    for cell in CALIBRATION_CELLS:
        group = calibration.loc[calibration["cell"] == cell]
        if group.empty:
            raise ValueError(f"the calibration pilot has no rows for {cell}")
        table[cell] = _cell_stats(group, float(group["truth"].iloc[0]))
    return pd.DataFrame.from_dict(table, orient="index")


def property_cell_table(rows: pd.DataFrame) -> pd.DataFrame:
    """P3: the statistics of each ``family/cell`` around its target."""
    table = {}
    for family, names in property_cells().items():
        for key in names:
            cell = key.split("/", 1)[1]
            group = rows.loc[(rows["property"] == family) & (rows["cell"] == cell)]
            if group.empty:
                raise ValueError(f"the property pilot has no rows for {key}")
            table[key] = _cell_stats(group, bias_target(family, cell))
    return pd.DataFrame.from_dict(table, orient="index")


def pilot_row_sizes(pilot: str, rows: pd.DataFrame) -> dict[str, set[int]]:
    """The sample sizes found in pilot rows, by unit.  Rows with no ``n`` column give none."""
    if "n" not in rows.columns:
        return {}
    if pilot == "P1":
        return {str(u): {int(v) for v in g["n"]} for u, g in rows.groupby("scenario")}
    if pilot == "P3":
        return {str(u): {int(v) for v in g["n"]} for u, g in rows.groupby("property")}
    unit = rows["cell"].astype(str).str.rsplit("__", n=1).str[-1]
    unit = unit.where(unit.isin(PILOT_UNITS["P2"]), "in_sample")
    return {str(u): {int(v) for v in g["n"]} for u, g in rows.groupby(unit)}


def summarize_primary_pilot(rows: pd.DataFrame) -> tuple[dict[str, int], dict[str, float], int]:
    """P1: the covered count and standardized bias of each ``scenario/label`` cell.

    Returns ``(covered, biases, replicates)``.  The bias is the mean error in empirical
    standard deviations of the estimate.
    """
    table = primary_cell_table(rows)
    counts = set(table["draws"].astype(int))
    if len(counts) != 1:
        raise ValueError(f"the primary pilot cells hold different counts: {sorted(counts)}")
    return (
        {key: int(value) for key, value in table["covered"].items()},
        {key: float(value) for key, value in table["bias"].items()},
        counts.pop(),
    )


def summarize_calibration_pilot(rows: pd.DataFrame) -> dict[str, Any]:
    """P2: the SE ratio and standardized bias of each calibration cell, and the C4 coverage.

    Returns a mapping with ``ratios``, ``biases``, ``covered``, ``spreads``,
    ``joint_coverage`` and ``replicates`` (the smallest count over the arms).
    """
    table = calibration_cell_table(rows)
    band = rows.loc[
        (rows["property"] == "simultaneous_coverage")
        & (rows["cell"] == f"{JOINT_LABEL}__simultaneous_band")
    ]
    return {
        "ratios": {key: float(v) for key, v in table["ratio"].items()},
        "biases": {key: float(v) for key, v in table["bias"].items()},
        "covered": {key: int(v) for key, v in table["covered"].items()},
        "spreads": {key: float(v) for key, v in table["spread"].items()},
        "joint_coverage": float(band["covered"].mean()),
        "replicates": int(table["draws"].min()),
    }


def summarize_property_pilot(rows: pd.DataFrame) -> tuple[dict[str, float], int]:
    """P3: the standardized bias of each ``family/cell`` around its target.

    Returns ``(biases, replicates)`` with the smallest count over the cells.
    """
    table = property_cell_table(rows)
    return (
        {key: float(value) for key, value in table["bias"].items()},
        int(table["draws"].min()),
    )


def apply_pilot_rules(
    *,
    primary: tuple[dict[str, int], dict[str, float], int] | None = None,
    calibration: Mapping[str, Any] | None = None,
    properties: tuple[dict[str, float], int] | None = None,
) -> dict[str, Any]:
    """The declared rules applied to summarized pilots: the sizes and the ``n`` rule verdicts."""
    out: dict[str, Any] = {}
    if primary is not None:
        covered, biases, replicates = primary
        out["primary_n_rule_fires"] = primary_n_rule_fires(biases, replicates)
        out["primary_replicates"] = primary_replicates(covered, biases, replicates)
    if calibration is not None:
        size = calibration_size(
            calibration["ratios"],
            calibration["biases"],
            calibration["replicates"],
            calibration["joint_coverage"],
        )
        out["calibration_replicates"] = size.replicates
        out["calibration_budget_met"] = size.budget_met
        out["c6_n_rule_fires"] = c6_rule_fires(calibration["biases"], calibration["replicates"])
    if properties is not None:
        biases, replicates = properties
        out["property_replicates"] = property_replicates(biases, replicates)
        out["n1_n_rule_fires"] = property_n_rule_fires(
            biases, replicates, "repeated_outcome_necessity"
        )
    return out


def time_fit_sets(seed: int = 0, n: int | None = None) -> dict[tuple[str, str], float]:
    """Seconds for one replication of each fit set, single process, for the budget table.

    ``n`` overrides the family size, for a smoke run.
    """
    seconds: dict[tuple[str, str], float] = {}
    for family, configuration, _ in FIT_SETS:
        size = fit_n(family, configuration) if n is None else n
        start = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _replication((family, configuration, 0, 1, seed, size))
        seconds[family, configuration] = time.perf_counter() - start
    return seconds


def declared_truths() -> Mapping[str, Mapping[str, float]]:
    """Every truth and control limit a cell reads, by kind."""
    return {
        "binary": {name: TRUTH[label_estimand(name)] for name in LABELS},
        "continuous": CONTINUOUS_TRUTH,
        "both_wrong": BOTH_WRONG_LIMIT,
        "plug_in": PLUG_IN_LIMIT,
        "reversed": REVERSED_TRUTH,
        "absorbing": ABSORBING_LIMIT,
        "history_free": HISTORY_FREE_LIMIT,
        "markov": MARKOV_LIMIT,
    }
