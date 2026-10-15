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
                                ``n = 8,000``, :data:`CALIBRATION_REPLICATES`.  The
                                derived ``shrunken_se_control`` and ``noise_control`` arms
                                read these rows (twenty-six controls)
C3 ``interval_calibration``     ``*_t3__crossfit``: C1 at five folds, five labels
C6 ``interval_calibration``     ``*_t3__continuous``: the three-value law
                                :data:`~tests.studies.ltmle_repeated_law.CONTINUOUS`,
                                ``q_bounds = (0, 1)``, in sample, five labels, ``n =``
                                :data:`CONTINUOUS_N`
C5 ``interval_calibration``     ``*_t3__crossfit_boosted``: five folds, a shallow
                                regularized booster for both regressions.  Gated one-sided
                                (plan 5.6): the SE ratio's 99% lower bound is at least the
                                lower band end, and so is the coverage interval's lower
                                end.  The two-sided result is the ``two_sided_passed``
                                column
C4 ``simultaneous_coverage``    ``all_reported``: the default band of the C1 fits over all
                                fifteen reported parameters, and its pointwise control
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
P2 C1, C3, C5, C6 (and C4) 8,000 each   ``CALIBRATION_REPLICATES`` or the cap, the C6 rule
P3 every property family   2,000 each   each family's size, and the N1 ``n`` rule
=========================  ===========  ====================================================

The union bound gives the discrimination checks 0.001 and each of the three sized instruments
:math:`d = 0.049 / 3`.  The calibration instrument has five terms: C1, C3, C5, C6 and C4.  The
property instrument has six families.  The study fails with probability at most 0.05.

While :data:`~tests.studies.canonical_sdr_repeated.PILOT_STATE` is ``"placeholder"``, the
tables below hold declared uniform planning values (SE ratio 1, standardized bias 0, coverage
0.95), and the sizes are the rules applied to them.  They are planning sizes, not results.  A
declared run is refused.  The pilot driver (``tests/canonical/sdr_repeated/pilot.py``) writes
the pilot rows in resumable parts, and :func:`summarize_primary_pilot`,
:func:`summarize_calibration_pilot` and :func:`summarize_property_pilot` turn them into these
tables.
"""

from __future__ import annotations

import functools
import time
import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.dummy import DummyClassifier

from cleverly.datasets import make_longitudinal_repeated
from cleverly.longitudinal import SDR, DynamicRegimen
from cleverly.utils.parallel import map_parallel
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
from tests.studies.evidence.properties import (
    REPLICATE_COLUMNS,
    control_row,
    replicate_row,
    se_ratio_interval,
    summary_interval,
)
from tests.studies.evidence.property_verdicts import (
    DIAGNOSTIC_ROLE,
    alternative_target_necessity_verdicts,
    alternative_target_verdicts,
    apply_shared_verdicts,
    calibration_controls,
    calibration_verdicts,
    finish,
    simultaneous_coverage_verdicts,
)
from tests.studies.evidence.seeds import stream_seed
from tests.studies.evidence.simultaneous import joint_coverage_rows
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
#: The sample size of the in-sample, cross-fit and booster ``interval_calibration`` arms.
CALIBRATION_N = 8_000
#: The sample size of the C6 arm (``continuous``).  It is declared apart from
#: :data:`CALIBRATION_N`, so the C6 ``n`` rule doubles this size alone.
CONTINUOUS_N_BASE = 8_000
#: How many times the C6 ``n`` rule has doubled :data:`CONTINUOUS_N` (0, 1 or 2).
CONTINUOUS_N_DOUBLINGS = 0
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
# Each size is the rule applied to the placeholder pilot tables of the section "pilot tables"
# below, until the pilots are recorded.  A fast test re-derives every one of them.

#: ``calibration_size`` on P2 (``CALIBRATION_BUDGET_MET`` is False when the rule reached the cap).
CALIBRATION_REPLICATES = 4_000
CALIBRATION_BUDGET_MET = True
#: ``property_replicates`` on P3 at the share ``family_share(6)``.
DOUBLE_ROBUST_REPLICATES = 800
CORRECTION_REPLICATES = 800
RULE_REPLICATES = 800
ABSORBING_REPLICATES = 800
HISTORY_FREE_REPLICATES = 800
FULL_HISTORY_REPLICATES = 800
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
    """The sample size a property family draws (the C6 arm: see :func:`fit_n`)."""
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
    """The sample size of one fit set: the C6 arm has its own size."""
    if family == "interval_calibration" and configuration == "continuous":
        return CONTINUOUS_N
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


def _replication(payload: tuple[str, str, int, int, int, int]) -> list[dict[str, Any]]:
    family, configuration, replicate, requested, seed, n = payload
    frame = sample(configuration, seed, n)
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
) -> list[tuple[tuple[str, str, int, int, int, int]]]:
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
                calibration_n=CALIBRATION_N,
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


def generate_property_rows(*, n_jobs: int = STUDY_JOBS) -> pd.DataFrame:
    """The declared property study.  Refused while the pilot tables are placeholders."""
    require_recorded_pilots("the declared property run")
    return assemble_property_rows(map_parallel(_replication, _payloads(), n_jobs=n_jobs))


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
        alternative_truths={f"{name}__both_wrong": BOTH_WRONG_LIMIT[name] for name in GATED_LABELS},
    )
    for family, labels, arms, limits, column, threshold in (
        (
            "correction_necessity",
            GATED_LABELS,
            CORRECTION_ARMS,
            PLUG_IN_LIMIT,
            "correction_displacement",
            CORRECTION_DISPLACEMENT,
        ),
        ("rule_necessity", RULE_LABELS, RULE_ARMS, REVERSED_TRUTH, None, NECESSITY_DISPLACEMENT),
        (
            "repeated_outcome_necessity",
            ABSORBING_LABELS,
            ABSORBING_ARMS,
            ABSORBING_LIMIT,
            None,
            NECESSITY_DISPLACEMENT,
        ),
        (
            "outcome_history_necessity",
            HISTORY_LABELS,
            HISTORY_FREE_ARMS,
            HISTORY_FREE_LIMIT,
            None,
            NECESSITY_DISPLACEMENT,
        ),
        (
            "full_history_necessity",
            GATED_LABELS,
            FULL_HISTORY_ARMS,
            MARKOV_LIMIT,
            None,
            NECESSITY_DISPLACEMENT,
        ),
    ):
        alternative_target_necessity_verdicts(
            summary,
            rows,
            STUDY,
            family=family,
            labels=labels,
            arms=arms,
            alternative_truths=limits,
            column=column or "necessity_displacement",
            threshold=threshold,
        )
    return finish(summary, rates)


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
# PLACEHOLDER while ``PILOT_STATE`` is "placeholder".  These are declared uniform planning
# values, not pilot results: every SE ratio is 1, every standardized bias is 0 and every
# coverage is 0.95.  They exist so that the sizing rules run end to end and the declared run is
# refused until the pilots replace them.  ``summarize_*_pilot`` produces the real tables.

#: Pilot P1 (primary): the covered count and the standardized bias of each ``scenario/label``
#: cell over :data:`PRIMARY_PILOT_REPLICATES` draws at ``n = PRIMARY_N`` from the pilot stream.
PRIMARY_PILOT_REPLICATES = 2_000
#: The sample size P1 ran at.  A recorded table must equal :data:`PRIMARY_N`.
PRIMARY_PILOT_N = 12_000
PRIMARY_PILOT_COVERED = {f"{scenario}/{name}": 1_900 for scenario in SCENARIOS for name in LABELS}
PRIMARY_PILOT_BIASES = {f"{scenario}/{name}": 0.0 for scenario in SCENARIOS for name in LABELS}
#: Mean reported standard error over the empirical spread of the estimate, by cell (reported).
PRIMARY_PILOT_RATIOS = {f"{scenario}/{name}": 1.0 for scenario in SCENARIOS for name in LABELS}

#: Pilot P2 (calibration): the SE ratio and the standardized bias of each gated calibration
#: cell over :data:`CALIBRATION_PILOT_REPLICATES` draws per arm, and the joint coverage of the
#: C4 band over the C1 fits.
CALIBRATION_PILOT_REPLICATES = 8_000
#: The sample size each P2 arm ran at.  A recorded table must equal the arm's declared size.
CALIBRATION_PILOT_N = {
    "in_sample": 8_000,
    "crossfit": 8_000,
    "continuous": 8_000,
    BOOSTER_ARM: 8_000,
}
CALIBRATION_CELLS = (
    *(f"{name}__in_sample" for name in LABELS),
    *(f"{name}__crossfit" for name in GATED_LABELS),
    *(f"{name}__continuous" for name in GATED_LABELS),
    *(f"{name}__{BOOSTER_ARM}" for name in GATED_LABELS),
)
CALIBRATION_PILOT_RATIOS = dict.fromkeys(CALIBRATION_CELLS, 1.0)
CALIBRATION_PILOT_BIASES = dict.fromkeys(CALIBRATION_CELLS, 0.0)
#: The covered count of each calibration cell over the pilot draws of its arm (reported).
CALIBRATION_PILOT_COVERED = dict.fromkeys(CALIBRATION_CELLS, 7_600)
#: The empirical spread of each calibration cell's estimate (reported).  The placeholder is the
#: efficiency bound over the root of the arm's size, which is the law's own spread.
CALIBRATION_PILOT_SPREADS = {
    cell: EFFICIENCY_SD[cell.split("__", 1)[0]] / 8_000**0.5 for cell in CALIBRATION_CELLS
}
CALIBRATION_PILOT_JOINT_COVERAGE = 0.95


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


def placeholder_spread(family: str, cell: str) -> float:
    """The law's own spread of a cell's estimate at the family's size, before any pilot.

    The positive arm and most controls estimate the label's parameter, so the efficient
    standard deviation of the label stands in.  The ``reversed_rule`` control estimates the
    reversed rule, and its own efficient standard deviation stands in.
    """
    name, arm = cell.split("__", 1)
    scale = REVERSED_EFFICIENCY_SD[name] if arm == RULE_ARMS[1] else EFFICIENCY_SD[name]
    return scale / PROPERTY_PILOT_N[family] ** 0.5


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
PROPERTY_PILOT_BIASES = {name: 0.0 for names in property_cells().values() for name in names}
PROPERTY_PILOT_RATIOS = {name: 1.0 for names in property_cells().values() for name in names}
PROPERTY_PILOT_COVERED = {name: 1_900 for names in property_cells().values() for name in names}
#: The empirical spread of each cell's estimate.  The discrimination test reads the gaps of the
#: controls in these spreads once the pilots are recorded.
PROPERTY_PILOT_SPREADS = {
    key: placeholder_spread(key.split("/", 1)[0], key.split("/", 1)[1])
    for names in property_cells().values()
    for key in names
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
