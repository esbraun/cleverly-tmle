r"""Independent repeated-sampling properties of repeated-outcome ``IPSI`` (X32b, PR A).

Every binary family samples :func:`cleverly.datasets.make_longitudinal_repeated` on its own
stream ``stream_seed(STUDY, "property_sample", family, configuration, replicate)``.  The
continuous arm samples :data:`~tests.studies.ltmle_repeated_law.CONTINUOUS`.  Every truth is
enumerated by :mod:`tests.studies.ltmle_repeated_law`.  The labels name the parameters:
``ipsi_0.5_t3`` is the mean of ``Y_3`` under the odds tilt 0.5, and ``ipsi_0.5_vs_1.0_t3`` is
its contrast with the natural course.  The five ``*_t3`` labels are the gated labels.

Declared before any run
-----------------------

==============================  =========================================================
family and cells                what is fitted, replications and size
==============================  =========================================================
C1 ``interval_calibration``     ``*__in_sample`` for all fifteen labels: in sample, the
                                primary learners, ``simultaneous=True``.  ``n = 8,000``,
                                :data:`CALIBRATION_REPLICATES` replications.  The derived
                                ``shrunken_se_control`` and ``noise_control`` arms read
                                these rows (thirty controls)
C3 ``interval_calibration``     ``*_t3__crossfit``: C1 at five folds, five labels
C6 ``interval_calibration``     ``*_t3__continuous``: the three-value law
                                :data:`~tests.studies.ltmle_repeated_law.CONTINUOUS`,
                                ``q_bounds = (0, 1)``, in sample, five labels
C4 ``simultaneous_coverage``    ``all_reported``: the default band of the C1 fits over all
                                fifteen reported parameters, and its pointwise control
C5 ``booster_calibration``      ``*_t3__crossfit_boosted``: five folds, a shallow
                                regularized booster for both outcome learners.  Role
                                ``diagnostic``: reported, never gated (plan 5.6).  2,000
                                replications
``small_sample_calibration``    C1 and C6 refitted at ``n = 2,000``, five labels.  Role
                                ``diagnostic``.  2,000 replications
``single_robustness``           ``{ipsi_0.5, ipsi_2.0}_t3`` and their contrasts with the
                                natural course, ``n = 2,000``.  Positive arm
                                ``w_only_m``: the outcome regression reads ``W`` and the
                                current arm alone (wrong), the mechanisms are the
                                correctly specified GLM.  The limit is the truth.  Control
                                ``one_step_necessity``: the tilt plug-in with the same
                                ``W``-only regression and the same fitted propensity,
                                written without cleverly (:func:`tilt_plug_ins`).  It
                                must miss the truth and reach :data:`ONE_STEP_LIMIT`
N1 ``repeated_outcome_neces     ``ipsi_{0.5,1.0,2.0}_t{2,3}__repeated`` against
sity``                          ``__absorbing_reading``, an end-of-study ``IPSI`` fit of
                                :math:`\max_{s \le h} Y_s` with the earlier outcomes in the
                                history.  The mechanisms are correct, so by single robustness
                                the limit is the absorbing tilt functional whatever the
                                regression is (:data:`ABSORBING_LIMIT`).  ``n`` is
                                :data:`ABSORBING_N`.  The ``n`` rule applies
N2 ``outcome_history_necessi    ``ipsi_{0.5,2.0}_t3__repeated`` against ``__history_free``,
ty``                            an end-of-study fit with every ``Y_s`` left out of every
                                history.  Cell means for the regression, the treatment
                                mechanism and the censoring mechanism on the coarse key, so
                                every nuisance is the coarse projection and the limit is the
                                coarse tilt functional (:data:`HISTORY_FREE_LIMIT`).  ``n``
                                is :data:`HISTORY_FREE_N`.  The ``n`` rule applies
==============================  =========================================================

There is no mixed double-robustness family.  ``IPSI`` is singly robust: it needs ``pi`` and
``omega``, and a wrong ``pi`` with a correct ``m`` leaves a bias (witness ``E7``).  There is no
N3 family either.  Markov outcome regressions leave the limit at the truth when ``pi`` and
``omega`` are correct, so the family would measure nothing.

Gating, declared before any run:

* the SE-ratio band and the coverage band gate every C1 cell, at every horizon.  The C3 and C6
  arms gate the ``t = 3`` labels, and C6 also answers to the coverage-keyed ``n`` rule;
* every other positive cell answers to its family's bias rule, and each control to the opposite
  direction of the same instrument.  A control also answers to its own limit;
* a red row is diagnosed.  A defect is fixed, and a finite-sample shortfall gets the ``n`` that a
  pilot shows removes it.  Either change is re-declared and re-run on fresh seeds.  No red row
  is deferred.

Sizes
-----

Every size is a rule of :mod:`tests.studies.repeated_sizing` applied to a pilot table of this
module.  The pilots draw from ``stream_seed(STUDY, "pilot", ...)``, which no published cell
reads.

=========================  ===========  ====================================================
pilot                      draws        sets
=========================  ===========  ====================================================
P4 primary, two scenarios  2,000 each   the primary ``n`` rule and ``PRIMARY_REPLICATES``
P5 C1, C3, C6 (and C4)     8,000 each   ``CALIBRATION_REPLICATES`` or the cap, the C6 rule
P6 every property family   2,000 each   each family's size, and the N1 and N2 ``n`` rules
=========================  ===========  ====================================================

While :data:`~tests.studies.canonical_ipsi_repeated.PILOT_STATE` is ``"placeholder"``, the
tables below hold declared uniform planning values (SE ratio 1, standardized bias 0, coverage
0.95), and the sizes are the rules applied to them.  They are planning sizes, not results.
A declared run is refused.  The pilot driver (``tests/canonical/ipsi_repeated/pilot.py``) writes
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
from sklearn.base import BaseEstimator, clone
from sklearn.utils.metaestimators import available_if

from cleverly.datasets import make_longitudinal_repeated
from cleverly.longitudinal import IPSI
from cleverly.utils.parallel import map_parallel
from tests.discrete_law_longitudinal import CellMeans
from tests.parallel import STUDY_JOBS
from tests.studies import ltmle_repeated_law as law
from tests.studies import repeated_sizing as sizing
from tests.studies.canonical_ipsi_repeated import (
    ABSORBING_ARMS,
    ABSORBING_LABELS,
    BOOSTER_ARM,
    COLUMNS,
    GATED_LABELS,
    HISTORY_FREE_ARMS,
    HISTORY_FREE_LABELS,
    JOINT_LABEL,
    LABELS,
    REFERENCE,
    REFERENCE_DELTA,
    SCENARIO_FOLDS,
    SCENARIOS,
    SINGLE_ROBUSTNESS_ARMS,
    SINGLE_ROBUSTNESS_LABELS,
    STUDY,
    TRUTH,
    fit_cleverly,
    label_estimand,
    label_parts,
    plans,
    require_recorded_pilots,
)
from tests.studies.canonical_ltmle import G_BOUNDS
from tests.studies.evidence.properties import (
    REPLICATE_COLUMNS,
    control_row,
    replicate_row,
    se_ratio_interval,
)
from tests.studies.evidence.property_verdicts import (
    DIAGNOSTIC_ROLE,
    alternative_target_necessity_verdicts,
    apply_shared_verdicts,
    calibration_controls,
    calibration_verdicts,
    finish,
    simultaneous_coverage_verdicts,
)
from tests.studies.evidence.seeds import stream_seed
from tests.studies.evidence.simultaneous import joint_coverage_rows
from tests.studies.fractional_glm import QuasiBinomialGLM
from tests.studies.ltmle_repeated_properties import CellMeansBackoff, booster

MARGINS = STUDY.margins
N = 2_000
#: The sample size of every ``interval_calibration`` arm and of the booster arm.
CALIBRATION_N = 8_000
#: The sample size of N1 and N2.  The ``n`` rule may double them, twice at most.
ABSORBING_N = 8_000
HISTORY_FREE_N = 8_000
#: The reported arms draw this many replications.
SMALL_SAMPLE_REPLICATES = 2_000
BOOSTER_REPLICATES = 2_000
FOLDS = 5
SHRUNKEN_SE_FACTOR = 0.70
#: The least paired displacement of a necessity family, in the positive arm's empirical
#: standard deviations: the standardized-bias margin.
NECESSITY_DISPLACEMENT = 0.25
CRITICAL = float(norm.ppf(1.0 - MARGINS.alpha / 2.0))
#: The configurations whose outcome regression reads ``W`` and the current arm alone.
BASELINE_CONFIGURATIONS = frozenset({"w_only_m"})

# ---------------------------------------------------------------------------- declared sizes
#
# Each size is the rule applied to the pilot tables of the section "pilot tables" below.  A fast
# test re-derives every one of them and states which pilot state it reads.

#: ``calibration_size`` on P5 (``CALIBRATION_BUDGET_MET`` is False when the rule reached the cap).
CALIBRATION_REPLICATES = 4_000
CALIBRATION_BUDGET_MET = True
#: ``property_bias_replicates`` on P6 at the share ``family_share(3)``.
SINGLE_ROBUSTNESS_REPLICATES = 600
ABSORBING_REPLICATES = 800
HISTORY_FREE_REPLICATES = 800

# ---------------------------------------------------------------------------- label parts


def _plan_value(
    model: law.RepeatedOutcomeLaw, name: str, method: str = "tilt_functional", **options: Any
) -> float:
    """A label's value on ``model``: a mean, or the contrast of two means."""
    delta, reference, horizon = label_parts(name)
    value = float(getattr(model, method)(delta, horizon, **options))
    if reference is not None:
        value -= float(getattr(model, method)(reference, horizon, **options))
    return value


def efficiency_sd(model: law.RepeatedOutcomeLaw, name: str) -> float:
    """The standard deviation of the influence curve of a label (Kim, Theorem 4.1)."""
    delta, reference, horizon = label_parts(name)
    return model.ipsi_eif_sd(delta, horizon, reference=reference)


CONTINUOUS_TRUTH = {
    name: _plan_value(law.CONTINUOUS, name, "tilt_counterfactual") for name in GATED_LABELS
}
#: The limit of the single-fold ``IPSI`` fit whose outcome regression reads ``W`` and the arm
#: and whose mechanisms are correct.  The one-step estimate reaches the truth (exact), and
#: the plug-in with the same regression does not: the ``one_step_necessity`` control.
ONE_STEP_LIMIT = {
    name: _plan_value(law.BINARY, name, "ipsi_plug_in_limit") for name in SINGLE_ROBUSTNESS_LABELS
}
#: The limit of a fit that reads the outcome as absorbing: the N1 control.
ABSORBING_LIMIT = {
    name: _plan_value(law.BINARY, name, "tilt_functional", absorbing=True)
    for name in ABSORBING_LABELS
}
#: The limit of a fit whose nuisances are correct for the history without ``Y_s``: the N2 control.
HISTORY_FREE_LIMIT = {
    name: _plan_value(law.BINARY, name, "tilt_functional", history="free")
    for name in HISTORY_FREE_LABELS
}
#: The efficient influence curve's standard deviation of each label.  It scales the noise
#: control.  The study claims no efficiency, so no efficiency band reads it.
EFFICIENCY_SD = {name: efficiency_sd(law.BINARY, name) for name in LABELS}

# ---------------------------------------------------------------------------- learners


class BaselineAndArm(BaseEstimator):
    """``learner`` fitted on ``W`` and the current arm alone.

    The design of an ``IPSI`` outcome regression is ``[W, Y1, L2, ..., Y_{t-1}, L_t,
    A_1, ..., A_t]`` (the earlier arms and the current arm follow the covariate history).  The
    first column is ``W`` and the last is ``A_t``.  A regression on both is wrong for the
    history, and it still lets the tilt weight the two arms, so the correction terms move.

    Parameters
    ----------
    learner : estimator
        The learner fitted on the two columns.  ``predict_proba`` is offered only when the
        learner has it.
    """

    def __init__(self, learner: Any) -> None:
        self.learner = learner

    @staticmethod
    def _select(X: Any) -> np.ndarray:
        matrix = np.asarray(X, dtype=float)
        return matrix[:, [0, matrix.shape[1] - 1]]

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> BaselineAndArm:
        self.fitted_ = clone(self.learner).fit(self._select(X), y, sample_weight=sample_weight)
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict(self, X: Any) -> np.ndarray:
        return np.asarray(self.fitted_.predict(self._select(X)))

    @available_if(lambda self: hasattr(self.learner, "predict_proba"))
    def predict_proba(self, X: Any) -> np.ndarray:
        return np.asarray(self.fitted_.predict_proba(self._select(X)))


def _learners(configuration: str) -> tuple[Any, Any, Any, Any]:
    """``(outcome, pseudo, treatment, censoring)`` learners of a configuration."""
    if configuration == BOOSTER_ARM:
        q, pseudo = booster("classifier"), booster("regressor")
    elif configuration in BASELINE_CONFIGURATIONS:
        q, pseudo = BaselineAndArm(CellMeans()), BaselineAndArm(CellMeans())
    else:
        q, pseudo = CellMeansBackoff(), CellMeansBackoff()
    return q, pseudo, QuasiBinomialGLM(), QuasiBinomialGLM()


def fit(
    frame: pd.DataFrame,
    configuration: str = "in_sample",
    *,
    simultaneous: bool = False,
    horizons: Sequence[int] | None = None,
) -> Any:
    """The repeated-outcome fit of one configuration."""
    q, pseudo, g, c = _learners(configuration)
    folds = FOLDS if configuration in {"crossfit", BOOSTER_ARM} else 1
    return IPSI(
        plans(),
        reference=REFERENCE,
        horizons=horizons,
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
    reads.  Only the outcome is the absorbing reading.  The mechanisms are correct, so the
    limit is the absorbing tilt functional whatever the outcome regression is.
    """
    panel = frame.copy()
    outcomes = panel[[f"Y{s}" for s in range(1, horizon + 1)]]
    panel["M"] = outcomes.max(axis=1, skipna=False)
    q, pseudo, g, c = _learners("in_sample")
    blocks: list[list[str]] = [[], *([f"Y{s - 1}", f"L{s}"] for s in range(2, horizon + 1))]
    return IPSI(
        plans(),
        reference=REFERENCE,
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

    Cell means for the regression and both mechanisms on the coarse key
    ``[W, L_2, ..., L_t]`` and the earlier arms: at most 96 cells at the third node.  Every
    nuisance is then the coarse projection, so the limit is the coarse tilt functional exactly.
    A coarse regression with GLM mechanisms would not be enough, because ``IPSI`` is not
    doubly robust.
    """
    return IPSI(
        [plan for plan in plans() if plan.delta != REFERENCE_DELTA],
        outcome_learner=CellMeans(),
        pseudo_learner=CellMeans(),
        treatment_learner=CellMeans(),
        censoring_learner=CellMeans(),
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


def _end_of_study_name(name: str) -> str:
    """The name an end-of-study fit reports for a label: the horizon leaves the bracket."""
    return label_estimand(name).rsplit(" @ t=", 1)[0] + "]"


# ---------------------------------------------------------------------------- plug-in


def tilt_plug_ins(
    frame: pd.DataFrame,
    deltas: Sequence[float],
    horizon: int,
    *,
    weights: np.ndarray | None = None,
) -> dict[float, float]:
    r"""The tilt plug-in :math:`P \bar m_1` with **no** correction terms, for each tilt.

    Backward from ``horizon``: regress the carried value on ``W`` among the rows observed
    through :math:`t` with :math:`A_t = a`, for both arms, and carry
    :math:`\bar m_t = q_t m_t(\cdot, 1) + (1 - q_t) m_t(\cdot, 0)` with the tilt probability
    :math:`q_t = \delta\hat\pi_t / (\delta\hat\pi_t + 1 - \hat\pi_t)`.  :math:`\hat\pi_t` is a
    main-terms quasibinomial GLM on the history :math:`[W, A_1, Y_1, L_2, \ldots, L_t]`, fitted
    on the rows still observed before :math:`t`, as the study's primary fit does.  An unseen
    cell takes the mean over the rows observed through :math:`t`.

    Written longhand with ``numpy`` and ``pandas`` plus the GLM, and not by calling the
    estimator with the correction disabled, so the control is not a statement about a branch of
    the code it audits.  With ``weights`` the frame is a weighted law, which is how a fast test
    checks the function against :meth:`~tests.studies.ltmle_repeated_law.RepeatedOutcomeLaw.
    ipsi_plug_in_limit`.
    """
    filled = frame.fillna(0.0).reset_index(drop=True)
    n = len(filled)
    mass = np.full(n, 1.0 / n) if weights is None else np.asarray(weights, dtype=float)
    arms = {t: filled[f"A{t}"].to_numpy() for t in range(1, horizon + 1)}
    stays = {t: filled[f"C{t}"].to_numpy() == 1.0 for t in range(1, horizon + 1)}
    baseline = filled["W"].to_numpy()
    risk = {1: np.ones(n, dtype=bool)}
    for t in range(2, horizon + 1):
        risk[t] = risk[t - 1] & stays[t - 1]
    pi: dict[int, np.ndarray] = {}
    for t in range(1, horizon + 1):
        design = filled[law.ipsi_history_columns(t, "full")].to_numpy(dtype=float)
        model = QuasiBinomialGLM().fit(
            design[risk[t]], arms[t][risk[t]], sample_weight=mass[risk[t]] * n
        )
        pi[t] = np.clip(np.asarray(model.predict(design), dtype=float), 0.0, 1.0)
    out: dict[float, float] = {}
    for delta in deltas:
        carried = filled[f"Y{horizon}"].to_numpy(dtype=float)
        for t in range(horizon, 0, -1):
            observed = risk[t] & stays[t]
            by_arm = []
            for arm_value in (0.0, 1.0):
                cell = observed & (arms[t] == arm_value)
                totals = (
                    pd.DataFrame({"num": mass * cell * carried, "den": mass * cell})
                    .groupby(baseline)[["num", "den"]]
                    .transform("sum")
                )
                fallback = np.sum(mass * observed * carried) / np.sum(mass * observed)
                with np.errstate(invalid="ignore", divide="ignore"):
                    mean = (totals["num"] / totals["den"]).to_numpy()
                by_arm.append(np.where(np.isnan(mean), fallback, mean))
            tilt = delta * pi[t] / (delta * pi[t] + 1.0 - pi[t])
            carried = tilt * by_arm[1] + (1.0 - tilt) * by_arm[0]
        out[float(delta)] = float(np.sum(mass * carried))
    return out


def one_step_plug_in(frame: pd.DataFrame, name: str) -> float:
    """The ``one_step_necessity`` control: the plug-in of a label, or of its contrast."""
    delta, reference, horizon = label_parts(name)
    deltas = (delta,) if reference is None else (delta, reference)
    plug = tilt_plug_ins(frame, deltas, horizon)
    value = plug[delta]
    if reference is not None:
        value -= plug[reference]
    return value


def fallback_mass(frame: pd.DataFrame, delta: float, time_point: int = 3) -> float:
    r"""The share of tilt-weighted reads at node ``time_point`` that meet an unseen outcome cell.

    A row at risk at node :math:`t` reads :math:`m_t(H_t, a)` at both arms with weight
    :math:`q_t(a \mid H_t)`.  The cell :math:`(H_t, a)` is unseen when no row observed through
    :math:`t` has that history and arm.  An unseen cell makes :class:`CellMeansBackoff` read
    its GLM.  :math:`\hat\pi_t` is the empirical share of treated rows in the history cell.
    P5 reports this for the continuous arm (plan 5.4, C6).
    """
    filled = frame.fillna(0.0).reset_index(drop=True)
    columns = law.ipsi_history_columns(time_point, "full")
    at_risk = np.ones(len(filled), dtype=bool)
    for s in range(1, time_point):
        at_risk &= filled[f"C{s}"].to_numpy() == 1.0
    seen = at_risk & (filled[f"C{time_point}"].to_numpy() == 1.0)
    treated = filled[f"A{time_point}"].to_numpy()
    cells = filled.loc[at_risk, columns]
    keys = pd.MultiIndex.from_frame(cells)
    share = pd.Series(treated[at_risk], index=keys).groupby(level=list(range(len(columns)))).mean()
    pi = share.reindex(keys).to_numpy()
    q = delta * pi / (delta * pi + 1.0 - pi)
    unseen: list[np.ndarray] = []
    for arm_value in (0.0, 1.0):
        observed = pd.MultiIndex.from_frame(filled.loc[seen & (treated == arm_value), columns])
        unseen.append(~keys.isin(observed))
    return float(np.mean(q * unseen[1] + (1.0 - q) * unseen[0]))


# ---------------------------------------------------------------------------- rows


def sample(configuration: str, seed: int, n: int = N) -> pd.DataFrame:
    """One draw of size ``n`` of the law a configuration reads."""
    if configuration == "continuous":
        return law.CONTINUOUS.sample(n, seed)
    frame, _ = make_longitudinal_repeated(n=n, seed=seed, censoring=True, backend="pandas")
    return frame


def family_n(family: str) -> int:
    """The sample size a property family draws."""
    if family in {"interval_calibration", "booster_calibration"}:
        return CALIBRATION_N
    if family == "repeated_outcome_necessity":
        return ABSORBING_N
    if family == "outcome_history_necessity":
        return HISTORY_FREE_N
    return N


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


def _replication(payload: tuple[str, str, int, int, int, int]) -> list[dict[str, Any]]:
    family, configuration, replicate, requested, seed, n = payload
    frame = sample(configuration, seed, n)
    rows: list[dict[str, Any]] = []
    if family in {"interval_calibration", "small_sample_calibration", "booster_calibration"}:
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
    if family == "single_robustness":
        result = fit(frame, "w_only_m", horizons=(3,))
        positive_arm, control_arm = SINGLE_ROBUSTNESS_ARMS
        for name in SINGLE_ROBUSTNESS_LABELS:
            estimand = label_estimand(name)
            rows.append(
                _row(
                    family,
                    f"{name}__{positive_arm}",
                    "positive",
                    replicate,
                    requested,
                    TRUTH[estimand],
                    result[estimand],
                    n,
                )
            )
            # The plug-in has no influence curve of its own, so the row carries the positive
            # fit's standard error and the family gates on bias alone.
            rows.append(
                control_row(
                    property_name=family,
                    cell=f"{name}__{control_arm}",
                    replicate=replicate,
                    n=n,
                    requested=requested,
                    truth=TRUTH[estimand],
                    estimate=one_step_plug_in(frame, name),
                    standard_error=float(result[estimand].std_error),
                    critical=CRITICAL,
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
        for name in HISTORY_FREE_LABELS:
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
    raise KeyError(family)


#: Every fit set: ``(family, configuration, replications)``.
FIT_SETS: tuple[tuple[str, str, int], ...] = (
    ("interval_calibration", "in_sample", CALIBRATION_REPLICATES),
    ("interval_calibration", "crossfit", CALIBRATION_REPLICATES),
    ("interval_calibration", "continuous", CALIBRATION_REPLICATES),
    ("booster_calibration", BOOSTER_ARM, BOOSTER_REPLICATES),
    ("small_sample_calibration", "in_sample", SMALL_SAMPLE_REPLICATES),
    ("small_sample_calibration", "continuous", SMALL_SAMPLE_REPLICATES),
    ("single_robustness", "paired", SINGLE_ROBUSTNESS_REPLICATES),
    ("repeated_outcome_necessity", "paired", ABSORBING_REPLICATES),
    ("outcome_history_necessity", "paired", HISTORY_FREE_REPLICATES),
)


def _payloads(
    fit_sets: tuple[tuple[str, str, int], ...] = FIT_SETS,
) -> list[tuple[tuple[str, str, int, int, int, int]]]:
    payloads = []
    for family, configuration, replicates in fit_sets:
        for replicate in range(replicates):
            seed = stream_seed(STUDY, "property_sample", family, configuration, replicate)
            payloads.append(
                ((family, configuration, replicate, replicates, seed, family_n(family)),)
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

#: The calibration families reported beside their SE-ratio interval and gated by nothing.
REPORTED_CALIBRATION = ("small_sample_calibration", "booster_calibration")


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


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    summary, rates = apply_shared_verdicts(
        rows, STUDY, extra_columns=("necessity_displacement",), rate_labels=()
    )
    # The C1 arm and its two derived controls read the instrument's rule for each kind.  The
    # C3 and C6 arms keep the two-sided rule ``apply_shared_verdicts`` wrote, which is the same
    # rule.
    primary = (summary["property"] == "interval_calibration") & ~summary["cell"].str.endswith(
        ("__crossfit", "__continuous")
    )
    subset = summary.loc[primary].copy()
    calibration_verdicts(subset, margins=MARGINS, positive_suffix="in_sample")
    summary.loc[subset.index, "passed"] = subset["passed"]

    for name in REPORTED_CALIBRATION:
        _calibration_report(summary, rows, name)
    simultaneous_coverage_verdicts(summary, margins=MARGINS)

    for family, labels, arms, limits in (
        (
            "single_robustness",
            SINGLE_ROBUSTNESS_LABELS,
            SINGLE_ROBUSTNESS_ARMS,
            ONE_STEP_LIMIT,
        ),
        ("repeated_outcome_necessity", ABSORBING_LABELS, ABSORBING_ARMS, ABSORBING_LIMIT),
        ("outcome_history_necessity", HISTORY_FREE_LABELS, HISTORY_FREE_ARMS, HISTORY_FREE_LIMIT),
    ):
        alternative_target_necessity_verdicts(
            summary,
            rows,
            STUDY,
            family=family,
            labels=labels,
            arms=arms,
            alternative_truths=limits,
            column="necessity_displacement",
            threshold=NECESSITY_DISPLACEMENT,
        )
    return finish(summary, rates)


# ---------------------------------------------------------------------------- label correlations


@functools.cache
def _curve(model: str, delta: float, horizon: int, reference: float | None) -> np.ndarray:
    laws = {"binary": law.BINARY, "continuous": law.CONTINUOUS}
    return laws[model].ipsi_eif(delta, horizon, reference=reference)


def _correlations(model: law.RepeatedOutcomeLaw) -> sizing.Correlations:
    def correlations(labels: Sequence[str]) -> np.ndarray:
        curves = []
        for name in labels:
            delta, reference, horizon = label_parts(name)
            curves.append(_curve(model.name, delta, horizon, reference))
        return sizing.influence_correlations(curves, model.paths["p"].to_numpy())

    return correlations


#: The correlation, under the binary law, of the Kim influence curves of labels.  Two estimates
#: of the labels from one sample have this correlation to first order.
label_correlations = _correlations(law.BINARY)
#: The same under the continuous law, which C6 reads.
continuous_label_correlations = _correlations(law.CONTINUOUS)
#: The correlation function of each calibration arm.
ARM_CORRELATIONS: dict[str, sizing.Correlations] = {
    "in_sample": label_correlations,
    "crossfit": label_correlations,
    "continuous": continuous_label_correlations,
}


def effective_cells(labels: Sequence[str]) -> float:
    r"""How many independent cells ``labels`` are worth when their estimates are averaged."""
    return sizing.effective_cells(labels, label_correlations)


# ---------------------------------------------------------------------------- pilot tables
#
# PLACEHOLDER while ``PILOT_STATE`` is "placeholder".  These are declared uniform planning
# values, not pilot results: every SE ratio is 1, every standardized bias is 0 and every
# coverage is 0.95.  They exist so that the sizing rules run end to end and the declared run is
# refused until the pilots replace them.  ``summarize_*_pilot`` produces the real tables.

#: Pilot P4 (primary): the covered count and the standardized bias of each ``scenario/label``
#: cell over :data:`PRIMARY_PILOT_REPLICATES` draws at ``n = PRIMARY_N`` from the pilot stream.
PRIMARY_PILOT_REPLICATES = 2_000
PRIMARY_PILOT_COVERED = {f"{scenario}/{name}": 1_900 for scenario in SCENARIOS for name in LABELS}
PRIMARY_PILOT_BIASES = {f"{scenario}/{name}": 0.0 for scenario in SCENARIOS for name in LABELS}

#: Pilot P5 (calibration): the SE ratio and the standardized bias of each gated calibration
#: cell over :data:`CALIBRATION_PILOT_REPLICATES` draws per arm at :data:`CALIBRATION_N`, and the
#: joint coverage of the C4 band over the C1 fits.
CALIBRATION_PILOT_REPLICATES = 8_000
CALIBRATION_CELLS = (
    *(f"{name}__in_sample" for name in LABELS),
    *(f"{name}__crossfit" for name in GATED_LABELS),
    *(f"{name}__continuous" for name in GATED_LABELS),
)
CALIBRATION_PILOT_RATIOS = dict.fromkeys(CALIBRATION_CELLS, 1.0)
CALIBRATION_PILOT_BIASES = dict.fromkeys(CALIBRATION_CELLS, 0.0)
CALIBRATION_PILOT_JOINT_COVERAGE = 0.95
#: The realized fallback-read mass of the C6 arm at the third node, by tilt, from P5.  Reported.
C6_FALLBACK_MASS: dict[str, float] = {}


def property_cells() -> dict[str, list[str]]:
    """The bias-checked cells of each property family: positive arm and control, by label."""
    pairs = (
        ("single_robustness", SINGLE_ROBUSTNESS_LABELS, SINGLE_ROBUSTNESS_ARMS),
        ("repeated_outcome_necessity", ABSORBING_LABELS, ABSORBING_ARMS),
        ("outcome_history_necessity", HISTORY_FREE_LABELS, HISTORY_FREE_ARMS),
    )
    return {
        family: [f"{family}/{name}__{arm}" for name in labels for arm in arms]
        for family, labels, arms in pairs
    }


#: Pilot P6 (property families): the standardized bias of each ``family/cell`` around its
#: target (the truth for a positive arm, the control's own limit for a control) over
#: :data:`PROPERTY_PILOT_REPLICATES` draws at the family's own size.
PROPERTY_PILOT_REPLICATES = 2_000
PROPERTY_PILOT_BIASES = {name: 0.0 for names in property_cells().values() for name in names}


def bias_target(family: str, cell: str) -> float:
    """The value a family's cell reads its bias around: the truth, or a control's limit."""
    name, arm = cell.split("__", 1)
    limits = {
        "single_robustness": ONE_STEP_LIMIT,
        "repeated_outcome_necessity": ABSORBING_LIMIT,
        "outcome_history_necessity": HISTORY_FREE_LIMIT,
    }[family]
    controls = {
        "single_robustness": SINGLE_ROBUSTNESS_ARMS[1],
        "repeated_outcome_necessity": ABSORBING_ARMS[1],
        "outcome_history_necessity": HISTORY_FREE_ARMS[1],
    }
    if arm == controls[family]:
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
    """The declared rule for ``PRIMARY_REPLICATES``: one implementation, clauses (a) to (d).

    The shares are :data:`~tests.studies.repeated_sizing.PRIMARY_CLAUSE_SHARE` for the coverage
    clause and for the bias clause, which reads the robust probability of the pilot biases.
    """
    return sizing.primary_replicates(
        MARGINS,
        covered,
        pilot_replicates,
        implementations=1,
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

    Each arm is a term of the union bound, read with the pooled reading.  The C4 joint band is
    one more term.  The coverage half reads the pilot bias.  C5 is reported, so it is not a term.
    """
    return sizing.calibration_size(
        MARGINS,
        ratios,
        pilot_replicates,
        ARM_CORRELATIONS,
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

    The bias pilots of C1 and C3 feed the spread.
    """
    groups = calibration_bias_groups(biases, pilot_replicates)
    return sizing.c6_rule_fires(
        MARGINS,
        groups["continuous"],
        extra=[groups["in_sample"], groups["crossfit"]],
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
    :func:`~tests.studies.repeated_sizing.family_share` for the three families.
    """
    return sizing.property_bias_replicates(
        MARGINS,
        biases,
        pilot_replicates,
        label_correlations,
        per_family=sizing.family_share(len(property_cells())),
    )


def property_n_rule_fires(biases: Mapping[str, float], pilot_replicates: int, family: str) -> bool:
    """Whether the ``n`` of ``family`` doubles (N1 or N2).

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
        "single_robustness": sizes["single_robustness"],
        "repeated_outcome_necessity": sizes["repeated_outcome_necessity"],
        "outcome_history_necessity": sizes["outcome_history_necessity"],
        "primary_n_rule": primary_n_rule_fires(PRIMARY_PILOT_BIASES, PRIMARY_PILOT_REPLICATES),
        "c6_n_rule": c6_rule_fires(CALIBRATION_PILOT_BIASES, CALIBRATION_PILOT_REPLICATES),
        "n1_n_rule": property_n_rule_fires(
            PROPERTY_PILOT_BIASES, PROPERTY_PILOT_REPLICATES, "repeated_outcome_necessity"
        ),
        "n2_n_rule": property_n_rule_fires(
            PROPERTY_PILOT_BIASES, PROPERTY_PILOT_REPLICATES, "outcome_history_necessity"
        ),
    }


# ---------------------------------------------------------------------------- pilots
#
# Every pilot draws from ``stream_seed(STUDY, "pilot", ...)``, which no published cell reads.
# The sample size is part of the stream label, so a pilot repeated at a doubled ``n`` draws
# fresh seeds.  ``tests/canonical/ipsi_repeated/pilot.py`` runs these tasks in resumable parts.


def primary_pilot_task(payload: tuple[str, int, int]) -> list[dict[str, Any]]:
    """P4: one primary fit on the pilot stream, one row for each estimand."""
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
                "estimate": float(estimate.psi),
                "std_error": float(estimate.std_error),
                "covered": int(low <= TRUTH[estimand] <= high),
            }
        )
    return rows


def calibration_pilot_task(payload: tuple[str, int, int]) -> dict[str, Any]:
    """P5: one calibration fit on the pilot stream.

    Returns the property rows of the arm, and for the continuous arm the realized fallback-read
    mass at the third node for the tilts 0.5 and 2.
    """
    arm, replicate, n = payload
    seed = stream_seed(STUDY, "pilot", "interval_calibration", arm, n, replicate)
    rows = _replication(("interval_calibration", arm, replicate, 0, seed, n))
    mass: dict[str, float] = {}
    if arm == "continuous":
        frame = sample(arm, seed, n)
        mass = {str(delta): fallback_mass(frame, delta) for delta in (0.5, 2.0)}
    return {"rows": rows, "mass": mass}


def property_pilot_task(payload: tuple[str, int, int]) -> list[dict[str, Any]]:
    """P6: one replication of a property family on the pilot stream."""
    family, replicate, n = payload
    seed = stream_seed(STUDY, "pilot", family, "paired", n, replicate)
    return _replication((family, "paired", replicate, 0, seed, n))


def _standardized_bias(group: pd.DataFrame, target: float) -> float:
    estimates = group["estimate"].to_numpy(dtype=float)
    return float((estimates.mean() - target) / estimates.std(ddof=1))


def summarize_primary_pilot(rows: pd.DataFrame) -> tuple[dict[str, int], dict[str, float], int]:
    """P4: the covered count and standardized bias of each ``scenario/label`` cell.

    Returns ``(covered, biases, replicates)``.  The bias is the mean error in empirical
    standard deviations of the estimate.
    """
    covered: dict[str, int] = {}
    biases: dict[str, float] = {}
    counts = set()
    for scenario in SCENARIOS:
        for name in LABELS:
            group = rows.loc[(rows["scenario"] == scenario) & (rows["label"] == name)]
            key = f"{scenario}/{name}"
            covered[key] = int(group["covered"].sum())
            biases[key] = _standardized_bias(group, TRUTH[label_estimand(name)])
            counts.add(len(group))
    if len(counts) != 1:
        raise ValueError(f"the primary pilot cells hold different counts: {sorted(counts)}")
    return covered, biases, counts.pop()


def summarize_calibration_pilot(
    rows: pd.DataFrame, masses: Sequence[Mapping[str, float]] = ()
) -> dict[str, Any]:
    """P5: the SE ratio and standardized bias of each calibration cell, and the C4 coverage.

    Returns a mapping with ``ratios``, ``biases``, ``joint_coverage``, ``replicates`` (the
    smallest count over the arms) and ``fallback_mass`` (the mean mass by tilt).
    """
    ratios: dict[str, float] = {}
    biases: dict[str, float] = {}
    calibration = rows.loc[rows["property"] == "interval_calibration"]
    counts = set()
    for cell in CALIBRATION_CELLS:
        group = calibration.loc[calibration["cell"] == cell]
        if group.empty:
            raise ValueError(f"the calibration pilot has no rows for {cell}")
        estimates = group["estimate"].to_numpy(dtype=float)
        ratios[cell] = float(group["std_error"].mean() / estimates.std(ddof=1))
        biases[cell] = _standardized_bias(group, float(group["truth"].iloc[0]))
        counts.add(len(group))
    band = rows.loc[
        (rows["property"] == "simultaneous_coverage")
        & (rows["cell"] == f"{JOINT_LABEL}__simultaneous_band")
    ]
    mass: dict[str, float] = {}
    if masses:
        for delta in masses[0]:
            mass[delta] = float(np.mean([entry[delta] for entry in masses]))
    return {
        "ratios": ratios,
        "biases": biases,
        "joint_coverage": float(band["covered"].mean()),
        "replicates": min(counts),
        "fallback_mass": mass,
    }


def summarize_property_pilot(rows: pd.DataFrame) -> tuple[dict[str, float], int]:
    """P6: the standardized bias of each ``family/cell`` around its target.

    Returns ``(biases, replicates)`` with the smallest count over the cells.
    """
    biases: dict[str, float] = {}
    counts = set()
    for family, names in property_cells().items():
        for key in names:
            cell = key.split("/", 1)[1]
            group = rows.loc[(rows["property"] == family) & (rows["cell"] == cell)]
            if group.empty:
                raise ValueError(f"the property pilot has no rows for {key}")
            biases[key] = _standardized_bias(group, bias_target(family, cell))
            counts.add(len(group))
    return biases, min(counts)


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
        out["n2_n_rule_fires"] = property_n_rule_fires(
            biases, replicates, "outcome_history_necessity"
        )
    return out


def time_fit_sets(seed: int = 0, n: int | None = None) -> dict[tuple[str, str], float]:
    """Seconds for one replication of each fit set, single process, for the budget table.

    ``n`` overrides the family size, for a smoke run.
    """
    seconds: dict[tuple[str, str], float] = {}
    for family, configuration, _ in FIT_SETS:
        size = family_n(family) if n is None else n
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
        "one_step": ONE_STEP_LIMIT,
        "absorbing": ABSORBING_LIMIT,
        "history_free": HISTORY_FREE_LIMIT,
    }
