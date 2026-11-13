r"""Declaration and driver of the ``mtp-policy-variance`` study.

The subject is ``variance="policy_weighted"`` on :class:`~cleverly.longitudinal.LTMLE` and
:class:`~cleverly.longitudinal.SDR`: the standard error from
:math:`\max(\hat\sigma^2_{IC}, \hat\sigma^2_{pw})`, with :math:`\hat\sigma^2_{pw}` the
change-of-measure sequential plug-in of :mod:`cleverly.longitudinal.policy_variance`.  The
estimate is the default's, and the interval contains the default Wald interval in every sample.
Every fit of the option also reports the default Wald standard error, so each fit writes two
rows: the option (the positive) and the default Wald interval (the control where marked).  The
study also holds the tail-size twins: in-regime checks of the default Wald interval for every
passing cell of ``mtp-boundaries`` and ``mtp-pricing`` whose n is below twice its tail size.

The law, the plans and the truths are those of :mod:`tests.studies.mtp_pricing_common` (law P)
and :mod:`tests.studies.mtp_boundaries_common` (law B and law B').  No comparator: R ``lmtp``
has no such variance.

**Primary scenarios** (:data:`SCENARIOS`, :data:`PRIMARY_REPLICATES` each): the coverage twins.
Each is the source row's law, plans, learners and estimator at its own n, the least multiple of
400 at or above twice the tail size of every quantity it gates (:func:`twin_sizes`, read from the
committed ``tests/diagnostics/fix5_heavy_ratios/`` CSVs).  A truth row passes on its bias
interval, the coverage floor and the SE screen :data:`TWIN_SE_SCREEN`.  The ``tau = 3`` and atom
twins fit with the option and publish its rows as the reported scenarios ``<twin>_policy``
(:data:`REPORTED`), from the same fits.  The large twins (``tau = 5`` and ``p4``) fit the
default only; their option is read on :data:`TAIL_OPTION_REPLICATES` property draws.  A
coverage-floor control (the twin's SE times :data:`CONTROL_SE_FACTOR`) on every gated twin
row is ``coverage-controls.csv`` (:func:`extra_artifacts`).

==========================  ===========  ================================================
scenario                    n            source rows twinned
==========================  ===========  ================================================
``pricing_ltmle_tau5_tail`` 72,800       ``mtp-pricing`` ``pricing_ltmle_tau5``, ``price_down``
``pricing_sdr_tau5_tail``   73,600       ``pricing_sdr_tau5``, ``price_down``
``pricing_ltmle_tau3_tail`` 5,600        ``pricing_ltmle_tau3``, ``price_down``
``pricing_sdr_tau3_tail``   5,600        ``pricing_sdr_tau3``, ``price_down``
``atom_ltmle_tail``         6,400        ``mtp-boundaries`` ``atom_ltmle``, ``ate[p5]``
``atom_sdr_tail``           6,400        ``atom_sdr``, ``ate[p5]``
``p4_ltmle_tail``           104,400      ``boundaries_ltmle``, ``p4``; law-B ``OracleRatio``
``p4_sdr_tail``             104,400      ``boundaries_sdr``, ``p4``; law-B ``OracleRatio``
==========================  ===========  ================================================

The property families are in :mod:`tests.studies.mtp_policy_variance_properties`.

**The variance-learner screen** (:func:`learner_screen`), declared before any run:
:data:`SCREEN_DRAWS` draws of each held-out configuration of :data:`SCREEN_CONFIGS` on the
``diagnostic`` stream, ``LTMLE`` and ``SDR``, each fit read under both candidates of
:data:`CANDIDATES` with the maximum.  Rule: drop a candidate with any cell's SE-ratio point
below 0.93 or above 1.25; among the rest take the smaller largest point; a tie within 0.01
goes to d3.  If both are dropped the option does not ship.  The gated laws (law P
``price_down`` at two and five nodes, law B ``p4``) are held out.  The screen's rows are
``tests/studies/mtp_policy_variance_screen.csv.gz``, and
:func:`screen_choice` reads them.  **Screen reading** (2026-11-13, 400 draws per
configuration, before any gated cell ran): both candidates stay inside the band on all 16 cells.
d5's largest SE-ratio point is 1.106 (``lawP:price_up:5``, ``SDR``, the contrast) against
d3's 1.162 (``lawP:price_down:3``, ``SDR``, the mean), and the smallest are 1.004 and 1.006
(``lawB:p3``), so the rule chooses **d5**, which :data:`~cleverly.longitudinal.policy_variance.
VARIANCE_LEARNER` already holds.

**Red-cell reading** (declared before any run).

* A red gated option row (``policy_variance`` and ``policy_calibration`` positives) is a defect
  suspect for the option: diagnose it with paired arms on fresh seeds, never rerun to change a
  verdict.  Its key gets no ``wald_policy`` license, and the warning edit lands only its BCa
  half.  A ``red: too wide`` reading is the option costing more than a 25% wider interval, also
  no license.
* A red Wald twin (a primary truth row, or an ``interval_calibration`` twin) is a red cell in
  regime, so a defect suspect for the default interval.  The cell it twins loses its gated
  reading, and its owner stays open.
* A Wald control that covers at nominal (``pricing_*_tau5`` Wald rows whose coverage upper
  bound reaches 0.95) means the study's premise is wrong: diagnose before registering.
* A control that does not fail (an SE control inside the clause, a coverage-floor control
  whose coverage upper bound reaches 0.90) is a red of the study, not of the estimator; the
  claim of the cell it guards is withheld until the control is repaired under a fresh
  declaration.
* The reporting cells' readings, stated before the run: ``sdr_default_pseudo`` Wald coverage
  0.90 to 0.95 (the fix5_heavy_ratios probe 4), with its trigger for a follow-up fix; ``pricing_learned_*`` option
  coverage 0.99 or more with SE ratio near 1.4 (probe 5); the twins' option rows
  ``conservative`` or calibrated, with coverage at least the Wald rows'.
* The expected readings of the two rows the round-1 pilot put outside their passing regions,
  which a paired-arm diagnosis on 3,000 fresh ``diagnostic`` draws read as draw noise (``LTMLE``
  and ``SDR`` agree to 0.0017 in the estimate on the same draws; every 99% SE-ratio interval
  contains 1 and every coverage interval 0.95): the ``pricing_ltmle_tau2`` option rows read
  ``calibrated``, the ``ey`` row possibly ``conservative``; the ``pricing_ltmle_tau3_tail``
  Wald twin passes.  Round 2 of the pilot (the sizing rule of
  :mod:`tests.studies.mtp_policy_variance_properties`) re-read both on the pilot stream, inside
  their passing regions: the ``tau2`` option SE ratio 0.997 (``ate``) and 1.006 (``ey``) with
  coverage 0.946 and 0.953 on 2,000 draws, and the ``tau3`` Wald twin's coverage 0.944 and
  0.940 with SE ratio 0.997 and 0.996 on 3,000 draws.
* No replicate count, law, learner, margin or seed moves after a verdict is seen.  No rerun buys
  a pass.
"""

from __future__ import annotations

import contextlib
import functools
import math
import warnings
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.stats import beta, norm
from sklearn.base import clone
from sklearn.linear_model import LinearRegression

from cleverly.learners import SuperLearner
from cleverly.longitudinal import LTMLE, SDR
from cleverly.longitudinal import estimator as estimator_module
from cleverly.longitudinal import policy_variance as pv
from tests.parallel import STUDY_JOBS
from tests.studies import canonical_mtp_boundaries as boundaries
from tests.studies import canonical_mtp_pricing as pricing
from tests.studies import mtp_boundaries_common as law_b
from tests.studies import mtp_pricing_common as law_p
from tests.studies import mtp_pricing_properties as pricing_properties
from tests.studies import mtp_study_tools as tools
from tests.studies.evidence import failures as evidence_failures
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate, replicate_seed, stream_seed
from tests.studies.fractional_glm import QuasiBinomialGLM

SLUG = "mtp-policy-variance"
SEED = 20265005
RESAMPLING_SEED = 2026500501
FAILURE_BOUND = 0.01
#: The primary (coverage-twin) replications (rules S1 and S2): at least 1,653 of 1,800 covered.
PRIMARY_REPLICATES = 1_800
#: The SE screen of a truth or coverage twin: the union-model band.
TWIN_SE_SCREEN = (0.1, 10.0)
#: The SE factor of the twins' coverage-floor control: the largest of the declared factors with
#: power at least 0.95 on every twin row (rev 4.1's rule, rule S4).
CONTROL_SE_FACTOR = 0.65
#: The coverage a coverage-floor control's 99% upper bound must fall below.
CONTROL_COVERAGE_CEILING = 0.90
#: The property draws on which the large twins' option rows are read.
TAIL_OPTION_REPLICATES = 400
CRITICAL = float(norm.ppf(0.975))
#: The step of every gated size and of every replication count.
STEP = 400

DIAGNOSTIC = ROOT / "tests" / "diagnostics" / "fix5_heavy_ratios"
SCREEN_ROWS = ROOT / "tests" / "studies" / "mtp_policy_variance_screen.csv.gz"

P_REF = law_p.REFERENCE
B_REF = law_b.REFERENCE
DOWN = "price_down"
UP = "price_up"


def ey(label: str) -> str:
    return f"ey_regimen[{label}]"


def ate(label: str, reference: str) -> str:
    return f"ate_regimen[{label} vs {reference}]"


# ------------------------------------------------------------------------ the tail sizes


@functools.cache
def _law_p_tails() -> dict[tuple[int, str, str, str], int]:
    frame = pd.read_csv(DIAGNOSTIC / "law_p_moments.csv")
    return {
        (int(r.tau), str(r.learners), str(r.estimator), str(r.quantity)): int(r.n_kappa)
        for r in frame.itertuples()
    }


@functools.cache
def _law_b_eif() -> dict[tuple[str, str], int]:
    frame = pd.read_csv(DIAGNOSTIC / "quadrature.csv")
    frame = frame[frame.order == 40]
    return {(str(r.plan), str(r.quantity)): int(r.n_kappa) for r in frame.itertuples()}


@functools.cache
def _law_b_own() -> dict[tuple[str, str, str, str], int]:
    frame = pd.read_csv(DIAGNOSTIC / "law_b_own_moments.csv")
    frame = frame[frame.order == 40]
    return {
        (str(r.law), str(r.learners), str(r.estimator), str(r.quantity)): int(r.n_kappa)
        for r in frame.itertuples()
    }


def gated_size(n_kappa: int) -> int:
    """The least multiple of :data:`STEP` at or above twice ``n_kappa``."""
    return STEP * math.ceil(2 * n_kappa / STEP)


def pricing_tail(tau: int, estimator: str, quantity: str) -> int:
    """A law-P primary quantity's tail size: the larger of the property proxy and its own fit's."""
    own = _law_p_tails()
    return max(own[tau, "property", "ltmle", quantity], own[tau, "primary", estimator, quantity])


def boundaries_tail(plan: str, quantity: str, learners: str, estimator: str) -> int:
    """A law-B or law-B' quantity's tail size: the larger of the EIF's and the fit's own curve's."""
    law = law_b.LAW_OF[plan]
    return max(
        _law_b_eif()[plan, quantity], _law_b_own()[law, learners, estimator, f"{quantity}[{plan}]"]
    )


def up_oracle_tail(tau: int) -> int:
    """``up_oracle_tau{tau}``'s tail size: the property learners' own ``LTMLE`` curve."""
    return _law_p_tails()[tau, "property", "ltmle", f"ey[{UP}]"]


def twin_sizes() -> dict[str, dict[str, int]]:
    """Each twin's gated size of each quantity it gates, read from the committed CSVs."""
    out: dict[str, dict[str, int]] = {}
    for tau in (5, 3):
        for estimator in ("ltmle", "sdr"):
            out[f"pricing_{estimator}_tau{tau}_tail"] = {
                ey(DOWN): gated_size(pricing_tail(tau, estimator, f"ey[{DOWN}]")),
                ate(DOWN, P_REF): gated_size(pricing_tail(tau, estimator, f"ate[{DOWN}]")),
            }
    for estimator in ("ltmle", "sdr"):
        out[f"atom_{estimator}_tail"] = {
            ate("p5", B_REF): gated_size(boundaries_tail("p5", "ate", "primary", estimator))
        }
        out[f"p4_{estimator}_tail"] = {
            ey("p4"): gated_size(boundaries_tail("p4", "ey", "primary", estimator)),
            ate("p4", B_REF): gated_size(boundaries_tail("p4", "ate", "primary", estimator)),
        }
    for label, estimator in (
        ("p4", "ltmle"),
        ("p4_sdr", "sdr"),
        ("p5", "ltmle"),
        ("p5_sdr", "sdr"),
    ):
        plan = label.removesuffix("_sdr")
        out[f"{label}_twin"] = {
            ate(plan, B_REF): gated_size(boundaries_tail(plan, "ate", "correct", estimator))
        }
    out["up_oracle_tau5_twin"] = {ey(UP): gated_size(up_oracle_tail(5))}
    # A lower bound only: the learned limit's own curve has no closed form.
    out["up_learned_classifier_tau5_twin"] = {ey(UP): gated_size(up_oracle_tail(5))}
    return out


def twin_n(name: str) -> int:
    """A twin's n: the larger of its quantities' gated sizes."""
    return max(twin_sizes()[name].values())


# ------------------------------------------------------------------------ the fits

Variance = Literal["influence_curve", "policy_weighted"]
OPTION: Variance = "policy_weighted"
DEFAULT: Variance = "influence_curve"


def pricing_fit(
    frame: pd.DataFrame,
    tau: int,
    labels: tuple[str, ...],
    estimator: str,
    *,
    variance: Variance,
    treatment_learner: Any = None,
) -> Any:
    """:func:`canonical_mtp_pricing.fit`'s primary model with the variance option.

    Primary learners (a quasibinomial GLM at the last node, least squares before it), the exact
    ratio on the classifier route unless ``treatment_learner`` names another, and ``LTMLE``'s
    submodel bound :data:`~tests.studies.canonical_mtp_pricing.PRIMARY_ALPHA`.
    """
    settings: dict[str, Any] = {
        "reference": P_REF,
        "outcome_learner": QuasiBinomialGLM(),
        "pseudo_learner": LinearRegression(),
        "treatment_learner": clone(
            law_p.OracleRatio() if treatment_learner is None else treatment_learner
        ),
        "n_folds": 1,
        "g_bounds": pricing.G_BOUNDS,
        "ratio": "classifier",
        "simultaneous": False,
        "random_state": 0,
        "variance": variance,
    }
    regimens = law_p.regimens(labels, tau)
    if estimator == "ltmle":
        model: Any = LTMLE(
            regimens, max_iter=100, tol=1e-10, alpha=pricing.PRIMARY_ALPHA, **settings
        )
    elif estimator == "sdr":
        model = SDR(regimens, **settings)
    else:
        raise ValueError(estimator)
    return model.fit(frame, outcome="Y", **pricing.roles(tau))


def default_pseudo() -> Any:
    """The pseudo learner ``SDR`` builds when no learner is given (``estimator.py``, SDR)."""
    return SuperLearner(task="regression", n_folds=5, clip=None, random_state=0, n_jobs=1)


def boundaries_fit(
    frame: pd.DataFrame,
    law: str,
    labels: tuple[str, ...],
    estimator: str,
    outcome: boundaries.Outcome,
    *,
    variance: Variance,
    ratio: Literal["density", "classifier"] = "density",
    pseudo: Literal["declared", "unit_range", "default"] = "declared",
) -> Any:
    """:func:`canonical_mtp_boundaries.fit` with the variance option.

    The density route reads the oracle hazard at ``oracle_bins(n)``; the classifier route the
    law-B :class:`~tests.studies.mtp_boundaries_common.OracleRatio`.  ``pseudo`` picks the
    first-node learner: the configuration's own, the ``[0, 1]`` projection of the rich basis
    (``SDR`` with the correct outcome, as ``mtp-boundaries``), or the default ``SDR`` builds.
    """
    q, first, g = boundaries.learners(frame, outcome)
    if pseudo == "unit_range":
        first = law_b.UnitRangePseudo()
    elif pseudo == "default":
        first = default_pseudo()
    settings: dict[str, Any] = {
        "reference": B_REF,
        "outcome_learner": clone(q),
        "pseudo_learner": clone(first),
        "treatment_learner": clone(law_b.OracleRatio() if ratio == "classifier" else g),
        "n_folds": 1,
        "g_bounds": boundaries.G_BOUNDS,
        "density_bins": boundaries.oracle_bins(len(frame)),
        "ratio": ratio,
        "simultaneous": False,
        "random_state": 0,
        "variance": variance,
    }
    if estimator == "ltmle":
        model: Any = LTMLE(law_b.regimens(labels), max_iter=100, tol=1e-10, **settings)
    elif estimator == "sdr":
        model = SDR(law_b.regimens(labels), **settings)
    else:
        raise ValueError(estimator)
    return model.fit(
        frame,
        outcome="Y",
        treatment=["A1", "A2"],
        baseline=["W1", "W2"],
        time_varying=[[], ["L2"]],
        continuous_treatment={"A1": law_b.SUPPORT, "A2": law_b.SUPPORT},
        dose_atoms={"A2": (law_b.ATOM,)} if law == "B_atom" else None,
    )


@dataclass(frozen=True)
class Config:
    """One fit configuration: a law, its plans, the learners, the estimator and the variance.

    Parameters
    ----------
    kind : str
        ``"pricing"`` (law P, primary learners), ``"pricing_learned"`` (law P, the screened
        quadratic logistic on the classifier route), ``"pricing_property"`` (law P ``price_up``,
        the ``mtp-pricing`` calibration learners of ``label``) or ``"boundaries"`` (law B or B').
    estimator : str
        ``"ltmle"`` or ``"sdr"``.
    tau : int
        Law P's nodes (2 for law B).
    plan : str
        The plan contrasted with the reference.
    outcome : str
        Law B's outcome learners (``"primary"`` or ``"correct"``); ``""`` on law P.
    ratio : str
        Law B's ratio route.
    pseudo : str
        Law B's first-node learner (:func:`boundaries_fit`).
    label : str
        The ``mtp-pricing`` calibration label of a ``"pricing_property"`` fit.
    """

    kind: str
    estimator: str
    tau: int
    plan: str
    outcome: str = ""
    ratio: str = "density"
    pseudo: str = "declared"
    label: str = ""

    @property
    def law(self) -> str:
        if self.kind == "boundaries":
            return law_b.LAW_OF[self.plan]
        return "P"

    @property
    def reference(self) -> str:
        return B_REF if self.kind == "boundaries" else P_REF

    @property
    def names(self) -> tuple[str, str]:
        """The plan's mean and its contrast with the reference."""
        return ey(self.plan), ate(self.plan, self.reference)

    def sample(self, n: int, seed: int) -> pd.DataFrame:
        if self.kind == "boundaries":
            return law_b.SAMPLERS[self.law](n, seed)
        return law_p.sample(n, seed, self.tau)

    def truth(self) -> dict[str, float]:
        if self.kind == "boundaries":
            return {
                ey(self.plan): law_b.truth(self.plan, self.law),
                ate(self.plan, B_REF): law_b.contrast(self.plan),
            }
        values = pricing.truth_of(self.tau)
        return {name: values[name] for name in self.names}

    def fit(self, frame: pd.DataFrame, variance: Variance) -> Any:
        labels = (self.reference, self.plan)
        if self.kind == "pricing":
            return pricing_fit(frame, self.tau, labels, self.estimator, variance=variance)
        if self.kind == "pricing_learned":
            learner = pricing_properties.classifier(
                pricing_properties.SCREENED_CLASSIFIER[self.tau]
            )
            return pricing_fit(
                frame,
                self.tau,
                labels,
                self.estimator,
                variance=variance,
                treatment_learner=learner,
            )
        if self.kind == "pricing_property":
            if variance != DEFAULT:
                raise ValueError("the calibration twins of mtp-pricing read the default only")
            return pricing_properties.calibration_fit(self.label, frame)
        if self.kind == "boundaries":
            return boundaries_fit(
                frame,
                self.law,
                labels,
                self.estimator,
                self.outcome,  # type: ignore[arg-type]
                variance=variance,
                ratio=self.ratio,  # type: ignore[arg-type]
                pseudo=self.pseudo,  # type: ignore[arg-type]
            )
        raise ValueError(self.kind)


def readings(result: Any, names: Sequence[str]) -> dict[str, dict[str, float]]:
    """Each estimand's estimate, both standard errors and the reported one.

    ``se_ic`` is the default Wald standard error and ``se`` the reported one: the option's
    maximum under ``variance="policy_weighted"``, else ``se_ic``.  ``se_pw`` is ``nan`` when the
    policy-weighted variance was not used (or not computed).
    """
    out: dict[str, dict[str, float]] = {}
    for name in names:
        estimate = result[name]
        record = getattr(estimate, "policy", None)
        se = float(estimate.std_error)
        if record is None:
            se_ic, se_pw = se, float("nan")
        else:
            se_ic = float(np.sqrt(max(record.influence_variance, 0.0)))
            se_pw = float(np.sqrt(record.policy_variance)) if record.usable else float("nan")
        low, high = estimate.ci
        out[name] = {
            "psi": float(estimate.psi),
            "se_ic": se_ic,
            "se_pw": se_pw,
            "se": se,
            "ci_lower": float(low),
            "ci_upper": float(high),
        }
    return out


# ------------------------------------------------------------------------ the primary twins

#: Each coverage twin: its fit configuration and the estimands its truth rows gate.
TWINS: dict[str, tuple[Config, tuple[str, ...]]] = {
    **{
        f"pricing_{est}_tau{tau}_tail": (
            Config("pricing", est, tau, DOWN),
            (ey(DOWN), ate(DOWN, P_REF)),
        )
        for tau in (5, 3)
        for est in ("ltmle", "sdr")
    },
    **{
        f"atom_{est}_tail": (
            Config("boundaries", est, 2, "p5", outcome="primary"),
            (ate("p5", B_REF),),
        )
        for est in ("ltmle", "sdr")
    },
    **{
        f"p4_{est}_tail": (
            Config("boundaries", est, 2, "p4", outcome="primary", ratio="classifier"),
            (ey("p4"), ate("p4", B_REF)),
        )
        for est in ("ltmle", "sdr")
    },
}
#: The twins whose fits carry the option on every replicate, published as ``<twin>_policy``
#: with ``ey`` beside the gated quantities (the atom twins report ``ey[p5]`` there).
OPTION_TWINS: tuple[str, ...] = (
    "pricing_ltmle_tau3_tail",
    "pricing_sdr_tau3_tail",
    "atom_ltmle_tail",
    "atom_sdr_tail",
)
#: The large twins, whose option rows are read on the property stream.
LARGE_TWINS: tuple[str, ...] = tuple(name for name in TWINS if name not in OPTION_TWINS)


def policy_scenario(twin: str) -> str:
    return f"{twin}_policy"


REPORTED: frozenset[str] = frozenset(policy_scenario(twin) for twin in OPTION_TWINS)

SCENARIOS: dict[str, tuple[str, ...]] = {
    **{twin: names for twin, (_, names) in TWINS.items()},
    **{policy_scenario(twin): TWINS[twin][0].names for twin in OPTION_TWINS},
}
SCENARIO_N: dict[str, int] = {
    **{twin: twin_n(twin) for twin in TWINS},
    **{policy_scenario(twin): twin_n(twin) for twin in OPTION_TWINS},
}
#: The smallest twin's size, the record's ``n``; every other size is in :data:`SCENARIO_N`.
PRIMARY_N = min(SCENARIO_N.values())


def owner(scenario: str) -> str:
    """The twin whose fit writes ``scenario``'s rows."""
    return scenario.removesuffix("_policy")


def _draw(key: tuple[str, int, int]) -> tuple[pd.DataFrame, dict[str, float]]:
    scenario, replicate, n = key
    return draw_scenario(scenario, n, replicate)


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    config = TWINS[owner(scenario)][0]
    return config.sample(n, seed), config.truth()


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def _row(
    scenario: str, replicate: int, n: int, name: str, truth: float, psi: float, se: float
) -> dict[str, Any]:
    low, high = psi - CRITICAL * se, psi + CRITICAL * se
    return {
        "implementation": STUDY.implementation,
        "scenario": scenario,
        "replicate": replicate,
        "n": n,
        "estimand": name,
        "truth": truth,
        "estimate": psi,
        "inference_estimate": psi,
        "std_error": se,
        "ci_lower": low,
        "ci_upper": high,
        "inference_scale": "identity",
        "covered": int(low <= truth <= high),
        "initial_estimate": float("nan"),
    }


def twin_rows(
    twin: str, replicate: int, frame: pd.DataFrame, truth: Mapping[str, float]
) -> list[dict[str, Any]]:
    """The truth rows of one replicate of a twin, and its option rows when it carries them."""
    config, names = TWINS[twin]
    option = twin in OPTION_TWINS
    result = config.fit(frame, OPTION if option else DEFAULT)
    values = readings(result, config.names)
    n = len(frame)
    rows = [
        _row(twin, replicate, n, name, truth[name], values[name]["psi"], values[name]["se_ic"])
        for name in names
    ]
    if option:
        scenario = policy_scenario(twin)
        rows += [
            _row(scenario, replicate, n, name, truth[name], values[name]["psi"], values[name]["se"])
            for name in config.names
        ]
    return rows


def _replicate(
    key: tuple[str, int, int], drawn: tuple[pd.DataFrame, dict[str, float]]
) -> list[dict[str, Any]]:
    scenario, replicate, _ = key
    frame, truth = drawn
    with quiet():
        return twin_rows(scenario, replicate, frame, truth)


def draw_and_fit(*, replicates: int, n: int, n_jobs: int = STUDY_JOBS) -> replication.PhaseResult:
    """Every twin's replicates: one fit per key writes the twin and its option scenario."""
    sizes = STUDY.sizes(n)
    keys = [(twin, r, sizes[twin]) for twin in TWINS for r in range(replicates)]
    outcomes = replication.fit_primary(
        keys,
        _replicate,
        draw=_draw,
        rows_at=None,
        n_jobs=n_jobs,
        implementation=STUDY.implementation,
        seed_of=lambda key: replicate_seed(STUDY, key[0], key[1]),
    )
    built = pd.DataFrame([row for rows in outcomes.kept for row in rows])
    return replication.result(outcomes, built.loc[:, list(REPLICATE_COLUMNS)])


# ------------------------------------------------------------------------ coverage controls

CONTROL_COLUMNS = (
    "scenario",
    "estimand",
    "replicates",
    "coverage",
    "coverage_ci_upper",
    "ceiling",
    "passed",
)
CONTROL_AUDIT = "coverage-floor control"


def clopper_pearson(covered: int, total: int, level: float = 0.99) -> tuple[float, float]:
    alpha = 1.0 - level
    low = float(beta.ppf(alpha / 2, covered, total - covered + 1)) if covered > 0 else 0.0
    high = float(beta.ppf(1 - alpha / 2, covered + 1, total - covered)) if covered < total else 1.0
    return low, high


def coverage_controls(rows: pd.DataFrame) -> pd.DataFrame:
    """Each gated twin row read with its SE times :data:`CONTROL_SE_FACTOR`."""
    out = []
    subject = rows.loc[
        (rows["implementation"] == STUDY.implementation) & ~rows["scenario"].isin(REPORTED)
    ]
    for (scenario, name), group in subject.groupby(["scenario", "estimand"], sort=True):
        half = CRITICAL * CONTROL_SE_FACTOR * group["std_error"]
        covered = int(((group["estimate"] - group["truth"]).abs() <= half).sum())
        total = len(group)
        upper = clopper_pearson(covered, total)[1]
        out.append(
            {
                "scenario": scenario,
                "estimand": name,
                "replicates": total,
                "coverage": covered / total,
                "coverage_ci_upper": upper,
                "ceiling": CONTROL_COVERAGE_CEILING,
                "passed": bool(upper < CONTROL_COVERAGE_CEILING),
            }
        )
    return pd.DataFrame(out, columns=list(CONTROL_COLUMNS))


def extra_artifacts(rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """``coverage-controls.csv``: the coverage-floor control of every gated twin row."""
    return {"coverage-controls.csv": coverage_controls(rows)}


def scientific_failures(artifacts: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """The coverage-floor controls that did not fail."""
    table = artifacts["coverage-controls.csv"]
    return {CONTROL_AUDIT: table.loc[~table["passed"].astype(bool)]}


def account_failures(failures: pd.DataFrame, artifacts: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """A lost replicate is read as covered by the control, its least favourable fill."""
    return evidence_failures.audits(
        scientific_failures(artifacts),
        missing={CONTROL_AUDIT: evidence_failures.missing_fits(failures, STUDY)},
    )


# ------------------------------------------------------------------------ the record


def _property_cells() -> dict[str, tuple[str, ...]]:
    """Every published property cell, by family.

    :mod:`tests.studies.mtp_policy_variance_properties` declares the fit sets; the design test
    checks that its cells are these.
    """

    def option(label: str, *, gated: bool, wald_control: bool) -> list[str]:
        out = []
        for q in ("ey", "ate"):
            out += [f"{label}__{q}__option", f"{label}__{q}__wald"]
            out += [f"{label}__{q}__coverage_floor_control"] if gated else []
            out += [f"{label}__{q}__inflated_se_control"] if wald_control else []
        return out

    policy = [
        cell
        for est in ("ltmle", "sdr")
        for label, gated, control in (
            (f"pricing_{est}_tau5", True, True),
            (f"pricing_{est}_tau2", True, False),
            (f"boundaries_{est}_p4", True, False),
            (f"pricing_learned_{est}_tau5", False, False),
        )
        for cell in option(label, gated=gated, wald_control=control)
    ]
    calibration = [
        f"p4_{est}__{kind}"
        for est in ("ltmle", "sdr")
        for kind in (
            "option",
            "wald",
            "shrunken_se_control",
            "inflated_se_control",
            "noise_control",
            "coverage_floor_control",
        )
    ]
    twins = [
        f"{label}__{kind}"
        for label in ("p4_twin", "p4_sdr_twin", "p5_twin", "p5_sdr_twin", "up_oracle_tau5_twin")
        for kind in ("correctly_specified", "shrunken_se_control", "noise_control")
    ]
    return {
        "policy_variance": tuple(policy),
        "policy_calibration": tuple(calibration),
        "interval_calibration": tuple(twins),
        "reporting_calibration": ("up_learned_classifier_tau5_twin__wald",),
        "sdr_default_pseudo": ("p4_sdr__superlearner__wald", "p4_sdr__superlearner__option"),
        "tail_option": tuple(
            f"{twin}__{q}__{kind}"
            for twin in LARGE_TWINS
            for q in ("ey", "ate")
            for kind in ("option", "wald")
        ),
    }


#: Every published property cell (:mod:`tests.studies.mtp_policy_variance_properties`).
PROPERTY_CELLS: dict[str, tuple[str, ...]] = _property_cells()

STUDY = StudyRecord(
    name="policy-weighted variance for heavy density ratios",
    slug=SLUG,
    artifacts=ROOT / "tests" / "canonical" / "mtp_policy_variance",
    document="docs/technical-reference/method-evidence/mtp-policy-variance.md",
    anchor="mtp-policy-variance",
    scenarios=dict(SCENARIOS),
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    scenario_n={name: n for name, n in SCENARIO_N.items() if n != PRIMARY_N},
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(se_ratio_sanity=TWIN_SE_SCREEN),
    implementation="cleverly-mtp-policy-variance",
    reference=None,
    reported_scenarios=REPORTED,
    modules=(
        "tests/studies/canonical_mtp_policy_variance.py",
        "tests/studies/mtp_policy_variance_properties.py",
        "tests/studies/canonical_mtp_boundaries.py",
        "tests/studies/canonical_mtp_pricing.py",
        "tests/studies/mtp_boundaries_common.py",
        "tests/studies/mtp_boundaries_properties.py",
        "tests/studies/mtp_pricing_common.py",
        "tests/studies/mtp_pricing_properties.py",
        "tests/studies/mtp_study_tools.py",
        "tests/studies/fractional_glm.py",
        "tests/studies/oracle_density_bins.py",
        "tests/diagnostics/fix5_heavy_ratios/law_p_moments.csv",
        "tests/diagnostics/fix5_heavy_ratios/quadrature.csv",
        "tests/diagnostics/fix5_heavy_ratios/law_b_own_moments.csv",
        "tests/studies/mtp_policy_variance_screen.csv.gz",
        "tests/studies/mtp_policy_variance_pilot.csv.gz",
    ),
    runner_module="tests.studies.canonical_mtp_policy_variance",
    properties_module="tests.studies.mtp_policy_variance_properties",
    property_cells=PROPERTY_CELLS,
    calibration_efficiency_ratio=False,
    publication_policy="gated",
    failure_bound={"cleverly-mtp-policy-variance": FAILURE_BOUND},
    extra_artifacts=("coverage-controls.csv",),
)

CONFIGURATION = {
    "construction": "ordinary, one fold, end of study",
    "variance": "policy_weighted on the option rows; influence_curve on the Wald rows",
    "variance_learner": dict(pv.VARIANCE_LEARNER),
    "twin_sizes": dict(SCENARIO_N),
    "twin_se_screen": list(TWIN_SE_SCREEN),
    "coverage_control_se_factor": CONTROL_SE_FACTOR,
}


# ------------------------------------------------------------------------ the learner screen

#: The two candidates of the variance learner, as in the fix5_heavy_ratios probe 5.
CANDIDATES: dict[str, dict[str, Any]] = {
    "d3": {"max_depth": 3, "max_iter": 60, "min_samples_leaf": 50},
    "d5": {"max_depth": 5, "max_iter": 200, "min_samples_leaf": 20},
}
#: The held-out configurations: law P at n = 5,000 (exact ratio, primary learners), law B at
#: n = 4,000 (correct outcome, oracle hazard, the ``[0, 1]`` first node for ``SDR``).
SCREEN_CONFIGS: dict[str, tuple[Config, Config, int]] = {
    "lawP:price_down:3": (
        Config("pricing", "ltmle", 3, DOWN),
        Config("pricing", "sdr", 3, DOWN),
        5_000,
    ),
    "lawP:price_up:5": (Config("pricing", "ltmle", 5, UP), Config("pricing", "sdr", 5, UP), 5_000),
    **{
        f"lawB:{plan}": (
            Config("boundaries", "ltmle", 2, plan, outcome="correct"),
            Config("boundaries", "sdr", 2, plan, outcome="correct", pseudo="unit_range"),
            4_000,
        )
        for plan in ("p2", "p3")
    },
}
SCREEN_DRAWS = 400
#: The screen rule's band on each cell's SE-ratio point, and the tie margin.
SCREEN_BAND = (0.93, 1.25)
SCREEN_TIE = 0.01
#: The candidate the screen chose on the committed rows (:func:`screen_choice`, 2026-11-13).
#: ``VARIANCE_LEARNER`` holds it.
SCREEN_CHOICE = "d5"
#: The candidate whose matrix the screen's fits report (the other is recomputed beside it).
SCREEN_FIT_CANDIDATE = "d5"


def quiet() -> contextlib.AbstractContextManager[None]:
    """Silence the warnings a fit raises (the sparse-influence warning above all)."""

    @contextlib.contextmanager
    def silenced() -> Iterator[None]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield

    return silenced()


@contextlib.contextmanager
def both_candidates(store: dict[str, list[Any]]) -> Iterator[None]:
    """Fit once and compute the policy-weighted matrix under each candidate.

    The fit reads the matrix of :data:`SCREEN_FIT_CANDIDATE`; ``store[name]`` collects each
    candidate's targeting-scale ``_PolicyMatrix`` in call order.  The nuisance fits do not
    depend on the variance learner, so a candidate's matrix equals that of a separate fit under
    it (the design test checks one draw).
    """
    original = estimator_module._policy_matrix
    shipped = pv.VARIANCE_LEARNER

    def computed(*args: Any, **kwargs: Any) -> Any:
        matrices = {}
        for name, settings in CANDIDATES.items():
            pv.VARIANCE_LEARNER = {**shipped, **settings}  # type: ignore[misc]
            try:
                matrices[name] = original(*args, **kwargs)
            finally:
                pv.VARIANCE_LEARNER = shipped  # type: ignore[misc]
            store.setdefault(name, []).append(matrices[name])
        return matrices[SCREEN_FIT_CANDIDATE]

    estimator_module._policy_matrix = computed
    try:
        yield
    finally:
        estimator_module._policy_matrix = original


def _candidate_se(estimate: Any, used: Any, other: Any) -> float:
    """The policy-weighted SE an estimate would read with another candidate's matrix.

    The reported matrix is ``scaler.range**2`` times the targeting-scale one, with each
    natural level's diagonal entry rewritten as its curve variance (``estimator.py``,
    ``_policy_estimates``).  The factor is read off a level that is not natural.
    """
    record = estimate.policy
    reported = np.asarray(record.covariance.matrix, dtype=float)
    raw_used = np.asarray(used.matrix, dtype=float)
    natural = np.asarray(used.natural, dtype=bool)
    moved = [i for i in range(len(natural)) if not natural[i] and raw_used[i, i] > 0.0]
    if not moved:
        return float("nan")
    factor = reported[moved[0], moved[0]] / raw_used[moved[0], moved[0]]
    matrix = factor * np.asarray(other.matrix, dtype=float)
    for i in np.flatnonzero(natural):
        matrix[i, i] = reported[i, i]
    variance = type(record.covariance)(record.covariance.names, matrix).quadratic(record.gradient)
    floor = 1e-12 * max(record.influence_variance, 0.0)
    return float(np.sqrt(variance)) if np.isfinite(variance) and variance > floor else float("nan")


def screen_rows(config_name: str, replicate: int) -> list[dict[str, Any]]:
    """One draw of a held-out configuration: both estimators, both candidates, both estimands."""
    ltmle_config, sdr_config, n = SCREEN_CONFIGS[config_name]
    seed = stream_seed(STUDY, "diagnostic", "screen", config_name, replicate)
    frame = ltmle_config.sample(n, seed)
    truth = ltmle_config.truth()
    rows = []
    for config in (ltmle_config, sdr_config):
        store: dict[str, list[Any]] = {}
        with quiet(), both_candidates(store):
            result = config.fit(frame, OPTION)
        (d3,), (d5,) = store["d3"], store["d5"]
        for name in config.names:
            estimate = result[name]
            values = readings(result, (name,))[name]
            rows.append(
                {
                    "config": config_name,
                    "estimator": config.estimator,
                    "replicate": replicate,
                    "estimand": name,
                    "truth": truth[name],
                    "psi": values["psi"],
                    "se_ic": values["se_ic"],
                    "se_pw_d5": values["se_pw"],
                    "se_pw_d3": _candidate_se(estimate, d5, d3),
                }
            )
    return rows


def learner_screen(
    parts: Path | None, *, draws: int = SCREEN_DRAWS, n_jobs: int = 1
) -> pd.DataFrame:
    """Every held-out configuration's :data:`SCREEN_DRAWS` draws, resumable in parts."""
    frames = []
    for config_name in SCREEN_CONFIGS:
        keys = [(config_name, r) for r in range(draws)]
        frames.append(
            tools.run_in_parts(
                screen_rows,
                keys,
                label=f"screen__{config_name.replace(':', '_')}",
                root=parts,
                chunk=20,
                n_jobs=n_jobs,
            )
        )
    return pd.concat(frames, ignore_index=True)


def option_se(se_ic: np.ndarray, se_pw: np.ndarray) -> np.ndarray:
    """The option's SE: the larger of the two, the curve's where the other is unused."""
    usable = np.isfinite(se_pw)
    return np.where(usable, np.maximum(se_ic, np.nan_to_num(se_pw)), se_ic)


def screen_summary(rows: pd.DataFrame) -> pd.DataFrame:
    """Each cell's Wald and candidate readings under the maximum."""
    out = []
    for (config, est, name), group in rows.groupby(["config", "estimator", "estimand"], sort=True):
        error = (group["psi"] - group["truth"]).to_numpy()
        sd = float(np.std(error, ddof=1))
        se_ic = group["se_ic"].to_numpy()
        row: dict[str, Any] = {
            "config": config,
            "estimator": est,
            "estimand": name,
            "replicates": len(group),
            "wald_se_ratio": float(se_ic.mean() / sd),
            "wald_coverage": float(np.mean(np.abs(error) <= CRITICAL * se_ic)),
        }
        for candidate in CANDIDATES:
            se = option_se(se_ic, group[f"se_pw_{candidate}"].to_numpy())
            row[f"{candidate}_se_ratio"] = float(se.mean() / sd)
            row[f"{candidate}_coverage"] = float(np.mean(np.abs(error) <= CRITICAL * se))
            row[f"{candidate}_unused"] = int(np.sum(~np.isfinite(group[f"se_pw_{candidate}"])))
        out.append(row)
    return pd.DataFrame(out)


def screen_choice(summary: pd.DataFrame) -> str:
    """Plan 6.2's rule on the screen's cells: ``"d3"``, ``"d5"`` or ``"none"``."""
    worst: dict[str, float] = {}
    for candidate in CANDIDATES:
        ratios = summary[f"{candidate}_se_ratio"]
        if ratios.min() < SCREEN_BAND[0] or ratios.max() > SCREEN_BAND[1]:
            continue
        worst[candidate] = float(ratios.max())
    if not worst:
        return "none"
    if len(worst) == 2 and worst["d5"] >= worst["d3"] - SCREEN_TIE:
        return "d3"
    return min(worst, key=worst.__getitem__)


def committed_screen() -> pd.DataFrame:
    """The committed screen rows."""
    return pd.read_csv(SCREEN_ROWS, float_precision="round_trip")


__all__ = ["STUDY", "draw_and_fit", "learner_screen"]
