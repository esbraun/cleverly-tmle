r"""Declaration and driver of the ``projection-methods`` study (P1 PR 3 plan rev 2, section 5).

The study measures the two methods P1 PR 3 adds for the projection of a
:class:`cleverly.ConditionalContrast` on ``Linear(["W1"])``, both in sample on law P
(:mod:`tests.studies.projection_methods_law`):

- the DR-TMLE projection on the difference scale with ``guard=("Q", "g")``, the univariate
  reduction and cubic reduced regressions: with the mechanism declared and a wrong outcome
  regression (``dr_known_4k``, the calibration arms and the primary), with both nuisances
  correct (``dr_both_4k``, bias), with the correct GLM mechanism and the wrong outcome regression
  (``dr_glm_4k``, the shipped one-wrong rule: the intercept conservative, the slope valid), and
  under the constructed slow mechanism at 1,000, 4,000 and 16,000 rows (``dr_slow_*``, the
  first-order ladder control);
- the per-arm outcome-adaptive C-TMLE projection with a correct outcome regression
  (``ctmle_points_4k``, bias of the points).

Everything a run reads is fixed here before it starts: the law and its exact truths, the fit
groups (:data:`GROUPS`) and the cells their fits fill (:func:`group_cells`), the learners, the
primary phase and the study record (:data:`STUDY`).  Each control and reading has its own cells,
and every group draws its own seed stream.

The study has no witness of the mechanism correction ``D_A``.  At the estimator's targeted
limit the ``d_star`` reading predicts ratios of 1.039 and 1.057, both inside the calibration
band, so the reading is published without a verdict.  The witness is the fast tier's exact-law
curve test, ``tests/unit/test_drtmle_projection.py``, which checks the reported curve row by
row against its closed form with the ``D_A`` term nonzero.

The calibration ``R`` of ``dr_known_4k`` comes from the 3,200-draw pilot
(:mod:`tests.studies.projection_methods_pilot`, with ``pilot_2`` where an arm needs it): X32a's
rule over the two one-row arms (plan 6.1).  :data:`BUDGETED_R` is the ``R`` at pilot ratios of
1, and the pilot's ``R`` replaces it where larger, never lower.  If no ``R`` up to 50,000 meets
the rule after ``pilot_2``, each row is read at its pooled pilot ratio and ``R`` is the smallest
multiple of 400 at which the product of the rows' pass probabilities reaches
:data:`POINT_READING_TARGET`, under the two-sided precondition of
:mod:`tests.studies.point_reading`.  The study is declared and not registered: no pilot and no run
has been made.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

from cleverly.bases import Linear, Modifiers
from cleverly.effect_modification import EffectModification
from cleverly.estimators import CTMLE, DRTMLE
from cleverly.estimators.tmle import correction_parts
from tests.parallel import STUDY_JOBS
from tests.studies import drtmle_one_wrong as law_l
from tests.studies import projection_methods_law as law
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS, reported_inference
from tests.studies.evidence.seeds import draw_replicate, replicate_seed, stream_seed

SLUG = "projection-methods"

#: The record seeds (plan 5.0): free against every registered seed and every pilot seed.
SEED = 20261531
RESAMPLING_SEED = 20261532

MODIFIERS = Modifiers(["W1"])
BASIS = Linear(["W1"])
TERMS = law.TERMS
COVARIATES = ["W1", "W2"]
G_BOUNDS = law_l.G_BOUNDS

#: The calibration ``R`` at pilot ratios of 1 over the two one-row arms (plan 5.4, first row).
BUDGETED_R = 3_600
#: The family target of the point-reading fallback (plan 5.0, kept from PR 2).
POINT_READING_TARGET = 0.95
#: Replications of the bias and one-wrong groups (plan 5.2).
BIAS_R = 2_400
#: Replications of each rung of the slow ladder (plan 5.2).
LADDER_R = 1_000
#: Pilot draws per gated calibration cell, and the ``pilot_2`` extension (plan 5.0).
PILOT_DRAWS = 3_200
PILOT_2_DRAWS = 3_200
#: An arm whose plug-in pass reads below this after the pilot runs ``pilot_2``.
PILOT_2_BELOW = 0.90
#: The shrunken control's factor (plan 5.3).
SHRUNKEN_FACTOR = 0.70

CRITICAL = float(stats.norm.ppf(0.975))

# ------------------------------------------------------------------------- the fit groups

CALIBRATION = "interval_calibration"
BIAS = "bias"
ONE_WRONG = "one_wrong_inference"
#: The family of the readings published with no verdict.
SPREAD = "spread"

#: The calibration arm of each ``dr_known_4k`` coefficient: one row per arm (plan 6.1).
DR_KNOWN_ARMS: Mapping[str, str] = {
    "(intercept)": "__dr_known_intercept",
    "W1": "__dr_known_slope",
}

#: The ladder of the one-wrong rule, keyed by the setting its cells carry (plan 5.3).
LADDER_SETTING = "dr_slow"
LADDERS: Mapping[str, tuple[int, ...]] = {LADDER_SETTING: law.LADDER}
#: The declared sign of each first-order control's bias, which the design test computes from the
#: population conditioning term.
PREDICTED_SIGNS: Mapping[str, int] = {
    f"{LADDER_SETTING}__intercept": -1,
    f"{LADDER_SETTING}__W1": -1,
}
#: The one-wrong kind of each ``dr_glm_4k`` coefficient (plan 5.3, rev 2 S7).
DR_GLM_KINDS: Mapping[str, str] = {"(intercept)": "conservative", "W1": "valid"}
#: The planning SE ratios and coverages of ``dr_glm_4k``: the quadrature's predicted ratios
#: (1.19 and 1.00) and ``2 Phi(1.96 rho) - 1`` at each.
PLANNING_RATIO: Mapping[str, float] = {"(intercept)": 1.19, "W1": 1.00}
PLANNING_COVERAGE: Mapping[str, float] = {"(intercept)": 0.980, "W1": 0.950}


@dataclass(frozen=True)
class Group:
    """One declared fit group: a size, its replications, its fit, and what its fits fill.

    Parameters
    ----------
    label : str
        The group key, which the seed streams and the cell names read.
    n : int
        Rows per replication.
    replicates : int
        Replications in the declared run.
    method : {"drtmle", "ctmle"}
        The projection's method.
    outcome : {"wrong", "correct"}
        The outcome regression: the GLM on ``(A, W1)`` or the correct GLM on
        ``(A, W1, W2, A W1)``.
    mechanism : {"known", "glm", "slow", "per_arm"}
        The declared mechanism, the correct GLM on ``(W1, W2)``, the slow offset learner, or
        C-TMLE's per-arm logistic design.
    arms : Mapping of str to str
        Each term's gated calibration arm.
    bias_rows : bool
        Whether the group's coefficients are positive ``bias`` rows.
    one_wrong : Mapping of str to str
        Each term's one-wrong kind (``conservative``, ``valid`` or ``first_order``).
    controls : Mapping of str to tuple of str
        Each control the group's fits compute, and the terms it publishes rows for.
    readings : Mapping of str to tuple of str
        Each reading published with no verdict, and its terms.
    seconds : float
        Single-thread seconds per replication: the 20-replicate smoke's on the declaration's
        container (:data:`SMOKE_TIMINGS`), rounded up to the hundredth.
    """

    label: str
    n: int
    replicates: int
    method: str
    outcome: str
    mechanism: str
    arms: Mapping[str, str] = field(default_factory=dict)
    bias_rows: bool = False
    one_wrong: Mapping[str, str] = field(default_factory=dict)
    controls: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    readings: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    seconds: float = 1.0

    @property
    def terms(self) -> tuple[str, ...]:
        """The basis terms the group's fit reports."""
        return TERMS

    def name(self, term: str) -> str:
        """The reported name of ``term``."""
        return f"cate_projection[{term}]"


#: The declared seconds of each ladder rung: the 20-replicate smoke's, rounded up.
LADDER_SECONDS: Mapping[int, float] = {1_000: 0.29, 4_000: 0.70, 16_000: 1.88}

#: The declared calibration ``R``: :data:`BUDGETED_R` until the pilot runs (plan 5.4).
DECLARED_R = BUDGETED_R
R = DECLARED_R

GROUPS: tuple[Group, ...] = (
    Group(
        "dr_known_4k",
        4_000,
        R,
        "drtmle",
        "wrong",
        "known",
        arms=DR_KNOWN_ARMS,
        bias_rows=True,
        controls={"shrunken_se": ("(intercept)",), "wrong_q_witness": TERMS},
        readings={"d_star_reading": TERMS},
        seconds=0.55,
    ),
    Group(
        "dr_both_4k",
        4_000,
        BIAS_R,
        "drtmle",
        "correct",
        "glm",
        bias_rows=True,
        readings={"plugin_reading": TERMS},
        seconds=0.32,
    ),
    *(
        Group(
            f"dr_slow_{n // 1_000}k",
            n,
            LADDER_R,
            "drtmle",
            "wrong",
            "slow",
            one_wrong=dict.fromkeys(TERMS, "first_order"),
            seconds=LADDER_SECONDS[n],
        )
        for n in law.LADDER
    ),
    Group(
        "dr_glm_4k", 4_000, BIAS_R, "drtmle", "wrong", "glm", one_wrong=DR_GLM_KINDS, seconds=0.50
    ),
    Group(
        "ctmle_points_4k",
        4_000,
        BIAS_R,
        "ctmle",
        "correct",
        "per_arm",
        bias_rows=True,
        readings={"plugin_reading": TERMS},
        seconds=0.03,
    ),
)

GROUP_BY_LABEL: Mapping[str, Group] = {group.label: group for group in GROUPS}

#: What each control computes from its group's fits, and the rule it must fail (plan 5.3).
CONTROL_CLAIMS: Mapping[str, str] = {
    "shrunken_se": "the reported SE times 0.70: its SE-ratio interval must fall below the band",
    "wrong_q_witness": "the plug-in at the wrong outcome regression: its bias must lie wholly "
    "outside the margin (+5.35 SD on the intercept, -4.86 SD on the slope), the witness that "
    "the wrong regression is wrong on both terms",
}
#: What each reading publishes, with no verdict (plan 5.3).
READING_CLAIMS: Mapping[str, str] = {
    "d_star_reading": "the SE of D* alone (the curve without the mechanism correction) beside "
    "the spread: predicted ratios 1.039 and 1.057 at the targeted limit, both inside the "
    "calibration band, so the reading is not a D_A witness (the exact-law curve test in "
    "tests/unit/test_drtmle_projection.py is); a fit that skips equation (9) but subtracts "
    "D_A would read 0.96 and 0.95",
    "plugin_reading": "the plug-in SE beside the spread, withheld by the fit's status",
}

#: The calibration controls, by the shared kind their cell names end in.
CALIBRATION_CONTROL_KINDS: Mapping[str, str] = {"shrunken_se": "shrunken_se_control"}


def term_label(term: str) -> str:
    """A cell-name fragment of ``term``."""
    return "intercept" if term == "(intercept)" else term


def calibration_cell(group: Group, term: str) -> str:
    """The ``interval_calibration`` cell of ``term`` (``<group>_<term>__<arm>``)."""
    return f"{group.label}_{term_label(term)}{group.arms[term]}"


def bias_cell(group: Group, term: str) -> str:
    """The positive ``bias`` cell of ``term`` on ``group``."""
    return f"{group.label}_{term_label(term)}"


def control_cell(control: str, group: Group, term: str) -> str:
    """The cell of ``control``'s row for ``term``.

    A calibration control is ``<group>_<term>__<kind>``; a bias control is
    ``<control>_<size>_<term>``.
    """
    if control in CALIBRATION_CONTROL_KINDS:
        return f"{group.label}_{term_label(term)}__{CALIBRATION_CONTROL_KINDS[control]}"
    size = group.label.rsplit("_", 1)[-1]
    return f"{control}_{size}_{term_label(term)}"


def reading_cell(reading: str, group: Group, term: str) -> str:
    """The ``spread`` cell of ``reading`` for ``term`` (``<group>_<term>__<reading>``)."""
    return f"{group.label}_{term_label(term)}__{reading}"


def one_wrong_setting(group: Group) -> str:
    """The setting a one-wrong cell carries: ``dr_slow_n1000`` on a rung, else ``dr_glm``."""
    if group.mechanism == "slow":
        return f"{LADDER_SETTING}_n{group.n}"
    return group.label.rsplit("_", 1)[0]


def one_wrong_cell(group: Group, term: str) -> str:
    """The ``one_wrong_inference`` cell of ``term``: ``<setting>__<term>__<kind>``."""
    return f"{one_wrong_setting(group)}__{term_label(term)}__{group.one_wrong[term]}"


#: The growth cells of the ladder, which read the rungs' rows and have none of their own.
GROWTH_CELLS: tuple[str, ...] = tuple(
    f"{LADDER_SETTING}__{term_label(term)}__growth" for term in TERMS
)


def group_cells(group: Group) -> dict[str, tuple[str, ...]]:
    """The property cells ``group``'s fits fill, by family."""
    cells: dict[str, list[str]] = {CALIBRATION: [], BIAS: [], ONE_WRONG: [], SPREAD: []}
    cells[CALIBRATION] += [calibration_cell(group, term) for term in group.arms]
    if group.bias_rows:
        cells[BIAS] += [bias_cell(group, term) for term in group.terms]
    cells[ONE_WRONG] += [one_wrong_cell(group, term) for term in group.one_wrong]
    for control, terms in group.controls.items():
        family = CALIBRATION if control in CALIBRATION_CONTROL_KINDS else BIAS
        cells[family] += [control_cell(control, group, term) for term in terms]
    for reading, terms in group.readings.items():
        cells[SPREAD] += [reading_cell(reading, group, term) for term in terms]
    return {family: tuple(names) for family, names in cells.items() if names}


def property_cells() -> dict[str, tuple[str, ...]]:
    """Every declared property cell, by family, in group order, with the growth cells."""
    out: dict[str, list[str]] = {}
    for group in GROUPS:
        for family, names in group_cells(group).items():
            out.setdefault(family, []).extend(names)
    out[ONE_WRONG].extend(GROWTH_CELLS)
    return {family: tuple(names) for family, names in out.items()}


PROPERTY_CELLS = property_cells()


def calibration_arms() -> dict[str, tuple[str, ...]]:
    """The gated calibration arms and their rows: one coefficient each (plan 6.1)."""
    out: dict[str, list[str]] = {}
    for group in GROUPS:
        for term, arm in group.arms.items():
            out.setdefault(arm, []).append(calibration_cell(group, term))
    return {arm: tuple(rows) for arm, rows in out.items()}


CALIBRATION_ARMS = calibration_arms()


def group_of_cell(cell: str) -> str:
    """The group whose fits fill ``cell``; a growth cell reads the ladder's largest rung."""
    for group in GROUPS:
        if any(cell in names for names in group_cells(group).values()):
            return group.label
    if cell in GROWTH_CELLS:
        return f"dr_slow_{law.LADDER[-1] // 1_000}k"
    raise KeyError(cell)


# ---------------------------------------------------------------------------- the learners


class CorrectOutcome(ClassifierMixin, BaseEstimator):
    """The correct outcome GLM: a logistic fit on ``(1, A, W1, W2, A W1)``.

    The fit reads the pooled design ``[A, W1, W2]``; law P's logit is linear in these columns.
    """

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> CorrectOutcome:
        self.beta_ = law_l.newton_logistic(self._features(X), np.asarray(y, dtype=float))
        self.classes_ = np.array([0.0, 1.0])
        return self

    @staticmethod
    def _features(X: Any) -> np.ndarray:
        values = np.asarray(X, dtype=float)
        return np.column_stack([values[:, :3], values[:, 0] * values[:, 1]])

    def predict_proba(self, X: Any) -> np.ndarray:
        features = self._features(X)
        p = expit(self.beta_[0] + features @ self.beta_[1:])
        return np.column_stack([1.0 - p, p])


class SlowMechanism(law_l.OffsetLogistic):
    """The correct mechanism GLM on ``(W1, W2)`` plus the logit offset ``n^(-1/3) W2 (1 + W1)``.

    A deliberate constructed error, not a fitted learner (plan 5.1).  ``n`` is the group's size,
    so the error shrinks with the ladder exactly as declared.
    """

    def __init__(self, n: int = 4_000) -> None:
        super().__init__(columns=(0, 1))
        self.n = n

    def predict_proba(self, design: Any) -> np.ndarray:
        values = np.asarray(design, dtype=float)
        eta = self.beta_[0] + values[:, :2] @ self.beta_[1:]
        p = expit(eta + law.slow_offset(values[:, 0], values[:, 1], self.n))
        return np.column_stack([1.0 - p, p])


def outcome_learner(group: Group) -> BaseEstimator:
    """The outcome learner of ``group`` on the pooled design ``[A, W1, W2]``."""
    if group.outcome == "wrong":
        return law_l.OffsetLogistic(columns=(0, 1))
    return CorrectOutcome()


def treatment_learner(group: Group) -> BaseEstimator:
    """The treatment learner of ``group`` on ``[W1, W2]``, or C-TMLE's per-arm design learner."""
    if group.mechanism == "known":
        return law_l.DeclaredOnly()
    if group.mechanism == "slow":
        return SlowMechanism(group.n)
    if group.mechanism == "per_arm":
        return LogisticRegression(max_iter=2_000)
    return law_l.OffsetLogistic(columns=(0, 1))


def projection() -> EffectModification:
    """The difference-scale projection on ``Linear(["W1"])``."""
    return EffectModification(BASIS, MODIFIERS, "difference")


def estimator(group: Group, random_state: int) -> Any:
    """The engine estimator of ``group``."""
    common: dict[str, Any] = {
        "outcome_learner": outcome_learner(group),
        "treatment_learner": treatment_learner(group),
        "cross_fit": False,
        "simultaneous": False,
        "g_bounds": G_BOUNDS,
        "random_state": random_state,
        "effect_modification": projection(),
    }
    if group.method == "ctmle":
        return CTMLE(strategy="oat", oat_design="per_arm", **common)
    return DRTMLE(
        guard=("Q", "g"),
        reduction="univariate",
        reduced_outcome_learner=law_l.CubicLeastSquares(),
        reduced_mechanism_residual_learner=law_l.CubicLeastSquares(),
        reduced_treatment_learner=law_l.CubicLogistic(),
        **common,
    )


# -------------------------------------------------------------------------------- the fits


def fit(group: Group, frame: pd.DataFrame, random_state: int) -> Any:
    """One in-sample engine fit of ``group``; the declared mechanism only on ``known``."""
    known = {0.0: "p0", 1.0: "p1"} if group.mechanism == "known" else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return (
            estimator(group, random_state)
            .fit(
                frame,
                outcome="Y",
                treatment="A",
                covariates=COVARIATES,
                treatment_probabilities=known,
            )
            .single()
        )


def phi(result: Any) -> np.ndarray:
    """The basis ``(1, W1)`` at the fit's rows."""
    w1 = np.asarray(result.data.covariates, dtype=float)[:, result.data.covariate_names.index("W1")]
    return np.column_stack([np.ones(len(w1)), w1])


def plugin_initial(result: Any) -> np.ndarray:
    """The plug-in projection of the initial outcome regression (the wrong-``Qbar`` witness)."""
    arms = result.nuisance.outcome.arms
    basis = phi(result)
    return np.linalg.solve(basis.T @ basis, basis.T @ (arms[1.0] - arms[0.0]))


def d_star_curves(result: Any) -> np.ndarray:
    r"""``D*`` alone of each coefficient: the reported curve with the corrections added back.

    The reported curve is :math:`M_n^{-1}[\ldots - \sum_a s(a)(D_{A,a} + D_{Y,a})]`
    (:func:`cleverly.inference.influence.effect_modification_coefficients`), so adding
    :math:`M_n^{-1}\{C_1 - C_0\}` with the corrections :math:`C_a` the fit subtracted
    (:func:`cleverly.estimators.tmle.correction_parts`) leaves the projection's own curve at
    the fit's targeted state.  ``(n, p)``.
    """
    fluctuation = result.fluctuations["effect_modification"]
    parts = correction_parts(
        result.data,
        result.nuisance,
        fluctuation,
        fluctuation.targeted,
        result.nuisance.scaler.scale(result.data.outcome),
    )
    if parts is None:
        raise ValueError("the D* reading needs a fit that solved the reduced equations")
    total = parts.total()
    basis = phi(result)
    gram = basis.T @ basis / len(basis)
    added = np.linalg.solve(gram, (np.asarray(total[1.0]) - np.asarray(total[0.0])).T).T
    reported = np.column_stack(
        [np.asarray(result.estimates[f"cate_projection[{t}]"].influence_curve) for t in TERMS]
    )
    return reported + added


def d_star_std_errors(result: Any) -> np.ndarray:
    """The SE of each coefficient read off ``D*`` alone, on the reported SE's own convention.

    The reported plug-in SE times the ratio of the two curves' root mean squares, so any
    finite-sample factor the package applies to its SE applies to this one as well.
    """
    reported = np.column_stack(
        [np.asarray(result.estimates[f"cate_projection[{t}]"].influence_curve) for t in TERMS]
    )
    ratio = np.sqrt(np.mean(d_star_curves(result) ** 2, axis=0) / np.mean(reported**2, axis=0))
    return np.array(
        [
            float(result.estimates[f"cate_projection[{t}]"].plugin_std_error) * r
            for t, r in zip(TERMS, ratio, strict=True)
        ]
    )


def fit_group(group: Group, frame: pd.DataFrame, *, random_state: int) -> dict[str, Any]:
    """Fit one replication of ``group`` and return what each of its cells reads.

    Returns
    -------
    dict
        ``"estimates"``: the reported estimate objects by term; ``"controls"`` and
        ``"readings"``: each one's value by term (a coefficient for a bias control, a standard
        error for a calibration control or a reading).
    """
    result = fit(group, frame, random_state)
    estimates = {term: result.estimates[group.name(term)] for term in group.terms}
    controls: dict[str, dict[str, float]] = {}
    for control, terms in group.controls.items():
        if control == "shrunken_se":
            values = [SHRUNKEN_FACTOR * float(estimates[t].std_error) for t in group.terms]
        elif control == "wrong_q_witness":
            values = list(plugin_initial(result))
        else:
            raise KeyError(control)
        controls[control] = {
            t: float(v) for t, v in zip(group.terms, values, strict=True) if t in terms
        }
    readings: dict[str, dict[str, float]] = {}
    for reading, terms in group.readings.items():
        if reading == "d_star_reading":
            values = list(d_star_std_errors(result))
        elif reading == "plugin_reading":
            values = [float(estimates[t].plugin_std_error) for t in group.terms]
        else:
            raise KeyError(reading)
        readings[reading] = {
            t: float(v) for t, v in zip(group.terms, values, strict=True) if t in terms
        }
    return {"estimates": estimates, "controls": controls, "readings": readings}


# ----------------------------------------------------------------------- the truths and draws


def truth(group: Group) -> dict[str, float]:
    """The exact coefficients, keyed by :meth:`Group.name` (one law for every group)."""
    return law.truth()


def draw_from_seed(group_label: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Draw one sample of a group from an explicit seed, with the group's truths."""
    group = GROUP_BY_LABEL[group_label]
    return law.sample(n, np.random.default_rng(seed)), truth(group)


# ----------------------------------------------------------------------- the primary phase

#: The primary phase (plan 5.0, 5.2): the ``dr_known_4k`` intercept on the primary stream.
PRIMARY_GROUP = "dr_known_4k"
PRIMARY_REPLICATES = 2_400
PRIMARY_N = GROUP_BY_LABEL[PRIMARY_GROUP].n
#: The coverage floor the primary reads (plan 5.2).
PRIMARY_COVERAGE_FLOOR = 0.90

SCENARIOS = {PRIMARY_GROUP: (GROUP_BY_LABEL[PRIMARY_GROUP].name("(intercept)"),)}

STUDY = StudyRecord(
    name="Projection methods (DR-TMLE, per-arm C-TMLE)",
    slug=SLUG,
    artifacts=ROOT / "tests" / "canonical" / "projection_methods",
    document="docs/technical-reference/method-evidence/projection-methods.md",
    anchor="projection-methods",
    scenarios=SCENARIOS,
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly-projection-methods",
    reference=None,
    failure_bound={"cleverly-projection-methods": 0.01},
    modules=(
        "tests/studies/projection_methods.py",
        "tests/studies/projection_methods_law.py",
        "tests/studies/projection_methods_pilot.py",
        "tests/studies/projection_methods_properties.py",
        "tests/studies/drtmle_one_wrong.py",
        "tests/studies/point_reading.py",
        "tests/studies/repeated_sizing.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
        "src/cleverly/estimators/drtmle.py",
        "src/cleverly/estimators/reduced.py",
        "src/cleverly/estimators/targeting.py",
        "src/cleverly/estimators/ctmle.py",
        "src/cleverly/estimators/tmle.py",
        "src/cleverly/estimators/_nuisance.py",
        "src/cleverly/fluctuation/submodel.py",
        "src/cleverly/fluctuation/reduced.py",
        "src/cleverly/inference/influence.py",
        "src/cleverly/effect_modification.py",
        "src/cleverly/_inference_status.py",
        "src/cleverly/_support_rows.py",
    ),
    runner_module="tests.studies.projection_methods",
    properties_module="tests.studies.projection_methods_properties",
    property_cells=PROPERTY_CELLS,
    bias_row_families=frozenset({BIAS}),
    calibration_efficiency_ratio=False,
    publication_policy="gated",
)

CONFIGURATION = {
    "law_P": "W1, W2 ~ N(0, 1); g0 = expit(0.6 W1 + 0.3 W2); "
    "Qbar0(a, W) = expit(-0.2 + a (0.5 + 0.6 W1) - W1 + 2 W2)",
    "basis": "Linear(['W1']) on the difference scale, in sample",
    "drtmle": "guard=('Q', 'g'), univariate reduction, cubic reduced regressions, "
    "g_bounds=(0.01, 0.99)",
    "ctmle": "strategy='oat', oat_design='per_arm', logistic per-arm design",
    "outcome_learners": {
        "wrong": "logistic GLM on (A, W1)",
        "correct": "logistic GLM on (A, W1, W2, A W1)",
    },
    "mechanism_learners": {
        "known": "the declared g0, by treatment_probabilities",
        "glm": "logistic GLM on (W1, W2)",
        "slow": "logistic GLM on (W1, W2) plus the logit offset n^(-1/3) W2 (1 + W1), "
        "a constructed error",
    },
    "groups": {
        group.label: f"n={group.n}, R={group.replicates}, {group.method}, outcome {group.outcome}, "
        f"mechanism {group.mechanism}"
        for group in GROUPS
    },
}


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Draw one primary replication from this study's declared seed."""
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def primary_rows(
    frame: pd.DataFrame, truths: Mapping[str, float], scenario: str, replicate: int
) -> list[dict[str, Any]]:
    """One primary row: the ``dr_known_4k`` intercept."""
    group = GROUP_BY_LABEL[scenario]
    fold_seed = stream_seed(STUDY, "fold_partition", "primary", scenario, replicate)
    estimates = fit_group(group, frame, random_state=fold_seed)["estimates"]
    rows = []
    for name in SCENARIOS[scenario]:
        term = name.removeprefix("cate_projection[").removesuffix("]")
        estimate = estimates[term]
        reference = float(truths[name])
        std_error, low, high = reported_inference(estimate)
        rows.append(
            {
                "implementation": STUDY.implementation,
                "scenario": scenario,
                "replicate": replicate,
                "n": len(frame),
                "estimand": name,
                "truth": reference,
                "estimate": float(estimate.psi),
                "inference_estimate": float(estimate.psi),
                "std_error": std_error,
                "ci_lower": low,
                "ci_upper": high,
                "inference_scale": "identity",
                "covered": int(low <= reference <= high),
                "initial_estimate": float("nan"),
            }
        )
    return rows


def _draw(key: tuple[str, int, int]) -> tuple[pd.DataFrame, dict[str, float]]:
    scenario, index, n = key
    return draw_scenario(scenario, n, index)


def _replicate(
    key: tuple[str, int, int], drawn: tuple[pd.DataFrame, dict[str, float]]
) -> list[dict[str, Any]]:
    scenario, index, _ = key
    frame, truths = drawn
    return primary_rows(frame, truths, scenario, index)


def draw_and_fit(*, replicates: int, n: int, n_jobs: int = STUDY_JOBS) -> replication.PhaseResult:
    """Draw and fit every declared primary replication."""
    outcomes = replication.fit_primary(
        [(scenario, index, n) for scenario in STUDY.scenarios for index in range(replicates)],
        _replicate,
        draw=_draw,
        rows_at=None,
        n_jobs=n_jobs,
        implementation=STUDY.implementation,
        seed_of=lambda key: replicate_seed(STUDY, key[0], key[1]),
    )
    built = pd.DataFrame([row for records in outcomes.kept for row in records])
    return replication.result(outcomes, built.loc[:, list(REPLICATE_COLUMNS)])


# ---------------------------------------------------------------------- the design numbers


def conservative_pass_probability(ratio: float, replicates: int) -> float:
    """P(the 99% SE-ratio interval lies wholly above the band) at a true ``ratio``.

    The interval is normal about ``ratio`` with SD ``ratio sqrt(cv^2 + (kurtosis - 1) / 4) /
    sqrt(R)`` at ``cv = 0.1`` and kurtosis 3, the approximation of
    :func:`tests.studies.conditional_projection.calibration_row_pass` (0.014 to 0.017 at
    ``R = 2,400``).
    """
    margins = STUDY.margins
    spread = ratio * math.sqrt(0.1**2 + 0.5) / math.sqrt(replicates)
    z = float(stats.norm.ppf(0.5 + margins.confidence_level / 2.0))
    upper = margins.calibration_se_ratio[1]
    return float(stats.norm.sf((upper + z * spread - ratio) / spread))


def sanity_pass_probability(ratio: float, replicates: int) -> float:
    """P(the SE ratio's point lies in the sanity band) at a true ``ratio``."""
    low, high = STUDY.margins.se_ratio_sanity
    spread = ratio * math.sqrt(0.1**2 + 0.5) / math.sqrt(replicates)
    return float(stats.norm.cdf((high - ratio) / spread) - stats.norm.cdf((low - ratio) / spread))


def design_numbers() -> dict[str, Any]:
    """Every design number that needs no pilot, computed rather than copied (plan 5.1 to 5.4)."""
    return {
        "law": law.design_table(),
        "calibration_arms": {arm: len(rows) for arm, rows in CALIBRATION_ARMS.items()},
        "cpu_hours": run_cpu_hours(),
        "pilot_cpu_hours": pilot_cpu_hours(),
    }


# ------------------------------------------------------------------------- the run's cost


def run_cpu_hours(calibration_r: int | None = None) -> float:
    """Single-thread CPU-hours of the declared property run and its primary phase.

    ``calibration_r`` replaces the calibration group's ``R`` to price another pilot outcome.
    """
    primary = GROUP_BY_LABEL[PRIMARY_GROUP].seconds * PRIMARY_REPLICATES
    total = 0.0
    for group in GROUPS:
        count = group.replicates
        if calibration_r is not None and group.arms:
            count = calibration_r
        total += group.seconds * count
    return (total + primary) / 3_600.0


def pilot_groups() -> tuple[str, ...]:
    """The groups the pilot fits: every group with a gated calibration arm."""
    return tuple(g.label for g in GROUPS if g.arms)


def pilot_cpu_hours(draws: int = PILOT_DRAWS) -> float:
    """Single-thread CPU-hours of the pilot: ``draws`` per pilot group."""
    return sum(GROUP_BY_LABEL[label].seconds * draws for label in pilot_groups()) / 3_600.0


def smoke_cpu_hours(replicates: int = 20) -> float:
    """Single-thread CPU-hours of a ``replicates``-draw smoke of every group."""
    return sum(g.seconds * replicates for g in GROUPS) / 3_600.0


#: Plan 5.0's cost rule, kept from PR 2: a group whose smoke cost exceeds this multiple of its
#: declared ``seconds`` runs as a second declared run.
COST_GATE = 1.5

#: The smoke timings (``group``, ``fits``, ``seconds_per_fit``), written by
#: ``tests/canonical/projection_methods/regenerate.py --property-smoke 20 --write-timings``.
SMOKE_TIMINGS = ROOT / "tests" / "studies" / "projection_methods_smoke.csv"

#: Every file the declared run writes into :attr:`StudyRecord.artifacts`.
RUN_FILES = (
    "replicates.parquet",
    "summary.csv",
    "equivalence.csv",
    "performance-tests.csv",
    "property-replicates.parquet",
    "properties.csv",
    "manifest.json",
    "run.log",
)
CONDITIONAL_RUN_FILES = (
    "property-failures.csv",
    "property-failure-accounting.csv",
    "property-failure-witnesses.parquet",
)
