"""Fixtures of ``tests/unit/test_longitudinal_repeats.py``: fits, samples and golden summaries.

Every fit estimates every nuisance with saturated cell means (:class:`CellMeans`), so a draw
differs from another draw only through its fold assignment.  The golden record of
``repeats=1`` in :mod:`tests.unit._longitudinal_repeats_golden` was written by
:func:`golden_records` before the repeated-draw refactor changed any source file.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

import tests.discrete_law_longitudinal as law
import tests.discrete_law_longitudinal_incremental as incremental
from cleverly.longitudinal import IPSI, LTMLE, SDR, DynamicRegimen
from cleverly.longitudinal.estimator import DEFAULT_LTMLE_G_BOUNDS, LongitudinalConfig
from cleverly.longitudinal.regimen import describe_plan
from cleverly.msm import MSM
from tests.discrete_law_longitudinal import CellMeans
from tests.studies.ltmle_crossfit_properties import sample
from tests.unit.test_long_panel import build as build_long_panel
from tests.unit.test_long_panel import long_panel

#: The static and dynamic plans of law D, reference ``never``.
REGIMENS: dict[str, Any] = {
    "never": 0,
    "always": 1,
    "treat_if_l2": DynamicRegimen(
        "treat_if_l2", (0, lambda history: history["L2"]), rule_kind="known"
    ),
}
REFERENCE = "never"
COLUMNS: dict[str, Any] = {
    "outcome": "Y",
    "treatment": ["A1", "A2"],
    "baseline": ["W"],
    "time_varying": [[], ["L2"]],
    "censoring": ["C1", "C2"],
}
#: The shipped default.  No true node probability of law D is below 0.25, so it never binds
#: at the truth, and a training cell with no treated row cannot put a weight of 1e8 on a row.
G_BOUNDS = (0.01, 1.0)
#: The working model of the MSM fixture: an intercept and a duration of treatment.
DURATION = {"never": 0.0, "always": 2.0, "treat_if_l2": 1.0}
#: The sample size and seed of the shared law-D frame.
N = 2_000
SEED = 4_601


def frame(n: int = N, seed: int = SEED) -> pd.DataFrame:
    """A sample of law D."""
    return sample(law.PROBS, n, seed)


def cells() -> dict[str, Any]:
    """Saturated cell means in every learner slot."""
    return {
        "outcome_learner": CellMeans(),
        "pseudo_learner": CellMeans(),
        "treatment_learner": CellMeans(),
        "censoring_learner": CellMeans(),
    }


def _settings(overrides: dict[str, Any]) -> dict[str, Any]:
    settings: dict[str, Any] = {
        **cells(),
        "n_folds": 5,
        "learner_folds": 2,
        "g_bounds": G_BOUNDS,
        "random_state": 11,
    }
    settings.update(overrides)
    if settings.get("repeats", 1) == 1:
        settings.pop("repeats", None)
    return settings


def ltmle(rows: pd.DataFrame, **overrides: Any) -> Any:
    """``LTMLE`` of the three law-D plans."""
    return LTMLE(REGIMENS, reference=REFERENCE, **_settings(overrides)).fit(rows, **COLUMNS)


def sdr(rows: pd.DataFrame, **overrides: Any) -> Any:
    """``SDR`` of the three law-D plans."""
    return SDR(REGIMENS, reference=REFERENCE, **_settings(overrides)).fit(rows, **COLUMNS)


def msm_model() -> MSM:
    """An intercept and a duration term over the three law-D plans, unit weights."""

    def design(label: Any, horizon: int, rows: Any) -> np.ndarray:
        del horizon
        return np.column_stack(
            [np.ones(len(rows)), np.full(len(rows), float(DURATION[str(label)]))]
        )

    def weight(label: Any, horizon: int, rows: Any) -> np.ndarray:
        del label, horizon
        return np.ones(len(rows))

    return MSM(
        design=design,
        terms=("(intercept)", "duration"),
        weights=weight,
        weights_kind="known",
        design_kind="known",
    )


def msm(rows: pd.DataFrame, **overrides: Any) -> Any:
    """``LTMLE`` with the two-term working model over the law-D plans."""
    return LTMLE(REGIMENS, msm=msm_model(), **_settings(overrides)).fit(rows, **COLUMNS)


#: The incremental plans: the natural course is the reference.
TILTS = incremental.DELTAS


def ipsi_frame(n: int = N, seed: int = SEED) -> pd.DataFrame:
    """A sample of the censored incremental law ``C``."""
    return incremental.sample(incremental.LAW_C, n, seed)


def ipsi(rows: pd.DataFrame, **overrides: Any) -> Any:
    """``IPSI`` of three constant tilts on law ``C``."""
    from tests.studies.longitudinal_incremental_common import Q_BOUNDS, constant_plans, regimens

    settings = _settings(overrides)
    settings.setdefault("q_bounds", Q_BOUNDS)
    return IPSI(regimens(constant_plans(TILTS)), reference=incremental.REFERENCE, **settings).fit(
        rows, **incremental.columns(incremental.LAW_C)
    )


def survival_frame(n: int = N, seed: int = SEED) -> pd.DataFrame:
    """A sample of the three-node survival law ``D`` of the incremental oracle."""
    return incremental.sample(incremental.LAW_D, n, seed)


def survival(rows: pd.DataFrame, **overrides: Any) -> Any:
    """``LTMLE`` of always and never at horizons 1 to 3 on the survival law."""
    return LTMLE(
        {"never": 0, "always": 1},
        reference="never",
        horizons=[1, 2, 3],
        **_settings(overrides),
    ).fit(rows, **incremental.columns(incremental.LAW_D))


def clustered_frame(n: int = N, seed: int = SEED) -> pd.DataFrame:
    """A sample of the clustered law, in clusters of ten."""
    return incremental.sample_clustered(n, 10, seed)


def clustered(rows: pd.DataFrame, **overrides: Any) -> Any:
    """``LTMLE`` of always and never on the clustered law, grouped by ``id``."""
    from tests.studies.longitudinal_incremental_common import Q_BOUNDS

    target = incremental.LAW_CK.halves[0]
    settings = _settings(overrides)
    settings.setdefault("q_bounds", Q_BOUNDS)
    return LTMLE({"never": 0, "always": 1}, reference="never", **settings).fit(
        rows, id="id", **incremental.columns(target)
    )


def panel_frame() -> pd.DataFrame:
    """The long panel of ``tests/unit/test_long_panel.py``."""
    return long_panel(n=600, seed=2)


def panel(rows: pd.DataFrame, **overrides: Any) -> Any:
    """``LTMLE`` of an always-offer and a never-offer plan on the long panel.

    The panel's covariates are continuous, so the learners are logistic and linear
    regressions rather than cell means.
    """
    from sklearn.linear_model import LinearRegression, LogisticRegression

    settings = _settings(overrides)
    settings.update(
        outcome_learner=LogisticRegression(max_iter=1000),
        pseudo_learner=LinearRegression(),
        treatment_learner=LogisticRegression(max_iter=1000),
        censoring_learner=LogisticRegression(max_iter=1000),
        n_folds=overrides.get("n_folds", 2),
    )
    return LTMLE(
        {"never": ("none",) * 3, "always": ("offer",) * 3}, reference="never", **settings
    ).fit(build_long_panel(rows))


#: Each golden fit: the sample it reads and the fit.
GOLDEN: dict[str, tuple[Callable[[], pd.DataFrame], Callable[..., Any]]] = {
    "ltmle": (frame, ltmle),
    "sdr": (frame, sdr),
    "ipsi": (ipsi_frame, ipsi),
    "msm": (frame, msm),
    "survival": (survival_frame, survival),
    "clustered": (clustered_frame, clustered),
    "panel": (panel_frame, panel),
}


def fit_recording(name: str, **overrides: Any) -> tuple[Any, list[str]]:
    """One golden fit and the text of every warning it raised."""
    rows_of, build = GOLDEN[name]
    rows = rows_of()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = build(rows, **overrides)
    texts = [f"{item.category.__name__}: {item.message}" for item in caught]
    return result, texts


def numbers(result: Any) -> dict[str, float]:
    """The floats a golden record pins: each estimate and two projections of its curve."""
    probe = np.random.default_rng(0).normal(size=result.data.n)
    out: dict[str, float] = {}
    for name, estimate in result.estimates.items():
        curve = np.asarray(estimate.influence_curve, dtype=float)
        out[f"{name}|psi"] = float(estimate.psi)
        out[f"{name}|se"] = float(estimate.plugin_std_error)
        out[f"{name}|square"] = float(np.mean(curve**2))
        out[f"{name}|probe"] = float(curve @ probe)
    if result.simultaneous is not None:
        out["band|critical"] = float(result.simultaneous.critical_value)
    return out


def texts(result: Any, warned: list[str]) -> dict[str, Any]:
    """The text a golden record pins: digests, warnings, settings and score rows."""
    contrast = any(estimate.scale != "level" for estimate in result.estimates.values())
    record: dict[str, Any] = {
        "fold_fingerprint": result.provenance.fold_fingerprint,
        "warnings": sorted(warned),
        "describe": list(describe_config(result.config, contrast=contrast)),
    }
    if result.config.estimator == "tmle":
        frame_ = result.diagnostics.score_equations().to_frame()
        record["score_rows"] = [
            "|".join(str(frame_[column].iloc[row]) for column in ("regimen", "horizon", "time"))
            for row in range(len(frame_))
        ]
    return record


def golden_records() -> dict[str, dict[str, Any]]:
    """The golden record of every fixture fit at ``repeats=1``."""
    out: dict[str, dict[str, Any]] = {}
    for name in GOLDEN:
        result, warned = fit_recording(name)
        out[name] = {"numbers": numbers(result), "texts": texts(result, warned)}
    return out


# ------------------------------------------------------------------ further designs
#: The wide roles of the shipped weighted and competing-risks samples.
SHARED: dict[str, Any] = {
    "treatment": ("A1", "A2"),
    "baseline": ("W1", "W2"),
    "time_varying": ((), ("L2",)),
    "censoring": ("C1", "C2"),
}


def _parametric(overrides: dict[str, Any]) -> dict[str, Any]:
    """Logistic and linear learners, two folds: the samples below have continuous covariates."""
    from sklearn.linear_model import LinearRegression, LogisticRegression

    settings: dict[str, Any] = {
        "outcome_learner": LogisticRegression(max_iter=1000),
        "pseudo_learner": LinearRegression(),
        "treatment_learner": LogisticRegression(max_iter=1000),
        "censoring_learner": LogisticRegression(max_iter=1000),
        "n_folds": 2,
        "random_state": 11,
    }
    settings.update(overrides)
    if settings.get("repeats", 1) == 1:
        settings.pop("repeats", None)
    return settings


def weighted_frame() -> pd.DataFrame:
    """The shipped survey-weighted longitudinal sample."""
    from cleverly.datasets import make_longitudinal_weighted

    return make_longitudinal_weighted(n=1400, seed=6)[0]


def weighted(rows: pd.DataFrame, **overrides: Any) -> Any:
    """``LTMLE`` with probability weights that were declared estimated."""
    return LTMLE({"always": 1, "never": 0}, reference="never", **_parametric(overrides)).fit(
        rows,
        outcome="Y",
        weights="w",
        weights_type="probability",
        weights_estimated=True,
        **SHARED,
    )


def competing_frame() -> pd.DataFrame:
    """The shipped competing-risks sample."""
    from cleverly.datasets import make_longitudinal_competing

    return make_longitudinal_competing(n=1800, seed=5)[0]


def competing(rows: pd.DataFrame, **overrides: Any) -> Any:
    """``LTMLE`` of the cause-specific risks of relapse and death."""
    return LTMLE({"always": 1, "never": 0}, reference="never", **_parametric(overrides)).fit(
        rows, outcome={"relapse": ("R1", "R2"), "death": ("D1", "D2")}, **SHARED
    )


def known_frame() -> pd.DataFrame:
    """The censored SMART sample whose mechanisms are known."""
    from tests.unit import _known_mechanism_support as smart

    return smart.smart_frame(censoring=True)


def known(rows: pd.DataFrame, **overrides: Any) -> Any:
    """``LTMLE`` with declared treatment and censoring probabilities."""
    from tests.unit import _known_mechanism_support as smart
    from tests.unit._natural_course_support import NeverFit

    columns = {**smart.SMART_COLUMNS, "censoring": ["C1", "C2"]}
    settings = _parametric(overrides)
    settings.update(treatment_learner=NeverFit(), censoring_learner=NeverFit())
    return LTMLE({"always": 1, "never": 0}, reference="never", **settings).fit(
        rows,
        treatment_probabilities=rows[["g1_1", "g2_1"]].to_numpy(dtype=float),
        censoring_probabilities=rows[["r1", "r2"]].to_numpy(dtype=float),
        **columns,
    )


def mtp_frame() -> pd.DataFrame:
    """A sample of the exact multivalued law of the modified-treatment-policy oracle."""
    from tests import discrete_law_longitudinal_mtp as exact

    return exact.frame()


def mtp(rows: pd.DataFrame, **overrides: Any) -> Any:
    """``LTMLE`` of a modified treatment policy on a dose read as continuous."""
    from tests import discrete_law_longitudinal_multivalue as multivalue
    from tests import longitudinal_mtp as exact_plans

    settings = _settings(overrides)
    settings.update(
        treatment_learner=multivalue.CellProbabilities(),
        n_folds=overrides.get("n_folds", 2),
        density_bins=overrides.get("density_bins"),
        ratio=overrides.get("ratio", "density"),
    )
    settings.pop("censoring_learner")
    atoms = {"A1": (0, 1, 2, 3), "A2": (0, 1, 2, 3)}
    return LTMLE(
        exact_plans.regimens(("natural", "up then history")), reference="natural", **settings
    ).fit(
        rows,
        outcome="Y",
        treatment=["A1", "A2"],
        baseline=["W"],
        time_varying=[[], ["L2"]],
        continuous_treatment=["A1", "A2"],
        dose_atoms=atoms,
    )


def settings_frame() -> pd.DataFrame:
    """The law-D sample at a smaller size."""
    return frame(n=1_200)


def settings(rows: pd.DataFrame, **overrides: Any) -> Any:
    """``LTMLE`` with every inference and bookkeeping setting away from its default."""
    chosen = {
        "horizons": None,
        "q_bounds": None,
        "alpha": 0.999,
        "alpha_sig": 0.1,
        "simultaneous": True,
        "n_multiplier": 500,
        "multiplier_kind": "mammen",
        "max_iter": 5,
        "tol": 1e-8,
        "run_id": "repeat witness",
        "n_jobs": 1,
        "bootstrap_resampling": "iid",
        "stopping": "scaled_var",
        "step_bound": 0.5,
        "step_search": False,
        "iterate": "best",
        **overrides,
    }
    return LTMLE(REGIMENS, reference=REFERENCE, **_settings(chosen)).fit(
        rows, family="binomial", **COLUMNS
    )


def time_to_event_frame() -> pd.DataFrame:
    """The one-row-per-unit layout of the exact point-survival law, on the unit grid."""
    from tests.discrete_law_point_survival import survival_law

    return survival_law(censor_first=False).long_frame((1.0, 2.0, 3.0))


def time_to_event(rows: pd.DataFrame, **overrides: Any) -> Any:
    """``LTMLE`` of two held arms on a container from ``from_time_to_event``."""
    from cleverly.longitudinal import LongitudinalData
    from tests.discrete_law_point_survival import CellMeans as SurvivalMeans
    from tests.discrete_law_point_survival import CellProbabilities

    data = LongitudinalData.from_time_to_event(
        rows, time="time", event="event", treatment="A", baseline=["W"]
    )
    settings = _settings(overrides)
    settings.update(
        outcome_learner=SurvivalMeans(),
        pseudo_learner=SurvivalMeans(),
        treatment_learner=CellProbabilities(),
        censoring_learner=SurvivalMeans(),
        n_folds=overrides.get("n_folds", 2),
    )
    return LTMLE({"a0": 0, "a1": 1}, reference="a0", **settings).fit(data)


#: Every design that ``test_each_draw_reproduces_a_single_draw_fit`` refits draw by draw.
def repeated_frame() -> pd.DataFrame:
    """The shipped repeated-outcome sample: a binary outcome at each of three nodes."""
    from cleverly.datasets import make_longitudinal_repeated

    return make_longitudinal_repeated(n=1_500, seed=7, backend="pandas")[0]


def repeated(rows: pd.DataFrame, **overrides: Any) -> Any:
    """``LTMLE`` of the X32a plans on a repeated outcome, a mean at each of three horizons."""
    from tests.studies import canonical_ltmle_repeated as x32a

    return LTMLE(x32a.REGIMENS, reference=x32a.REFERENCE, **_parametric(overrides)).fit(
        rows, **x32a.COLUMNS
    )


DESIGNS: dict[str, tuple[Callable[[], pd.DataFrame], Callable[..., Any]]] = {
    **GOLDEN,
    "weighted": (weighted_frame, weighted),
    "competing": (competing_frame, competing),
    "known": (known_frame, known),
    "mtp": (mtp_frame, mtp),
    "settings": (settings_frame, settings),
    "time_to_event": (time_to_event_frame, time_to_event),
    "repeated": (repeated_frame, repeated),
}


def describe_config(self: LongitudinalConfig, *, contrast: bool) -> list[str]:
    """The settings lines the longitudinal summary printed before it was laid out in tables.

    A byte-for-byte copy of the removed ``LongitudinalConfig.describe``: the repeats golden
    and the configuration tests read the fields through it.

    Parameters
    ----------
    contrast : bool
        Whether the report holds a contrast, which the reference line describes. A
        report of regime means prints no reference line.

    Returns
    -------
    list of str
        One line per setting.
    """
    plans = ", ".join(f"{regimen.label}=({describe_plan(regimen)})" for regimen in self.regimens)
    lines = [
        f"time points: {self.n_times}",
        f"outcome family: {self.family}",
        f"regimens: {plans}",
    ]
    if self.estimator == "sdr":
        lines.insert(
            0,
            "estimator: sequentially doubly robust (Diaz, Williams, Hoffman and Schenck "
            "2023, Section 5.3), no fluctuation",
        )
    if self.estimator == "ipsi":
        lines.insert(
            0,
            "estimator: Kennedy (2019) Algorithm 1, the mean of the influence function "
            "at the fitted nuisances (Kim, Kennedy and Naimi 2021, Algorithm 1, under "
            "censoring), no fluctuation",
        )
    if self.estimator == "gcomp":
        lines.insert(
            0,
            "estimator: sequential g-computation, the iterated conditional expectation "
            "plug-in (Bang and Robins 2005, Section 4), no fluctuation; an interval "
            "assumes every outcome GLM is correctly specified",
        )
    if self.estimator == "parametric_gformula":
        lines.insert(
            0,
            "estimator: parametric g-formula, forward simulation of the fitted covariate, "
            "outcome and hazard laws (Robins 1986; McGrath et al. 2020), no fluctuation; "
            "an interval assumes every model is correctly specified",
        )
    if self.estimator == "ipw":
        form = (
            "Hajek (normalized by the weight sum)"
            if self.normalization == "hajek"
            else "Horvitz-Thompson (normalized by n)"
        )
        lines.insert(
            0,
            f"estimator: inverse probability weighting, {form}; an interval assumes "
            "every treatment and censoring GLM is correctly specified",
        )
    if self.held_treatment is not None:
        lines.insert(
            1,
            f"treatment: one baseline decision {self.held_treatment!r}, held over "
            f"{self.n_times} node(s)",
        )
    if self.time_grid is not None:
        lines.insert(1, f"time grid: {', '.join(f'{g:g}' for g in self.time_grid)}")
    if self.panel is not None:
        lines.insert(1, self.panel)
    if self.outcome_kind != "end_of_study":
        lines.insert(
            1,
            (
                f"outcome: repeated, a non-absorbing outcome at {', '.join(self.outcome_names)}"
                if self.outcome_kind == "repeated"
                else f"outcome: survival, event indicator at {', '.join(self.outcome_names)}"
            ),
        )
        reported = ", ".join(str(horizon) for horizon in self.horizons)
        lines.insert(2, f"horizons reported: t = {reported}")
        if self.causes:
            lines[1] = (
                "outcome: competing risks, absorbing causes "
                f"{', '.join(self.causes)} at {', '.join(self.outcome_names)}"
            )
            lines.insert(3, f"causes reported: {', '.join(self.causes)}")
    dynamic = {
        regimen.label
        for regimen in self.regimens
        if isinstance(regimen, DynamicRegimen)
        and (regimen.has_rule or regimen.has_policy or not regimen.has_tilt)
    }
    if any(isinstance(regimen, DynamicRegimen) and regimen.has_tilt for regimen in self.regimens):
        lines.append(
            "  an 'o' node is an odds tilt: each unit draws its arm from "
            "q_t = delta*pi_t/(delta*pi_t + 1 - pi_t), with pi_t the treatment mechanism "
            "(Kennedy 2019, Equation (1)); every unit at risk is in the plan"
        )
    if dynamic:
        # Worth a line rather than leaving "d" to be guessed at, because it changes
        # how the follower counts below should be read: a static regimen's followers
        # are whoever happened to receive that sequence, and a rule's are whoever the
        # rule would have treated -- a set this sample determines.
        # A repeated outcome's earlier nodes open the later blocks, so they are in H_t.
        history = (
            "L_1, ..., L_t, with Y_{t-1} first in each block L_t, t > 1"
            if self.outcome_kind == "repeated"
            else "L_1, ..., L_t"
        )
        lines.append(
            f"  a 'd' is a rule d_t(H_t) read off [W, {history}]; its followers "
            "are a covariate-dependent set, so the counts below describe this sample"
        )
        # Only for the regimens that have a rule in them, and only when one does:
        # printing a digest of a plan the line above already spells out in full would
        # add noise to every static report to say nothing.
        for label, digest in self.plan_fingerprints:
            if label in dynamic:
                lines.append(f"  assigned arms, {label}: {digest}")
    policies = [
        regimen.label
        for regimen in self.regimens
        if isinstance(regimen, DynamicRegimen) and regimen.has_policy
    ]
    if policies:
        lines.append(
            "  a 'q' is a known policy q_t(a | H_t) read off [W, L_1, ..., L_t] and the "
            "earlier treatments; each unit draws its arm there, and the plan's units are "
            "those whose observed arm the policy can draw"
        )
    for label, nodes in self.policy_point_mass_nodes:
        listed = ", ".join(str(node) for node in nodes)
        lines.append(
            f"  policy {label}: one-hot at node(s) {listed}, so resolved as the rule it equals"
        )
    if self.msm_terms is not None:
        # A working model has no reference regimen -- what the intercept is against is
        # whatever the design makes it -- so the working model stands where that line
        # would, rather than beside a field that names nothing.
        lines.append(
            f"working model: {len(self.msm_terms)} term(s) "
            f"{', '.join(self.msm_terms)}, link={self.msm_link}"
        )
    elif contrast:
        # A report of means holds no contrast for the reference to describe. The field
        # stays set, because a replay of the result reads it.
        lines.append(f"reference: {self.reference}")
    lines += [
        # A single fold is not cross-fitting, and printing "1 fold(s)" reads as
        # though it were: the nuisances are then fitted on the rows they predict
        # for, which the reported variance does not account for.
        "cross-fitting: none; nuisances fitted in sample"
        if self.n_folds <= 1
        else f"cross-fitting: {self.n_folds} fold(s)"
        + (f", median over {self.repeats} draws" if self.repeats > 1 else ""),
        # Both factors enter the product before its prefix is bounded, as in ltmle's
        # CalcCumG.  Saying "mechanism at every node" used to conceal the order.
        f"g_bounds: fixed [{self.g_bounds[0]:.4g}, {self.g_bounds[1]:.4g}] on each "
        + (
            "partial path treatment-and-censoring probability"
            if self.estimator == "sdr"
            else "cumulative censoring and non-tilt treatment probability; the tilt "
            "factor is unbounded"
            if self.estimator == "ipsi"
            else "cumulative treatment-and-censoring probability"
        )
        + (
            " (package default, an R ltmle-compatible heuristic; "
            "inspect the truncation diagnostics)"
            if self.g_bounds == DEFAULT_LTMLE_G_BOUNDS
            else ""
        ),
    ]
    if len(self.outcome_names) > 1:
        lines.append(f"survival model: {self.survival_model}")
        lines.append(
            f"targeting: {self.targeting}"
            + (f", step size {self.step_size:g}" if self.targeting == "one_step" else "")
        )
        lines.append(
            "isotonic projection: "
            + ("each curve's levels and band limits" if self.isotonic else "none")
        )
    if self.known_factors:
        lines.append(
            f"known factors (declared, no learner): {', '.join(self.known_factors)}; no "
            "truncation bound moved a declared value"
        )
    if self.node_bounds:
        lines.append(
            "node_bounds: "
            + "; ".join(
                f"{'' if cause is None else f'{cause} '}t={time:g} in [{lower:g}, {upper:g}]"
                for cause, time, lower, upper in self.node_bounds
            )
            + (
                " (bounded cause-specific hazards at the named nodes, on the hazard scale)"
                if self.survival_model == "hazard"
                else " (bounded logistic fluctuation at the named nodes)"
            )
        )
    if self.q_bounds is not None:
        lines.append(f"q_bounds: [{self.q_bounds[0]:.4g}, {self.q_bounds[1]:.4g}]")
    lines.append(f"confidence level: {(1 - self.alpha_sig) * 100:g}%")
    if self.random_state is not None:
        lines.append(f"random_state: {self.random_state}")
    elif self.multiplier_seed is not None:
        lines.append(f"multiplier seed: {self.multiplier_seed} (drawn, random_state=None)")
    return lines
