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
        "describe": list(result.config.describe(contrast=contrast)),
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
