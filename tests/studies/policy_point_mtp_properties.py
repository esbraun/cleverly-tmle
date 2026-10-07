"""Property families of ``policy-point-mtp``.

The estimand of every single-parameter family is ``ate_policy[x1.25 vs natural course]``
(label ``x1_25``) unless the table says otherwise.

==============================  ===========================================================
family                          cells, replications and size
==============================  ===========================================================
``double_robustness``           ``x1_25`` in four nuisance configurations; 1,000 at
                                n = 2,000
``root_n_and_efficiency``       ``x1_25`` at n = 500, 2,000 and 8,000; 600 each.  n = 500 is
                                the ladder's control rung
``root_n_rate``                 the two rate rows of ``x1_25``, from the ladder
``interval_calibration``        ``x1_25``, ``piecewise``, the declared policy ``halve`` and
                                ``x1_25`` on the classifier ratio route with the
                                correctly specified :class:`OracleLogOdds`
                                (``classifier_route``); 2,000 at n = 2,000 each, with the two
                                derived controls of each
``type_i_error``                ``x1_25`` under a law where the dose does not move the
                                outcome; 600 at n = 4,000
``power``                       ``x1_25`` under the law; 600 at n = 4,000
``targeting_necessity``         ``x1_25``, targeted and the same fit's untargeted plug-in,
                                with a correct density only; 1,000 at n = 2,000
``inverse_necessity``           ``halve`` with the declared inverse, and the same fit with the
                                inverse dropped from Equation (3); a correct density only;
                                1,000 at n = 2,000
==============================  ===========================================================

The plan's ``ratio_route`` is the ``classifier_route`` label of ``interval_calibration``.  The
efficiency bounds (:data:`EFFICIENCY_SD`) are the standard deviation of the closed-form
influence function :math:`(r - 1)(Y - \\bar Q) + \\bar Q(d(A, W), W) - \\bar Q(A, W)`, over
:data:`BOUND_DRAWS` draws.  The failure rule is ``longitudinal-mtp``'s.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression

from cleverly.interventions import policy as policy_module
from cleverly.utils.parallel import map_parallel
from tests.parallel import STUDY_JOBS
from tests.studies import canonical_policy_point_mtp as study
from tests.studies.evidence.properties import (
    REPLICATE_COLUMNS,
    PropertyCell,
    control_row,
    replicate_row,
)
from tests.studies.evidence.property_verdicts import (
    apply_shared_verdicts,
    calibration_controls,
    calibration_verdicts,
    finish,
    necessity_verdicts,
)
from tests.studies.evidence.seeds import stream_seed

STUDY = study.STUDY
DOUBLE_ROBUST_REPLICATES = 1_000
DOUBLE_ROBUST_N = 2_000
RATE_REPLICATES = 600
RATE_SIZES = (500, 2_000, 8_000)
CALIBRATION_REPLICATES = 2_000
CALIBRATION_N = 2_000
NULL_REPLICATES = 600
NULL_N = 4_000
TARGETING_REPLICATES = 1_000
TARGETING_N = 2_000
INVERSE_REPLICATES = 1_000
INVERSE_N = 2_000
BOUND_DRAWS = 400_000
BOUND_SEED = 20261053

EFFICIENCY_RATIO_BAND = (0.90, 1.10)
SHRUNKEN_SE_FACTOR = 0.70
TARGETING_DISPLACEMENT = 0.10
INVERSE_DISPLACEMENT = 0.10
CRITICAL = float(norm.ppf(1.0 - STUDY.margins.alpha / 2.0))

REFERENCE = study.REFERENCE
NAMES = {
    "x1_25": "ate_policy[x1.25 vs natural course]",
    "piecewise": "ate_policy[piecewise vs natural course]",
    "halve": "ate_policy[halve below 3 vs natural course]",
    "classifier_route": "ate_policy[x1.25 vs natural course]",
}
CALIBRATION_LABELS = ("x1_25", "piecewise", "halve", "classifier_route")
X125 = NAMES["x1_25"]
HALVE = NAMES["halve"]


# ---------------------------------------------------------------------------- bounds


def _true_ratio(label: str, a: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """Equation (3) on the truncated normal, at the observed dose."""
    g = study.truncated_pdf(a, mean)
    safe = np.where(g > 0.0, g, 1.0)
    if label == "x1.25":
        moved = study.truncated_pdf(a / study.FACTOR, mean) / study.FACTOR * (a <= study.CAP)
        return np.where(g > 0, moved / safe, 0.0) + (a * study.FACTOR > study.CAP)
    if label == "piecewise":
        moved = study.truncated_pdf(a + study.DROP, mean) * (a + study.DROP > study.KNEE)
        return np.where(g > 0, moved / safe, 0.0) + (a <= study.KNEE)
    if label == "halve below 3":
        source = 2.0 * a - study.HALVE_BELOW
        moved = 2.0 * study.truncated_pdf(source, mean) * (source <= study.HALVE_BELOW)
        return np.where(g > 0, moved / safe, 0.0) + (a > study.HALVE_BELOW)
    return np.ones_like(a)


class OracleLogOdds(BaseEstimator):
    r"""The correctly specified classifier of the stacked ratio route, on one oracle feature.

    The stacked classification of Section 5.4 of Díaz et al. (2023) has the odds
    :math:`g^d(a \mid w) / g(a \mid w)`.  This learner fits a logistic regression on the one
    feature :math:`\log r(a, w)`, the true log ratio of ``label``, so the true odds are the
    model at ``(0, 1)``.  The natural course has labels independent of the feature, and the
    model holds it at ``(0, 0)``.  The design is ``[A, W1, W2]``.

    Parameters
    ----------
    label : str
        The policy whose ratio is the feature.
    """

    def __init__(self, label: str = "x1.25") -> None:
        self.label = label

    def _feature(self, X: Any) -> np.ndarray:
        design = np.asarray(X, dtype=float)
        mean = study.dose_mean(design[:, 1], design[:, 2])
        ratio = _true_ratio(self.label, design[:, 0], mean)
        return np.log(np.clip(ratio, 1e-13, None)).reshape(-1, 1)

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> OracleLogOdds:
        self.model_ = LogisticRegression(penalty=None, max_iter=1000).fit(
            self._feature(X), np.asarray(y), sample_weight=sample_weight
        )
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        return np.asarray(self.model_.predict_proba(self._feature(X)), dtype=float)


def contrast_sd(label: str) -> float:
    """The efficiency bound of ``E[Y^label] - E[Y]``, as a standard deviation."""
    frame = study.sample(BOUND_DRAWS, BOUND_SEED)
    a = frame["A"].to_numpy()
    w1, w2 = frame["W1"].to_numpy(), frame["W2"].to_numpy()
    y = frame["Y"].to_numpy()
    q = study.outcome_mean(a, w1, w2)
    r = _true_ratio(label, a, study.dose_mean(w1, w2))
    curve = (r - 1.0) * (y - q) + study.outcome_mean(study.MAPS[label](a), w1, w2) - q
    return float(np.std(curve))


#: The efficiency bound of each label, pinned from :func:`contrast_sd`;
#: ``tests/unit/test_policy_point_mtp_design.py`` recomputes them.
EFFICIENCY_SD: dict[str, float] = {
    "x1_25": 0.4427816926812578,
    "piecewise": 0.25011202370817515,
    "halve": 0.23671532524961356,
    "classifier_route": 0.4427816926812578,
}


# ---------------------------------------------------------------------------- laws


def null_sample(n: int, seed: int) -> pd.DataFrame:
    """The law with a constant dose in the outcome: no policy moves ``Y``."""
    frame = study.sample(n, seed)
    rng = np.random.default_rng([seed, 7])
    probability = expit(-0.4 + 0.5 * frame["W1"].to_numpy() + 0.3 * frame["W2"].to_numpy())
    return frame.assign(Y=rng.binomial(1, probability).astype(float))


@dataclass(frozen=True)
class DeclaredLaw:
    """The law a declared cell reads, with its exact truth of every estimand.

    Parameters
    ----------
    name : str
        ``"point"``, ``"point_null"``, ``"point_classifier"``, ``"point_piecewise"``,
        ``"point_halve"`` or ``"point_inverse"``.
    """

    name: str

    def truth(self) -> dict[str, float]:
        if self.name == "point_null":
            return {X125: 0.0}
        return dict(study.TRUTH)


# ---------------------------------------------------------------------------- fits


@contextmanager
def _inverse_dropped(active: bool) -> Iterator[None]:
    """Equation (3) read at the dose itself instead of at the declared inverse, if ``active``."""
    if not active:
        yield
        return
    original = policy_module._PieceBranch.ratio

    def no_inverse(self: Any, values: Any, frame: Any, density: Any) -> Any:
        b = np.asarray(values, dtype=float).reshape(-1)
        denominator = density(b)
        numerator = np.zeros(b.size)
        for piece in self.pieces:
            if piece.map is None:
                continue
            slope = policy_module._per_row(
                policy_module._call(piece.derivative, b, frame()), b.size
            )
            inside = policy_module._member(b, *self.bounds(piece, frame, b.size), piece.closed)
            numerator = numerator + np.where(inside, density(b) * np.abs(slope), 0.0)
        safe = np.where(denominator > 0.0, denominator, 1.0)
        covariate = np.where(denominator > 0.0, numerator / safe, 0.0)
        for piece in self.pieces:
            if piece.map is None:
                covariate = covariate + policy_module._member(
                    b, *self.bounds(piece, frame, b.size), piece.closed
                )
        return covariate

    policy_module._PieceBranch.ratio = no_inverse  # type: ignore[method-assign]
    try:
        yield
    finally:
        policy_module._PieceBranch.ratio = original  # type: ignore[method-assign]


def _pair(label: str) -> tuple[Any, ...]:
    chosen = {policy.name: policy for policy in study.policies()}
    return (chosen[REFERENCE], chosen[label])


def initial_contrast(result: Any, label: str) -> float:
    """The untargeted plug-in contrast of a fit over ``(natural course, label)``."""
    nuisance = result.nuisance
    natural = nuisance.scaler.unscale_levels(nuisance.outcome.arms[0.0])
    policy = nuisance.scaler.unscale_levels(nuisance.outcome.arms[1.0])
    return float(np.mean(policy) - np.mean(natural))


# ---------------------------------------------------------------------------- the grid

#: Every fit set: ``(family, stream label, configuration, n, replicates, law)``.
FIT_SETS: tuple[tuple[str, str, str, int, int, str], ...] = (
    *(
        ("double_robustness", c, c, DOUBLE_ROBUST_N, DOUBLE_ROBUST_REPLICATES, "point")
        for c in ("both_correct", "outcome_correct", "density_correct", "both_wrong")
    ),
    *(
        ("root_n_and_efficiency", f"n_{size}", "both_correct", size, RATE_REPLICATES, "point")
        for size in RATE_SIZES
    ),
    (
        "interval_calibration",
        "correctly_specified",
        "both_correct",
        CALIBRATION_N,
        CALIBRATION_REPLICATES,
        "point",
    ),
    *(
        ("interval_calibration", label, "both_correct", CALIBRATION_N, CALIBRATION_REPLICATES, law)
        for label, law in (
            ("piecewise", "point_piecewise"),
            ("halve", "point_halve"),
            ("classifier_route", "point_classifier"),
        )
    ),
    ("type_i_error", "sharp_null", "both_correct", NULL_N, NULL_REPLICATES, "point_null"),
    ("power", "alternative", "both_correct", NULL_N, NULL_REPLICATES, "point"),
    (
        "targeting_necessity",
        "targeted",
        "density_correct",
        TARGETING_N,
        TARGETING_REPLICATES,
        "point",
    ),
    (
        "inverse_necessity",
        "paired",
        "density_correct",
        INVERSE_N,
        INVERSE_REPLICATES,
        "point_inverse",
    ),
)


def _seed(family: str, label: str, replicate: int) -> int:
    return stream_seed(STUDY, "property_sample", family, label, replicate)


def declared_cells() -> tuple[PropertyCell, ...]:
    """Every sampled cell, with the law it reads, its estimand, its size and its stream."""
    cells: list[PropertyCell] = []

    def add(
        family: str, cell: str, role: str, law: str, n: int, reps: int, label: str, name: str
    ) -> None:
        cells.append(
            PropertyCell(
                property=family,
                cell=cell,
                dgp=DeclaredLaw(law),
                outcome_learner=lambda: None,
                treatment_learner=lambda: None,
                n=n,
                replicates=reps,
                seed=_seed(family, label, 0),
                role=role,
                estimand=name,
            )
        )

    for configuration in ("both_correct", "outcome_correct", "density_correct", "both_wrong"):
        role = "control" if configuration == "both_wrong" else "positive"
        add(
            "double_robustness",
            f"x1_25__{configuration}",
            role,
            "point",
            DOUBLE_ROBUST_N,
            DOUBLE_ROBUST_REPLICATES,
            configuration,
            X125,
        )
    for size in RATE_SIZES:
        role = "control" if size == min(RATE_SIZES) else "positive"
        add(
            "root_n_and_efficiency",
            f"x1_25__n_{size}",
            role,
            "point",
            size,
            RATE_REPLICATES,
            f"n_{size}",
            X125,
        )
    for label, law, stream in (
        ("x1_25", "point", "correctly_specified"),
        ("piecewise", "point_piecewise", "piecewise"),
        ("halve", "point_halve", "halve"),
        ("classifier_route", "point_classifier", "classifier_route"),
    ):
        for cell, role in (
            ("correctly_specified", "positive"),
            ("shrunken_se_control", "control"),
            ("noise_control", "control"),
        ):
            add(
                "interval_calibration",
                f"{label}__{cell}",
                role,
                law,
                CALIBRATION_N,
                CALIBRATION_REPLICATES,
                stream,
                NAMES[label],
            )
    add(
        "type_i_error",
        "x1_25__sharp_null",
        "positive",
        "point_null",
        NULL_N,
        NULL_REPLICATES,
        "sharp_null",
        X125,
    )
    add(
        "power",
        "x1_25__alternative",
        "positive",
        "point",
        NULL_N,
        NULL_REPLICATES,
        "alternative",
        X125,
    )
    add(
        "targeting_necessity",
        "x1_25__targeted",
        "positive",
        "point",
        TARGETING_N,
        TARGETING_REPLICATES,
        "targeted",
        X125,
    )
    add(
        "targeting_necessity",
        "x1_25__untargeted",
        "control",
        "point",
        TARGETING_N,
        TARGETING_REPLICATES,
        "targeted",
        X125,
    )
    add(
        "inverse_necessity",
        "halve__declared_inverse",
        "positive",
        "point_inverse",
        INVERSE_N,
        INVERSE_REPLICATES,
        "paired",
        HALVE,
    )
    add(
        "inverse_necessity",
        "halve__inverse_dropped_control",
        "control",
        "point_inverse",
        INVERSE_N,
        INVERSE_REPLICATES,
        "paired",
        HALVE,
    )
    return tuple(cells)


# ---------------------------------------------------------------------------- rows


def _fit_set_rows(payload: tuple[str, str, str, int, int, int, str]) -> list[dict[str, Any]]:
    family, label, configuration, replicate, n, requested, law = payload
    seed = _seed(family, label, replicate)
    common_row = {"replicate": replicate, "n": n, "requested": requested}

    def row(cell: str, role: str, truth: float, estimate: Any) -> dict[str, Any]:
        return replicate_row(
            property_name=family,
            cell=cell,
            role=role,
            truth=truth,
            estimate=estimate,
            alpha=STUDY.margins.alpha,
            **common_row,
        )

    if law == "point_classifier":
        result = study.fit(
            study.sample(n, seed),
            chosen=_pair("x1.25"),
            ratio="classifier",
            treatment_learner=OracleLogOdds("x1.25"),
        )
        return [
            row(
                "classifier_route__correctly_specified", "positive", study.TRUTH[X125], result[X125]
            )
        ]
    if law in {"point_piecewise", "point_halve"}:
        policy = "piecewise" if law == "point_piecewise" else "halve below 3"
        name = NAMES["piecewise" if law == "point_piecewise" else "halve"]
        result = study.fit(study.sample(n, seed), chosen=_pair(policy))
        return [row(f"{label}__correctly_specified", "positive", study.TRUTH[name], result[name])]
    if family == "inverse_necessity":
        frame = study.sample(n, seed)
        declared = study.fit(frame, configuration=configuration, chosen=_pair("halve below 3"))
        with _inverse_dropped(True):
            dropped = study.fit(frame, configuration=configuration, chosen=_pair("halve below 3"))
        truth = study.TRUTH[HALVE]
        return [
            row("halve__declared_inverse", "positive", truth, declared[HALVE]),
            control_row(
                property_name=family,
                cell="halve__inverse_dropped_control",
                truth=truth,
                estimate=float(dropped[HALVE].psi),
                standard_error=float(dropped[HALVE].std_error),
                critical=CRITICAL,
                **common_row,
            ),
        ]
    frame = null_sample(n, seed) if law == "point_null" else study.sample(n, seed)
    result = study.fit(frame, configuration=configuration, chosen=_pair("x1.25"))
    truth = 0.0 if law == "point_null" else study.TRUTH[X125]
    stream = "correctly_specified" if family == "interval_calibration" else label
    role = (
        "control"
        if label == "both_wrong" or (family == "root_n_and_efficiency" and n == min(RATE_SIZES))
        else "positive"
    )
    rows = [row(f"x1_25__{stream}", role, truth, result[X125])]
    if family == "targeting_necessity":
        rows.append(
            control_row(
                property_name=family,
                cell="x1_25__untargeted",
                truth=truth,
                estimate=initial_contrast(result, "x1.25"),
                standard_error=float(result[X125].std_error),
                critical=CRITICAL,
                **common_row,
            )
        )
    return rows


def _payloads(budget: int | None) -> list[tuple[Any, ...]]:
    out: list[tuple[Any, ...]] = []
    for family, label, configuration, n, replicates, law in FIT_SETS:
        requested = replicates if budget is None else budget
        out.extend(
            ((family, label, configuration, r, n, requested, law),) for r in range(requested)
        )
    return out


def generate_property_rows(*, n_jobs: int = STUDY_JOBS, budget: int | None = None) -> pd.DataFrame:
    """Fit every property replication; ``budget`` caps each fit set for a pre-run check."""
    outcomes = map_parallel(_fit_set_rows, _payloads(budget), n_jobs=n_jobs)
    rows = pd.DataFrame([row for result in outcomes for row in result])
    rows = pd.concat(
        [
            rows,
            calibration_controls(
                rows,
                STUDY,
                labels=CALIBRATION_LABELS,
                efficiency_bounds=EFFICIENCY_SD,
                calibration_n=CALIBRATION_N,
                shrunken_se_factor=SHRUNKEN_SE_FACTOR,
                critical=CRITICAL,
            ),
        ],
        ignore_index=True,
    )
    return rows.loc[:, list(REPLICATE_COLUMNS)].sort_values(
        ["property", "cell", "replicate"], ignore_index=True
    )


def failure_probe(draws: int, *, n_jobs: int = 1, start: int = 0) -> dict[str, int]:
    """Failed fits per fit set over streams ``start`` to ``start + draws - 1``."""
    payloads = [
        ((family, label, configuration, r, n, draws, law),)
        for family, label, configuration, n, _, law in FIT_SETS
        for r in range(start, start + draws)
    ]
    outcomes = map_parallel(_probe_one, payloads, n_jobs=n_jobs)
    counts = {f"{family}/{label}": 0 for family, label, *_ in FIT_SETS}
    for (payload,), failed in zip(payloads, outcomes, strict=True):
        counts[f"{payload[0]}/{payload[1]}"] += int(failed)
    return counts


def _probe_one(payload: tuple[str, str, str, int, int, int, str]) -> bool:
    try:
        rows = _fit_set_rows(payload)
    except Exception:
        return True
    return not all(np.isfinite(row["estimate"]) and np.isfinite(row["std_error"]) for row in rows)


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The shared verdicts and the study's own declared rules."""
    summary, rates = apply_shared_verdicts(
        rows,
        STUDY,
        extra_columns=(
            "targeting_displacement",
            "inverse_displacement",
            "coverage_gain_ci_lower",
            "coverage_gain_ci_upper",
        ),
        rate_labels=("x1_25",),
        efficiency_bounds=EFFICIENCY_SD,
    )
    calibration_verdicts(summary, margins=STUDY.margins, efficiency_band=EFFICIENCY_RATIO_BAND)
    necessity_verdicts(
        summary,
        rows,
        family="targeting_necessity",
        labels=("x1_25",),
        arms=("targeted", "untargeted"),
        column="targeting_displacement",
        threshold=TARGETING_DISPLACEMENT,
    )
    necessity_verdicts(
        summary,
        rows,
        family="inverse_necessity",
        labels=("halve",),
        arms=("declared_inverse", "inverse_dropped_control"),
        column="inverse_displacement",
        threshold=INVERSE_DISPLACEMENT,
    )
    return finish(summary, rates)
