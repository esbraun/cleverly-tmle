r"""Are the nuisance models any good?

TMLE is doubly robust, not magic.  Under identification and positivity the point
estimate is consistent if *one* of ``g`` and ``Qbar`` is consistent; asymptotic
linearity and a valid Wald interval need both, converging fast enough that the
*product* of their errors is ``o(n^{-1/2})``.

The gap between those two is the reason to read these numbers.  In the doubly-robust-
but-not-efficient case -- one nuisance inconsistent -- the estimate still converges to
the truth, but the influence-curve standard error generally does not describe it, so the
reported interval is not merely wide, it is wrong.  Double robustness protects the point
estimate, not the inference.  These diagnostics use the out-of-fold predictions the fit
already produced, so they cost nothing.

What each number is for:

**Propensity model discrimination (AUC).**  Not "higher is better".  An AUC near 0.5
means treatment is close to randomised given ``W``, which is *good* for overlap.  An AUC
near 1 means treatment is nearly determined by ``W``, which means poor overlap and a
fragile estimate -- read it together with
:meth:`~cleverly.sensitivity.SensitivityAnalysis.positivity`.

**Calibration.**  Discrimination is irrelevant if the probabilities themselves are
wrong: the clever covariate divides by ``g(W)``, so a systematically overconfident
propensity model biases every weight.  The calibration slope from a logistic
recalibration of the out-of-fold predictions should be near 1; the calibration table
shows *where* it goes wrong.

**Outcome model R-squared / Brier score.**  Bounds how much variance reduction the
targeting step can buy.  A near-zero R-squared means the estimate is effectively
inverse-probability weighting, with the variance that implies.

**Super Learner weights.**  Which candidates the ensemble actually used.  All weight
on ``mean`` is a warning: nothing in the library predicted better than the marginal
average, so the adjustment is doing very little.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .._typing import BoolArray, FloatArray
from ..estimators.base import format_table
from ..utils.bounds import logit

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..data.causal_data import CausalData
    from ..estimators.base import TMLEResult

__all__ = ["NuisanceDiagnostics", "NuisanceModelReport", "nuisance_diagnostics"]

#: Number of bins in the calibration table.
_CALIBRATION_BINS = 10


@dataclass(frozen=True)
class NuisanceModelReport:
    """Fit quality for one nuisance model."""

    name: str
    kind: str
    metrics: dict[str, float]
    calibration: dict[str, list[float]]
    learner_weights: dict[str, float]
    learner_risks: dict[str, float]

    def row(self) -> list[str]:
        order = ("auc", "brier", "log_loss", "r2", "mse", "calibration_slope")
        return [self.name] + [
            f"{self.metrics[key]:.4f}" if key in self.metrics else "-" for key in order
        ]


@dataclass(frozen=True)
class NuisanceDiagnostics:
    """Out-of-fold fit quality for every nuisance model in a TMLE fit."""

    models: tuple[NuisanceModelReport, ...]
    #: How many cross-fitting draws the fit averaged over.  The reports above describe the
    #: **first** draw, for the reason
    #: :attr:`~cleverly.sensitivity.PositivityReport.n_repeats` gives: a model's
    #: out-of-fold calibration is a property of that model, and one fitted under a
    #: different split is a different model rather than another measurement of the same
    #: one.  Averaging their AUCs would report a number no model achieved.
    n_repeats: int = 1

    def __getitem__(self, name: str) -> NuisanceModelReport:
        for model in self.models:
            if model.name == name:
                return model
        raise KeyError(f"no nuisance model named {name!r}; have {[m.name for m in self.models]}")

    def to_frame(self, data: Any = None) -> Any:
        from ..utils.frames import frame_from_dict

        keys: list[str] = []
        for model in self.models:
            for key in model.metrics:
                if key not in keys:
                    keys.append(key)
        payload: dict[str, Any] = {
            "model": [model.name for model in self.models],
            "kind": [model.kind for model in self.models],
        }
        for key in keys:
            payload[key] = [model.metrics.get(key, float("nan")) for model in self.models]
        if data is not None:
            return data.frame_like(payload)
        return frame_from_dict(payload)

    def calibration_frame(self, name: str, data: Any = None) -> Any:
        """Binned observed-vs-predicted table for one model."""
        from ..utils.frames import frame_from_dict

        payload = dict(self[name].calibration)
        if data is not None:
            return data.frame_like(payload)
        return frame_from_dict(payload)

    def summary(self) -> str:
        lines = [
            "Nuisance model diagnostics (out of fold)",
            "-" * 40,
        ]
        if self.n_repeats > 1:
            lines.append(f"describing draw 1 of {self.n_repeats}; each draw fits its own models")
        lines.append(
            format_table(
                ["model", "auc", "brier", "log_loss", "r2", "mse", "cal_slope"],
                [model.row() for model in self.models],
            )
        )
        for model in self.models:
            if not model.learner_weights:
                continue
            used = {name: weight for name, weight in model.learner_weights.items() if weight > 1e-3}
            lines.append("")
            lines.append(
                f"{model.name}: super learner weights "
                + ", ".join(f"{name}={weight:.3f}" for name, weight in used.items())
            )
        lines.append("")
        lines.append(self.verdict())
        return "\n".join(lines)

    def verdict(self) -> str:
        """A reading of the diagnostics that says what to do about them."""
        notes: list[str] = []
        for model in self.models:
            auc = model.metrics.get("auc")
            slope = model.metrics.get("calibration_slope")
            if model.name == "propensity" and auc is not None:
                if auc > 0.9:
                    notes.append(
                        f"the propensity model separates the arms almost perfectly "
                        f"(AUC {auc:.3f}); this signals a positivity problem, not a good fit"
                    )
                elif auc < 0.55:
                    notes.append(
                        f"treatment is nearly unpredictable from W (AUC {auc:.3f}); overlap is "
                        "excellent and confounding by these covariates is limited"
                    )
            if model.name == "missingness" and auc is not None and auc > 0.9:
                # The same reading as for the propensity, and for the same reason: this
                # probability divides the clever covariate, so predicting it almost
                # perfectly means some rows had virtually no chance of being observed and
                # the estimate is extrapolating to them.
                notes.append(
                    f"the missingness model predicts almost perfectly (AUC {auc:.3f}); some "
                    "units had virtually no chance of a recorded outcome, so 1/P(Delta=1|A,W) "
                    "gives them extreme leverage -- check res.sensitivity.positivity()"
                )
            if slope is not None and not 0.7 <= slope <= 1.4:
                notes.append(
                    f"{model.name} is poorly calibrated (slope {slope:.2f}, ideal 1.0); its "
                    "predicted probabilities are systematically off, which biases the weights"
                )
            if model.learner_weights.get("mean", 0.0) > 0.8:
                notes.append(
                    f"{model.name} put {model.learner_weights['mean']:.0%} of its weight on the "
                    "marginal mean -- no candidate beat predicting the average, so this model "
                    "is contributing almost nothing"
                )
            r2 = model.metrics.get("r2")
            if model.name == "outcome" and r2 is not None and r2 < 0.05:
                notes.append(
                    f"the outcome model explains little variance (R^2 {r2:.3f}); the estimate is "
                    "close to inverse-probability weighting and will be correspondingly noisy"
                )
        if not notes:
            return "VERDICT: nuisance fits look reasonable."
        return "VERDICT:\n" + "\n".join(f"  - {note}" for note in notes)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return self.summary()


def _at_realised_treatment(data: CausalData, mechanism: FloatArray) -> FloatArray:
    r"""A per-treatment mechanism read at the treatment each unit actually received.

    That is the prediction a calibration report has an outcome to compare against: the
    model was fitted on :math:`(A_i, W_i)`, so it is :math:`\hat\pi(A_i, W_i)` that
    :math:`\Delta_i` is evidence about, not the value at some other arm.

    On a ``shifts=`` fit the mechanism is ``(n, S + 1)`` and column ``0`` *is* the value at
    the row's own dose, so there is nothing to select.  Reading ``[:, 1]`` there would
    silently report the mechanism at some shift's assigned dose against the observed
    outcome, selected by ``dose == 1.0`` -- a plausible number for a quantity nobody asked
    for.
    """
    values = np.asarray(mechanism, dtype=float)
    if data.is_continuous_treatment:
        return np.asarray(values[:, 0], dtype=float)
    # Binary-only, as it has always been: a K-armed fit reports arm 0's value for every
    # row outside arms 0 and 1. Pre-existing and out of scope here; named so it is not
    # mistaken for something this branch introduced.
    return np.asarray(np.where(data.treatment == 1.0, values[:, 1], values[:, 0]), dtype=float)


def nuisance_diagnostics(result: TMLEResult) -> NuisanceDiagnostics:
    """Out-of-fold diagnostics for every nuisance model in the fit."""
    data = result.data
    nuisance = result.nuisance
    models: list[NuisanceModelReport] = []

    if data.is_binary_treatment:
        models.append(
            _binary_report(
                "propensity",
                nuisance.propensity.arm(1.0),
                data.treatment,
                data.weights,
                nuisance.diagnostics.get("propensity"),
            )
        )
    else:
        # One one-vs-rest report per arm, rather than a single multi-class summary.
        # Positivity is an arm-by-arm property -- the estimate can rest on a badly
        # calibrated denominator for one arm while the pooled log loss looks fine --
        # and a per-arm report is what says which arm to go and look at.
        for arm in nuisance.arms:
            models.append(
                _binary_report(
                    f"propensity[{data.arm_label(arm)}]",
                    nuisance.propensity.arm(arm),
                    (data.treatment == arm).astype(float),
                    data.weights,
                    nuisance.diagnostics.get("propensity"),
                )
            )

    if nuisance.missingness is not None:
        models.append(
            _binary_report(
                "missingness",
                _at_realised_treatment(data, nuisance.missingness),
                data.observed.astype(float),
                data.weights,
                nuisance.diagnostics.get("missingness"),
            )
        )

    if nuisance.intermediate is not None and data.intermediate is not None:
        arm_probability = _at_realised_treatment(data, nuisance.intermediate)
        models.append(
            _binary_report(
                "intermediate",
                arm_probability,
                data.intermediate,
                data.weights,
                nuisance.diagnostics.get("intermediate"),
            )
        )

    scaled = nuisance.scaler.scale(data.outcome)
    if data.family == "binomial":
        models.append(
            _binary_report(
                "outcome",
                nuisance.outcome.observed,
                data.outcome,
                data.weights,
                nuisance.diagnostics.get("outcome"),
                mask=data.observed,
            )
        )
    else:
        models.append(
            _continuous_report(
                "outcome",
                nuisance.outcome.observed,
                scaled,
                data.weights,
                nuisance.diagnostics.get("outcome"),
                mask=data.observed,
            )
        )
    return NuisanceDiagnostics(models=tuple(models), n_repeats=result.n_repeats)


def _aggregate_learner_info(
    diagnostics: Any,
) -> tuple[dict[str, float], dict[str, float]]:
    """Average Super Learner weights and risks across cross-fitting folds."""
    if not diagnostics:
        return {}, {}
    entries = diagnostics if isinstance(diagnostics, list) else [diagnostics]
    weights: dict[str, list[float]] = {}
    risks: dict[str, list[float]] = {}
    for entry in entries:
        names = getattr(entry, "names", None)
        if names is None:
            continue
        for index, name in enumerate(names):
            weights.setdefault(name, []).append(float(entry.weights[index]))
            risks.setdefault(name, []).append(float(entry.cv_risk[index]))
    return (
        {name: float(np.mean(values)) for name, values in weights.items()},
        {name: float(np.mean(values)) for name, values in risks.items()},
    )


def _binary_report(
    name: str,
    predicted: FloatArray,
    actual: FloatArray,
    weights: FloatArray,
    diagnostics: Any,
    *,
    mask: BoolArray | None = None,
) -> NuisanceModelReport:
    """Discrimination, calibration and proper-scoring metrics for a probability model."""
    index = slice(None) if mask is None else np.asarray(mask, dtype=bool)
    p = np.clip(np.asarray(predicted, dtype=float)[index], 1e-12, 1.0 - 1e-12)
    y = np.asarray(actual, dtype=float)[index]
    w = np.asarray(weights, dtype=float)[index]

    metrics = {
        "auc": _weighted_auc(p, y, w),
        "brier": float(np.average((p - y) ** 2, weights=w)),
        "log_loss": float(-np.average(y * np.log(p) + (1.0 - y) * np.log(1.0 - p), weights=w)),
        "calibration_slope": _calibration_slope(p, y, w),
        "mean_predicted": float(np.average(p, weights=w)),
        "mean_observed": float(np.average(y, weights=w)),
    }
    learner_weights, learner_risks = _aggregate_learner_info(diagnostics)
    return NuisanceModelReport(
        name=name,
        kind="probability",
        metrics=metrics,
        calibration=_calibration_table(p, y, w),
        learner_weights=learner_weights,
        learner_risks=learner_risks,
    )


def _continuous_report(
    name: str,
    predicted: FloatArray,
    actual: FloatArray,
    weights: FloatArray,
    diagnostics: Any,
    *,
    mask: BoolArray | None = None,
) -> NuisanceModelReport:
    """Variance-explained metrics for a conditional-mean model."""
    index = slice(None) if mask is None else np.asarray(mask, dtype=bool)
    p = np.asarray(predicted, dtype=float)[index]
    y = np.asarray(actual, dtype=float)[index]
    w = np.asarray(weights, dtype=float)[index]

    mse = float(np.average((y - p) ** 2, weights=w))
    variance = float(np.average((y - np.average(y, weights=w)) ** 2, weights=w))
    metrics = {
        "mse": mse,
        "r2": float(1.0 - mse / variance) if variance > 0 else float("nan"),
        "calibration_slope": _regression_slope(p, y, w),
        "mean_predicted": float(np.average(p, weights=w)),
        "mean_observed": float(np.average(y, weights=w)),
    }
    learner_weights, learner_risks = _aggregate_learner_info(diagnostics)
    return NuisanceModelReport(
        name=name,
        kind="conditional mean",
        metrics=metrics,
        calibration=_calibration_table(p, y, w),
        learner_weights=learner_weights,
        learner_risks=learner_risks,
    )


def _weighted_auc(predicted: FloatArray, actual: FloatArray, weights: FloatArray) -> float:
    """Weighted area under the ROC curve, via the Mann--Whitney identity."""
    positive = actual == 1.0
    if not positive.any() or positive.all():
        return float("nan")
    order = np.argsort(predicted, kind="stable")
    p_sorted = predicted[order]
    y_sorted = positive[order]
    w_sorted = weights[order]

    # Mid-ranks so ties contribute 0.5, matching the usual AUC convention.
    ranks = np.empty(p_sorted.shape[0], dtype=float)
    cumulative = np.cumsum(w_sorted)
    start = 0
    while start < p_sorted.shape[0]:
        stop = start
        while stop + 1 < p_sorted.shape[0] and p_sorted[stop + 1] == p_sorted[start]:
            stop += 1
        below = cumulative[start - 1] if start > 0 else 0.0
        block = cumulative[stop] - below
        ranks[start : stop + 1] = below + 0.5 * block
        start = stop + 1

    weight_positive = w_sorted[y_sorted].sum()
    weight_negative = w_sorted[~y_sorted].sum()
    if weight_positive <= 0 or weight_negative <= 0:
        return float("nan")
    concordant = float(np.sum(w_sorted[y_sorted] * ranks[y_sorted]))
    concordant -= 0.5 * weight_positive**2
    return float(concordant / (weight_positive * weight_negative))


def _calibration_slope(predicted: FloatArray, actual: FloatArray, weights: FloatArray) -> float:
    """Slope of a logistic recalibration of the predictions; 1.0 is perfect."""
    from ..fluctuation.iterative import _newton_logistic

    x = np.column_stack([np.ones_like(predicted), logit(predicted)])
    epsilon, _converged, _detail = _newton_logistic(x, actual, np.zeros_like(predicted), weights)
    return float(epsilon[1])


def _regression_slope(predicted: FloatArray, actual: FloatArray, weights: FloatArray) -> float:
    """Slope of a weighted regression of the outcome on the prediction."""
    centred_x = predicted - np.average(predicted, weights=weights)
    centred_y = actual - np.average(actual, weights=weights)
    denominator = float(np.sum(weights * centred_x**2))
    if denominator <= 0:
        return float("nan")
    return float(np.sum(weights * centred_x * centred_y) / denominator)


def _calibration_table(
    predicted: FloatArray, actual: FloatArray, weights: FloatArray
) -> dict[str, list[float]]:
    """Binned observed-vs-predicted means, by quantile of the prediction."""
    n = predicted.shape[0]
    bins = min(_CALIBRATION_BINS, max(2, n // 20))
    edges = np.quantile(predicted, np.linspace(0.0, 1.0, bins + 1))
    edges[0] -= 1e-12
    edges[-1] += 1e-12
    index = np.clip(np.digitize(predicted, edges[1:-1], right=True), 0, bins - 1)

    rows: dict[str, list[float]] = {
        "bin": [],
        "n": [],
        "mean_predicted": [],
        "mean_observed": [],
    }
    for b in range(bins):
        mask = index == b
        if not mask.any():
            continue
        rows["bin"].append(float(b))
        rows["n"].append(float(mask.sum()))
        rows["mean_predicted"].append(float(np.average(predicted[mask], weights=weights[mask])))
        rows["mean_observed"].append(float(np.average(actual[mask], weights=weights[mask])))
    return rows
