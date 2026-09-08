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

**Propensity model discrimination (AUC).**  Not "higher is better".  For an ordinary
estimated treatment law, an AUC near 0.5 means treatment is close to randomised given
``W``, while an AUC near 1 can signal poor overlap.  A C-TMLE propensity is instead a
selected working mechanism: its AUC describes the denominator the method used, not
treatment given the complete adjustment set.  Read either case together with
:meth:`~cleverly.assessment.DiagnosticsFacade.support`.

**Calibration.**  Discrimination is irrelevant if the probabilities themselves are
wrong: the clever covariate divides by ``g(W)``.  For an ordinary treatment-law estimate,
systematic miscalibration is therefore evidence against the fitted weights.  C-TMLE's
selected working mechanism keeps the same descriptive values without receiving that
treatment-law interpretation.  The calibration table shows *where* predictions differ
from observations.

**Outcome model R-squared / Brier score.**  Bounds how much variance reduction the
targeting step can buy.  A near-zero R-squared means the estimate is effectively
inverse-probability weighting, with the variance that implies.

**Super Learner weights.**  Which candidates the ensemble actually used.  All weight
on ``mean`` is a warning: nothing in the library predicted better than the marginal
average, so the adjustment is doing very little.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from .._typing import BoolArray, FloatArray
from ..data.weighting import REPORTED_DRAW
from ..utils.bounds import logit
from ..utils.frames import emit_frame
from ..utils.records import sentinel_equality
from ..utils.text import format_draw, format_table

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..data.causal_data import CausalData
    from ..estimators.base import TMLEResult
    from ..estimators.ctmle import CTMLEOutcomeAdaptiveFit, CTMLESelection

__all__ = [
    "NUISANCE_SELECTION_MISSING",
    "SPREAD_NOT_FINITE",
    "SPREAD_NO_PARAMETERS",
    "SPREAD_SINGLE_DRAW",
    "SPREAD_UNAVAILABLE_DRAWS",
    "NuisanceDiagnostics",
    "NuisanceModelReport",
    "RepeatSpreadRow",
    "nuisance_diagnostics",
]

#: Number of bins in the calibration table.
_CALIBRATION_BINS = 10

# ------------------------------------------------------------------ omission reasons
#
# Named for the reason :data:`~cleverly.data.weighting.SCORE_LOAD_MISSING` and its
# siblings are: the producer here, the consumers in
# :func:`~cleverly.validation.nuisance.NuisanceDiagnostics.summary` and
# :mod:`cleverly.assessment`, and the test that asserts a report states its cause all name
# one object rather than three hand-written copies of one sentence.

#: The fit declares itself collaborative and retains no ``ctmle`` artifact to describe.
NUISANCE_SELECTION_MISSING = "the fitted result retains no ctmle method artifact"

#: The fit drew the cross-fitting split once, so there is no between-draw spread. Not a
#: fault: it is the ordinary state of an ordinary fit.
SPREAD_SINGLE_DRAW = "the fit used one cross-fitting draw"

#: Prefix naming the parameters some draw did not report, so their spread is over fewer
#: draws than the fit made and is not computed at all.
SPREAD_UNAVAILABLE_DRAWS = "draw-specific estimates are unavailable for: "

#: Prefix naming the parameters whose spread exists but is not finite. A ratio-scale draw
#: at or below zero has no inference-scale value, and a non-finite ``psi`` has none on any
#: scale, so :meth:`~cleverly.estimators.TMLEResult.repeat_spread` yields ``nan`` there.
SPREAD_NOT_FINITE = "the split spread is not finite for: "

#: The result reports no parameter, so there is nothing to take a spread of.
SPREAD_NO_PARAMETERS = "no reported parameter is available"

TreatmentModelRole = Literal["estimated_treatment_law", "collaborative_working_model"]


@dataclass(frozen=True)
class NuisanceModelReport:
    """Fit quality for one nuisance model.

    Parameters
    ----------
    name : str
        Which nuisance this report describes.
    kind : str
        Whether it is a regression or a classification.
    metrics : dict of str to float
        Fit metrics at the report's evaluation rows.
    calibration : dict of str to list of float
        Binned mean prediction and observed rate.
    learner_weights : dict of str to float
        Ensemble weight per candidate.
    learner_risks : dict of str to float
        Cross-validated risk per candidate.
    """

    name: str
    kind: str
    metrics: dict[str, float]
    calibration: dict[str, list[float]]
    learner_weights: dict[str, float]
    learner_risks: dict[str, float]

    def row(self) -> list[str]:
        """Return this model's metrics as one row of the diagnostics table.

        Returns
        -------
        list of str
            The formatted cells, in the column order
            :meth:`NuisanceDiagnostics.summary` prints.
        """
        order = ("auc", "brier", "log_loss", "r2", "mse", "calibration_slope")
        return [self.name] + [
            f"{self.metrics[key]:.4f}" if key in self.metrics else "-" for key in order
        ]


def _is_propensity(model: NuisanceModelReport) -> bool:
    """Whether a report describes a treatment mechanism rather than another nuisance.

    Matches the multi-arm reports too, which are named ``propensity[<label>]``.  This is
    the test for *whose interpretation a role changes*, so it covers every report the role
    applies to: a collaborative fit selects one mechanism however many arms the report
    splits it into.
    """
    return model.name.startswith("propensity")


def _is_binary_propensity(model: NuisanceModelReport) -> bool:
    """Whether a report is the single binary-treatment propensity the AUC rules read.

    Deliberately narrower than :func:`_is_propensity`, and narrow in the way the two AUC
    rules have always been.  Both gates are calibrated against a binary mechanism, and a
    one-vs-rest report is not on that scale: a rare arm separates from the pooled rest
    almost perfectly for the very reason it is rare, so ``0.9`` would fire on the arm
    count rather than on a positivity problem.

    Extending the AUC rules to a K-armed fit needs its own thresholds and its own
    evidence, exactly as :func:`_at_realised_treatment` records for the mechanism it
    reads.  Named here so the asymmetry with :func:`_is_propensity` reads as the
    pre-existing scope it is, rather than as something a reader has to rediscover.
    """
    return model.name == "propensity"


@sentinel_equality
@dataclass(frozen=True)
class RepeatSpreadRow:
    r"""Split sensitivity for one parameter in a repeated cross-fitted result.

    Parameters
    ----------
    estimand : str
        Stable alias of the reported parameter.
    n_repeats : int
        Number of cross-fitting split draws on the same sample.
    standard_deviation : float
        Sample standard deviation of :math:`\hat\psi_r` across the draws, on the
        inference scale of the estimand. That is :math:`\log\hat\psi_r` for a ratio, so
        the value is comparable with ``reported_standard_error``.
    reported_standard_error : float
        Standard error on the median-combined result, on the inference scale.
    ratio_to_standard_error : float
        Split standard deviation divided by the reported standard error. This ratio is
        descriptive and has no pass threshold.
    """

    estimand: str
    n_repeats: int
    standard_deviation: float
    reported_standard_error: float
    ratio_to_standard_error: float

    def row(self) -> list[str]:
        """Return this parameter's spread as one row of the split-sensitivity table.

        Returns
        -------
        list of str
            The formatted cells, in the column order
            :meth:`NuisanceDiagnostics.summary` prints. A cell with no finite value reads
            ``"-"``, as every other table in this package renders a missing cell.
        """
        values = (
            self.standard_deviation,
            self.reported_standard_error,
            self.ratio_to_standard_error,
        )
        return [self.estimand, str(self.n_repeats)] + [
            f"{value:.6g}" if np.isfinite(value) else "-" for value in values
        ]


@dataclass(frozen=True)
class NuisanceDiagnostics:
    """Out-of-fold fit quality for every nuisance model in a TMLE fit.

    Parameters
    ----------
    models : tuple of NuisanceModelReport
        One report per fitted nuisance model.
    n_repeats : int
        Cross-fitting draws the fit combined. The reports describe the first.
    backend : str or None
        Dataframe backend :meth:`to_frame` returns when ``data`` is omitted.
    selection : CTMLESelection or CTMLEOutcomeAdaptiveFit or None
        The fitted :class:`~cleverly.estimators.CTMLESelection` or
        :class:`~cleverly.estimators.CTMLEOutcomeAdaptiveFit`. ``None`` means the method
        is not collaborative, or the stored artifact is unavailable.
    treatment_role : {"estimated_treatment_law", "collaborative_working_model"} or None
        Statistical role of the propensity predictions in ``models``. A collaborative
        working mechanism is not the complete treatment law.
    repeat_spread : tuple of RepeatSpreadRow
        One descriptive split-sensitivity row per reported parameter. A one-draw fit
        retains an empty tuple rather than a false zero spread.
    selection_omission : str or None
        Machine-readable reason an expected C-TMLE artifact is absent.
    repeat_spread_omission : str or None
        Machine-readable reason no split-spread row is available.
    reported_repeat : int
        One-based draw described by the nuisance models and selection artifact.
    """

    models: tuple[NuisanceModelReport, ...]
    #: How many cross-fitting draws the fit combined. The reports above describe the
    #: **first** draw, for the reason
    #: :attr:`~cleverly.sensitivity.PositivityReport.n_repeats` gives: a model's
    #: out-of-fold calibration is a property of that model, and one fitted under a
    #: different split is a different model rather than another measurement of the same
    #: one.  Averaging their AUCs would report a number no model achieved.
    n_repeats: int = 1
    #: Name of the dataframe backend the fit's data arrived in, so that
    #: :meth:`to_frame` honours "results come back in the backend you passed in"
    #: without a caller having to thread the container back in by hand.
    backend: str | None = None
    #: The exact method object already retained by the fitted result. Selector paths keep
    #: every candidate and risk rather than flattening the selected index into prose.
    selection: CTMLESelection | CTMLEOutcomeAdaptiveFit | None = field(default=None, compare=False)
    #: The propensity report's role. The role changes its interpretation, not its values.
    treatment_role: TreatmentModelRole | None = None
    #: Unlike the nuisance models and selection, these rows read every retained draw.
    repeat_spread: tuple[RepeatSpreadRow, ...] = ()
    selection_omission: str | None = None
    repeat_spread_omission: str | None = None
    #: Nuisance models and the selection artifact describe this draw. One is the only
    #: draw the fitted result retains method-specific extras for.
    reported_repeat: int = REPORTED_DRAW

    def __getitem__(self, name: str) -> NuisanceModelReport:
        for model in self.models:
            if model.name == name:
                return model
        raise KeyError(f"no nuisance model named {name!r}; have {[m.name for m in self.models]}")

    def to_frame(self, data: Any = None) -> Any:
        """Return tabular output in the input dataframe backend.

        Parameters
        ----------
        data : Any
            A dataframe or fitted container whose backend to match. ``None`` uses the
            backend recorded on this object.

        Returns
        -------
        dataframe
            One row per nuisance model, with its fit metrics.
        """
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
        return emit_frame(payload, data, backend=self.backend)

    def calibration_frame(self, name: str, data: Any = None) -> Any:
        """Binned observed-vs-predicted table for one model.

        Parameters
        ----------
        name : str
            Which nuisance model to tabulate.
        data : Any
            A dataframe or fitted container whose backend to match. ``None`` uses the
            backend recorded on this object.

        Returns
        -------
        dataframe
            One row per bin, with the mean prediction and the observed rate in it.
        """
        payload = dict(self[name].calibration)
        return emit_frame(payload, data, backend=self.backend)

    def repeat_spread_frame(self, data: Any = None) -> Any:
        """Return one split-sensitivity row per reported parameter.

        Parameters
        ----------
        data : Any
            A dataframe or fitted container whose backend to match. ``None`` uses the
            backend recorded on this object.

        Returns
        -------
        dataframe
            Parameter aliases, draw counts, split standard deviations, reported standard
            errors, and their descriptive ratios.
        """
        payload = {
            "estimand": [row.estimand for row in self.repeat_spread],
            "n_repeats": [row.n_repeats for row in self.repeat_spread],
            "standard_deviation": [row.standard_deviation for row in self.repeat_spread],
            "reported_standard_error": [row.reported_standard_error for row in self.repeat_spread],
            "ratio_to_standard_error": [row.ratio_to_standard_error for row in self.repeat_spread],
        }
        return emit_frame(payload, data, backend=self.backend)

    def summary(self) -> str:
        """Return a printable summary.

        Returns
        -------
        str
            A printable table, one line per nuisance model.
        """
        lines = [
            "Nuisance model diagnostics (out of fold)",
            "-" * 40,
        ]
        if self.n_repeats > 1:
            lines.append(
                f"describing {format_draw(self.reported_repeat, self.n_repeats)}; "
                "each draw fits its own models"
            )
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
        if self._working_model:
            lines.extend(
                [
                    "",
                    "The propensity metrics describe the selected C-TMLE working mechanism.",
                    "They do not describe treatment given the complete adjustment set.",
                ]
            )
        if self.selection is not None:
            # No draw suffix here. The header above already states which draw every
            # method-specific artifact in this report describes, and a repeated
            # collaborative fit printed that fact twice.
            lines.append(self.selection.describe())
        elif self.selection_omission is not None:
            lines.append(f"C-TMLE selection unavailable: {self.selection_omission}")
        if self.repeat_spread:
            lines.extend(
                [
                    "",
                    "Repeated-split sensitivity. The sd and reported se columns are on "
                    "the estimand's inference scale, which is the log scale for a ratio.",
                    format_table(
                        ["estimand", "draws", "sd", "reported se", "sd/se"],
                        [row.row() for row in self.repeat_spread],
                    ),
                    "The split ratio is descriptive and has no pass threshold.",
                ]
            )
        if self.n_repeats > 1 and self.repeat_spread_omission is not None:
            # Guarded on the draw count rather than on the reason, because the one-draw
            # reason is the ordinary state of an ordinary fit and would otherwise print
            # under every summary this package produces.
            lines.append(f"split spread unavailable: {self.repeat_spread_omission}")
        lines.append("")
        lines.append(self.verdict())
        return "\n".join(lines)

    def verdict(self) -> str:
        """A reading of the diagnostics that says what to do about them.

        Returns
        -------
        str
            A reading of the diagnostics that says what to do about them.
        """
        notes = list(self.findings)
        for model in self.models:
            if not _is_binary_propensity(model) or self._working_mechanism(model):
                continue
            auc = model.metrics.get("auc")
            if auc is not None and auc < 0.55:
                notes.append(
                    f"treatment is nearly unpredictable from W (AUC {auc:.3f}); overlap is "
                    "excellent and confounding by these covariates is limited"
                )
        if not notes:
            if self._working_model:
                return (
                    "VERDICT: C-TMLE working-model metrics are descriptive; inspect the "
                    "selection and support reports."
                )
            return "VERDICT: nuisance fits look reasonable."
        return "VERDICT:\n" + "\n".join(f"  - {note}" for note in notes)

    @property
    def _working_model(self) -> bool:
        """Whether the propensity reports describe a selected C-TMLE working mechanism."""
        return self.treatment_role == "collaborative_working_model"

    def _working_mechanism(self, model: NuisanceModelReport) -> bool:
        """Whether this report is a collaborative fit's own selected propensity.

        Exactly two claims are suppressed for such a report, and both are claims about
        the *treatment law* that a working mechanism does not make. An intercept-only
        C-TMLE selection legitimately gives an AUC near 0.5 and a calibration slope near
        ``-2``, and reporting "overlap is excellent" or "poorly calibrated" from those is
        a false positive about a model nobody fitted.

        Nothing else is suppressed. The high-AUC positivity note stays, because
        ``CTMLE._nuisances`` puts the selected mechanism on ``nuisance.propensity`` and it
        is therefore the denominator the clever covariate divides by: an AUC near one
        there means the estimator's own weights are near-degenerate. The Super Learner
        mean-weight note stays too, because it is a fact about a learner library rather
        than an interpretation of the treatment law.
        """
        return self._working_model and _is_propensity(model)

    @property
    def findings(self) -> tuple[str, ...]:
        """Return findings that meet an existing diagnostic warning rule."""
        notes: list[str] = []
        for model in self.models:
            auc = model.metrics.get("auc")
            slope = model.metrics.get("calibration_slope")
            working_mechanism = self._working_mechanism(model)
            if _is_binary_propensity(model) and auc is not None and auc > 0.9:
                notes.append(
                    f"the propensity model separates the arms almost perfectly "
                    f"(AUC {auc:.3f}); this signals a positivity problem, not a good fit"
                )
            if model.name == "missingness" and auc is not None and auc > 0.9:
                notes.append(
                    f"the missingness model predicts almost perfectly (AUC {auc:.3f}); some "
                    "units had virtually no chance of a recorded outcome, so 1/P(Delta=1|A,W) "
                    "gives them extreme leverage -- check res.diagnostics.support()"
                )
            if not working_mechanism and slope is not None and not 0.7 <= slope <= 1.4:
                notes.append(
                    f"{model.name} is poorly calibrated (slope {slope:.2f}, ideal 1.0); its "
                    "predicted probabilities are systematically off, which biases the weights"
                )
            mean_weight = model.learner_weights.get("mean", 0.0)
            if mean_weight > 0.8:
                notes.append(
                    f"{model.name} put {mean_weight:.0%} of its weight on the marginal mean -- "
                    "no candidate beat predicting the average, so this model contributes little"
                )
            r2 = model.metrics.get("r2")
            if model.name == "outcome" and r2 is not None and r2 < 0.05:
                notes.append(
                    f"the outcome model explains little variance (R^2 {r2:.3f}); the estimate is "
                    "close to inverse-probability weighting and will be correspondingly noisy"
                )
        return tuple(notes)

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
    """Out-of-fold diagnostics for every nuisance model in the fit.

    Parameters
    ----------
    result : TMLEResult
        A fitted result.

    Returns
    -------
    NuisanceDiagnostics
        One report per nuisance model, read off the stored fits.
    """
    data = result.data
    nuisance = result.nuisance
    models: list[NuisanceModelReport] = []
    selection = result.ctmle_selection
    collaborative = result.fitted_method == "collaborative_tmle" or selection is not None
    selection_omission = NUISANCE_SELECTION_MISSING if collaborative and selection is None else None

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
    elif not data.is_continuous_treatment:
        # One one-vs-rest report per arm, rather than a single multi-class summary.
        # Positivity is an arm-by-arm property -- the estimate can rest on a badly
        # calibrated denominator for one arm while the pooled log loss looks fine --
        # and a per-arm report is what says which arm to go and look at.
        # Continuous MSM counterfactual codes name integration doses, not arms.
        # Their fitted mechanism is a density; no propensity calibration exists.
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
    spread_rows, spread_omission = _spread_rows(result)
    return NuisanceDiagnostics(
        models=tuple(models),
        n_repeats=result.n_repeats,
        backend=result.data.backend,
        selection=selection,
        treatment_role=(
            "collaborative_working_model"
            if collaborative
            else "estimated_treatment_law"
            if not data.is_continuous_treatment
            else None
        ),
        repeat_spread=spread_rows,
        selection_omission=selection_omission,
        repeat_spread_omission=spread_omission,
    )


def _spread_rows(result: TMLEResult) -> tuple[tuple[RepeatSpreadRow, ...], str | None]:
    """Split-sensitivity rows for a repeated fit, and the reason any are missing.

    Returns ``(rows, reason)``, the contract
    :func:`~cleverly.interventions.support._intervention_loads` already uses: a caller
    that receives no row always receives a machine-readable cause for it, and never has
    to infer one from an empty tuple.

    A ``nan`` cell is a stated outcome and not a silence.  Two states produce one: a draw
    that reported no ``psi`` for the parameter, and a spread that exists but is not finite
    on the inference scale, which is what a non-positive ratio draw gives.  Only one
    reason can be carried, so the missing draw is reported first: it says the spread was
    never computed, which is the stronger statement about the row.
    """
    if result.n_repeats < 2:
        return (), SPREAD_SINGLE_DRAW
    spreads = result.repeat_spread()
    rows: list[RepeatSpreadRow] = []
    unavailable: list[str] = []
    not_finite: list[str] = []
    for name in result.estimates:
        standard_error = float(result.estimates[name].std_error)
        spread = spreads.get(name, float("nan"))
        if name not in spreads:
            unavailable.append(name)
        elif not np.isfinite(spread):
            not_finite.append(name)
        rows.append(
            RepeatSpreadRow(
                estimand=name,
                n_repeats=result.n_repeats,
                standard_deviation=spread,
                reported_standard_error=standard_error,
                ratio_to_standard_error=(
                    spread / standard_error
                    if np.isfinite(spread) and np.isfinite(standard_error) and standard_error > 0.0
                    else float("nan")
                ),
            )
        )
    if not rows:
        return (), SPREAD_NO_PARAMETERS
    if unavailable:
        return tuple(rows), SPREAD_UNAVAILABLE_DRAWS + ", ".join(unavailable)
    if not_finite:
        return tuple(rows), SPREAD_NOT_FINITE + ", ".join(not_finite)
    return tuple(rows), None


def _aggregate_learner_info(
    diagnostics: Any,
) -> tuple[dict[str, float], dict[str, float]]:
    """Average Super Learner weights and risks across cross-fitting folds."""
    if not diagnostics:
        return {}, {}
    entries = diagnostics if isinstance(diagnostics, (list, tuple)) else [diagnostics]
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
