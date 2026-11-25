"""The assessment of a fitted effect curve: out-of-fold calibration, risk and stability.

:attr:`cleverly.effect_curves.EffectCurveResult.diagnostics` routes here.  Every reading is a
point reading with no interval (P1 plan 1.3 and 3.5): the out-of-fold refits of the producing
procedure are compared with the doubly robust pseudo-outcome :math:`\\tilde D` that the
producing fit's own cross-fitted nuisances give on the held-out rows, which never saw them.

- :meth:`CurveDiagnostics.calibration`: the least-squares fit of :math:`\\tilde D` on
  ``(1, prediction)`` over the held-out predictions, and the mean prediction against the mean
  :math:`\\tilde D` within quantile bins of the predictions.  On the log-ratio scale the fit
  minimises the one-step log-ratio loss instead.
- :meth:`CurveDiagnostics.risk_comparison`: the out-of-fold doubly robust risk of the curve
  against the constant (the average effect), the T-learner (the projection of the outcome
  regressions' contrast) and, for a difference, the DR-learner (the regression of
  :math:`\\tilde D` on the same basis).
- :meth:`CurveDiagnostics.stability`: refits on the same data under other cross-fitting split
  seeds and, for a cosine or B-spline basis, under the other family at a matched size.  It
  measures sensitivity to the split, not sampling noise.
- :meth:`CurveDiagnostics.modifier_stability`: refits without each modifier, and the change
  in the out-of-fold risk.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from ._assessment_cache import _cached
from ._typing import FloatArray, IntArray
from .assessment import AssessmentCapability, DiagnosticsFacade, _argument_resolved
from .exceptions import CapabilityError, Refusal, RefusalReason
from .utils.frames import frame_from_dict

__all__ = [
    "STABILITY_WARNING",
    "CurveCalibration",
    "CurveDiagnostics",
    "CurveRiskComparison",
    "CurveStability",
    "ModifierStability",
]

#: The across-refit SD of the predictions, relative to the curve's own spread, above which
#: :meth:`CurveDiagnostics.stability` warns that the curve is sensitive to the cross-fitting
#: split (P1 plan 3.5 and 5.4).  It is not a threshold on sampling noise.
STABILITY_WARNING = 0.5


def _readable(text: str) -> str:
    return text if len(text) <= 100 else text[:97] + "..."


@dataclass(frozen=True)
class CurveCalibration:
    """The out-of-fold calibration of a curve against the doubly robust pseudo-outcome.

    Parameters
    ----------
    scale : str
        The contrast's scale.
    intercept : float
        The calibration intercept; 0 for a calibrated curve.
    slope : float
        The calibration slope; 1 for a calibrated curve.
    bins : pandas.DataFrame
        Per quantile bin of the predictions: ``mean_prediction``, the bin's contrast of the
        mean pseudo-outcomes (``pseudo_outcome_contrast``, a difference or a log ratio) and
        ``rows``.  Descriptive.
    n_folds : int
        Outer folds of the refits.
    random_state : int
        The seed of the fold assignment and of each refit.
    backend : str
        The dataframe backend of the producing fit's data, which :meth:`to_frame` emits in.
    """

    scale: str
    intercept: float
    slope: float
    bins: pd.DataFrame = field(compare=False)
    n_folds: int
    random_state: int
    backend: str = "pandas"

    def to_frame(self) -> Any:
        """The binned table, in the producing fit's backend.

        Returns
        -------
        dataframe
            One row per bin.
        """
        return frame_from_dict(
            {name: self.bins[name].to_numpy() for name in self.bins.columns},
            backend=self.backend,
        )

    def plot(self, **kwargs: Any) -> Any:
        """Plot each bin's measured effect against its mean predicted effect.

        Parameters
        ----------
        **kwargs
            Passed to :func:`cleverly.plotting.calibration`: ``title=`` and ``ax=``.

        Returns
        -------
        Figure
            One panel with the bins, the 45-degree line of perfect calibration, and the
            calibration slope and intercept in the subtitle.

        Examples
        --------
        >>> import matplotlib
        >>> matplotlib.use("Agg")
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> import pandas as pd
        >>> from sklearn.linear_model import LogisticRegression
        >>> from cleverly import CausalStudy, ConditionalContrast, PointTreatment
        >>> from cleverly.bases import Linear, Modifiers
        >>> rng = np.random.default_rng(5)
        >>> w = rng.uniform(-1, 1, (300, 2))
        >>> a = rng.binomial(1, 1 / (1 + np.exp(-0.5 * w[:, 0])))
        >>> y = rng.binomial(1, 0.3 + 0.1 * w[:, 1] + a * (0.15 + 0.1 * w[:, 0]))
        >>> frame = pd.DataFrame({"Y": y, "A": a, "W1": w[:, 0], "W2": w[:, 1]})
        >>> study = CausalStudy(
        ...     frame, design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2"))
        ... )
        >>> contrast = ConditionalContrast(Modifiers(["W1", "W2"]), model=Linear(["W1", "W2"]))
        >>> curve = (
        ...     study.identify(contrast)
        ...     .estimate(
        ...         outcome_learner=LogisticRegression(),
        ...         treatment_learner=LogisticRegression(),
        ...         n_folds=2,
        ...         random_state=0,
        ...     )
        ...     .effect_curve()
        ... )
        >>> fig = curve.diagnostics.calibration(n_folds=2, bins=4, random_state=0).plot()
        >>> fig.axes[0].get_xlabel()
        'Mean predicted effect in the bin'
        >>> plt.close(fig)
        """
        from .plotting import calibration

        return calibration(self, **kwargs)

    def __repr__(self) -> str:
        return _readable(
            f"CurveCalibration(slope={self.slope:.3g}, intercept={self.intercept:.3g}, "
            f"folds={self.n_folds}, point readings)"
        )


@dataclass(frozen=True)
class CurveRiskComparison:
    """The out-of-fold doubly robust risk of the curve and of its comparators.

    Parameters
    ----------
    scale : str
        The contrast's scale.
    risks : Mapping of str to float
        The risk of ``curve``, ``constant``, ``t_learner`` and, for a difference,
        ``dr_learner``.  Lower is better; only differences between them are meaningful.
    omitted : str
        Why a comparator is absent, or the empty string.
    n_folds : int
        Outer folds of the refits.
    random_state : int
        The seed of the fold assignment and of each refit.
    backend : str
        The dataframe backend of the producing fit's data, which :meth:`to_frame` emits in.
    """

    scale: str
    risks: dict[str, float] = field(compare=False)
    omitted: str
    n_folds: int
    random_state: int
    backend: str = "pandas"

    def to_frame(self) -> Any:
        """The risks, and each one's difference from the curve's, in the fit's backend.

        Returns
        -------
        dataframe
            One row per learner.
        """
        curve = self.risks["curve"]
        return frame_from_dict(
            {
                "learner": list(self.risks),
                "risk": list(self.risks.values()),
                "minus_curve": [value - curve for value in self.risks.values()],
            },
            backend=self.backend,
        )

    def __repr__(self) -> str:
        parts = ", ".join(f"{name}={value:.3g}" for name, value in self.risks.items())
        return _readable(f"CurveRiskComparison({parts})")


@dataclass(frozen=True)
class CurveStability:
    """The spread of the curve's predictions across refits on the same data.

    The refits differ only in the cross-fitting split, so the reading measures sensitivity to
    the split, not sampling noise: a low value does not mean the curve is well determined.  A
    curve of pure noise at a constant effect reads low, because its wiggle comes from the
    draw, which every refit shares (the ``ep-assessment`` pilot reads 0.09).  Whether the
    effect varies at all is the question of ``heterogeneity_test`` and the projection's
    intervals.

    Parameters
    ----------
    relative_sd : float
        The mean across rows of the across-refit SD of the predictions, over the SD of the
        curve's own in-sample predictions.
    warning : bool
        Whether ``relative_sd`` exceeds :data:`STABILITY_WARNING`: a split-sensitivity
        warning.
    seeds : tuple of int
        The split seeds of the refits.
    alternative : str
        The alternative basis refitted at a matched size, or why there is none.
    alternative_relative_sd : float or None
        The same ratio for the alternative basis against the declared curve.
    """

    relative_sd: float
    warning: bool
    seeds: tuple[int, ...]
    alternative: str
    alternative_relative_sd: float | None

    def __repr__(self) -> str:
        return _readable(
            f"CurveStability(relative_sd={self.relative_sd:.3g}, warning={self.warning}, "
            f"refits={len(self.seeds)})"
        )


@dataclass(frozen=True)
class ModifierStability:
    """The change in out-of-fold risk when each modifier is dropped from the basis.

    Parameters
    ----------
    risks : Mapping of str to float
        The out-of-fold risk of the curve refitted without each modifier, keyed by it, and
        of the declared curve under ``"(none)"``.
    random_state : int
        The seed of the fold assignment and of each refit.
    backend : str
        The dataframe backend of the producing fit's data, which :meth:`to_frame` emits in.
    """

    risks: dict[str, float] = field(compare=False)
    random_state: int
    backend: str = "pandas"

    def to_frame(self) -> Any:
        """The risk without each modifier, and its change from the declared curve's.

        Returns
        -------
        dataframe
            One row per dropped modifier, in the producing fit's backend.
        """
        full = self.risks["(none)"]
        names = [name for name in self.risks if name != "(none)"]
        return frame_from_dict(
            {
                "dropped": names,
                "risk": [self.risks[name] for name in names],
                "change": [self.risks[name] - full for name in names],
            },
            backend=self.backend,
        )

    def __repr__(self) -> str:
        return _readable(
            f"ModifierStability(modifiers={len(self.risks) - 1}, "
            f"declared_risk={self.risks['(none)']:.3g})"
        )


def _refuse(text: str, alternative: str) -> CapabilityError:
    return CapabilityError(text, reason=RefusalReason.INVALID_DECLARATION, alternative=alternative)


def _raise(refusal: Refusal | None) -> None:
    """Raise ``refusal`` itself, so the call carries the record its row declares."""
    if refusal is not None:
        raise CapabilityError(refusal=refusal)


# ------------------------------------------------------------ the fit-wide rules
#
# Each predicate below is what one curve operation raises from, read without fitting.  The
# capability rows read the same predicate (:func:`cleverly.assessment._curve_fit_wide_rules`),
# so a row never reads available where the call refuses.  A curve that keeps no producing fit
# or estimator meets none of them: the replay gate refuses its rows instead.


def _estimator_of(curve: Any) -> Any:
    """The producing estimator, or ``None`` when the curve keeps no fit or estimator."""
    return getattr(getattr(curve, "fit", None), "estimator", None)


def modifier_stability_refusal(curve: Any) -> Refusal | None:
    """Why :meth:`CurveDiagnostics.modifier_stability` refuses this curve, or ``None``.

    Parameters
    ----------
    curve : EffectCurveResult
        The curve.

    Returns
    -------
    Refusal or None
        The record when the basis reads fewer than two modifiers, or when dropping a
        modifier leaves an additive basis empty.
    """
    estimator = _estimator_of(curve)
    if estimator is None:
        return None
    basis = estimator.effect_modification.basis
    columns = tuple(getattr(basis, "columns", ()))
    if len(columns) < 2 or not hasattr(basis, "columns"):
        return _refuse(
            "the basis reads fewer than two modifiers, or does not declare them as "
            "columns, so no modifier can be dropped from it.",
            "Declare a shipped basis of at least two modifiers, such as Linear(['V1', 'V2']).",
        ).refusal
    for name in columns:
        refusal = _without_refusal(basis, name)
        if refusal is not None:
            return refusal
    return None


def _zero_spread_refusal(curve: Any) -> Refusal | None:
    if float(np.std(curve.in_sample, ddof=1)) != 0.0:
        return None
    return _refuse(
        "the curve is constant on the fit rows, so its spread is zero and a relative "
        "SD is undefined.",
        "Read the coefficients' standard errors instead.",
    ).refusal


def stability_refusal(curve: Any) -> Refusal | None:
    """Why :meth:`CurveDiagnostics.stability` refuses this curve, or ``None``.

    Parameters
    ----------
    curve : EffectCurveResult
        The curve.

    Returns
    -------
    Refusal or None
        The record when the curve is constant on the fit rows, or when the alternative
        basis refit's sieve check refuses: an EP sieve size above the alternative family's
        largest member at the fit's rows.
    """
    zero = _zero_spread_refusal(curve)
    if zero is not None:
        return zero
    estimator = _estimator_of(curve)
    if estimator is None:
        return None
    other = _alternative_basis(estimator.effect_modification.basis)
    if other is None or not hasattr(estimator, "with_basis"):
        return None
    from .estimators.ep import EPEstimator, size_refusal

    rebuilt = estimator.with_basis(other)
    if not isinstance(rebuilt, EPEstimator):
        return None
    n = int(curve.fit.data.n)
    for size in rebuilt.sizes(n):
        error = size_refusal(size, other, n)
        if error is not None:
            return error.refusal
    return None


def _targeting_ate(fit: Any, estimator: Any, seed: int) -> tuple[Any, Refusal | None]:
    """The ATE estimator of the targeting curve's evaluation rows, or why it is refused."""
    from .estimators import TMLE
    from .targeting_curves import TARGETING_REPEATS_REFUSAL, check_targeting_settings

    spec = fit.nuisance.effect_modification
    kwargs = _ate_kwargs(estimator, reference=float(spec.pair[1]), seed=seed)
    if int(kwargs.get("repeats", 1)) > 1:
        return None, TARGETING_REPEATS_REFUSAL
    ate_estimator = TMLE(**kwargs)
    try:
        check_targeting_settings(ate_estimator)
    except CapabilityError as error:
        return None, error.refusal
    return ate_estimator, None


def targeting_curve_refusal(curve: Any) -> Refusal | None:
    """Why :meth:`CurveDiagnostics.targeting_curve` refuses this curve, or ``None``.

    Parameters
    ----------
    curve : EffectCurveResult
        The curve.

    Returns
    -------
    Refusal or None
        The record when the producing fit repeats its cross-fitting split, or carries a
        fluctuation setting the curve's ATE fit would drop.
    """
    estimator = _estimator_of(curve)
    if estimator is None:
        return None
    return _targeting_ate(curve.fit, estimator, 0)[1]


def _units(data: Any) -> int:
    units = np.arange(data.n) if data.cluster is None else np.asarray(data.cluster)
    return int(np.unique(units).size)


def _folds_refusal(n_units: int, n_folds: int) -> Refusal | None:
    if n_units >= n_folds:
        return None
    return _refuse(
        f"{n_units} independent units cannot fill {n_folds} folds.", "Pass a smaller n_folds."
    ).refusal


def _split_fraction_refusal(split: Any) -> Refusal | None:
    """The fold-average refusal of a ``split`` that is not a fraction in ``(0, 1)``."""
    from .targeting_curves import TARGETING_SPLIT_REFUSAL

    if (
        isinstance(split, bool)
        or not isinstance(split, (float, np.floating))
        or not 0.0 < float(split) < 1.0
    ):
        return TARGETING_SPLIT_REFUSAL
    return None


def _split_refusal(split: Any, n_units: int) -> Refusal | None:
    fraction = _split_fraction_refusal(split)
    if fraction is not None:
        return fraction
    n_train_units = int(np.floor(float(split) * n_units))
    if not 2 <= n_train_units <= n_units - 2:
        return _refuse(
            f"a split of {split} leaves {n_train_units} of {n_units} units to train on.",
            "Pass a split that leaves at least two units in each part.",
        ).refusal
    return None


def _engine_compare_refusal(curve: Any, compare: Any) -> Refusal | None:
    if compare is None or getattr(curve, "identified_effect", None) is not None:
        return None
    return _refuse(
        "compare= reads an external score column of the producing study, and this "
        "curve was fitted by the engine, with no study.",
        "Fit the curve through CausalStudy.identify(...).estimate(...).",
    ).refusal


class CurveDiagnostics(DiagnosticsFacade):
    """The assessment surface of an :class:`~cleverly.effect_curves.EffectCurveResult`.

    A :class:`~cleverly.assessment.DiagnosticsFacade`: its rows pass the method, replay and
    request gates, a refused row raises its own record, and :meth:`run_all` runs the rows a
    caller has paid for.  The summaries read the producing fit; the five curve operations
    refit the producing estimator.

    Parameters
    ----------
    curve : EffectCurveResult
        A curve fitted by :meth:`cleverly.estimators.TMLEResult.effect_curve` or by the EP
        learner (``method="ep"``).
    """

    #: The operations whose call refuses some argument values and not others.
    _request_gates: ClassVar[Mapping[str, str]] = {
        **DiagnosticsFacade._request_gates,
        "calibration": "_folds_gated",
        "risk_comparison": "_folds_gated",
        "modifier_stability": "_folds_gated",
        "targeting_curve": "_targeting_gated",
    }

    def __init__(self, curve: Any) -> None:
        super().__init__(curve)

    @property
    def curve(self) -> Any:
        """The assessed curve."""
        return self._result

    def _folds_gated(
        self, capability: AssessmentCapability, arguments: Mapping[str, Any]
    ) -> AssessmentCapability:
        """Resolve ``n_folds`` from the predicate the fold split raises from.

        A curve that keeps no producing fit has no units to count; its replay gate refuses
        the row, and the call raises that record.
        """
        if self.curve.fit is None:
            return capability
        n_units = _units(self.curve.fit.data)
        default = self._argument_default(capability.operation, "n_folds")
        return _argument_resolved(
            capability,
            "n_folds",
            arguments.get("n_folds"),
            lambda value: _folds_refusal(n_units, default if value is None else int(value)),
            (2,),
        )

    def _targeting_gated(
        self, capability: AssessmentCapability, arguments: Mapping[str, Any]
    ) -> AssessmentCapability:
        """Resolve ``split`` and ``compare`` from the predicates the call raises from."""
        if self.curve.fit is None:
            return capability
        n_units = _units(self.curve.fit.data)
        resolved = _argument_resolved(
            capability,
            "split",
            arguments.get("split"),
            lambda value: _split_refusal(0.5 if value is None else value, n_units),
            (0.5,),
        )
        if not resolved.available:
            return resolved
        return _argument_resolved(
            resolved,
            "compare",
            arguments.get("compare"),
            lambda value: _engine_compare_refusal(self.curve, value),
            (None,),
        )

    # ------------------------------------------------------------ the producing fit

    def _fit(self, slot: str = "refit_nuisances") -> Any:
        """The producing fit, or the replay gate's refusal when this curve cannot do ``slot``.

        The call reads the fit and its estimator itself, as well as the replay record, so a
        row that reads available on a curve without them still meets the refusal here.
        """
        from .assessment import replay_refusal, replayability

        replay = replayability(self.curve)
        fit = self.curve.fit
        kept = fit is not None and (
            slot == "summarize_existing_artifacts" or getattr(fit, "estimator", None) is not None
        )
        if not kept or not getattr(replay, slot):
            raise CapabilityError(refusal=replay_refusal(slot, replay))
        return fit

    def support(self) -> Any:
        """The overlap of the producing fit's mechanism, as the point support report reads it.

        Returns
        -------
        PositivityReport
            The producing fit's support report.
        """
        self._require("support")
        # The producing fit's facade files the answer in its own cache, so the curve's cache
        # would hold a second reference to it.
        return self._fit("summarize_existing_artifacts").diagnostics.support()

    def nuisance_models(self) -> Any:
        """The producing fit's held-out nuisance diagnostics.

        Returns
        -------
        NuisanceDiagnostics
            The producing fit's report.
        """
        self._require("nuisance_models")
        # The producing fit's facade files the answer in its own cache, so the curve's cache
        # would hold a second reference to it.
        return self._fit("summarize_existing_artifacts").diagnostics.nuisance_models()

    def score_equations(self) -> Any:  # type: ignore[override]
        """The score equations the producing fit solved.

        A projection fit reports its targeting's score equations.  An EP fit reports the
        weighted mean score of each debiasing feature before and after its debiasing
        regression, which solves them.

        Returns
        -------
        ScoreVerdict or DebiasingScores
            The producing fit's report.
        """
        self._require("score_equations")

        fit = self._fit("summarize_existing_artifacts")
        if self.curve.fitted_method != "ep":
            # Filed in the producing fit's cache, as support() is.
            return fit.diagnostics.score_equations()

        def compute() -> Any:
            from .estimators.ep import DebiasingScores

            return DebiasingScores.of(fit, backend=self.curve.backend)

        return _cached(self.curve, "diagnostics.score_equations", (), {}, compute)

    def influence(self) -> Any:
        """Refused: a curve's influence curves are its producing fit's.

        Returns
        -------
        InfluenceReport
            Never; the call always raises.

        Raises
        ------
        CapabilityError
            On a projection curve, an ``API_REDIRECTION`` to the producing fit, whose
            coefficients carry the influence curves; on an EP curve, which has no influence
            curve, an ``INVALID_DECLARATION``.
        """
        raise CapabilityError(refusal=curve_influence_refusal(self.curve))

    # ---------------------------------------------------------------- refits

    def _pieces(self) -> tuple[Any, Any, Any, tuple[FloatArray, ...]]:
        from .effect_curves import _pseudo_outcomes

        fit = self._fit()
        data = fit.data
        spec = fit.nuisance.effect_modification
        pseudo = _pseudo_outcomes(
            fit.nuisance,
            data,
            spec.pair,
            g_bounds=tuple(fit.config.g_bounds),
            missingness_bound=float(fit.config.missingness_bound),
        )
        return fit, data, spec, pseudo

    def _frame(self, data: Any) -> pd.DataFrame:
        return pd.DataFrame(np.asarray(data.covariates, dtype=float), columns=data.covariate_names)

    def _folds(self, data: Any, n_folds: int, seed: int) -> IntArray:
        units = np.arange(data.n) if data.cluster is None else np.asarray(data.cluster)
        labels, inverse = np.unique(units, return_inverse=True)
        _raise(_folds_refusal(len(labels), n_folds))
        order = np.random.default_rng(seed).permutation(len(labels))
        assignment = np.empty(len(labels), dtype=np.int64)
        assignment[order] = np.arange(len(labels)) % n_folds
        return assignment[inverse]

    def _estimator(self, fit: Any, basis: Any = None) -> Any:
        estimator = fit.estimator
        if basis is None:
            return estimator
        if hasattr(estimator, "with_basis"):
            return estimator.with_basis(basis)
        rebuilt = copy.copy(estimator)
        rebuilt.effect_modification = replace(estimator.effect_modification, basis=basis)
        return rebuilt

    def _out_of_fold(
        self, n_folds: int, seed: int, basis: Any = None
    ) -> tuple[FloatArray, IntArray]:
        fit, data, _, _ = self._pieces()
        folds = self._folds(data, n_folds, seed)
        frame = self._frame(data)
        estimator = self._estimator(fit, basis)
        predictions = np.empty(data.n)
        for k in range(n_folds):
            test = folds == k
            refit = estimator.refit(data.subset(np.flatnonzero(~test)), random_state=seed + k)
            predictions[test] = refit.effect_curve().predict(frame.loc[test])
        return predictions, folds

    @staticmethod
    def _seed(random_state: int | None) -> int:
        return 0 if random_state is None else int(random_state)

    def _risk(self, values: FloatArray, scale: str, pseudo: tuple[FloatArray, ...]) -> float:
        _, _, d_t, d_r = pseudo
        if scale == "difference":
            return float(np.mean(values**2 - 2.0 * values * (d_t - d_r)))
        return float(np.mean((d_t + d_r) * np.logaddexp(0.0, values) - d_t * values))

    def calibration(
        self, *, n_folds: int = 5, bins: int = 5, random_state: int | None = None
    ) -> CurveCalibration:
        """The out-of-fold calibration of the curve against the pseudo-outcome.

        Parameters
        ----------
        n_folds : int
            Outer folds of the refits.
        bins : int
            Quantile bins of the out-of-fold predictions in the table.
        random_state : int or None
            Seed of the folds and the refits; ``None`` uses 0.

        Returns
        -------
        CurveCalibration
            The slope, the intercept and the binned table, all point readings.
        """
        self._require("calibration", {"n_folds": n_folds})
        seed = self._seed(random_state)
        return _cached(
            self.curve,
            "diagnostics.calibration",
            (),
            {"n_folds": n_folds, "bins": bins, "random_state": seed},
            lambda: self._calibration(n_folds, bins, seed),
        )

    def _calibration(self, n_folds: int, bins: int, seed: int) -> CurveCalibration:
        _, _, _, pseudo = self._pieces()
        predictions, _ = self._out_of_fold(n_folds, seed)
        _, _, d_t, d_r = pseudo
        design = np.column_stack([np.ones(len(predictions)), predictions])
        difference = self.curve.scale == "difference"
        if difference:
            coefficients = np.linalg.lstsq(design, d_t - d_r)[0]
        else:
            coefficients = _log_ratio_calibration(design, d_t, d_r)
        edges = np.unique(np.quantile(predictions, np.linspace(0.0, 1.0, bins + 1)))
        index = np.clip(np.searchsorted(edges, predictions, side="right") - 1, 0, len(edges) - 2)
        table = pd.DataFrame({"bin": index, "prediction": predictions, "t": d_t, "r": d_r})
        grouped = table.groupby("bin")
        means = grouped[["prediction", "t", "r"]].mean()
        # The bin's contrast of the mean pseudo-outcomes: their difference, or the log of
        # their ratio, which is the one-step estimate of the bin's log ratio.
        observed = means["t"] - means["r"] if difference else np.log(means["t"] / means["r"])
        frame = pd.DataFrame(
            {
                "mean_prediction": means["prediction"],
                "pseudo_outcome_contrast": observed,
                "rows": grouped.size(),
            }
        ).reset_index(drop=True)
        return CurveCalibration(
            scale=self.curve.scale,
            intercept=float(coefficients[0]),
            slope=float(coefficients[1]),
            bins=frame,
            n_folds=n_folds,
            random_state=seed,
            backend=self.curve.backend,
        )

    def risk_comparison(
        self, *, n_folds: int = 5, random_state: int | None = None
    ) -> CurveRiskComparison:
        """The out-of-fold doubly robust risk of the curve and of three comparators.

        Parameters
        ----------
        n_folds : int
            Outer folds of the refits.
        random_state : int or None
            Seed of the folds and the refits; ``None`` uses 0.

        Returns
        -------
        CurveRiskComparison
            The risks, as point readings.
        """
        self._require("risk_comparison", {"n_folds": n_folds})
        seed = self._seed(random_state)
        return _cached(
            self.curve,
            "diagnostics.risk_comparison",
            (),
            {"n_folds": n_folds, "random_state": seed},
            lambda: self._risk_comparison(n_folds, seed),
        )

    def _risk_comparison(self, n_folds: int, seed: int) -> CurveRiskComparison:
        _, data, _, pseudo = self._pieces()
        q_t, q_r, d_t, d_r = pseudo
        predictions, folds = self._out_of_fold(n_folds, seed)
        phi = np.asarray(self.curve.model.design(self._frame(data)), dtype=float)
        scale = self.curve.scale
        constant = np.empty(data.n)
        t_learner = np.empty(data.n)
        dr_learner = np.empty(data.n)
        for k in range(n_folds):
            test, train = folds == k, folds != k
            if scale == "difference":
                constant[test] = np.mean(d_t[train] - d_r[train])
                t_coef = np.linalg.lstsq(phi[train], (q_t - q_r)[train])[0]
                dr_coef = np.linalg.lstsq(phi[train], (d_t - d_r)[train])[0]
                dr_learner[test] = phi[test] @ dr_coef
            else:
                constant[test] = np.log(np.mean(d_t[train]) / np.mean(d_r[train]))
                t_coef = _log_ratio_calibration(phi[train], q_t[train], q_r[train])
            t_learner[test] = phi[test] @ t_coef
        risks = {
            "curve": self._risk(predictions, scale, pseudo),
            "constant": self._risk(constant, scale, pseudo),
            "t_learner": self._risk(t_learner, scale, pseudo),
        }
        omitted = ""
        if scale == "difference":
            risks["dr_learner"] = self._risk(dr_learner, scale, pseudo)
        else:
            omitted = (
                "No DR-learner is compared on the log-ratio scale: the ratio has no "
                "pseudo-outcome whose regression estimates it (van der Laan, Carone and "
                "Luedtke 2024, section 3.2)."
            )
        return CurveRiskComparison(
            scale=scale,
            risks=risks,
            omitted=omitted,
            n_folds=n_folds,
            random_state=seed,
            backend=self.curve.backend,
        )

    def stability(self, *, n_seeds: int = 5, random_state: int | None = None) -> CurveStability:
        """The spread of the in-sample curve across refits under other split seeds.

        Every refit uses the same rows; only the cross-fitting split changes.  The reading
        measures sensitivity to the split, not sampling noise, so a low value does not mean
        the curve is well determined.  For whether the effect varies at all, fit the
        projection and read ``heterogeneity_test()`` and its intervals.

        Parameters
        ----------
        n_seeds : int
            Refits, each under its own split seed.
        random_state : int or None
            The first seed; ``None`` uses 0.

        Returns
        -------
        CurveStability
            The relative SD, and the alternative basis's reading where one exists.
        """
        _raise(stability_refusal(self.curve))
        self._require("stability")
        seed = self._seed(random_state)
        return _cached(
            self.curve,
            "diagnostics.stability",
            (),
            {"n_seeds": n_seeds, "random_state": seed},
            lambda: self._stability(n_seeds, seed),
        )

    def _stability(self, n_seeds: int, seed: int) -> CurveStability:
        fit, data, _, _ = self._pieces()
        frame = self._frame(data)
        seeds = tuple(seed + 1 + j for j in range(n_seeds))
        curves = np.column_stack(
            [fit.estimator.refit(data, random_state=s).effect_curve().predict(frame) for s in seeds]
        )
        spread = float(np.std(self.curve.in_sample, ddof=1))
        relative = float(np.mean(np.std(curves, axis=1, ddof=1)) / spread)
        alternative, alternative_sd = self._alternative(fit, data, frame, spread, seed)
        return CurveStability(
            relative_sd=relative,
            warning=relative > STABILITY_WARNING,
            seeds=seeds,
            alternative=alternative,
            alternative_relative_sd=alternative_sd,
        )

    def _alternative(
        self, fit: Any, data: Any, frame: pd.DataFrame, spread: float, seed: int
    ) -> tuple[str, float | None]:
        basis = fit.estimator.effect_modification.basis
        other = _alternative_basis(basis)
        if other is None:
            return (
                f"none: a {type(basis).__name__} basis has no alternative family at a matched "
                "size; only Cosine and BSpline do",
                None,
            )
        refit = self._estimator(fit, other).refit(data, random_state=seed)
        moved = refit.effect_curve().predict(frame) - self.curve.in_sample
        return repr(other), float(np.sqrt(np.mean(moved**2)) / spread)

    def modifier_stability(
        self, *, n_folds: int = 5, random_state: int | None = None
    ) -> ModifierStability:
        """The out-of-fold risk of the curve refitted without each modifier.

        Parameters
        ----------
        n_folds : int
            Outer folds of the refits.
        random_state : int or None
            Seed of the folds and the refits; ``None`` uses 0.

        Returns
        -------
        ModifierStability
            The risks, as point readings.
        """
        _raise(modifier_stability_refusal(self.curve))
        self._require("modifier_stability", {"n_folds": n_folds})
        seed = self._seed(random_state)
        return _cached(
            self.curve,
            "diagnostics.modifier_stability",
            (),
            {"n_folds": n_folds, "random_state": seed},
            lambda: self._modifier_stability(n_folds, seed),
        )

    def _modifier_stability(self, n_folds: int, seed: int) -> ModifierStability:
        fit, _, _, pseudo = self._pieces()
        basis = fit.estimator.effect_modification.basis
        columns = tuple(basis.columns)
        scale = self.curve.scale
        full, _ = self._out_of_fold(n_folds, seed)
        risks = {"(none)": self._risk(full, scale, pseudo)}
        for name in columns:
            predictions, _ = self._out_of_fold(n_folds, seed, basis=_without(basis, name))
            risks[name] = self._risk(predictions, scale, pseudo)
        return ModifierStability(risks=risks, random_state=seed, backend=self.curve.backend)

    def targeting_curve(
        self,
        *,
        split: float = 0.5,
        compare: str | None = None,
        compare_computed_from: Any = None,
        score_kind: str | None = None,
        grid: Any = None,
        groups: Any = 5,
        n_half_samples: int = 2000,
        alpha: float = 0.05,
        random_state: int | None = None,
        max_memory_bytes: int | None = None,
    ) -> Any:
        """How much treating by the curve first gains over treating at random, on held-out rows.

        A declared split: the units (whole clusters on a clustered fit) go to a training part
        of share ``split`` by ``random_state``, the producing estimator is refit there, and its
        curve scores the other rows.  The curve of
        :meth:`cleverly.study.IdentifiedEffect.targeting_curve` is then estimated on those
        evaluation rows from an ATE fit on them alone, with the producing fit's learners.  It
        reads the curve fitted on the training part, not the curve fitted on all rows.

        Parameters
        ----------
        split : float
            The training share, in ``(0, 1)``.
        compare : str or None
            An external score column of the producing study to compare on the same rows.
        compare_computed_from : sequence of str or None
            The adjustment columns ``compare`` was computed from.
        score_kind : {"external"} or None
            The declaration that ``compare`` was fitted on other rows.
        grid : sequence of float or None
            Treated fractions in ``(0, 1]``, increasing; ``0.1, ..., 1.0`` when omitted.
        groups : int, sequence of float or None
            Group effects: a number of equal groups, or their upper edges on the grid.
        n_half_samples : int
            Half-sample bootstrap draws.
        alpha : float
            The intervals hold at ``1 - alpha``, and the band jointly over the grid.
        random_state : int or None
            Seed of the split, the refits and the draws; ``None`` uses 0.
        max_memory_bytes : int or None
            The most memory the half-sample draws may add above the arrays already held, in
            bytes; ``None`` uses the memory available now.  The draws run in blocks sized to
            it, and a request whose setup, outputs and one draw would exceed it is refused
            before the refits.  The score is not known before the refit, so this check
            bounds its tie count by the evaluation rows.
            It bounds the draws only, not the refits, and never lowers ``n_half_samples``.

        Returns
        -------
        TargetingCurve
            The curve, with ``route="split"``.

        Raises
        ------
        CapabilityError
            If ``split`` is not a fraction (a fold average is refused), the producing fit
            repeats its cross-fitting split, or ``compare`` is asked of an engine fit.
        MethodConfigurationError
            If one half-sample draw at a time is predicted to exceed the memory budget.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from sklearn.linear_model import LinearRegression, LogisticRegression
        >>> from cleverly import CausalStudy, ConditionalContrast, PointTreatment
        >>> from cleverly.bases import Linear, Modifiers
        >>> rng = np.random.default_rng(0)
        >>> n = 1000
        >>> W = rng.uniform(-1, 1, n)
        >>> A = rng.binomial(1, 0.5, n)
        >>> Y = A * (1 + W) + rng.normal(0, 0.5, n)
        >>> frame = pd.DataFrame({"Y": Y, "A": A, "W": W})
        >>> study = CausalStudy(
        ...     frame, design=PointTreatment(outcome="Y", treatment="A", adjustment=("W",))
        ... )
        >>> effect = study.identify(ConditionalContrast(Modifiers(["W"]), model=Linear(["W"])))
        >>> fit = effect.estimate(
        ...     outcome_learner=LinearRegression(),
        ...     treatment_learner=LogisticRegression(),
        ...     outcome_by_arm=True,
        ...     q_bounds=(-4.0, 6.0),
        ...     random_state=0,
        ... )
        >>> targeting = fit.effect_curve().diagnostics.targeting_curve(
        ...     n_half_samples=200, random_state=0
        ... )
        >>> targeting.route, targeting.n_train, targeting.n_eval
        ('split', 500, 500)
        >>> bool(targeting.area("autoc")["estimate"] > 0)
        True
        """
        # A fold average is a malformed request, refused before any row is read, as
        # ``refute`` checks its arguments first.  Then the fit-wide rules, each raising the
        # record its row declares, and then the row this request resolves to.
        _raise(_split_fraction_refusal(split))
        _raise(targeting_curve_refusal(self.curve))
        self._require("targeting_curve", {"split": split, "compare": compare})
        seed = self._seed(random_state)
        arguments = {
            "split": split,
            "compare": compare,
            "compare_computed_from": compare_computed_from,
            "score_kind": score_kind,
            "grid": grid,
            "groups": groups,
            "n_half_samples": n_half_samples,
            "alpha": alpha,
            "random_state": seed,
            "max_memory_bytes": max_memory_bytes,
        }
        return _cached(
            self.curve,
            "diagnostics.targeting_curve",
            (),
            arguments,
            lambda: self._targeting_curve(**arguments),
        )

    def _targeting_curve(
        self,
        *,
        split: float,
        compare: str | None,
        compare_computed_from: Any,
        score_kind: str | None,
        grid: Any,
        groups: Any,
        n_half_samples: int,
        alpha: float,
        random_state: int,
        max_memory_bytes: int | None,
    ) -> Any:
        from .targeting_curves import check_draw_memory, check_grid, check_groups, targeting_fit

        seed = random_state
        fit, data, spec, _ = self._pieces()
        estimator = self._estimator(fit)
        ate_estimator, refusal = _targeting_ate(fit, estimator, seed)
        _raise(refusal)
        other = None
        _raise(_engine_compare_refusal(self.curve, compare))
        if compare is not None:
            effect = self.curve.identified_effect
            other = np.asarray(
                effect._external_score(compare, compare_computed_from, score_kind), dtype=float
            )
            if len(other) != data.n:
                raise _refuse(
                    f"the study holds {len(other)} rows and the producing fit {data.n}",
                    "Compare on the study whose rows the curve was fitted on.",
                )
        units = np.arange(data.n) if data.cluster is None else np.asarray(data.cluster)
        labels, inverse = np.unique(units, return_inverse=True)
        n_units = len(labels)
        _raise(_split_refusal(split, n_units))
        n_train_units = int(np.floor(float(split) * n_units))
        order = np.random.default_rng(seed).permutation(n_units)
        in_train = np.zeros(n_units, dtype=bool)
        in_train[order[:n_train_units]] = True
        train = np.flatnonzero(in_train[inverse])
        evaluation = np.flatnonzero(~in_train[inverse])
        grid_ = check_grid(grid)
        check_draw_memory(
            n=len(evaluation),
            units=n_units - n_train_units,
            blocks=len(evaluation),
            scores=1 if other is None else 2,
            draws=int(n_half_samples),
            grid=grid_,
            edges=check_groups(groups, grid_),
            max_memory_bytes=max_memory_bytes,
            ties_bounded=True,
        )
        frame = self._frame(data)
        refit = estimator.refit(data.subset(train), random_state=seed)
        refit_curve = refit.effect_curve()
        score = np.asarray(refit_curve.predict(frame.iloc[evaluation]), dtype=float)
        ate = ate_estimator.fit(data.subset(evaluation)).single()
        scores = {self.curve.prediction_column: score}
        if other is not None and compare is not None:
            scores[compare] = other[evaluation]
        return targeting_fit(
            ate,
            ate.data,
            scores,
            pair=(float(spec.pair[0]), float(spec.pair[1])),
            grid=grid,
            groups=groups,
            n_half_samples=n_half_samples,
            alpha=alpha,
            random_state=seed,
            route="split",
            n_train=len(train),
            score_curve=refit_curve,
            max_memory_bytes=max_memory_bytes,
        )


def _ate_kwargs(estimator: Any, *, reference: float, seed: int) -> dict[str, Any]:
    """The producing estimator's settings for a plain ATE fit, with its learners."""
    import inspect

    from .estimators import TMLE

    if hasattr(estimator, "tmle_kwargs"):
        kwargs = dict(estimator.tmle_kwargs)
    else:
        names = [name for name in inspect.signature(TMLE.__init__).parameters if name != "self"]
        kwargs = {name: getattr(estimator, name) for name in names if hasattr(estimator, name)}
    kwargs.update(
        {
            "estimands": ("ate",),
            "reference": reference,
            "effect_modification": None,
            "split_plan": None,
            "random_state": seed,
        }
    )
    return kwargs


def _drops(basis: Any, name: str) -> bool:
    """Whether ``basis`` reads ``name`` alone, so nothing of it is left without ``name``."""
    return tuple(getattr(basis, "columns", ())) == (name,)


def _without(basis: Any, name: str) -> Any:
    """``basis`` without the modifier ``name``.

    An :class:`~cleverly.bases.Additive` basis drops each component that reads only ``name``
    and rebuilds each that reads it among others.  A :class:`~cleverly.bases.Tensor` keeps
    the other factor when one factor reads only ``name``, and rebuilds a factor that reads it
    among others.  An :class:`~cleverly.bases.Interactions` basis rebuilds its additive basis
    and keeps its order where enough components remain.  Every other shipped basis is a
    dataclass of columns and domains.  :func:`_without_refusal` names what cannot be dropped.
    """
    from .bases import Additive, Interactions, Tensor

    _raise(_without_refusal(basis, name))
    if name not in tuple(getattr(basis, "columns", ())):
        return basis
    if isinstance(basis, Additive):
        parts = [
            _without(component, name) for component in basis.bases if not _drops(component, name)
        ]
        return Additive(*parts, intercept=basis.intercept)
    if isinstance(basis, Tensor):
        if _drops(basis.left, name):
            return _without(basis.right, name)
        if _drops(basis.right, name):
            return _without(basis.left, name)
        return Tensor(_without(basis.left, name), _without(basis.right, name))
    if isinstance(basis, Interactions):
        additive = _without(basis.basis, name)
        order = min(basis.order, len(additive.bases))
        return additive if order == 1 else Interactions(additive, order)
    kept = [column for column in basis.columns if column != name]
    fields: dict[str, Any] = {"columns": kept}
    domains = getattr(basis, "domains", None)
    if isinstance(domains, dict):
        fields["domains"] = {key: value for key, value in domains.items() if key in kept}
    elif isinstance(domains, tuple) and domains:
        fields["domains"] = tuple(item for item in domains if item[0] in kept)
    return replace(basis, **fields)


def _without_refusal(basis: Any, name: str) -> Refusal | None:
    """Why :func:`_without` cannot drop ``name`` from ``basis``, or ``None``.

    Read without building the reduced basis, so a capability row can ask it.  An additive
    basis whose every component reads ``name`` alone, or a tensor product whose two factors
    each read ``name`` alone, leaves nothing behind.  A
    :class:`~cleverly.bases.UserBasis` evaluates one function of all its columns, which
    cannot be rebuilt without one of them.
    """
    from .bases import Additive, Interactions, Tensor, UserBasis

    if name not in tuple(getattr(basis, "columns", ())):
        return None
    if isinstance(basis, UserBasis):
        return _refuse(
            f"a UserBasis evaluates one function of its columns, so it cannot be rebuilt "
            f"without {name}.",
            "Declare a shipped basis, such as Linear or Additive, to compare modifiers.",
        ).refusal
    if isinstance(basis, Interactions):
        return _without_refusal(basis.basis, name)
    if isinstance(basis, Tensor):
        factors = (basis.left, basis.right)
        if all(_drops(factor, name) for factor in factors):
            return _refuse(
                f"both factors of the tensor product read {name} alone, so nothing is left "
                "without it.",
                "Declare a basis with a factor for each modifier.",
            ).refusal
        for factor in factors:
            inner = None if _drops(factor, name) else _without_refusal(factor, name)
            if inner is not None:
                return inner
        return None
    if not isinstance(basis, Additive):
        return None
    kept = False
    for component in basis.bases:
        if _drops(component, name):
            continue
        inner = _without_refusal(component, name)
        if inner is not None:
            return inner
        kept = True
    if kept:
        return None
    return _refuse(
        f"every component of the basis reads {name}, so nothing is left without it.",
        "Declare a basis with a component for each modifier.",
    ).refusal


def _alternative_basis(basis: Any) -> Any:
    """The other shipped family at a matched size: Cosine for BSpline and back, else ``None``."""
    from .bases import BSpline, Cosine

    if isinstance(basis, Cosine):
        degree = min(3, basis.size)
        return BSpline(
            basis.columns,
            n_interior_knots=basis.size - degree,
            degree=degree,
            domains=basis.domains,
            intercept=basis.intercept,
        )
    if isinstance(basis, BSpline):
        return Cosine(
            basis.columns,
            size=basis.n_interior_knots + basis.degree,
            domains=basis.domains,
            intercept=basis.intercept,
        )
    return None


def curve_influence_refusal(curve: Any) -> Refusal:
    """Why ``curve.diagnostics.influence()`` is refused.

    Parameters
    ----------
    curve : EffectCurveResult
        The curve.

    Returns
    -------
    Refusal
        An ``API_REDIRECTION`` to the producing fit on a projection curve, whose coefficients
        carry the influence curves, and an ``INVALID_DECLARATION`` on an EP curve.
    """
    if curve.fitted_method == "ep":
        return Refusal(
            RefusalReason.INVALID_DECLARATION,
            "an EP curve has no influence curve: its source proves no limit law for the "
            "curve or its coefficients, so it reports no interval.",
        )
    return Refusal(
        RefusalReason.API_REDIRECTION,
        "a projection curve's influence curves are those of its coefficients, which the "
        "producing fit reports.",
        alternative="Call curve.fit.diagnostics.influence().",
    )


def _log_ratio_calibration(design: FloatArray, d_t: FloatArray, d_r: FloatArray) -> FloatArray:
    """Minimise the one-step log-ratio loss ``sum (D_t + D_r) log(1 + e^eta) - D_t eta``.

    Newton steps with a step-halving line search; the pseudo-outcomes may be negative, so the
    loss need not be convex, and the iteration stops at a stationary point.
    """
    total = d_t + d_r
    beta = np.zeros(design.shape[1])

    def loss(value: FloatArray) -> float:
        eta = design @ value
        return float(np.mean(total * np.logaddexp(0.0, eta) - d_t * eta))

    current = loss(beta)
    for _ in range(100):
        eta = design @ beta
        e = 1.0 / (1.0 + np.exp(-eta))
        gradient = design.T @ (total * e - d_t) / len(eta)
        hessian = design.T @ ((total * e * (1.0 - e))[:, None] * design) / len(eta)
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = gradient
        size = 1.0
        while size > 1e-8:
            candidate = beta - size * step
            value = loss(candidate)
            if value <= current:
                break
            size /= 2.0
        else:
            break
        beta, change = candidate, current - value
        current = value
        if np.max(np.abs(size * step)) < 1e-12 or change < 1e-15:
            break
    return beta
