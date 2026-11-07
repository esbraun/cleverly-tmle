r"""Fitted conditional effect curves, their intervals, and the assessment of a fixed score.

A projection TMLE of a :class:`cleverly.ConditionalContrast` reports coefficients
:math:`\hat\beta` of a declared basis :math:`\varphi(V)`.  The curve
:math:`v \mapsto \varphi(v)^\top\hat\beta` is linear in :math:`\hat\beta`, so its influence
curve at a point is :math:`\varphi(v)^\top D_\beta`, its pointwise interval is Wald, and a
band over a declared finite grid is the shipped multiplier band of those curves (see
``docs/technical-reference/conditional-effects.md``).  :class:`EffectCurveResult` holds that
curve, predicts it at new rows, and saves and loads like every result.

:func:`score_assessment` answers a different question: how good is a score :math:`s(V)`
that was fitted on other rows?  Its risk difference against the constant ATE and its
calibration projection onto :math:`(1, s)` are fixed-dimension parameters with intervals,
estimated by one-step estimators from the cross-fitted nuisances of an ATE fit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import numpy as np
from scipy.special import expit
from scipy.stats import chi2

from ._typing import FloatArray, IntArray
from .bases import Modifiers
from .exceptions import CapabilityError, DataError, RefusalReason
from .inference.delta import reference_quantile
from .inference.influence import ParameterEstimate, make_estimate
from .inference.multiplier import MultiplierKind, multiplier_critical_value
from .inference.results import estimate_covariance
from .utils.frames import as_frame, backend_of, frame_from_dict, is_dataframe

if TYPE_CHECKING:
    from ._curve_assessment import CurveDiagnostics

__all__ = [
    "CurveBand",
    "EffectCurveResult",
    "HeterogeneityTest",
    "ProjectionCurve",
    "ScoreAssessment",
    "heterogeneity_test",
    "score_assessment",
]

#: The prefix of the prediction column :meth:`EffectCurveResult.to_frame` writes.  The
#: assessment of an external score reads it as the mark of a score fitted on these rows.
PREDICTION_PREFIX = "effect_curve:"


@dataclass(frozen=True)
class HeterogeneityTest:
    """The joint Wald test that every non-constant coefficient of the projection is zero.

    It tests "no heterogeneity that the declared basis can express", not "no
    heterogeneity": a contrast that varies orthogonally to the basis leaves every
    coefficient but the intercept at zero.

    Parameters
    ----------
    statistic : float
        The Wald chi-square statistic.
    df : int
        Its degrees of freedom, the number of non-constant terms.
    p_value : float
        The upper-tail chi-square probability.
    terms : tuple of str
        The tested terms.
    scale : {"difference", "log_ratio"}
        The scale of the projected contrast.
    """

    statistic: float
    df: int
    p_value: float
    terms: tuple[str, ...]
    scale: str

    def __str__(self) -> str:
        return (
            f"Wald chi-square = {self.statistic:.3f} on {self.df} df, p = {self.p_value:.4g} "
            f"(terms {', '.join(self.terms)})"
        )


def heterogeneity_test(
    estimates: dict[str, ParameterEstimate],
    names: tuple[str, ...],
    terms: tuple[str, ...],
    intercepts: tuple[int, ...],
    *,
    scale: str,
    cluster: IntArray | None = None,
) -> HeterogeneityTest:
    """The Wald test of the non-constant coefficients, from their joint covariance.

    Parameters
    ----------
    estimates : dict of str to ParameterEstimate
        The fit's estimates.
    names : tuple of str
        The reported name of each coefficient, in basis order.
    terms : tuple of str
        The basis terms, in the same order.
    intercepts : tuple of int
        The positions of the intercept, which the test leaves out.
    scale : str
        The scale of the projected contrast.
    cluster : ndarray or None
        Cluster codes, for the cluster-level covariance.

    Returns
    -------
    HeterogeneityTest
        The statistic, its degrees of freedom and its p-value.

    Raises
    ------
    CapabilityError
        If the basis has no intercept column, or no non-constant term.
    """
    if not intercepts:
        raise CapabilityError(
            "the basis has no intercept column, so its coefficients being zero says the "
            "contrast is zero, not that it is constant; a test of heterogeneity needs the "
            "constant function to be one term of the basis, so that the other terms carry "
            "only the variation",
            reason=RefusalReason.INVALID_DECLARATION,
            alternative=(
                "Declare the basis with its intercept (the default), for example Linear(...) "
                "or Indicators(...) without intercept=False."
            ),
        )
    keep = [j for j in range(len(terms)) if j not in intercepts]
    if not keep:
        raise CapabilityError(
            "the basis has no non-constant term, so there is no heterogeneity to test",
            reason=RefusalReason.INVALID_DECLARATION,
            alternative="Declare a basis with at least one term that varies with the modifiers.",
        )
    selected = tuple(names[j] for j in keep)
    covariance = estimate_covariance(estimates, list(selected), cluster=cluster)
    beta = np.array([estimates[name].psi for name in selected])
    statistic = float(beta @ np.linalg.solve(covariance, beta))
    df = len(keep)
    return HeterogeneityTest(
        statistic=statistic,
        df=df,
        p_value=float(chi2.sf(statistic, df)),
        terms=tuple(terms[j] for j in keep),
        scale=scale,
    )


@dataclass(frozen=True)
class CurveBand:
    """A simultaneous band of the projection curve over a declared finite grid.

    Parameters
    ----------
    grid : dict of str to ndarray
        The modifier values of each grid point.
    estimate : ndarray
        The curve at each point.
    lower : ndarray
        The lower end of the band.
    upper : ndarray
        The upper end of the band.
    critical_value : float
        The multiplier quantile of the max-t statistic.
    alpha : float
        The joint level is ``1 - alpha``.
    n_replicates : int
        Multiplier draws.
    kind : str
        The multiplier distribution.
    """

    grid: dict[str, np.ndarray]
    estimate: FloatArray
    lower: FloatArray
    upper: FloatArray
    critical_value: float
    alpha: float
    n_replicates: int
    kind: str


@dataclass(frozen=True)
class ProjectionCurve:
    r"""The fitted projection :math:`v \mapsto \varphi(v)^\top\hat\beta` and its curves.

    Parameters
    ----------
    basis : Basis
        The resolved basis.
    modifiers : Modifiers
        The modifier schema the basis reads.
    scale : {"difference", "log_ratio"}
        The scale of the projected contrast.
    terms : tuple of str
        The basis terms.
    beta : ndarray
        ``(p,)`` coefficients on the outcome's own scale.
    covariance : ndarray
        ``(p, p)`` joint covariance of the coefficients.
    influence : ndarray
        ``(n, p)`` influence curves of the coefficients, which a band reads.
    cluster : ndarray or None
        Cluster codes of the fit, or ``None``.
    reference_df : int or None
        Degrees of freedom of the t reference the coefficients' intervals read, or ``None``
        for the normal reference.
    """

    basis: Any
    modifiers: Modifiers
    scale: str
    terms: tuple[str, ...]
    beta: FloatArray
    covariance: FloatArray
    influence: FloatArray
    cluster: IntArray | None = None
    reference_df: int | None = None

    def design(self, frame: Any) -> FloatArray:
        """The basis at the rows of ``frame``, checked against the modifier schema.

        Parameters
        ----------
        frame : dataframe or mapping
            Rows holding the modifier columns.

        Returns
        -------
        ndarray
            ``(m, p)`` design.
        """
        return np.asarray(self.basis.evaluate(frame, self.modifiers), dtype=float)

    def predict(self, frame: Any) -> FloatArray:
        """The curve at the rows of ``frame``.

        Parameters
        ----------
        frame : dataframe or mapping
            Rows holding the modifier columns.

        Returns
        -------
        ndarray
            One prediction per row.
        """
        return self.design(frame) @ self.beta

    def std_error(self, frame: Any) -> FloatArray:
        """The pointwise standard error of the curve at the rows of ``frame``.

        Parameters
        ----------
        frame : dataframe or mapping
            Rows holding the modifier columns.

        Returns
        -------
        ndarray
            One standard error per row.
        """
        phi = self.design(frame)
        return np.sqrt(np.einsum("ip,pq,iq->i", phi, self.covariance, phi))

    def band(
        self,
        grid: Any,
        *,
        alpha: float = 0.05,
        n_replicates: int = 1000,
        kind: MultiplierKind = "rademacher",
        random_state: int | None = None,
    ) -> CurveBand:
        """A simultaneous band over the declared grid ``grid``.

        Parameters
        ----------
        grid : dataframe or mapping
            The grid points, one row each, holding the modifier columns.
        alpha : float
            The band holds jointly at ``1 - alpha``.
        n_replicates : int
            Multiplier draws.
        kind : {"rademacher", "mammen", "normal"}
            The multiplier distribution.
        random_state : int or None
            Seed for the draws.

        Returns
        -------
        CurveBand
            The curve and its band at each grid point.
        """
        phi = self.design(grid)
        estimate = phi @ self.beta
        se = np.sqrt(np.einsum("ip,pq,iq->i", phi, self.covariance, phi))
        if np.any(se <= 0):
            raise DataError("a grid point has a zero standard error; drop it from the grid")
        curves = self.influence @ phi.T
        critical = multiplier_critical_value(
            curves,
            se,
            n=int(curves.shape[0]),
            cluster=self.cluster,
            alpha=alpha,
            n_replicates=n_replicates,
            kind=kind,
            random_state=random_state,
        )
        values = self.modifiers.values(grid, self.basis.columns)
        return CurveBand(
            grid={name: np.asarray(value) for name, value in values.items()},
            estimate=estimate,
            lower=estimate - critical * se,
            upper=estimate + critical * se,
            critical_value=float(critical),
            alpha=alpha,
            n_replicates=n_replicates,
            kind=str(kind),
        )


@dataclass(frozen=True)
class EffectCurveResult:
    """A fitted conditional effect curve: its model, its predictions and how it was fitted.

    Two fits produce it.  A projection fit (``method="tmle"``) produces it through
    :meth:`cleverly.estimators.TMLEResult.effect_curve`: the curve is the projection of the
    conditional contrast onto the declared basis, with pointwise intervals and a band.  The
    EP learner (``method="ep"``) returns it directly: the curve has no interval, and with
    ``model=None`` it estimates the conditional contrast itself, approximated within the
    final learner's class.  It is not a :class:`cleverly.CausalResult`: it has no scalar
    estimate, and the coefficients with their intervals stay on the
    :class:`~cleverly.estimators.TMLEResult` of a projection fit.

    Parameters
    ----------
    estimand_name : str
        ``"cate"`` or ``"log_crr"``.
    modifiers : Modifiers
        The modifier schema.
    model : ProjectionCurve or EPCurve
        The fitted curve, which predicts at new rows.
    scale : {"difference", "log_ratio"}
        The scale of the contrast.
    in_sample : ndarray
        The curve at the rows of the fit.
    target_sentence : str
        What the curve estimates, in one sentence.
    inference_status : str
        The inference status of the coefficients.
    status : {"supported", "experimental"}
        The support label of the coefficients the curve is built from.
    status_reason : str
        Why the curve carries that label, in plain words.
    lower_bound : float
        The lower end of the range a contrast of means of the outcome can take.
    upper_bound : float
        The upper end of that range.
    band : CurveBand or None
        The simultaneous band over the grid the fit declared, or ``None``.
    backend : str
        The dataframe backend of the fit's data.
    identified_effect : Any
        The identified effect the fit came from, or ``None`` for an engine fit.
    provenance : Provenance or None
        The producing fit's provenance record.
    fit : Any
        The producing fit, which the curve's assessment reads and refits: a
        :class:`~cleverly.estimators.TMLEResult`, or an
        :class:`~cleverly.estimators.ep.EPFit`.
    fitted_method : {"tmle", "ep"}
        The method that produced the curve.
    sieve : SieveRecord or None
        How an EP fit set its sieve size, and what its cross-validation read; ``None`` for a
        projection fit.
    clipped_share : float
        The share of in-sample predictions an EP fit with ``model=None`` clipped to the range
        of a contrast of means; 0 for every other curve.
    fit_warnings : tuple of FitWarning
        The warnings the fit raised, recorded as a point fit records them.
    """

    estimand_name: str
    modifiers: Modifiers
    model: Any
    scale: str
    in_sample: FloatArray
    target_sentence: str
    inference_status: str
    status: str
    status_reason: str
    lower_bound: float
    upper_bound: float
    band: CurveBand | None = None
    backend: str = "pandas"
    identified_effect: Any = field(default=None, repr=False, compare=False)
    provenance: Any = field(default=None, repr=False, compare=False)
    fit: Any = field(default=None, repr=False, compare=False)
    fitted_method: str = "tmle"
    sieve: Any = None
    clipped_share: float = 0.0
    fit_warnings: tuple[Any, ...] = field(default=(), compare=False)
    assessment_cache: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    #: The assessment family whose capability rows a curve reads.
    assessment_family: ClassVar[str] = "effect_modification"

    @property
    def diagnostics(self) -> CurveDiagnostics:
        """The curve's assessment: support, out-of-fold calibration, risk and stability.

        Each refit reruns the producing estimator, so these need the fit this curve came
        from; a loaded curve keeps none.  ``diagnostics.capabilities`` lists the rows.
        """
        from ._curve_assessment import CurveDiagnostics

        return CurveDiagnostics(self)

    @property
    def prediction_column(self) -> str:
        """The name of the prediction column :meth:`to_frame` writes."""
        return f"{PREDICTION_PREFIX}{self.estimand_name}"

    @property
    def outside_share(self) -> float:
        """The share of in-sample predictions outside the range of a contrast of means.

        A projection is a best approximation and can leave that range; it is not clipped.
        On the log ratio there is no such range and the share is 0.
        """
        if self.scale != "difference":
            return 0.0
        width = self.upper_bound - self.lower_bound
        outside = np.abs(self.in_sample) > width
        return float(np.mean(outside))

    def predict(self, data: Any, *, interval: bool = False, alpha: float = 0.05) -> Any:
        """The curve at new rows, with a pointwise Wald interval on request.

        The rows are checked against the modifier schema: a missing column, an
        undeclared level, or a value outside its domain under ``outside="refuse"`` raises,
        and ``outside="clip"`` clips.  A projection curve gives an interval; an EP curve
        refuses ``interval=True``, because its source proves no limit law for the curve.

        Parameters
        ----------
        data : dataframe or mapping
            Rows holding the modifier columns.
        interval : bool
            ``False`` returns the predictions.  ``True`` returns a frame with the
            estimate, its standard error and the interval.
        alpha : float
            The pointwise interval holds at ``1 - alpha``, against the coefficients' own
            reference: the normal, or t on a few-cluster fit.

        Returns
        -------
        ndarray or dataframe
            The predictions, or a frame in the input's backend.

        Raises
        ------
        CapabilityError
            If ``interval=True`` on an EP curve.

        See Also
        --------
        cleverly.estimators.TMLEResult.effect_curve : What returns this result.
        cleverly.estimators.TMLEResult.heterogeneity_test : The test of the non-constant
            terms.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from sklearn.linear_model import LogisticRegression
        >>> from cleverly import CausalStudy, ConditionalContrast, PointTreatment
        >>> from cleverly.bases import Linear, Modifiers
        >>> rng = np.random.default_rng(0)
        >>> W = rng.uniform(-1, 1, 400)
        >>> A = rng.binomial(1, 0.5, 400)
        >>> Y = rng.binomial(1, 0.3 + 0.2 * A * (W > 0))
        >>> frame = pd.DataFrame({"Y": Y, "A": A, "W": W})
        >>> study = CausalStudy(
        ...     frame, design=PointTreatment(outcome="Y", treatment="A", adjustment=("W",))
        ... )
        >>> effect = study.identify(ConditionalContrast(Modifiers(["W"]), model=Linear(["W"])))
        >>> fit = effect.estimate(
        ...     outcome_learner=LogisticRegression(),
        ...     treatment_learner=LogisticRegression(),
        ...     cross_fit=False,
        ...     random_state=0,
        ... )
        >>> curve = fit.effect_curve()
        >>> curve.predict({"W": np.array([-0.5, 0.5])}).shape
        (2,)
        >>> list(curve.predict({"W": np.array([0.5])}, interval=True).columns)
        ['estimate', 'std_err', 'ci_lower', 'ci_upper']
        """
        if interval and self.fitted_method == "ep":
            from .estimators.ep import EP_INTERVAL_REFUSAL

            raise CapabilityError.from_refusal(EP_INTERVAL_REFUSAL)
        estimate = self.model.predict(data)
        if not interval:
            return estimate
        se = self.model.std_error(data)
        # The reference the coefficients' own intervals read: t on a few-cluster fit.
        z = reference_quantile(alpha, self.model.reference_df)
        backend = backend_of(as_frame(data)) if is_dataframe(data) else self.backend
        return frame_from_dict(
            {
                "estimate": estimate,
                "std_err": se,
                "ci_lower": estimate - z * se,
                "ci_upper": estimate + z * se,
            },
            backend=backend,
        )

    def plot(self, **kwargs: Any) -> Any:
        """Plot the conditional effect over one modifier.

        The curve is drawn over the modifier with its pointwise interval (a projection fit
        only; an EP curve carries none), the average of the curve over the fitted rows as a
        reference line, a rug of the modifier's values, and the band at the grid points the
        fit declared. A curve that reads other modifiers is drawn with each held at a value
        you give in ``at=``.

        Parameters
        ----------
        **kwargs
            Passed to :func:`cleverly.plotting.curve`: ``modifier=``, ``by=``, ``at=``,
            ``alpha=``, ``outcome_label=``, ``units=`` and ``title=``.

        Returns
        -------
        Figure
            One panel, or one panel per level of ``by``.

        Raises
        ------
        ValueError
            When the curve reads modifiers that ``modifier=``, ``by=`` and ``at=`` leave
            undeclared.

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
        >>> rng = np.random.default_rng(0)
        >>> W = rng.uniform(-1, 1, 400)
        >>> A = rng.binomial(1, 0.5, 400)
        >>> Y = rng.binomial(1, 0.3 + 0.2 * A * (W > 0))
        >>> frame = pd.DataFrame({"Y": Y, "A": A, "W": W})
        >>> study = CausalStudy(
        ...     frame, design=PointTreatment(outcome="Y", treatment="A", adjustment=("W",))
        ... )
        >>> effect = study.identify(ConditionalContrast(Modifiers(["W"]), model=Linear(["W"])))
        >>> curve = effect.estimate(
        ...     outcome_learner=LogisticRegression(),
        ...     treatment_learner=LogisticRegression(),
        ...     cross_fit=False,
        ...     random_state=0,
        ... ).effect_curve()
        >>> fig = curve.plot()
        >>> fig.axes[0].get_title(loc="left")
        'Conditional effect by W'
        >>> plt.close(fig)
        """
        from .plotting import curve

        return curve(self, **kwargs)

    def to_frame(self) -> Any:
        """The in-sample predictions, one row per fit row, in the fit's backend.

        Returns
        -------
        dataframe
            One column, named ``effect_curve:<estimand>``.
        """
        return frame_from_dict({self.prediction_column: self.in_sample}, backend=self.backend)

    def summary(self, *, detail: Literal["short", "full"] = "short") -> Any:
        """A printable report of the curve.

        Parameters
        ----------
        detail : {"short", "full"}
            ``"full"`` adds the band over the declared grid, when there is one.

        Returns
        -------
        Summary
            The report.
        """
        from ._summary._effect_curve import effect_curve_summary

        return effect_curve_summary(self, detail=detail)

    def _repr_html_(self) -> str:
        return str(self.summary().as_html())

    def __repr__(self) -> str:
        band = "" if self.band is None else f", band={int(self.band.estimate.size)} points"
        method = "" if self.fitted_method == "tmle" else f", method={self.fitted_method}"
        return (
            f"EffectCurveResult({self.estimand_name}, terms={len(self.model.terms)}, "
            f"rows={int(self.in_sample.size)}{band}{method})"
        )

    def save(self, path: Any) -> Any:
        """Write this result to a trusted joblib artifact; see :func:`cleverly.load`.

        Parameters
        ----------
        path : path-like
            Destination file.

        Returns
        -------
        Path
            The destination path, as given.
        """
        from .estimators.serialize import save as _save

        return _save(self, path)


@dataclass(frozen=True)
class ScoreAssessment:
    """The risk difference and calibration of a fixed external score, with intervals.

    Parameters
    ----------
    column : str
        The score column.
    scale : {"difference", "log_ratio"}
        The scale the score predicts.
    estimates : dict of str to ParameterEstimate
        ``risk_difference``, ``calibration[(intercept)]`` and ``calibration[<column>]``.
    backend : str
        The dataframe backend of the study's data.
    """

    column: str
    scale: str
    estimates: dict[str, ParameterEstimate]
    backend: str = "pandas"

    def __getitem__(self, name: str) -> ParameterEstimate:
        return self.estimates[name]

    def summary(self, *, detail: Literal["short", "full"] = "short") -> Any:
        """A printable report of the check.

        Parameters
        ----------
        detail : {"short", "full"}
            ``"full"`` adds the p-value of each estimate.

        Returns
        -------
        Summary
            The report.
        """
        from ._summary._effect_curve import score_assessment_summary

        return score_assessment_summary(self, detail=detail)

    def _repr_html_(self) -> str:
        return str(self.summary().as_html())

    def __repr__(self) -> str:
        risk = self.estimates["risk_difference"].psi
        return f"ScoreAssessment({self.column!r}, scale={self.scale!r}, risk_difference={risk:.4g})"

    def to_frame(self) -> Any:
        """One row per parameter, in the study's backend.

        Returns
        -------
        dataframe
            The estimates with their intervals.
        """
        rows = [estimate.to_dict() for estimate in self.estimates.values()]
        return frame_from_dict(
            {key: [row[key] for row in rows] for key in rows[0]}, backend=self.backend
        )


def _pseudo_outcomes(
    nuisance: Any,
    data: Any,
    pair: tuple[float, float],
    *,
    g_bounds: tuple[float, float] | None = None,
    missingness_bound: float | None = None,
) -> tuple[FloatArray, ...]:
    """The arm regressions and their AIPW pseudo-outcomes, on the outcome's own scale.

    ``D_a = Delta 1{A = a} / (g(a | W) Pi(a, W)) (Y - Q_a) + Q_a`` from the fit's
    out-of-fold nuisances.  ``g`` is truncated into ``g_bounds`` and ``Pi`` away from
    ``missingness_bound``, the fit's own bounds, as the projection fit reads them; ``None``
    reads the untruncated mechanism (an exact law in a test).
    """
    t, r = pair
    scaler = nuisance.scaler
    fit = nuisance.outcome
    q = {a: scaler.unscale_levels(np.asarray(fit.arms[a], dtype=float)) for a in (t, r)}
    treatment = np.asarray(data.treatment, dtype=float)
    observed = np.asarray(data.observed, dtype=bool)
    y = np.where(observed, np.asarray(data.outcome, dtype=float), 0.0)
    propensity = (
        nuisance.propensity.values
        if g_bounds is None
        else nuisance.propensity.bounded(tuple(g_bounds))
    )
    missingness: Any = None
    if nuisance.missingness is not None:
        missingness = (
            np.asarray(nuisance.missingness, dtype=float)
            if missingness_bound is None
            else np.asarray(nuisance.bounded_missingness(missingness_bound), dtype=float)
        )
    out = []
    for a in (t, r):
        column = nuisance.propensity.column_for(a)
        g = np.asarray(propensity, dtype=float)[:, column]
        pi: Any = 1.0
        if missingness is not None:
            pi = missingness[:, column] if missingness.ndim == 2 else missingness
        weight = observed * (treatment == a) / (g * pi)
        out.append(weight * (y - q[a]) + q[a])
    return q[t], q[r], out[0], out[1]


def score_assessment(
    nuisance: Any,
    data: Any,
    score: FloatArray,
    column: str,
    *,
    scale: str = "difference",
    pair: tuple[float, float] = (1.0, 0.0),
    alpha: float = 0.05,
    g_bounds: tuple[float, float] | None = None,
    missingness_bound: float | None = None,
) -> ScoreAssessment:
    r"""One-step estimates of a fixed score's risk difference and calibration.

    On the difference scale, with :math:`\tilde D = D_t - D_r`, the risk of a score is
    :math:`R(s) = E[s^2 - 2 s \tau]` and the constant comparison is the ATE, the
    minimiser over constants, so the difference is :math:`E[s^2] - 2E[s\tau] + ATE^2`
    with curve :math:`(s^2 - 2 s\tilde D - R(s)) + 2\,ATE\,(\tilde D - ATE)`
    (``docs/technical-reference/conditional-effects.md``; the estimate of the constant adds
    nothing, by the envelope argument).  On the
    log-ratio scale :math:`L(t) = (\bar Q_t + \bar Q_r)\log(1 + e^t) - \bar Q_t t` and
    the constant is :math:`c = \log(E D_t / E D_r)`.  The calibration is the projection
    onto :math:`(1, s)`, by the one-step estimating equation with the pseudo-outcomes in
    place of the regressions.

    Parameters
    ----------
    nuisance : NuisanceEstimates
        The out-of-fold nuisances of an ATE fit on the same rows.
    data : CausalData
        The prepared data.
    score : ndarray
        The score at each row.
    column : str
        The score's column name.
    scale : {"difference", "log_ratio"}
        The scale the score predicts.
    pair : tuple of float
        The arm codes ``(t, r)``.
    alpha : float
        The intervals hold at ``1 - alpha``.
    g_bounds : tuple of float or None
        The fit's truncation of the treatment mechanism; ``None`` reads it untruncated.
    missingness_bound : float or None
        The fit's floor on the missingness mechanism; ``None`` reads it unbounded.

    Returns
    -------
    ScoreAssessment
        The risk difference and the two calibration coefficients.
    """
    s = np.asarray(score, dtype=float).reshape(-1)
    n = s.size
    w = np.asarray(data.weights, dtype=float)
    w = w / w.mean()
    cluster = data.cluster

    def mean(values: FloatArray) -> float:
        return float(np.mean(w * values))

    q_t, q_r, d_t, d_r = _pseudo_outcomes(
        nuisance, data, pair, g_bounds=g_bounds, missingness_bound=missingness_bound
    )
    phi = np.column_stack([np.ones(n), s])
    estimates: dict[str, ParameterEstimate] = {}
    if scale == "difference":
        d = d_t - d_r
        ate = mean(d)
        risk = mean(s**2 - 2.0 * s * d)
        psi = risk + ate**2
        curve = w * ((s**2 - 2.0 * s * d - risk) + 2.0 * ate * (d - ate))
        gram = phi.T @ (w[:, None] * phi) / n
        beta = np.linalg.solve(gram, phi.T @ (w * d) / n)
        calibration = np.linalg.solve(gram, (w[:, None] * phi * (d - phi @ beta)[:, None]).T).T
    else:
        mu_t, mu_r = mean(d_t), mean(d_r)
        c = float(np.log(mu_t / mu_r))
        total = q_t + q_r

        def loss(theta: FloatArray) -> FloatArray:
            softplus = np.logaddexp(0.0, theta)
            correction = (d_t - q_t) * (softplus - theta) + (d_r - q_r) * softplus
            return total * softplus - q_t * theta + correction

        risk_s = loss(s)
        risk_c = loss(np.full(n, c))
        psi = mean(risk_s) - mean(risk_c)
        curve = w * ((risk_s - mean(risk_s)) - (risk_c - mean(risk_c)))
        beta = np.zeros(2)
        for _ in range(100):
            e = expit(phi @ beta)
            equation = phi.T @ (w * ((d_t + d_r) * e - d_t)) / n
            jacobian = phi.T @ ((w * total * e * (1.0 - e))[:, None] * phi) / n
            step = np.linalg.solve(jacobian, equation)
            beta = beta - step
            if np.max(np.abs(step)) < 1e-12:
                break
        e = expit(phi @ beta)
        jacobian = phi.T @ ((w * total * e * (1.0 - e))[:, None] * phi) / n
        contribution = w[:, None] * phi * ((d_t + d_r) * e - d_t)[:, None]
        calibration = -np.linalg.solve(jacobian, contribution.T).T
    estimates["risk_difference"] = make_estimate(
        "risk_difference", float(psi), curve, n=n, cluster=cluster, alpha=alpha
    )
    for j, term in enumerate(("(intercept)", column)):
        name = f"calibration[{term}]"
        estimates[name] = make_estimate(
            name, float(beta[j]), calibration[:, j], n=n, cluster=cluster, alpha=alpha
        )
    return ScoreAssessment(column, scale, estimates, backend=data.backend)
