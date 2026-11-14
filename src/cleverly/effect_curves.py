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
calibration projection onto :math:`(1, s)` are fixed-dimension parameters with intervals, and
so is the paired risk difference of two such scores.  All are TMLE substitution estimates from
one pooled fluctuation of the cross-fitted outcome regression of an ATE fit.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, ClassVar, Final, Literal

import numpy as np
from scipy.special import expit
from scipy.stats import chi2
from scipy.stats import f as f_distribution

from ._pooled_targeting import PooledFit, arm_span_basis, solve_pooled, span_basis
from ._support import Support
from ._typing import FloatArray, IntArray
from .bases import Modifiers
from .estimators.targeting import _STALL_FACTOR, _UNSOLVED, DEFAULT_MAX_OUTER
from .exceptions import CapabilityError, DataError, RefusalReason, refuse_inference
from .fluctuation._score import StopContext, rule_ratio_at
from .inference.cluster import cluster_inference_status, cluster_reference_df
from .inference.delta import reference_quantile
from .inference.influence import ParameterEstimate, family_of, make_estimate
from .inference.multiplier import MultiplierKind, multiplier_critical_value
from .inference.results import estimate_covariance
from .utils.frames import as_frame, backend_of, frame_from_dict, is_dataframe

if TYPE_CHECKING:
    from ._curve_assessment import CurveDiagnostics

__all__ = [
    "SCORE_ASSESSMENT_MEASURED",
    "SCORE_PAIRED_MEASURED",
    "CurveBand",
    "EffectCurveResult",
    "HeterogeneityTest",
    "ProjectionCurve",
    "ProjectionDraw",
    "ScoreAssessment",
    "heterogeneity_test",
    "score_assessment",
    "score_assessment_codes",
    "score_assessment_support",
]

#: The prefix of the prediction column :meth:`EffectCurveResult.to_frame` writes.  The
#: assessment of an external score reads it as the mark of a score fitted on these rows.
PREDICTION_PREFIX = "effect_curve:"


@dataclass(frozen=True)
class HeterogeneityTest:
    r"""The joint Wald test that every non-constant coefficient of the projection is zero.

    It tests "no heterogeneity that the declared basis can express", not "no
    heterogeneity": a contrast that varies orthogonally to the basis leaves every
    coefficient but the intercept at zero.

    On a clustered fit with fewer than
    :data:`~cleverly._inference_status.FEW_CLUSTER_THRESHOLD` clusters, where each coefficient's
    interval reads a Student t reference on :math:`\nu = J - 2` degrees of freedom, the
    statistic :math:`T` is read by the Hotelling law: :math:`T(\nu - q + 1)/(\nu q)` against
    :math:`F(q, \nu - q + 1)`, with :math:`q` the number of tested terms.  At one term this
    is the square of that t reference.  A basis with more tested terms than :math:`\nu` has no
    reference and is refused.

    Parameters
    ----------
    statistic : float
        The Wald chi-square statistic.
    df : int
        Its degrees of freedom, the number of non-constant terms.
    p_value : float
        The upper-tail probability of the reference.
    terms : tuple of str
        The tested terms.
    scale : {"difference", "log_ratio"}
        The scale of the projected contrast.
    reference_df : int or None
        The degrees of freedom :math:`\nu = J - 2` of the covariance, which the Hotelling
        reference reads, or ``None`` for the chi-square reference.
    residual_margin : float
        Under a declared stopping rule, the remaining score of the tested coefficients in the
        statistic's units, :math:`m = (s^\top \Sigma^{-1} s)^{1/2}` for their scores
        :math:`s`; ``0.0`` otherwise.  The p-value then reads
        :math:`\max(0, \sqrt{T} - m)^2` in place of :math:`T`.
    """

    statistic: float
    df: int
    p_value: float
    terms: tuple[str, ...]
    scale: str
    reference_df: int | None = None
    residual_margin: float = 0.0

    def __str__(self) -> str:
        reference = (
            ""
            if self.reference_df is None
            else (f", Hotelling F reference on {self.df} and {self.reference_df - self.df + 1} df")
        )
        margin = (
            ""
            if self.residual_margin == 0.0
            else f", read at the stopping rule's residual {self.residual_margin:.3g}"
        )
        return (
            f"Wald chi-square = {self.statistic:.3f} on {self.df} df, p = {self.p_value:.4g} "
            f"(terms {', '.join(self.terms)}{reference}{margin})"
        )


def heterogeneity_test(
    estimates: dict[str, ParameterEstimate],
    names: tuple[str, ...],
    terms: tuple[str, ...],
    intercepts: tuple[int, ...],
    *,
    scale: str,
    cluster: IntArray | None = None,
    covariance: FloatArray | None = None,
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
    covariance : ndarray or None
        The joint covariance of the tested (non-constant) coefficients, in basis order.
        ``None`` reads :func:`estimate_covariance` of their influence curves; a repeated fit
        passes the matrix rule over its draws.

    Returns
    -------
    HeterogeneityTest
        The statistic, its degrees of freedom and its p-value.

    Raises
    ------
    CapabilityError
        If the basis has no intercept column, or no non-constant term, or a clustered fit has
        more tested terms than the J - 2 degrees of freedom of its Hotelling reference.
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
    df = len(keep)
    # The coefficients' own t reference, J - 2 below the few-cluster threshold, which the fit
    # stamped on each estimate; None reads the asymptotic reference.
    stamped = [estimates[name].reference_df for name in selected]
    reference_df = min((d for d in stamped if d is not None), default=None)
    if cluster is not None and reference_df is not None and df > reference_df:
        raise CapabilityError(
            f"the test reads the joint covariance of {df} coefficients from "
            f"{reference_df + 2} clusters; the Hotelling reference on {reference_df} degrees "
            f"of freedom needs at most {reference_df} tested terms, and with more there is no "
            "reference law",
            reason=RefusalReason.INFERENCE_GAP,
            alternative=(
                "Declare a basis with at most J - 2 non-constant terms for J clusters, or read "
                "each coefficient's own interval."
            ),
        )
    if covariance is None:
        covariance = estimate_covariance(estimates, list(selected), cluster=cluster)
    beta = np.array([estimates[name].psi for name in selected])
    statistic = float(beta @ np.linalg.solve(covariance, beta))
    # Under a declared stopping rule the coefficients' error is (P_n - P_0) D - s, with s the
    # scores the walk left.  In the norm the statistic reads, the triangle inequality gives
    # sqrt(T at the truth) <= sqrt(T of the noise) + m, so the test reads sqrt(T) - m.
    margin = 0.0
    if any(estimates[name].rule_widened for name in selected):
        scores = np.array([estimates[name].score for name in selected])
        margin = float(np.sqrt(max(0.0, float(scores @ np.linalg.solve(covariance, scores)))))
    # ``np.clip`` keeps a ``nan`` statistic ``nan``, where the builtin ``max`` would not.
    read = (
        statistic if margin == 0.0 else float(np.clip(np.sqrt(statistic) - margin, 0.0, None)) ** 2
    )
    if reference_df is None:
        p_value = float(chi2.sf(read, df))
    else:
        # Hotelling: T (nu - q + 1) / (nu q) ~ F(q, nu - q + 1), with nu = J - 2.
        denominator = reference_df - df + 1
        p_value = float(
            f_distribution.sf(read * denominator / (reference_df * df), df, denominator)
        )
    return HeterogeneityTest(
        statistic=statistic,
        df=df,
        p_value=p_value,
        terms=tuple(terms[j] for j in keep),
        scale=scale,
        reference_df=reference_df,
        residual_margin=margin,
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
class ProjectionDraw:
    """One cross-fitting draw's projection coefficients, their covariance and their curves.

    Parameters
    ----------
    beta : ndarray
        ``(p,)`` the draw's coefficients.
    covariance : ndarray
        ``(p, p)`` the draw's joint covariance of the coefficients.
    influence : ndarray
        ``(n, p)`` the draw's influence curves of the coefficients.
    """

    beta: FloatArray
    covariance: FloatArray
    influence: FloatArray


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
    rule_widened : bool
        Whether the coefficients come from a fit under a declared stopping rule, so each
        interval and band adds :meth:`residual_margin` to its half-width.
    draws : tuple of ProjectionDraw
        Each cross-fitting draw of a fit with ``repeats`` above 1, or ``()``.  With two or
        more draws the curve at a point is the median rule over the draws' values
        :math:`\varphi(v)^\top\beta_r` with variances
        :math:`\varphi(v)^\top\Sigma_r\varphi(v)`, and the band takes the median of the
        draws' critical values (P1 plan 2.3).  ``beta``, ``covariance`` and ``influence``
        then hold the reported median coefficients, their matrix-rule covariance and the
        central draw's curves.
    inference : str
        The inference status of the coefficients: a status that withholds inference in the
        reported coefficients or in any draw.  :meth:`std_error` and :meth:`band` refuse
        under a status that supplies no inference, and the curve's points stay available.
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
    rule_widened: bool = False
    draws: tuple[ProjectionDraw, ...] = ()
    inference: str = "influence_curve"

    @property
    def repeated(self) -> bool:
        """Whether the curve combines two or more cross-fitting draws."""
        return len(self.draws) > 1

    def _median_rule(self, phi: FloatArray) -> tuple[FloatArray, FloatArray]:
        """The median point and the median-rule variance at each row of ``phi``."""
        values = np.column_stack([phi @ draw.beta for draw in self.draws])
        variances = np.column_stack(
            [np.einsum("ip,pq,iq->i", phi, draw.covariance, phi) for draw in self.draws]
        )
        point = np.median(values, axis=1)
        variance = np.median(variances + (values - point[:, None]) ** 2, axis=1)
        return point, variance

    def at_design(self, phi: FloatArray) -> FloatArray:
        """The curve at the rows of an evaluated basis design.

        Parameters
        ----------
        phi : ndarray
            ``(m, p)`` basis design.

        Returns
        -------
        ndarray
            One value per row.
        """
        design = np.asarray(phi, dtype=float)
        if self.repeated:
            return self._median_rule(design)[0]
        return np.asarray(design @ self.beta, dtype=float)

    def residual_margin(self, frame: Any) -> FloatArray:
        r"""What a declared stopping rule adds to the half-width at each row of ``frame``.

        The curve at :math:`v` is the linear functional :math:`\varphi(v)^\top\hat\beta`,
        so the score it leaves is :math:`\varphi(v)^\top s`, with :math:`s` the
        coefficients' scores (the means of their influence curves), and the margin is its
        absolute value.  Zero when :attr:`rule_widened` is false.

        Parameters
        ----------
        frame : dataframe or mapping
            Rows holding the modifier columns.

        Returns
        -------
        ndarray
            ``(m,)`` margins.
        """
        phi = self.design(frame)
        if not self.rule_widened:
            return np.zeros(phi.shape[0])
        return np.asarray(np.abs(phi @ np.mean(self.influence, axis=0)), dtype=float)

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
        return self.at_design(self.design(frame))

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

        Raises
        ------
        CapabilityError
            If :attr:`inference` supplies no inference.
        """
        refuse_inference(self.inference, operation="the curve's standard error")
        phi = self.design(frame)
        if self.repeated:
            return np.sqrt(self._median_rule(phi)[1])
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

        Raises
        ------
        CapabilityError
            If :attr:`inference` supplies no inference.
        """
        refuse_inference(self.inference, operation="the curve's band")
        phi = self.design(grid)
        estimate = self.at_design(phi)
        se = self.std_error(grid)
        if np.any(se <= 0):
            raise DataError("a grid point has a zero standard error; drop it from the grid")

        def critical_value(influence: FloatArray, se_at: FloatArray) -> float:
            curves = influence @ phi.T
            return float(
                multiplier_critical_value(
                    curves,
                    se_at,
                    n=int(curves.shape[0]),
                    cluster=self.cluster,
                    alpha=alpha,
                    n_replicates=n_replicates,
                    kind=kind,
                    random_state=random_state,
                )
            )

        if self.repeated:
            # Each draw's critical value from its own curves and standard errors, with the
            # one seed, and their median around the median-rule curve (``median_bands``).
            critical = float(
                np.median(
                    [
                        critical_value(
                            draw.influence,
                            np.sqrt(np.einsum("ip,pq,iq->i", phi, draw.covariance, phi)),
                        )
                        for draw in self.draws
                    ]
                )
            )
        else:
            critical = critical_value(self.influence, se)
        values = self.modifiers.values(grid, self.basis.columns)
        half = critical * se + self.residual_margin(grid)
        return CurveBand(
            grid={name: np.asarray(value) for name, value in values.items()},
            estimate=estimate,
            lower=estimate - half,
            upper=estimate + half,
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
        # Under a declared stopping rule each interval adds the score the curve leaves there.
        half = z * se + self.model.residual_margin(data)
        backend = backend_of(as_frame(data)) if is_dataframe(data) else self.backend
        return frame_from_dict(
            {
                "estimate": estimate,
                "std_err": se,
                "ci_lower": estimate - half,
                "ci_upper": estimate + half,
            },
            backend=backend,
        )

    def plot(self, **kwargs: Any) -> Any:
        """Plot the conditional effect over one modifier.

        The curve is drawn over the modifier with its pointwise interval (a projection fit
        only; an EP curve carries none), a rug of the modifier's values, and the band at the
        grid points the fit declared. A difference-scale projection whose model includes a
        constant term also draws its average effect as a reference line: the mean of the
        curve over the fitted rows, which equals the fit's plug-in average effect only then.
        A curve that reads other modifiers is drawn with each held at a value you give in
        ``at=``.

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


#: The configurations whose single-score parameters (the risk difference against the best
#: constant and the calibration) a validation study measures, keyed by
#: ``(scale, outcome, mechanism)``.  Each value names the studies and their groups,
#: ``"<study>: <groups>"``, joined by ``"; "``.  Every group fits cross-fitted nuisances on iid,
#: unweighted rows with complete outcomes and a binary treatment.
SCORE_ASSESSMENT_MEASURED: Final[Mapping[tuple[str, str, str], str]] = {
    ("difference", "binary", "estimated"): (
        "conditional-projection: assess_score_2k; score-assessment: paired_2k"
    ),
    ("difference", "binary", "known"): "score-assessment: paired_trial_2k",
    ("difference", "continuous", "estimated"): "score-assessment: paired_cont_2k",
    ("log_ratio", "binary", "estimated"): "score-assessment: single_crr_4k, paired_crr_4k",
}

#: The configurations whose paired risk difference of two scores a validation study measures,
#: in the form of :data:`SCORE_ASSESSMENT_MEASURED`.  A paired row of any other configuration
#: reads experimental, whatever its single-score rows read.
SCORE_PAIRED_MEASURED: Final[Mapping[tuple[str, str, str], str]] = {
    ("difference", "binary", "estimated"): "score-assessment: paired_2k",
    ("difference", "binary", "known"): "score-assessment: paired_trial_2k",
    ("difference", "continuous", "estimated"): (
        "score-assessment: paired_cont_2k, equal_cont_2k, power_cont_2k"
    ),
    ("log_ratio", "binary", "estimated"): "score-assessment: paired_crr_4k",
}

#: The reasons an estimate of a score assessment reads experimental, in the order they are
#: listed.
SCORE_REASONS: Final[Mapping[str, str]] = {
    "unmeasured": "no validation study of the score assessment measures this configuration",
    "unmeasured_pair": (
        "no validation study measures the paired risk difference of two scores in this "
        "configuration"
    ),
    "weights": "observation weights, which no study of the score assessment measures",
    "clusters": "clustered rows, which no study of the score assessment measures",
    "missing": "missing outcomes: the fluctuation divides by the response mechanism, with no study",
    "multi_arm": "a pair of arms of a multi-arm treatment, with no study",
    "not_cross_fitted": "the ATE fit is not cross-fitted, which no study measures",
    "reduced_span": (
        "the log-ratio fluctuation settled only after dropping a collinear direction; no study "
        "measures coverage there"
    ),
}

#: The limits every supported score assessment states.
SCORE_LIMITS: Final[tuple[str, ...]] = (
    "The studies fit declared parametric or spline nuisances; the default learner library is "
    "not measured.",
    "Sample size 2,000 for the intervals on the difference scale and 4,000 on the log-ratio scale.",
    "The scores are fixed functions fitted on other rows; the interval covers one pair named "
    "before the fit, not the best of several.",
)


def _sentence(fragment: str) -> str:
    return fragment[:1].upper() + fragment[1:] + "."


def _measured(entry: str) -> list[tuple[str, str]]:
    """The ``(study, groups)`` pairs of a table entry."""
    pairs = []
    for item in entry.split(";"):
        study, groups = item.split(":", 1)
        pairs.append((study.strip(), groups.strip()))
    return pairs


def _score_support(codes: Sequence[str], entry: str | None) -> Support:
    if codes or entry is None:
        return Support(
            "experimental",
            " ".join(_sentence(SCORE_REASONS[code]) for code in codes),
            limits=SCORE_LIMITS,
        )
    pairs = _measured(entry)
    named = "; ".join(f"{study} ({groups})" for study, groups in pairs)
    return Support(
        "supported",
        f"Validation studies measure this configuration: {named}.",
        tuple(dict.fromkeys(study for study, _ in pairs)),
        limits=SCORE_LIMITS,
    )


def score_assessment_codes(
    *,
    scale: str,
    outcome: str,
    known_mechanism: bool,
    weighted: bool,
    clustered: bool,
    missing: bool,
    arms: int,
    cross_fitted: bool,
    reduced_span: bool = False,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The reason codes of a score assessment's single-score rows and of its paired row.

    Parameters
    ----------
    scale : {"difference", "log_ratio"}
        The scale the scores predict.
    outcome : {"binary", "continuous"}
        The outcome type.
    known_mechanism : bool
        Whether the treatment mechanism was declared known.
    weighted : bool
        Whether the rows carry observation weights.
    clustered : bool
        Whether the rows are clustered.
    missing : bool
        Whether some outcomes are missing.
    arms : int
        The number of treatment arms.
    cross_fitted : bool
        Whether the ATE fit cross-fitted its nuisances.
    reduced_span : bool, default False
        Whether the log-ratio fluctuation settled only on a reduced span (the span record's
        ``reduced`` is positive).

    Returns
    -------
    tuple
        The codes (keys of :data:`SCORE_REASONS`) of the single-score rows, then of the paired
        row.  Empty codes mean the row reads ``supported``.
    """
    key = (scale, outcome, "known" if known_mechanism else "estimated")
    single = [] if key in SCORE_ASSESSMENT_MEASURED else ["unmeasured"]
    for flag, code in (
        (weighted, "weights"),
        (clustered, "clusters"),
        (missing, "missing"),
        (arms > 2, "multi_arm"),
        (not cross_fitted, "not_cross_fitted"),
        (reduced_span, "reduced_span"),
    ):
        if flag:
            single.append(code)
    paired = list(single)
    if key in SCORE_ASSESSMENT_MEASURED and key not in SCORE_PAIRED_MEASURED:
        paired.insert(0, "unmeasured_pair")
    return tuple(single), tuple(paired)


def score_assessment_support(
    *,
    scale: str,
    outcome: str,
    known_mechanism: bool,
    weighted: bool,
    clustered: bool,
    missing: bool,
    arms: int,
    cross_fitted: bool,
    reduced_span: bool = False,
) -> tuple[Support, Support]:
    """The support of a score assessment's single-score rows and of its paired row.

    A single-score row reads ``supported`` when its configuration is one of
    :data:`SCORE_ASSESSMENT_MEASURED` and the fit has a binary treatment, iid unweighted rows,
    complete outcomes and cross-fitted nuisances, and its fluctuation settled on the full span.
    The paired row also needs its configuration in :data:`SCORE_PAIRED_MEASURED`.  Otherwise a
    row reads ``experimental``, with every reason of :data:`SCORE_REASONS` that applies.
    Whether ``compare=`` was passed does not enter: the single-score rows are the same TMLE
    along a span that holds their features.

    Parameters
    ----------
    scale : {"difference", "log_ratio"}
        The scale the scores predict.
    outcome : {"binary", "continuous"}
        The outcome type.
    known_mechanism : bool
        Whether the treatment mechanism was declared known.
    weighted : bool
        Whether the rows carry observation weights.
    clustered : bool
        Whether the rows are clustered.
    missing : bool
        Whether some outcomes are missing.
    arms : int
        The number of treatment arms.
    cross_fitted : bool
        Whether the ATE fit cross-fitted its nuisances.
    reduced_span : bool, default False
        Whether the log-ratio fluctuation settled only on a reduced span, which no study
        measures.

    Returns
    -------
    tuple of Support
        The support of each single-score row, then that of the paired row.

    Examples
    --------
    >>> from cleverly.effect_curves import score_assessment_support
    >>> single, paired = score_assessment_support(
    ...     scale="difference", outcome="continuous", known_mechanism=False,
    ...     weighted=False, clustered=False, missing=False, arms=2, cross_fitted=True,
    ... )
    >>> single.label, paired.label
    ('supported', 'supported')
    >>> single, _ = score_assessment_support(
    ...     scale="log_ratio", outcome="continuous", known_mechanism=False,
    ...     weighted=False, clustered=False, missing=False, arms=2, cross_fitted=True,
    ... )
    >>> single.label
    'experimental'
    """
    single, paired = score_assessment_codes(
        scale=scale,
        outcome=outcome,
        known_mechanism=known_mechanism,
        weighted=weighted,
        clustered=clustered,
        missing=missing,
        arms=arms,
        cross_fitted=cross_fitted,
        reduced_span=reduced_span,
    )
    key = (scale, outcome, "known" if known_mechanism else "estimated")
    return (
        _score_support(single, SCORE_ASSESSMENT_MEASURED.get(key)),
        _score_support(paired, SCORE_PAIRED_MEASURED.get(key)),
    )


@dataclass(frozen=True)
class ScoreAssessment:
    """The risk difference and calibration of a fixed external score, with intervals.

    With ``compare=``, it also holds the second score's parameters and the paired risk
    difference of the two, all from one targeted fit.

    Parameters
    ----------
    column : str
        The score column.
    scale : {"difference", "log_ratio"}
        The scale the score predicts.
    estimates : dict of str to ParameterEstimate
        ``risk_difference``, ``calibration[(intercept)]`` and ``calibration[<column>]``, and
        with ``compare=`` the paired row ``risk_difference[<column> - <compare>]``, the risk of
        ``column`` minus that of ``compare``: negative when ``column`` predicts better.
    status : {"supported", "experimental"}
        The worst support label of the estimates, those of ``compared`` included.
    status_reason : str
        The reasons for an ``experimental`` status, joined by ``"; "``; empty otherwise.
    span : dict of str to float
        The fluctuation's span record: ``columns``, ``rank``, ``out_of_span``,
        ``score_norm`` and ``reduced``, the number of thinnest directions a log-ratio fit
        dropped to settle (0 on the full span).
    compare : str or None
        The second score column, or ``None``.
    compared : ScoreAssessment or None
        The three parameters of ``compare``, from the same fluctuation, or ``None``.
    backend : str
        The dataframe backend of the study's data.
    """

    column: str
    scale: str
    estimates: dict[str, ParameterEstimate]
    status: str
    status_reason: str
    span: dict[str, float]
    compare: str | None = None
    compared: ScoreAssessment | None = None
    backend: str = "pandas"

    def __getitem__(self, name: str) -> ParameterEstimate:
        return self.estimates[name]

    @property
    def paired_name(self) -> str | None:
        """The key of the paired row in :attr:`estimates`, or ``None`` without ``compare=``."""
        return None if self.compare is None else f"risk_difference[{self.column} - {self.compare}]"

    def rows(self) -> list[tuple[str, str, ParameterEstimate]]:
        """Every estimate with the score it describes, in report order.

        Returns
        -------
        list of tuple
            ``(score, name, estimate)``: the column's three parameters, then those of
            ``compare``, then the paired row, whose score is ``"<column> - <compare>"``.
        """
        paired = self.paired_name
        out = [(self.column, name, e) for name, e in self.estimates.items() if name != paired]
        if self.compared is not None and paired is not None:
            out += [(self.compared.column, name, e) for name, e in self.compared.estimates.items()]
            out.append((f"{self.column} - {self.compare}", paired, self.estimates[paired]))
        return out

    def summary(self, *, detail: Literal["short", "full"] = "short") -> Any:
        """A printable report of the check.

        Parameters
        ----------
        detail : {"short", "full"}
            ``"full"`` adds the p-value of each estimate and the fluctuation's span record.

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
        paired = self.paired_name
        if paired is None:
            reading = f"{self.column!r}, risk_difference={self['risk_difference'].psi:.4g}"
        else:
            reading = f"{self.column!r} vs {self.compare!r}, paired={self[paired].psi:.4g}"
        return f"ScoreAssessment({reading}, scale={self.scale!r}, status={self.status!r})"

    def to_frame(self) -> Any:
        """One row per parameter, in the study's backend.

        Returns
        -------
        dataframe
            A ``score`` column (the column, ``compare``, or ``"<column> - <compare>"``) and
            each estimate's fields (:meth:`ParameterEstimate.to_dict`), in the order of
            :meth:`rows`.  ``support`` and ``support_reason`` carry each row's status.
        """
        records = []
        for score, _, estimate in self.rows():
            record = {"score": score, **estimate.to_dict()}
            record.setdefault("support_reason", "")
            records.append(record)
        keys = list(dict.fromkeys(key for record in records for key in record))
        return frame_from_dict(
            {key: [record.get(key) for record in records] for key in keys}, backend=self.backend
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


#: The result the pooled fluctuation's error messages name.
_WHAT: Final = "the score assessment"


def _difference_columns(scores: Sequence[FloatArray]) -> FloatArray:
    """The numerator columns on the difference scale: the constant and each score."""
    return np.column_stack([np.ones(scores[0].size), *scores])


def _log_ratio_numerators(
    scores: Sequence[FloatArray], betas: Sequence[FloatArray]
) -> tuple[FloatArray, FloatArray]:
    r"""The arm-specific numerator pairs on the log-ratio scale.

    :math:`(1, 0)` and :math:`(0, 1)` for the two arm means, which fix the constant;
    :math:`(\ell_t(s), \ell_r(s))` with :math:`\ell_t(\theta) = \log(1 + e^{-\theta})` and
    :math:`\ell_r(\theta) = \log(1 + e^\theta)` for each score's risk; and
    :math:`(\varphi\{e - 1\}, \varphi e)` with :math:`\varphi = (1, s)` and
    :math:`e = \operatorname{expit}(\varphi^\top\beta)` for each score's calibration.
    """
    n = scores[0].size
    ones, zeros = np.ones(n), np.zeros(n)
    at_t, at_r = [ones, zeros], [zeros, ones]
    for s, beta in zip(scores, betas, strict=True):
        at_t.append(np.logaddexp(0.0, -s))
        at_r.append(np.logaddexp(0.0, s))
        e = expit(beta[0] + beta[1] * s)
        for column in (ones, s):
            at_t.append(column * (e - 1.0))
            at_r.append(column * e)
    return np.column_stack(at_t), np.column_stack(at_r)


def _log_ratio_projection(
    q_t: FloatArray, q_r: FloatArray, score: FloatArray, weights: FloatArray
) -> FloatArray:
    r"""The log-CRR calibration of ``score`` under ``(q_t, q_r)``, by Newton's method.

    It minimises :math:`P_n w\{(q_t + q_r)\log(1 + e^{\varphi^\top\beta}) - q_t\varphi^\top\beta\}`
    over :math:`\beta`, with :math:`\varphi = (1, s)`.  With positive arm regressions the loss
    is strictly convex, so the root is unique.
    """
    n = score.size
    phi = np.column_stack([np.ones(n), score])
    total = q_t + q_r
    beta = np.zeros(2)
    for _ in range(100):
        e = expit(phi @ beta)
        gradient = phi.T @ (weights * (total * e - q_t)) / n
        hessian = phi.T @ ((weights * total * e * (1.0 - e))[:, None] * phi) / n
        step = np.linalg.solve(hessian, gradient)
        beta = beta - step
        if np.max(np.abs(step)) < 1e-12:
            break
    return beta


@dataclass(frozen=True)
class _TargetedScores:
    """The pooled fluctuation of a score assessment.

    ``betas`` holds the log-ratio calibration coefficients the last clever covariate used,
    one pair per score, and is empty on the difference scale.
    """

    pooled: PooledFit
    betas: tuple[FloatArray, ...]
    span: dict[str, float]


def _span_record(
    columns: int, rank: int, out_of_span: float, pooled: PooledFit, reduced: int = 0
) -> dict[str, float]:
    return {
        "columns": float(columns),
        "rank": float(rank),
        "out_of_span": out_of_span,
        "score_norm": pooled.score_norm,
        "reduced": float(reduced),
    }


def _target_scores(
    fit: Any,
    data: Any,
    scores: Sequence[FloatArray],
    *,
    scale: str,
    pair: tuple[float, float],
) -> _TargetedScores:
    """One pooled fluctuation of the fit's outcome regression along every score's columns."""
    if scale == "difference":
        columns = _difference_columns(scores)
        basis, out_of_span = span_basis(columns)
        pooled = solve_pooled(fit, data, basis, -basis, pair, what=_WHAT)
        span = _span_record(columns.shape[1], basis.shape[1], out_of_span, pooled)
        return _TargetedScores(pooled, (), span)
    return _alternate_log_ratio(fit, data, scores, pair)


#: The largest relative distance of a numerator from the span, after the thinnest direction is
#: dropped, at which a log-ratio fit still reports.  Each numerator's score equation then holds
#: up to that fraction of the column's norm times the residual's; :func:`_numerator_score_ratio`
#: judges the equations themselves.
_REDUCED_SPAN_BOUND: Final = 1e-3


def _numerator_score_ratio(
    pooled: PooledFit, numerators_t: FloatArray, numerators_r: FloatArray, data: Any
) -> float:
    r"""The largest score equation of the numerators over its ``scaled_var`` threshold.

    Numerator :math:`j`'s score at the targeted fit is
    :math:`P_n w\{w_t(Y - \bar Q^*_t)\,a_{tj} + w_r(Y - \bar Q^*_r)\,a_{rj}\}`, and its threshold
    is :math:`\widehat{sd}_j / (\sqrt{n_u}\min(\log n_u, 10))` with :math:`\widehat{sd}_j` the
    SD of the per-unit contributions (:func:`cleverly.fluctuation._score.rule_ratio_at`).  At
    most 1 means every equation holds to :math:`o_P(n^{-1/2})` in SE units.
    """
    w = np.asarray(data.weights, dtype=float)
    w = w / w.mean()
    residual_t = w * pooled.weight_t * (pooled.y - pooled.q_t)
    residual_r = w * pooled.weight_r * (pooled.y - pooled.q_r)
    contributions = residual_t[:, None] * numerators_t + residual_r[:, None] * numerators_r
    units = None if data.cluster is None else np.asarray(data.cluster)
    _, ratio = rule_ratio_at(
        contributions.mean(axis=0), contributions, StopContext(units, 1.0), "scaled_var"
    )
    return ratio


def _alternate_log_ratio(
    fit: Any, data: Any, scores: Sequence[FloatArray], pair: tuple[float, float]
) -> _TargetedScores:
    r"""Alternate the log-ratio fluctuation with the calibration projections until both settle.

    The calibration columns read :math:`\beta`, which is the projection of the targeted fit,
    so this repeats {numerators at :math:`\beta`, fluctuate from :math:`\bar Q^0`,
    :math:`\beta \leftarrow` projection of :math:`\bar Q^*`} as
    :func:`cleverly.estimators.targeting.solve_with_projection` does: each round restarts from
    :math:`\bar Q^0`, it stops when the relative shift in :math:`\beta` is at most the fit's
    ``tol`` or a round no longer improves it materially, and at most
    :data:`~cleverly.estimators.targeting.DEFAULT_MAX_OUTER` rounds run.

    Expanded to second order in :math:`s`, a score's numerators (the arm constants, the risk
    pair and the two calibration pairs) are linearly dependent at two calibration slopes:
    where :math:`e' = 0` and where :math:`e'` is half of :math:`\operatorname{expit}'` at the
    score's centre (about :math:`\beta_1 = 0.5` when :math:`e(s)` sits near
    :math:`\operatorname{expit}(s)`).  A fit whose slope lands there has a direction of the
    span only third-order thin, and the
    span jumps as :math:`\beta` crosses the slope, so the rounds cycle without settling (12 of
    the ``score-assessment`` pilot's 1,600 log-ratio fits).  Such a fit runs the rounds again
    from its best one along the span without its thinnest direction, at that fixed rank, then
    without its two thinnest, up to one per score.  It reports the first that settles with every
    numerator within :data:`_REDUCED_SPAN_BOUND` of the span (the span record's
    ``out_of_span``) **and** every numerator's score equation, the dropped directions' included,
    inside its ``scaled_var`` threshold (:func:`_numerator_score_ratio`), so the remainder stays
    :math:`o_P(n^{-1/2})` at every n; otherwise it refuses.  The span record's ``reduced`` counts
    the dropped directions, and such a fit reads ``experimental``.  Every fit the first rounds
    settle is unchanged.
    """
    w = np.asarray(data.weights, dtype=float)
    w = w / w.mean()
    tol = float(fit.config.targeting_spec.tol)
    initial_t, initial_r, _, _ = _pseudo_outcomes(fit.nuisance, data, pair)

    def solve(betas: Sequence[FloatArray], rank: int | None) -> tuple[PooledFit, dict[str, float]]:
        at_t, at_r = _log_ratio_numerators(scores, betas)
        basis_t, basis_r, out_of_span = arm_span_basis(at_t, at_r, rank)
        pooled = solve_pooled(fit, data, basis_t, basis_r, pair, what=_WHAT)
        return pooled, _span_record(
            at_t.shape[1], basis_t.shape[1], out_of_span, pooled, 0 if rank is None else dropped
        )

    def rounds_from(
        betas: tuple[FloatArray, ...], rank: int | None
    ) -> tuple[
        float, int, tuple[FloatArray, ...], PooledFit, dict[str, float], tuple[FloatArray, ...]
    ]:
        pooled, span = solve(betas, rank)
        shift, previous, rounds = 0.0, None, 0
        best: tuple[float, tuple[FloatArray, ...]] = (math.inf, betas)
        while rounds < DEFAULT_MAX_OUTER:
            rounds += 1
            solved = tuple(_log_ratio_projection(pooled.q_t, pooled.q_r, s, w) for s in scores)
            old, new = np.concatenate(betas), np.concatenate(solved)
            shift = float(np.max(np.abs(new - old))) / (1.0 + float(np.max(np.abs(old))))
            best = min(best, (shift, betas), key=lambda item: item[0])
            if shift <= tol:
                break
            if previous is not None and shift > _STALL_FACTOR * previous:
                break
            previous = shift
            betas = solved
            pooled, span = solve(betas, rank)
        return shift, rounds, betas, pooled, span, best[1]

    def scores_hold(betas: tuple[FloatArray, ...], pooled: PooledFit) -> bool:
        at_t, at_r = _log_ratio_numerators(scores, betas)
        return _numerator_score_ratio(pooled, at_t, at_r, data) <= 1.0

    dropped = 0
    start = tuple(_log_ratio_projection(initial_t, initial_r, s, w) for s in scores)
    shift, rounds, betas, pooled, span, best = rounds_from(start, None)
    full = int(span["rank"])
    # Each score has one risk pair, so at most one thin direction each.
    for dropped in range(1, len(scores) + 1):
        if shift <= _UNSOLVED or full - dropped < 1:
            break
        reduced = rounds_from(best, full - dropped)
        rounds += reduced[1]
        if (
            reduced[0] <= _UNSOLVED
            and reduced[4]["out_of_span"] <= _REDUCED_SPAN_BOUND
            and scores_hold(reduced[2], reduced[3])
        ):
            shift, _, betas, pooled, span, _ = reduced
    # As in solve_with_projection: a shift above _UNSOLVED fails even under a declared tol above
    # it, and after a cap exit the last round's projection is not compared with its beta again.
    if shift > _UNSOLVED:
        raise DataError(
            "the score assessment's log-ratio fluctuation and its calibration projection did "
            f"not settle: the calibration coefficients still moved by {shift:.1e} (relative) "
            f"after {rounds} rounds"
        )
    return _TargetedScores(pooled, betas, span)


def _one_step_estimates(
    q_t: FloatArray,
    q_r: FloatArray,
    d_t: FloatArray,
    d_r: FloatArray,
    scores: Mapping[str, FloatArray],
    *,
    scale: str,
    weights: FloatArray,
) -> dict[str, dict[str, float]]:
    r"""The one-step estimates of each score's parameters from arm regressions and AIPW scores.

    The auxiliary of the TMLE: at the targeted fit and its AIPW scores it equals the
    substitution estimates to the solver's tolerance, and at the initial fit it is the
    one-step estimator.  On the difference scale, with :math:`\tilde D = D_t - D_r`, the risk
    difference is :math:`P_n[s^2 - 2s\tilde D] + (P_n\tilde D)^2` and the calibration solves
    :math:`P_n\varphi(\tilde D - \varphi^\top\beta) = 0`.  On the log-ratio scale the constant
    is :math:`\log(P_n D_t / P_n D_r)`, the risk reads the loss with its AIPW correction, and the
    calibration solves :math:`P_n\varphi\{(D_t + D_r)e - D_t\} = 0` by Newton steps whose
    Jacobian reads :math:`q_t + q_r`.

    Parameters
    ----------
    q_t : ndarray
        The regression of arm ``t``.
    q_r : ndarray
        The regression of arm ``r``.
    d_t : ndarray
        The AIPW score of arm ``t``.
    d_r : ndarray
        The AIPW score of arm ``r``.
    scores : mapping of str to ndarray
        The scores by name, the column first.
    scale : {"difference", "log_ratio"}
        The scale the scores predict.
    weights : ndarray
        The observation weights.

    Returns
    -------
    dict
        By score name, its ``risk_difference``, ``calibration[(intercept)]`` and
        ``calibration[<name>]``.  With two scores the first also holds
        ``risk_difference[<first> - <second>]``.
    """
    w = np.asarray(weights, dtype=float)
    w = w / w.mean()
    names = list(scores)
    n = w.size

    def mean(values: FloatArray) -> float:
        return float(np.mean(w * values))

    d = d_t - d_r
    total = q_t + q_r

    def loss(theta: FloatArray) -> FloatArray:
        softplus = np.logaddexp(0.0, theta)
        correction = (d_t - q_t) * (softplus - theta) + (d_r - q_r) * softplus
        return total * softplus - q_t * theta + correction

    out: dict[str, dict[str, float]] = {}
    risks: dict[str, float] = {}
    for name in names:
        s = np.asarray(scores[name], dtype=float)
        phi = np.column_stack([np.ones(n), s])
        if scale == "difference":
            risk = mean(s**2 - 2.0 * s * d)
            psi = risk + mean(d) ** 2
            gram = phi.T @ (w[:, None] * phi) / n
            beta = np.linalg.solve(gram, phi.T @ (w * d) / n)
        else:
            c = float(np.log(mean(d_t) / mean(d_r)))
            risk = mean(loss(s))
            psi = risk - mean(loss(np.full(n, c)))
            beta = np.zeros(2)
            for _ in range(100):
                e = expit(phi @ beta)
                equation = phi.T @ (w * ((d_t + d_r) * e - d_t)) / n
                jacobian = phi.T @ ((w * total * e * (1.0 - e))[:, None] * phi) / n
                step = np.linalg.solve(jacobian, equation)
                beta = beta - step
                if np.max(np.abs(step)) < 1e-12:
                    break
        risks[name] = risk
        out[name] = {
            "risk_difference": float(psi),
            "calibration[(intercept)]": float(beta[0]),
            f"calibration[{name}]": float(beta[1]),
        }
    if len(names) == 2:
        first, second = names
        out[first][f"risk_difference[{first} - {second}]"] = risks[first] - risks[second]
    return out


def _cluster_inference(cluster: Any, weights: FloatArray) -> tuple[str, int | None]:
    """The inference status and the t reference's degrees of freedom of a clustered fit.

    Below three clusters with positive weight mass there is no interval, and no reference.
    """
    if cluster is None:
        return "influence_curve", None
    status = cluster_inference_status(cluster, weights=weights)
    if status == "no_cluster_variance_plugin":
        return status, None
    return status, cluster_reference_df(cluster, weights)


def score_assessment(
    fit: Any,
    data: Any,
    scores: Mapping[str, FloatArray],
    *,
    scale: str = "difference",
    pair: tuple[float, float] = (1.0, 0.0),
    alpha: float = 0.05,
) -> ScoreAssessment:
    r"""TMLE estimates of a fixed score's risk difference and calibration, or of two scores'.

    The risks are linear in the arm regressions with known weight functions,
    :math:`R = E[k(W) + c_t(W)\bar Q_t + c_r(W)\bar Q_r]`.  On the difference scale
    :math:`R(s) = E[s^2 - 2s\tau]`; on the log-ratio scale
    :math:`R_L(s) = E[\bar Q_t\ell_t(s) + \bar Q_r\ell_r(s)]` with
    :math:`\ell_t(\theta) = \log(1 + e^{-\theta})` and :math:`\ell_r(\theta) = \log(1 + e^\theta)`.
    One logistic fluctuation of the ATE fit's cross-fitted :math:`\bar Q^0`, along the span of
    every score's numerator columns (:mod:`cleverly._pooled_targeting`), solves the score
    equation of every reported functional, and each estimate is the substitution from the
    targeted :math:`\bar Q^*`: the risk difference against the best constant (the ATE, or
    :math:`c = \log(E\bar Q_t / E\bar Q_r)`), the calibration projection onto :math:`(1, s)`,
    and with two scores the paired risk difference :math:`R(s_1) - R(s_2)`.  On the log-ratio
    scale the calibration columns read :math:`\beta`, so the fluctuation alternates with the
    projections.  The influence curves are those of
    ``docs/technical-reference/conditional-effects.md`` at :math:`\bar Q^*` and the fit's
    truncated mechanism; the paired curve is the difference of the two single-score curves, in
    which the constant's terms cancel.

    The column's own rows depend on whether a second score is passed: the span grows, so the
    targeted fit differs in finite samples.  Both are TMLEs with the same influence curve.

    Parameters
    ----------
    fit : TMLEResult
        The single ATE fit on these rows, with ``repeats=1``.
    data : CausalData
        The rows, as the fit prepared them.
    scores : mapping of str to ndarray
        The score, and optionally a second one to compare, by column name.
    scale : {"difference", "log_ratio"}
        The scale the scores predict.
    pair : tuple of float
        The arm codes ``(t, r)``.
    alpha : float
        The intervals hold at ``1 - alpha``.

    Returns
    -------
    ScoreAssessment
        The first score's risk difference and two calibration coefficients, and with two
        scores the second's in ``compared`` and the paired row.

    Raises
    ------
    DataError
        For a score without one finite value per row, two scores that are equal on every row,
        or a fluctuation that does not converge.
    """
    names = list(scores)
    if not 1 <= len(names) <= 2:
        raise DataError(f"a score assessment takes one score or two; got {len(names)}")
    arrays = [np.asarray(scores[name], dtype=float).reshape(-1) for name in names]
    for name, values in zip(names, arrays, strict=True):
        if values.shape[0] != data.n or not np.all(np.isfinite(values)):
            raise DataError(f"the score {name!r} must hold one finite value per row")
    if len(arrays) == 2 and np.array_equal(arrays[0], arrays[1]):
        raise DataError(
            f"the score column {names[0]!r} and the compare= column {names[1]!r} hold the same "
            "value on every row, so their risk difference is 0 with no spread"
        )
    targeted = _target_scores(fit, data, arrays, scale=scale, pair=pair)
    pooled = targeted.pooled
    raw = np.asarray(data.weights, dtype=float)
    w = raw / raw.mean()
    n = w.size
    q_t, q_r = pooled.q_t, pooled.q_r
    d_t, d_r = pooled.targeted_t, pooled.targeted_r

    def mean(values: FloatArray) -> float:
        return float(np.mean(w * values))

    # Each score's (name, point, curve) triples; then the paired row's.
    blocks: list[list[tuple[str, float, FloatArray]]] = []
    risks: list[tuple[float, FloatArray]] = []
    if scale == "difference":
        tau, dtilde = q_t - q_r, d_t - d_r
        ate = mean(tau)
        for name, s in zip(names, arrays, strict=True):
            risk = mean(s**2 - 2.0 * s * tau)
            augmented = s**2 - 2.0 * s * dtilde
            curve = w * ((augmented - risk) + 2.0 * ate * (dtilde - ate))
            phi = np.column_stack([np.ones(n), s])
            gram = phi.T @ (w[:, None] * phi) / n
            beta = np.linalg.solve(gram, phi.T @ (w * tau) / n)
            calibration = np.linalg.solve(
                gram, (w[:, None] * phi * (dtilde - phi @ beta)[:, None]).T
            ).T
            risks.append((risk, augmented))
            blocks.append(_score_block(name, risk + ate**2, curve, beta, calibration))
    else:
        c = float(np.log(mean(q_t) / mean(q_r)))

        def plug_in(theta: FloatArray) -> FloatArray:
            return q_t * np.logaddexp(0.0, -theta) + q_r * np.logaddexp(0.0, theta)

        def augmented_loss(theta: FloatArray) -> FloatArray:
            return d_t * np.logaddexp(0.0, -theta) + d_r * np.logaddexp(0.0, theta)

        constant = np.full(n, c)
        risk_c = mean(plug_in(constant))
        centred_c = augmented_loss(constant) - risk_c
        total = q_t + q_r
        for name, s, beta in zip(names, arrays, targeted.betas, strict=True):
            risk = mean(plug_in(s))
            augmented = augmented_loss(s)
            curve = w * ((augmented - risk) - centred_c)
            phi = np.column_stack([np.ones(n), s])
            e = expit(phi @ beta)
            jacobian = phi.T @ ((w * total * e * (1.0 - e))[:, None] * phi) / n
            contribution = w[:, None] * phi * ((d_t + d_r) * e - d_t)[:, None]
            calibration = -np.linalg.solve(jacobian, contribution.T).T
            risks.append((risk, augmented))
            blocks.append(_score_block(name, risk - risk_c, curve, beta, calibration))
    paired: tuple[str, float, FloatArray] | None = None
    if len(names) == 2:
        (risk_1, augmented_1), (risk_2, augmented_2) = risks
        psi = risk_1 - risk_2
        paired = (
            f"risk_difference[{names[0]} - {names[1]}]",
            psi,
            w * ((augmented_1 - augmented_2) - psi),
        )
    configuration: dict[str, Any] = {
        "scale": scale,
        "outcome": "binary" if getattr(data, "family", "") == "binomial" else "continuous",
        "known_mechanism": getattr(data, "known_treatment", None) is not None,
        "weighted": bool(np.any(np.abs(raw - 1.0) > 1e-12)),
        "clustered": data.cluster is not None,
        "missing": bool(not np.all(np.asarray(data.observed, dtype=bool))),
        "arms": len(tuple(fit.nuisance.arms)),
        "cross_fitted": bool(getattr(fit.config, "cross_fit", True)),
        "reduced_span": targeted.span["reduced"] > 0,
    }
    single_codes, paired_codes = score_assessment_codes(**configuration)
    single_support, paired_support = score_assessment_support(**configuration)
    inference, reference_df = _cluster_inference(data.cluster, raw)

    def estimate(name: str, psi: float, curve: FloatArray, support: Support) -> ParameterEstimate:
        return make_estimate(
            name,
            float(psi),
            curve,
            n=n,
            cluster=data.cluster,
            alpha=alpha,
            inference=inference,  # type: ignore[arg-type]
            reference_df=reference_df,
            support=support,
            outcome_family=family_of(data),
            # The score assessment solves its own projection; no stopping rule applies.
            rule_widened=False,
        )

    def built(
        name: str,
        block: list[tuple[str, float, FloatArray]],
        extra: tuple[str, float, FloatArray] | None,
        codes: tuple[str, ...],
    ) -> ScoreAssessment:
        estimates = {
            label: estimate(label, psi, curve, single_support) for label, psi, curve in block
        }
        if extra is not None:
            estimates[extra[0]] = estimate(*extra, paired_support)
        return ScoreAssessment(
            column=name,
            scale=scale,
            estimates=estimates,
            status="experimental" if codes else "supported",
            status_reason="; ".join(SCORE_REASONS[code] for code in codes),
            span=targeted.span,
            backend=data.backend,
        )

    compared = None if len(names) == 1 else built(names[1], blocks[1], None, single_codes)
    primary = built(
        names[0], blocks[0], paired, paired_codes if paired is not None else single_codes
    )
    return replace(primary, compare=None if compared is None else names[1], compared=compared)


def _score_block(
    name: str, risk_difference: float, curve: FloatArray, beta: FloatArray, calibration: FloatArray
) -> list[tuple[str, float, FloatArray]]:
    return [
        ("risk_difference", risk_difference, curve),
        ("calibration[(intercept)]", float(beta[0]), calibration[:, 0]),
        (f"calibration[{name}]", float(beta[1]), calibration[:, 1]),
    ]
