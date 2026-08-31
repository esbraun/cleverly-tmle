r"""Solving the fluctuation: the targeting step.

Given an initial outcome regression and a clever covariate, find the
:math:`\epsilon` that maximises the (quasi-)likelihood along the submodel.  For a
logistic fluctuation this is a weighted logistic regression with the initial
prediction as an offset, solved here by Newton--Raphson with an explicit Hessian
and a backtracking line search.

Why not hand this to :mod:`statsmodels` or scikit-learn?  Neither exposes a
regression with an *offset* and no intercept, which is exactly what the submodel
requires; and the problem is one- or two-dimensional with a closed-form Hessian,
so a direct Newton solve is both faster and more accurate than a general-purpose
optimiser.  A brute-force grid search is used as a reference in the tests.

Because the solution is a maximum-likelihood estimate *within* the submodel, its
score is exactly zero -- meaning the resulting estimator solves the efficient
influence-function equation.  When the targeted predictions hit the ``[0, 1]``
boundary the score can fail to vanish; the outer loop then re-fluctuates from the
updated fit, which is what makes this the "iterative" TMLE.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias, cast

import numpy as np

from .._typing import BoolArray, FloatArray, FluctuationKind, IntArray
from ..exceptions import ConvergenceWarning
from ..utils.bounds import expit, logit, shrink_probabilities
from ._score import quasi_loglik, relative_score, score_columns, score_scale
from .submodel import Submodel, check_arms, weighted_form

__all__ = [
    "CarryItem",
    "Fluctuation",
    "FoldFluctuation",
    "InitialFit",
    "apply_logistic",
    "check_matching_arms",
    "dominant_failure",
    "solve_fluctuation",
]

TargetingLabel = Literal["iterative", "one_step", "linear"]

#: One item of ``carry``: a further initial fit moved by the same steps as the fitted one.
#:
#: A bare :class:`InitialFit` travels along the **production** submodel, which is what the
#: nested reduced-regression construction wants -- its fold-free copies describe the same
#: rows, so they take the same clever covariate.  A ``(fit, submodel)`` pair travels along
#: one of its own, which is what an *evaluation companion* needs: its rows are not the
#: fitted rows, so ``1_a / g^*`` at them is a different array even though ``epsilon`` is the
#: same number.  Both are moved by the identical step sequence, which is the whole reason
#: either goes through the solver rather than being reconstructed from ``(initial, epsilon)``
#: afterwards -- see :attr:`Fluctuation.carried`.
CarryItem: TypeAlias = "InitialFit | tuple[InitialFit, Submodel]"

#: How a targeting step failed, when it did.  Naming the mode matters because the
#: fixes differ: ``separation_suspected`` and ``bounds_pinned`` point at positivity,
#: ``singular_hessian`` at a rank-deficient clever covariate (two columns that are
#: multiples of one another), and ``max_iter_reached`` at a tolerance that is simply
#: too tight for the conditioning of the problem.
#:
#: These are *reported*, not raised.  Raising would break the sensitivity sweeps that
#: deliberately push the truncation bound into bad territory to show what happens.
TargetingFailure = Literal[
    "singular_hessian",
    "line_search_exhausted",
    "nonfinite_step",
    "separation_suspected",
    "bounds_pinned",
    "max_iter_reached",
]

#: ``|epsilon|`` past which the fluctuation is treated as running away, which is what
#: separation looks like from here: the logistic MLE is at infinity, so every Newton
#: step moves further out and the targeted predictions pin against ``alpha``.
_SEPARATION_EPSILON = 30.0

#: Reciprocal condition number below which the Hessian is treated as singular.
_ILL_CONDITIONED = 1e-12

_FAILURE_TEXT: dict[str, str] = {
    "singular_hessian": (
        "the Newton Hessian was singular or near-singular, so epsilon is barely "
        "identified; usually two clever-covariate columns that are nearly collinear"
    ),
    "line_search_exhausted": (
        "the line search halved 30 times without improving the quasi-likelihood, which "
        "means the Newton direction was not a descent direction at working precision"
    ),
    "nonfinite_step": (
        "the Newton step was not finite, typically an infinite clever covariate from a "
        "propensity at zero or one"
    ),
    "separation_suspected": (
        "epsilon ran away, the signature of separation: the logistic MLE within the "
        "submodel is at infinity, so no finite fluctuation solves the score equation"
    ),
    "bounds_pinned": (
        "the targeted predictions are pinned against their [1 - alpha, alpha] bounds, so "
        "further fluctuation cannot move the score; check positivity"
    ),
    "max_iter_reached": (
        "the iteration cap was reached with the score still above tolerance; the fit may "
        "simply need more iterations, or the tolerance may be tighter than the "
        "conditioning of the problem supports"
    ),
}


@dataclass
class NewtonDetail:
    """What the inner Newton solve saw, kept rather than discarded.

    The Hessian is formed on every iteration and was previously thrown away, which
    left no way to tell an ill-conditioned clever covariate from a merely slow one.
    """

    failure: TargetingFailure | None = None
    hessian_condition: float = float("nan")
    epsilon_std_error: FloatArray | None = None
    loglik: float = float("nan")

    def observe(self, hessian: FloatArray) -> None:
        """Record the conditioning and the coefficient standard errors."""
        try:
            self.hessian_condition = float(np.linalg.cond(hessian))
        except np.linalg.LinAlgError:  # pragma: no cover - non-finite Hessian
            self.hessian_condition = float("inf")
        if not np.isfinite(self.hessian_condition) or (
            self.hessian_condition > 1.0 / _ILL_CONDITIONED
        ):
            self.failure = "singular_hessian"
            self.epsilon_std_error = None
            return
        try:
            self.epsilon_std_error = np.sqrt(np.abs(np.diag(np.linalg.inv(hessian))))
        except np.linalg.LinAlgError:  # pragma: no cover - guarded by the condition
            self.epsilon_std_error = None


#: Relative slack allowed when checking that a Newton step did not reduce the
#: quasi-log-likelihood.  See :func:`_newton_logistic` for why it must be relative.
_LINE_SEARCH_SLACK = 1e-11


@dataclass(frozen=True)
class InitialFit:
    """The initial outcome regression, at the observed treatment and at each arm.

    ``arms`` maps a treatment level to the predictions obtained by setting the treatment
    to that level for everybody, so a binary treatment carries ``{0.0: ..., 1.0: ...}``.
    Keying them rather than naming two fields is what lets :meth:`shrunk`,
    :func:`apply_logistic`, :func:`~cleverly.fluctuation.submodel.restrict` and the
    row-slicing helpers be written once without counting arms.

    Every array lives on the ``[0, 1]`` scale: for a binary outcome that is the natural
    scale, and for a continuous one it is the scaled outcome (see
    :class:`cleverly.utils.bounds.OutcomeScaler`).
    """

    observed: FloatArray
    arms: dict[float, FloatArray]

    def __post_init__(self) -> None:
        check_arms(self.observed, self.arms, "initial fit")

    def map_arms(self, fn: Callable[[FloatArray], FloatArray]) -> InitialFit:
        """Apply ``fn`` to the observed predictions and to every arm's, keys preserved."""
        return InitialFit(
            fn(self.observed), {level: fn(values) for level, values in self.arms.items()}
        )

    def shrunk(self, alpha: float) -> InitialFit:
        """Pull predictions away from 0 and 1 so ``logit`` stays finite."""
        return self.map_arms(lambda values: shrink_probabilities(values, alpha))

    @property
    def levels(self) -> tuple[float, ...]:
        """The arm levels, ascending."""
        return tuple(sorted(self.arms))

    @property
    def n(self) -> int:
        return int(self.observed.shape[0])


def dominant_failure(reasons: Sequence[str], failed: Sequence[int]) -> TargetingFailure | None:
    """The most common failure mode across folds, for the summary line.

    A single label cannot describe ten folds, so the per-fold detail stays on
    :attr:`Fluctuation.folds`; this is only what to print when there is room for one word.

    Shared by both fold-targeting paths -- :meth:`cleverly.TMLE._solve_by_fold` and the
    longitudinal outer recursion -- because a fit that reports the *first* fold's reason
    rather than the prevailing one is reporting whichever fold the scheduler returned
    first, which is not a property of the fit.
    """
    if not failed:
        return None
    modes = [reasons[index] for index in failed if reasons[index] != "unknown"]
    if not modes:
        return "max_iter_reached"
    return cast("TargetingFailure", max(set(modes), key=modes.count))


@dataclass(frozen=True)
class FoldFluctuation:
    """One validation fold's contribution to a cross-validated targeting step.

    Recorded so a CV-TMLE fit can be inspected fold by fold.  A fluctuation
    coefficient that swings wildly across folds is the signature of an unstable
    clever covariate -- something the pooled ``epsilon`` averages away and hides.

    Parameters
    ----------
    index : ndarray
        Row positions of this validation fold, into the fitted sample.
    epsilon : ndarray
        Fluctuation coefficients solved on this fold alone, one per clever-covariate
        column.
    score : ndarray
        Solved score per column over this fold's rows, the fold's own estimating
        equation.
    converged : bool
        Whether this fold's own solve reached its tolerance.
    n_iter : int
        Solver iterations this fold took.
    trace : tuple of float
        Relative score norm through this fold's solve.
    score_scale : ndarray or None
        Per-column largest score this fold's rows could produce, or ``None`` when the
        solver reported no scale.

    Attributes
    ----------
    n : int
    score_norm : float
    relative_score_norm : float

    See Also
    --------
    cleverly.fluctuation.Fluctuation : The targeting step these folds make up.
    """

    index: IntArray
    epsilon: FloatArray
    score: FloatArray
    converged: bool
    n_iter: int
    #: Complete relative-score trajectory for this fold. ``trace[0]`` is the score before
    #: its first update, under the same contract as :attr:`Fluctuation.trace`.
    trace: tuple[float, ...] = ()
    #: Component-wise scale used to decide whether this fold reached its score root.
    score_scale: FloatArray | None = None

    @property
    def n(self) -> int:
        """How many rows this validation fold holds."""
        return int(self.index.shape[0])

    @property
    def score_norm(self) -> float:
        """Largest absolute score component, in the outcome's own units."""
        return float(np.max(np.abs(self.score))) if self.score.size else 0.0

    @property
    def relative_score_norm(self) -> float:
        """Largest score component relative to its maximum possible magnitude."""
        if self.score.size == 0:
            return 0.0
        if self.score_scale is None:
            return self.score_norm
        return float(np.max(np.abs(self.score) / np.maximum(self.score_scale, 1e-300)))


@dataclass(frozen=True)
class Fluctuation:
    """The result of a targeting step.

    Parameters
    ----------
    epsilon : ndarray
        Fitted fluctuation coefficients, one per clever-covariate column.
    targeted : InitialFit
        Targeted predictions at the observed treatment, and at ``A = 1`` / ``A = 0``.
    score : ndarray
        Mean of ``w * h * (Y - Q*)`` per column -- the estimating equation the
        targeting step is meant to zero out.  Reported rather than asserted so
        :mod:`cleverly.validation.score` can check it against the standard error.
    converged : bool
        Whether the *relative* score norm reached ``tol``.
    n_iter : int
        Outer iterations of the solver: Newton steps for ``"iterative"``, walk steps
        for ``"one_step"``, ``1`` for ``"linear"``.
    trace : tuple of float
        Relative score norm through one solve.  ``trace[0]`` is always the score at
        ``epsilon = 0``, whichever solver ran.  Empty when this object aggregates several
        independent fold solves; their complete trajectories are in ``folds``.
    method : {"iterative", "one_step", "linear"}
        Which solver produced this step.
    names : tuple of str
        Name of each clever-covariate column, in the order ``epsilon`` reports them.
    carried : tuple of InitialFit
        Further initial fits moved along the **same** submodel by the **same** steps, in
        the order they were passed as ``carry``.  Empty on every fit that did not ask for
        one, which is every fit but a
        :class:`~cleverly.DRTMLE` with ``reduced_crossfit="nested"``.

        It exists because a step sequence cannot be reconstructed from its endpoint.  The
        solve applies :func:`apply_logistic` once per Newton step and shrinks after each,
        so a caller holding ``(initial, epsilon)`` recovers ``targeted`` only when no
        intermediate iterate touched the shrinkage bound -- and the fits that reach it are
        exactly the weak-overlap ones a reference construction is compared on.  Carrying the
        arrays *through* the solver is the only way to move them by the transformation that
        was actually applied rather than by one that usually equals it.
    score_scale : ndarray or None
        Per-column ``mean(|w * h|)``, the largest the score could possibly be given
        that the residual is bounded by one on the ``[0, 1]`` outcome scale.  Dividing
        by it turns the score into a dimensionless quantity, which is what makes a
        single default tolerance meaningful across problems whose clever covariates
        differ by orders of magnitude.
    folds : tuple of FoldFluctuation
        Per-fold detail, populated only by the cross-validated (``targeting_scheme=
        "fold"``) targeting step and empty otherwise.
    score_initial : ndarray or None
        The score *before* targeting.  Reported so the reader can see how far
        the step actually moved: a score that started near zero means the initial fit
        already solved the equation and targeting had nothing to do, which is a
        different situation from one that started large and was driven down.
    n_solver_calls : int
        How many times a solver was invoked -- ``1`` for a pooled fit, and one per
        fold for a fold-targeted one, where ``n_iter`` is the sum across folds and on
        its own would be indistinguishable from a single long solve.
    failure : str or None
        Why the step stopped, when it did not converge.  ``None`` when it converged.
    hessian_condition : float
        Condition number of the last Newton Hessian.  Large means the clever
        covariate columns are nearly collinear and ``epsilon`` is barely identified.
    epsilon_std_error : ndarray or None
        Standard errors of the fluctuation coefficients from the inverse Hessian.
        A diagnostic only: the parameter's inference comes from the influence curve,
        not from ``epsilon``.
    loglik : float
        Quasi-log-likelihood at the fitted ``epsilon``.  The submodel is fit by
        maximising it, so it must not decrease along the path.
    mechanism : object or None
        The treatment-mechanism half of the targeting, for a group whose parameter is
        defined through the mechanism.
    projection : object or None
        The working model's coefficients for an ``msm`` group whose link makes the
        clever covariate depend on them.
    reduction : object or None
        Equation (10)'s fluctuation and the reduced regressions it was solved against,
        for a :class:`~cleverly.DRTMLE` fit.

    Attributes
    ----------
    score_norm : float
    relative_score_norm : float
    initial_score_norm : float

    See Also
    --------
    cleverly.fluctuation.FoldFluctuation : One validation fold's own solve.
    cleverly.estimators.TMLEResult : The fit these are read out of.
    """

    epsilon: FloatArray
    targeted: InitialFit
    score: FloatArray
    converged: bool
    n_iter: int
    trace: tuple[float, ...]
    method: TargetingLabel
    names: tuple[str, ...]
    carried: tuple[InitialFit, ...] = ()
    score_scale: FloatArray | None = None
    folds: tuple[FoldFluctuation, ...] = ()
    score_initial: FloatArray | None = None
    n_solver_calls: int = 1
    failure: TargetingFailure | None = None
    hessian_condition: float = float("nan")
    epsilon_std_error: FloatArray | None = None
    loglik: float = float("nan")
    #: The treatment-mechanism half of the targeting, for a group whose parameter is
    #: defined through the mechanism (today only ``ipsi``); ``None`` for every other
    #: group, whose targeting is finished when this fluctuation converges.  Carried
    #: here rather than on the nuisances so that ``result.nuisance.propensity`` stays
    #: the initial cross-fitted mechanism, exactly as ``outcome`` stays the initial
    #: regression -- see :mod:`cleverly.fluctuation.mechanism`.
    mechanism: Any | None = None
    #: The working model's coefficients and how the alternation that found them went, for
    #: an ``msm`` group whose link makes the clever covariate depend on them; ``None`` for
    #: the identity link and for every other group.  A sibling of ``mechanism`` above and
    #: carried for the same reason -- it is the other half of a targeting step that has
    #: two -- but *not* the same kind of object: the mechanism is a nuisance that was
    #: tilted, while this is the reported parameter, solved for rather than fluctuated.
    #: See :func:`~cleverly.estimators.targeting.solve_with_projection`.
    projection: Any | None = None
    #: The extra half of a doubly-robust fit: equation (10)'s fluctuation, the reduced
    #: regressions it was solved against, and the mechanism tilt of equation (9).  ``None``
    #: for every fit that is not a :class:`~cleverly.DRTMLE`.  A third sibling of the two
    #: above, carried here for the same reason and with the same consequence -- neither the
    #: targeted mechanism nor the refitted reductions are written back onto the nuisances,
    #: so ``result.nuisance`` keeps describing the models that were actually fitted.
    #: See :func:`~cleverly.estimators.targeting.solve_with_reduction`.
    reduction: Any | None = None

    @property
    def score_norm(self) -> float:
        """Largest absolute score component, in the outcome's own units."""
        return float(np.max(np.abs(self.score))) if self.score.size else 0.0

    @property
    def relative_score_norm(self) -> float:
        """Largest score component relative to its maximum possible magnitude."""
        if self.score.size == 0:
            return 0.0
        if self.score_scale is None:
            return self.score_norm
        return float(np.max(np.abs(self.score) / np.maximum(self.score_scale, 1e-300)))

    @property
    def initial_score_norm(self) -> float:
        """Largest absolute score component before targeting."""
        if self.score_initial is None or self.score_initial.size == 0:
            return float("nan")
        return float(np.max(np.abs(self.score_initial)))

    def coefficients(self) -> dict[str, float]:
        """Return the fitted ``epsilon``, keyed by clever-covariate column name.

        Returns
        -------
        dict of str to float
            One coefficient for each column named in ``names``, in that order.
        """
        return dict(zip(self.names, self.epsilon.tolist(), strict=True))

    def describe_failure(self) -> str:
        """One sentence naming the failure and what usually causes it.

        Returns
        -------
        str
            ``"converged"`` when the step converged, and the explanation otherwise.
        """
        if self.failure is None:
            return "converged"
        return _FAILURE_TEXT[self.failure]


def solve_fluctuation(
    outcome: FloatArray,
    initial: InitialFit,
    submodel: Submodel,
    weights: FloatArray,
    observed: BoolArray | None = None,
    *,
    kind: FluctuationKind = "logistic",
    target_weights: bool = False,
    alpha: float = 0.9995,
    max_iter: int = 20,
    tol: float = 1e-10,
    warn: bool = True,
    carry: Sequence[CarryItem] = (),
) -> Fluctuation:
    """Run the targeting step.

    Parameters
    ----------
    outcome:
        Outcome on the ``[0, 1]`` scale.  Values at unobserved rows are ignored.
    initial:
        Initial outcome regression predictions.
    submodel:
        Clever covariates from :mod:`cleverly.fluctuation.submodel`.
    weights:
        Observation weights, normalised to mean one.
    observed:
        Mask of rows with an observed outcome; ``None`` means all rows.  Rows with a
        missing outcome contribute nothing to the fluctuation regression -- their
        clever covariate is multiplied by ``Delta = 0`` in the estimating equation.
    kind:
        ``"logistic"`` keeps the targeted predictions inside ``[0, 1]``;
        ``"linear"`` fluctuates on the identity scale, matching R's
        ``fluctuation="linear"``, and can leave the unit interval.
    target_weights:
        Use the weighted form of the fluctuation (R's ``target.gwt``).
    carry:
        Further initial fits to move by the same steps, returned on
        :attr:`Fluctuation.carried`.  Nothing here reads them: they take the step and
        contribute nothing to it, so a fit that passes none is unchanged in every array
        and every diagnostic.  A bare fit travels along *this* submodel and a
        ``(fit, submodel)`` pair along one of its own -- see :data:`CarryItem`.  And see
        :attr:`Fluctuation.carried` for why an endpoint plus an ``epsilon`` is not a
        substitute for either.
    """
    y = np.asarray(outcome, dtype=float).reshape(-1)
    n = y.shape[0]
    mask = np.ones(n, dtype=bool) if observed is None else np.asarray(observed, dtype=bool)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if submodel.n != n or w.shape[0] != n or initial.n != n:
        raise ValueError(
            "outcome, initial fit, submodel and weights must all have the same length; got "
            f"{n}, {initial.n}, {submodel.n}, {w.shape[0]}"
        )
    if not mask.any():
        raise ValueError("no observed outcomes: the fluctuation has nothing to fit")

    scoring_submodel = submodel
    if kind == "linear":
        return _solve_linear(y, initial, submodel, w, mask, carry=carry)

    fit_submodel, fit_weights = weighted_form(submodel, w) if target_weights else (submodel, w)

    current = initial.shrunk(alpha)
    # The submodel a carried item travels along is fixed before the first step, so an
    # evaluation companion cannot pick up a covariate from a later iterate than the one
    # its own state was built at.
    carry_fits, carry_submodels = split_carry(carry, fit_submodel)
    carried = tuple(fit.shrunk(alpha) for fit in carry_fits)
    epsilon = np.zeros(fit_submodel.dim)
    scale = score_scale(scoring_submodel.observed, w, mask)
    # trace[0] is the score at epsilon = 0, so the same entry means the same thing
    # here and in the one-step solver.
    score_before = score_columns(y, current.observed, scoring_submodel.observed, w, mask)
    trace: list[float] = [relative_score(score_before, scale)]
    iterations = 0
    detail = NewtonDetail()

    for iterations in range(1, max_iter + 1):  # noqa: B007 - reported after the loop
        step, step_converged, detail = _newton_logistic(
            fit_submodel.observed[mask],
            y[mask],
            logit(current.observed[mask]),
            fit_weights[mask],
            tol=min(tol, 1e-12),
        )
        epsilon = epsilon + step
        current = apply_logistic(current, fit_submodel, step, alpha)
        carried = tuple(
            apply_logistic(fit, own, step, alpha)
            for fit, own in zip(carried, carry_submodels, strict=True)
        )
        score = score_columns(y, current.observed, scoring_submodel.observed, w, mask)
        trace.append(relative_score(score, scale))
        if trace[-1] <= tol or (step_converged and np.max(np.abs(step)) <= tol):
            break

    score = score_columns(y, current.observed, scoring_submodel.observed, w, mask)
    relative = relative_score(score, scale)
    converged = bool(relative <= tol)
    failure = (
        None
        if converged
        else _classify(epsilon, current, detail, alpha, iterations, max_iter, mask)
    )
    if not converged and warn:
        warnings.warn(
            f"targeting step did not drive the relative score below {tol:g} after "
            f"{iterations} iteration(s) (relative score = {relative:.3g}): "
            f"{_FAILURE_TEXT[failure] if failure else 'cause not identified'}. "
            "Inspect res.fluctuations[group].failure and res.diagnostics.support().",
            ConvergenceWarning,
            stacklevel=2,
        )
    return Fluctuation(
        epsilon=epsilon,
        targeted=current,
        carried=carried,
        score=score,
        converged=converged,
        n_iter=iterations,
        trace=tuple(trace),
        method="iterative",
        names=submodel.names,
        score_scale=scale,
        score_initial=score_before,
        failure=failure,
        hessian_condition=detail.hessian_condition,
        epsilon_std_error=detail.epsilon_std_error,
        loglik=detail.loglik,
    )


def _classify(
    epsilon: FloatArray,
    current: InitialFit,
    detail: NewtonDetail,
    alpha: float,
    iterations: int,
    max_iter: int,
    mask: BoolArray,
) -> TargetingFailure:
    """Name the failure mode of a targeting step that did not converge.

    The inner solver's own verdict wins where it has one -- it saw the Hessian and
    the line search.  The two modes it cannot see are diagnosed from the endpoint:
    an ``epsilon`` that has run away is separation, and predictions sitting on the
    shrinkage bounds mean no further fluctuation can move the score.
    """
    if detail.failure in ("singular_hessian", "nonfinite_step", "line_search_exhausted"):
        return detail.failure
    if epsilon.size and np.max(np.abs(epsilon)) >= _SEPARATION_EPSILON:
        return "separation_suspected"
    edge = 1.0 - alpha
    scored = current.observed[mask]
    pinned = np.mean((scored <= edge * 1.000001) | (scored >= alpha * 0.999999))
    if pinned > 0.01:
        return "bounds_pinned"
    if iterations >= max_iter:
        return "max_iter_reached"
    return detail.failure or "max_iter_reached"


def split_carry(
    carry: Sequence[CarryItem], default: Submodel
) -> tuple[tuple[InitialFit, ...], tuple[Submodel, ...]]:
    """The carried fits and the submodel each of them travels along.

    A bare :class:`InitialFit` takes ``default`` -- the submodel the fitted arrays are
    moving along, which is right whenever the carried arrays describe the same rows.  A
    ``(fit, submodel)`` pair takes its own, which is what rows the fit never saw need.  See
    :data:`CarryItem`.

    The two are separated here rather than at each call site so that both solvers and every
    future one read one definition of what a carried item is.
    """
    fits: list[InitialFit] = []
    submodels: list[Submodel] = []
    for item in carry:
        if isinstance(item, tuple):
            fit, own = item
            fits.append(fit)
            submodels.append(own)
        else:
            fits.append(item)
            submodels.append(default)
    return tuple(fits), tuple(submodels)


def apply_logistic(
    fit: InitialFit, submodel: Submodel, epsilon: FloatArray, alpha: float
) -> InitialFit:
    """Move the predictions along the logistic submodel by ``epsilon``."""
    check_matching_arms(fit, submodel)
    return InitialFit(
        expit(logit(fit.observed) + submodel.observed @ epsilon),
        {
            level: expit(logit(values) + submodel.arms[level] @ epsilon)
            for level, values in fit.arms.items()
        },
    ).shrunk(alpha)


def check_matching_arms(fit: InitialFit, submodel: Submodel) -> None:
    """Refuse to fluctuate a fit along a submodel that describes different arms.

    Without the check, a mismatch is not an error but a wrong answer: the arm present in
    only one of the two is dropped from the comprehension, and the resulting fit has
    fewer counterfactual predictions than the estimand needs -- discovered later, as a
    ``KeyError`` far from the cause, or not at all.
    """
    if set(fit.arms) != set(submodel.arms):
        raise ValueError(
            f"the fit carries arms {sorted(fit.arms)} but the submodel carries "
            f"{sorted(submodel.arms)}; a fluctuation moves each arm along its own "
            "covariate, so the two must describe the same ones"
        )


def _newton_logistic(
    x: FloatArray,
    y: FloatArray,
    offset: FloatArray,
    weights: FloatArray,
    *,
    max_iter: int = 50,
    tol: float = 1e-12,
) -> tuple[FloatArray, bool, NewtonDetail]:
    """Weighted logistic MLE with an offset and no intercept.

    Returns the coefficient vector, whether the gradient reached ``tol``, and a
    :class:`NewtonDetail` recording *how* it stopped.  The quasi-binomial
    log-likelihood is valid for any ``y`` in ``[0, 1]``, which is what lets the same
    routine target a scaled continuous outcome (Gruber & van der Laan, 2010).
    """
    k = x.shape[1]
    epsilon = np.zeros(k)
    detail = NewtonDetail()
    if x.size == 0 or np.allclose(x, 0.0):
        return epsilon, True, detail

    total_weight = weights.sum()
    if total_weight <= 0:
        return epsilon, True, detail

    loglik = quasi_loglik(y, expit(offset), weights)
    detail.loglik = loglik
    for _ in range(max_iter):
        eta = offset + x @ epsilon
        p = expit(eta)
        gradient = x.T @ (weights * (y - p))
        if np.max(np.abs(gradient)) / total_weight <= tol:
            return epsilon, True, detail

        variance = weights * p * (1.0 - p)
        hessian = x.T @ (x * variance[:, None])
        detail.observe(hessian)
        step = _solve_step(hessian, gradient)
        if step is None:
            detail.failure = "nonfinite_step"
            return epsilon, False, detail

        # Backtracking: a Newton step can overshoot when the clever covariate has
        # extreme values, and the quasi-likelihood must never decrease.  The slack is
        # *relative* to the magnitude of the log-likelihood: near the optimum the
        # improvement per step falls below the floating-point granularity of a number
        # of size |loglik|, and an absolute slack would reject every remaining step
        # and stall the solver short of the root.
        slack = _LINE_SEARCH_SLACK * max(1.0, abs(loglik))
        scale = 1.0
        for _ in range(30):
            candidate = epsilon + scale * step
            candidate_loglik = quasi_loglik(y, expit(offset + x @ candidate), weights)
            if candidate_loglik >= loglik - slack:
                epsilon, loglik = candidate, candidate_loglik
                detail.loglik = loglik
                break
            scale *= 0.5
        else:
            detail.failure = "line_search_exhausted"
            return epsilon, False, detail

        if np.max(np.abs(scale * step)) <= tol:
            return epsilon, True, detail
    detail.failure = "max_iter_reached"
    return epsilon, False, detail


def _solve_step(hessian: FloatArray, gradient: FloatArray) -> FloatArray | None:
    """Newton step, falling back to a pseudo-inverse for a singular Hessian."""
    try:
        step = np.linalg.solve(hessian, gradient)
    except np.linalg.LinAlgError:
        step = np.linalg.pinv(hessian) @ gradient
    if not np.all(np.isfinite(step)):
        return None
    return np.asarray(step, dtype=float)


def _solve_linear(
    y: FloatArray,
    initial: InitialFit,
    submodel: Submodel,
    weights: FloatArray,
    mask: BoolArray,
    *,
    carry: Sequence[CarryItem] = (),
) -> Fluctuation:
    """Fluctuate on the identity scale: a single weighted least-squares solve.

    This is R's ``fluctuation="linear"``.  The normal equations *are* the estimating
    equation, so the score is solved exactly in one step -- at the cost of targeted
    predictions that may fall outside the outcome's range.
    """
    check_matching_arms(initial, submodel)
    x = submodel.observed[mask]
    residual = y[mask] - initial.observed[mask]
    w = weights[mask]
    lhs = x.T @ (x * w[:, None])
    rhs = x.T @ (w * residual)
    step = _solve_step(lhs, rhs)
    epsilon = np.zeros(submodel.dim) if step is None else step

    def moved(fit: InitialFit, along: Submodel) -> InitialFit:
        return InitialFit(
            fit.observed + along.observed @ epsilon,
            {level: values + along.arms[level] @ epsilon for level, values in fit.arms.items()},
        )

    targeted = moved(initial, submodel)
    carry_fits, carry_submodels = split_carry(carry, submodel)
    carried = tuple(moved(fit, own) for fit, own in zip(carry_fits, carry_submodels, strict=True))
    escaped = any(values.min() < 0.0 or values.max() > 1.0 for values in targeted.arms.values())
    if escaped:
        warnings.warn(
            "the linear fluctuation produced targeted predictions outside the outcome's "
            "range. fluctuation='logistic' (the default) is bounded by construction and is "
            "preferred unless you specifically need the linear submodel.",
            UserWarning,
            stacklevel=3,
        )
    score = score_columns(y, targeted.observed, submodel.observed, weights, mask)
    scale = score_scale(submodel.observed, weights, mask)
    score_before = score_columns(y, initial.observed, submodel.observed, weights, mask)
    converged = bool(relative_score(score, scale) <= 1e-8)
    return Fluctuation(
        epsilon=epsilon,
        targeted=targeted,
        carried=carried,
        score=score,
        converged=converged,
        n_iter=1,
        trace=(relative_score(score_before, scale), relative_score(score, scale)),
        method="linear",
        names=submodel.names,
        score_scale=scale,
        score_initial=score_before,
        # One weighted least-squares solve: the only way it fails is a singular
        # normal-equations matrix, which _solve_step already fell back on.
        failure=None if converged else ("singular_hessian" if step is None else "max_iter_reached"),
    )
