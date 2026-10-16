r"""Did the targeting step actually work?

TMLE's guarantees are built on the estimator solving the efficient influence-function
equation,

.. math:: P_n \hat D^*(O) = \frac{1}{n}\sum_i \hat D^*_i = 0.

If that equation is not solved, the estimate is not the TMLE -- it is a plug-in
estimate with an unquantified bias, and its influence-curve standard error does not
describe it.  Unlike the identification assumptions this is a *checkable* condition,
and there is no reason not to check it on every fit.

What a pass does and does not mean
----------------------------------

Read it as necessary, not sufficient.  What is checked is that the fluctuation reached
the root of the equation the library posed -- with :math:`\hat D^*` being whatever
:mod:`cleverly.inference.influence` computes.  A wrong clever covariate used
*consistently*, in both the targeting step and the influence curve, would solve its own
equation to machine precision and pass here.  So this is an implementation invariant of
the fit in front of you, not evidence that the estimating equation behind it is the
right one.

That the equation is the right one is established elsewhere, and separately:

* ``tests/unit/test_influence_gateaux.py`` checks :math:`\hat D^*` against the numerical
  Gateaux derivative of the target parameter on a finite-support law -- the definition of
  the efficient influence function, computed without any of the library's machinery;
* ``tests/unit/test_remainder.py`` checks that the resulting estimating equation has the
  second-order remainder that double robustness consists of;
* ``tests/e2e/test_oracle.py`` cross-checks the point estimate against an independently
  written AIPW estimator, and ``tests/e2e/test_double_robustness.py`` runs the full
  misspecification grid end to end.

A *failure* here, by contrast, is informative on its own and non-circular: it means the
solver did not converge, or the fluctuation could not reach the root.

On a doubly-robust fit it is not the efficient equation
---------------------------------------------------------

Everything above is written for :math:`\hat D^*`, and a :class:`~cleverly.DRTMLE` fit
solves up to three equations rather than one -- three under the default guard, and two
under either guard alone.  The extra ones are not refinements of the efficient score
equation: the curve their solution leaves is
:math:`D = D^* - D^*_Q - D^*_g`, one correction per equation the guard asked for and so
one term shorter under a single guard, the **estimator's** asymptotic influence function at the
nuisance limits, and it is generally not the canonical gradient at :math:`P_0`.  When both
nuisances are consistent, the corrections converge to zero and the curve approaches the
canonical gradient.  At the true nuisance functions, they vanish row by row.  So the verdict
here branches on
:attr:`ScoreCheck.corrected` and does not sign such a fit off as having solved "the
estimated efficient score equation"; see
:func:`~cleverly.inference.influence.reduced_corrections` and
:mod:`cleverly.estimators.drtmle`.

The check compares :math:`|P_n \hat D^*|` against the standard error.  The natural scale is
:math:`\widehat{se} / \sqrt{n}`: the score has to be small relative to the estimate's
own sampling variability, not merely small in absolute terms, since the units of the
outcome are arbitrary.  A converged fit typically lands near machine precision, many
orders of magnitude below the tolerance.

The usual reasons for a failure:

* **positivity.** Targeted predictions pin against ``[0, 1]`` and cannot move further,
  so the fluctuation cannot reach the root.  Check
  :meth:`~cleverly.assessment.DiagnosticsFacade.support`.
* **a fluctuation that has to travel far**, from a poor initial fit -- try
  ``targeting="one_step"``, which rebuilds the direction along the path.
* **too coarse a step** in the one-step walk -- lower ``step_size``.

Under a declared stopping rule
------------------------------

``Targeting(stopping="scaled_var")`` and ``"sample_size"`` replace the bar above.  The
fluctuation rows then read the rule: a row's bar is the threshold the solver stopped at
(:attr:`~cleverly.fluctuation.Fluctuation.thresholds`), and a row passes when its largest score
over its threshold is at most one.  The rule also judges the ``(strata)`` row of a stratified
working model and the ``(reduced)`` row of a doubly-robust fit, whose outcome solves it stopped
too; a ``(mechanism)`` row keeps the default bar.  A fit that met its rule passes although its
score is above the default bar, and a fit that missed its rule fails although its score is
below the default bar.  The influence-curve rows cannot fail under a rule, because the rule
leaves a residual by design.  They become ``"diagnostic"`` rows that report the residual in
standard errors (``residual_se``) and the coverage a Wald interval keeps at that residual
(``coverage_bound``).  Under ``scaled_var`` they also report the residual the rule implies
before the fit is read (``rule_bound_se``).  Only an arm mean and a linear contrast of arm
means have an a priori bound.  A contrast has none that is small in general, because the
threshold of a column uses the full arm curve, and the curves of two arms share terms that the
contrast cancels.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .._inference_status import InferenceStatus, status_record, supplies_inference
from ..fluctuation._score import (
    RESIDUAL_LINE,
    residual_se,
    rule_failure_line,
    wald_coverage_bound,
)
from ..inference.influence import ParameterEstimate, spread_name
from ..utils.frames import emit_frame
from ..utils.records import sentinel_equality
from ..utils.text import format_negligible, format_table
from .drtmle import CorrectionCheck, correction_check

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .._typing import FloatArray
    from ..estimators.base import TMLEResult
    from ..fluctuation.iterative import Fluctuation

__all__ = ["ScoreCheck", "ScoreCheckRow", "score_check", "score_threshold"]

#: The score must be below ``tolerance * se / sqrt(n)`` to pass.  A converged
#: fluctuation is normally near machine precision, so this is a loose gate that only
#: catches genuine failures.
DEFAULT_TOLERANCE = 1e-3


def score_threshold(
    std_errors: Iterable[float], n: int, *, tolerance: float = DEFAULT_TOLERANCE
) -> float:
    r"""The magnitude a score may reach and still pass, at one fit's reported precision.

    :math:`\text{tolerance} \times \widehat{se} / \sqrt n`.  The natural scale is the
    estimate's own sampling variability rather than an absolute number, because the units of
    the outcome are arbitrary; :math:`\widehat{se}` is the **largest** finite standard error
    the fit reports, so one bar covers every estimand a fit publishes rather than a different
    one per row.

    Exported because a study auditing this package against another implementation has to
    apply *this* rule to both sides, and the registered DR-TMLE study did not: it declared a
    constant ``1e-4`` while the R runner declared ``1 / n``, so one published column held two
    quantities and neither was the bar a user's fit answers to.  A copy of the arithmetic in
    the study would have re-created the same drift one release later.

    Parameters
    ----------
    std_errors : Iterable[float]
        The fit's reported standard errors.  Non-finite entries are skipped, and a collection
        with no finite entry falls back to one, which is the unscaled bar.
    n : int
        Rows the fit was computed on.
    tolerance : float
        Score magnitude in standard errors.  :data:`DEFAULT_TOLERANCE` unless a caller has a
        reason.

    Returns
    -------
    float
        The threshold, on the scale the scores are reported in.

    Examples
    --------
    >>> from cleverly.validation.score import score_threshold
    >>> f"{score_threshold([0.02, 0.01], 3000):.3e}"
    '3.651e-07'
    """
    reference = max((value for value in std_errors if np.isfinite(value)), default=1.0)
    return float(tolerance * reference / np.sqrt(float(n)))


@sentinel_equality
@dataclass(frozen=True)
class ScoreCheckRow:
    """The score diagnostic for one estimand or one targeted family.

    ``score_initial`` and ``hessian_condition`` are ``nan`` on the rows that do not carry
    them, which is a sentinel and not an unknown -- see
    :func:`~cleverly.utils.records.sentinel_equality` for why comparing two of these rows
    needs saying so rather than inheriting the generated ``__eq__``.

    Parameters
    ----------
    name : str
        Estimand alias or targeted family.
    kind : str
        Which score equation this row checks.
    score : float
        The solved score, on the scale ``threshold`` is given in.
    threshold : float
        Magnitude the score may reach and still pass.
    std_error : float
        Standard error the threshold is expressed in, read through
        :attr:`~cleverly.ParameterEstimate.plugin_std_error`. It is a scale for the
        tolerance. On a fit that supplies no inference it is the plug-in diagnostic and
        not a standard error, and :meth:`ScoreCheck.to_frame` does not publish it.
    passed : bool
        Whether the score stayed inside it.
    converged : bool
        Whether targeting reached its own tolerance.
    n_iter : int
        Targeting iterations taken.
    method : str
        Targeting algorithm used.
    score_initial : float
        The score before targeting, for comparison.
    failure : str
        What went wrong, when something did.
    hessian_condition : float
        Condition number of the targeting Hessian. A large value says the solve was
        ill-conditioned even where the score came out small.
    folds_converged : tuple of bool
        Whether each fold's own targeting converged, under fold-specific targeting.
    stopping : str
        The stopping rule that judged the row.  ``"relative"`` is the shipped bar, and any
        other value means ``threshold`` is the threshold the solver stopped at.
    residual_se : float
        On a ``"diagnostic"`` row of a fit under a declared rule, ``|P_n IC| / se`` of the
        estimand.  ``nan`` elsewhere.
    coverage_bound : float
        The coverage a Wald interval keeps at that residual (:func:`wald_coverage_bound`).
    rule_bound_se : float
        Under ``scaled_var``, the residual the rule implies before the fit is read:
        ``sum_j |g_j| threshold_j / se``, over the score columns the estimand is a combination
        of.  It equals ``c`` for a unit-weight arm mean, and it has no a priori value for a
        contrast.  On a stratified fit it sums the stratum blocks.  ``nan`` for a
        doubly-robust fit, a fit with several draws, an incremental tilt, a working-model
        coefficient, and a shift, policy, regime or ratio row (``_rule_bound_se`` gives each
        reason).
    """

    name: str
    kind: str
    score: float
    threshold: float
    std_error: float
    passed: bool
    converged: bool
    n_iter: int
    method: str
    #: The same score before the fluctuation moved anything.  A step that starts near
    #: zero had nothing to do, which is a different situation from one that started
    #: large and was driven down -- and only the latter is evidence targeting worked.
    score_initial: float = float("nan")
    #: Why the step stopped, when it did not converge.  Empty when it did.
    failure: str = ""
    #: Conditioning of the last Newton Hessian.  Large means ``epsilon`` is barely
    #: identified even where the score looks solved.
    hessian_condition: float = float("nan")
    #: For a fold-targeted fit, how many of the per-fold solves converged.
    folds_converged: tuple[int, int] | None = None
    #: The rule that judged this row.  Beside ``threshold`` so a reader of a frame can tell a
    #: row held to the default bar from a row held to its fit's declared rule.
    stopping: str = "relative"
    #: The three columns of the estimand rows of a fit under a declared rule.  ``nan`` is a
    #: sentinel for "this row does not carry it".
    residual_se: float = float("nan")
    coverage_bound: float = float("nan")
    rule_bound_se: float = float("nan")

    @property
    def reduction(self) -> float:
        """How far targeting moved the score, as a factor.  Above 1 means it shrank."""
        if not np.isfinite(self.score_initial) or self.score <= 0:
            return float("nan")
        return abs(self.score_initial) / abs(self.score)

    @property
    def ratio(self) -> float:
        """Score relative to its threshold; below 1 passes."""
        if self.threshold <= 0:
            return float("nan")
        return abs(self.score) / self.threshold


@dataclass(frozen=True)
class ScoreCheck:
    """Whether the targeting step solved the score equations it was supposed to.

    "The efficient score equation" for a plain fit, and deliberately not that phrase for a
    doubly-robust one -- see :attr:`corrected`.

    Parameters
    ----------
    rows : tuple of ScoreCheckRow
        One row per estimand or targeted family.
    tolerance : float
        Score magnitude, in standard errors, a row may reach and pass.
    n : int
        Number of observations.
    corrected : bool
        Whether the fit carried DR-TMLE correction equations.
    corrections : CorrectionCheck or None
        Those equations' own check, when it did.
    backend : str or None
        Dataframe backend :meth:`to_frame` returns when ``data`` is omitted.
    inference : str
        Inference status of the checked fit. One of
        :data:`~cleverly.inference.influence.InferenceStatus`, which the
        :doc:`inference reference </technical-reference/inference>` lists. A failing
        verdict names the spread the fit reports by it, so a fit that supplies no
        inference reads "plug-in standard errors".
    stopping : str
        The stopping rule the fit declared, ``"relative"`` for the shipped one.
    alpha_sig : float
        The significance level of the fit's intervals.
    """

    rows: tuple[ScoreCheckRow, ...]
    tolerance: float
    n: int
    #: Whether this fit reports the **corrected** curve
    #: :math:`D = D^* - D^*_Q - D^*_g` -- or the one term of it a single guard solves for,
    #: which :meth:`_curve` derives from the rows -- rather than :math:`D^*`.  True for a
    #: :class:`~cleverly.DRTMLE` fit with a non-empty guard, and what the verdict branches
    #: on: the equations such a fit solves are its own three, and the curve they leave is
    #: the *estimator's* influence function at the nuisance limits, which is generally not
    #: the efficient one.  Signing that off as "the estimated efficient score equation"
    #: was the one place the package asserted the thing
    #: :func:`~cleverly.inference.influence.reduced_corrections` exists to deny.
    corrected: bool = False
    #: The per-arm corrections behind this check's ``identity`` and ``correction`` rows,
    #: for a reader who wants the recomputation itself rather than the verdict on it.
    #: ``None`` on a check built by hand; empty rows on any fit that reports no
    #: corrections, which is every fit but a guarded :class:`~cleverly.DRTMLE` one.
    corrections: CorrectionCheck | None = None
    #: Name of the dataframe backend the fit's data arrived in, so that
    #: :meth:`to_frame` honours "results come back in the backend you passed in"
    #: without a caller having to thread the container back in by hand.
    backend: str | None = None
    #: A default, so a check built by hand reads as the ordinary fit it describes.
    inference: InferenceStatus = "influence_curve"
    #: The stopping rule the fit declared, ``"relative"`` for the shipped one.  Under a
    #: declared rule the fluctuation rows are held to the rule's thresholds and the estimand
    #: rows are diagnostics.
    stopping: str = "relative"
    #: The significance level of the fit's intervals, which the coverage bound reads.
    alpha_sig: float = 0.05

    @property
    def passed(self) -> bool:
        """True when every row is within tolerance."""
        return all(row.passed for row in self.rows)

    @property
    def declared(self) -> bool:
        """Whether the fit declared a stopping rule other than the shipped one."""
        return self.stopping != "relative"

    @property
    def rule_failures(self) -> tuple[ScoreCheckRow, ...]:
        """The failing rows that a declared rule judged, which are the rule not being met."""
        return tuple(
            row for row in self.failures if row.kind == "fluctuation" and row.stopping != "relative"
        )

    def rule_line(self) -> str:
        """The pinned line of a fit under a declared rule, or ``""`` under the shipped one.

        Returns
        -------
        str
            The largest score residual in standard errors, the estimand it is on, and the
            coverage a Wald interval keeps at that residual (:data:`RESIDUAL_LINE`).
        """
        if not self.declared:
            return ""
        rows = [row for row in self.rows if np.isfinite(row.residual_se)]
        if not rows:
            return ""
        worst = max(rows, key=lambda row: row.residual_se)
        return RESIDUAL_LINE.format(
            rule=self.stopping,
            b=worst.residual_se,
            name=worst.name,
            level=1.0 - self.alpha_sig,
            bound=worst.coverage_bound,
        )

    def rule_failure_line(self) -> str:
        """The pinned line when the targeting did not meet the rule, else ``""``.

        Returns
        -------
        str
            The line ``RULE_FAILURE_LINE`` of ``cleverly.fluctuation._score``.  Its detail
            names each equation that missed its rule and how many times its threshold its
            largest score is.  An empty string means the targeting met the rule, or the fit
            declared none.
        """
        failures = self.rule_failures
        if not failures:
            return ""
        detail = "; ".join(f"{row.name} is {row.ratio:.3g} times its threshold" for row in failures)
        return rule_failure_line(self.stopping, detail + ". See res.diagnostics.score_equations()")

    @property
    def identity_failures(self) -> tuple[ScoreCheckRow, ...]:
        """Failures that are software defects rather than unsolved equations.

        A row here says the score the targeting step recorded and the term the reported
        curve carries are not the same functional of the state the fit returned --
        the DR-TMLE correction identity. Iterating longer cannot fix one, which is why it is
        worded apart from every other failure this class reports.
        """
        return tuple(row for row in self.failures if row.kind == "identity")

    def __bool__(self) -> bool:
        return self.passed

    @property
    def failures(self) -> tuple[ScoreCheckRow, ...]:
        """Return rows that failed their check."""
        return tuple(row for row in self.rows if not row.passed)

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
            One row per checked score equation.
        """
        payload = {
            "name": [row.name for row in self.rows],
            "kind": [row.kind for row in self.rows],
            "score": [row.score for row in self.rows],
            "threshold": [row.threshold for row in self.rows],
            "ratio": [row.ratio for row in self.rows],
            "passed": [row.passed for row in self.rows],
            "n_iter": [row.n_iter for row in self.rows],
            "method": [row.method for row in self.rows],
        }
        if self.declared:
            # Only a fit under a declared rule has these columns, so an ordinary fit's frame
            # is the one it always was.
            payload["stopping"] = [row.stopping for row in self.rows]
            payload["residual_se"] = [row.residual_se for row in self.rows]
            payload["coverage_bound"] = [row.coverage_bound for row in self.rows]
            payload["rule_bound_se"] = [row.rule_bound_se for row in self.rows]
        return emit_frame(payload, data, backend=self.backend)

    def one_line(self) -> str:
        """The verdict as a block a report can append, without the table.

        Phrased here rather than at the call site so that every place a verdict is spoken
        -- this module's own report, and :meth:`cleverly.TMLEResult.summary` -- says the
        same thing.  It names the failing rows, because "the check failed" without them
        sends the reader back to the method call this line exists to spare them.

        Returns
        -------
        str
            The verdict alone, for a report to append without the table.
        """
        if self.declared:
            return self._rule_one_line()
        failures = self.failures
        if not failures:
            return (
                f"score check: PASS -- all {len(self.rows)} within tolerance "
                f"(worst |score| {format_negligible(self._worst_ratio(), 1.0, digits=2)} "
                "of its threshold)."
            )
        named = "; ".join(
            f"{row.name} |score| {abs(row.score):.3e} against {row.threshold:.3e}"
            for row in failures
        )
        return "\n".join(
            [
                f"score check: FAIL -- {len(failures)} of {len(self.rows)} not solved.",
                f"  {named}",
                *self._identity_lines(),
                f"  The {spread_name('standard errors', self.inference)} above are read off "
                "an influence curve whose mean is",
                "  not zero, so they do not describe this estimate.  See",
                "  res.diagnostics.score_equations() for the table and cleverly.validation.score",
                "  for the usual causes.",
            ]
        )

    def _rule_one_line(self) -> str:
        """:meth:`one_line` under a declared stopping rule.

        A row the rule judged that missed it is the pinned failure line.  A row held to the
        default bar that failed (a DR-TMLE mechanism or reduction equation, or an identity) is
        named after it, because the rule says nothing about those equations.
        """
        failures = self.failures
        if not failures:
            ratios = [
                row.ratio
                for row in self.rows
                if row.kind == "fluctuation" and np.isfinite(row.ratio)
            ]
            worst = format_negligible(max(ratios), 1.0, digits=2) if ratios else "0"
            return (
                f"score check: PASS -- the {self.stopping} stopping rule was met "
                f"(worst score {worst} of its threshold)."
            )
        lines = []
        if self.rule_failures:
            lines.append(self.rule_failure_line())
        others = [row for row in failures if row not in self.rule_failures]
        if others:
            named = "; ".join(
                f"{row.name} |score| {abs(row.score):.3e} against {row.threshold:.3e}"
                for row in others
            )
            lines.append(
                f"score check: FAIL -- {len(others)} equation(s) held to the default bar "
                f"not solved: {named}"
            )
            lines.extend(self._identity_lines())
        return "\n".join(lines)

    def _identity_lines(self) -> list[str]:
        """The state-identity failures, said as what they are and not as a convergence one.

        An unsolved equation is a fit that did not get there; a broken identity is a fit
        that solved something else.  Reporting the second in the first's words -- "try
        one_step, lower the step size" -- would send a reader looking for a numerical
        problem that is not there, which is precisely what happened for two revisions while
        the loop reported ``1e-11`` and the curve was out by ``2e-04``.
        """
        failures = self.identity_failures
        if not failures:
            return []
        return [
            f"  {len(failures)} of those {'is' if len(failures) == 1 else 'are'} a state "
            "identity, not an unsolved equation: the score the targeting",
            "  step recorded and the term the reported curve carries are not the same",
            "  functional of the state this fit returned.  That is a defect in the",
            "  implementation and iterating longer will not fix",
            "  it. Inspect the score rows; guarded DR-TMLE also exposes corrections().",
        ]

    def _worst_ratio(self) -> float:
        ratios = [row.ratio for row in self.rows if np.isfinite(row.ratio)]
        return max(ratios) if ratios else float("nan")

    def _curve(self) -> str:
        """The corrected curve this fit reports, as its terms rather than as a constant.

        ``D = D* - D*_Q - D*_g`` under both guards and one term shorter under either alone,
        derived from which corrections are in the curve rather than written out here --
        so it cannot drift from what
        :meth:`~cleverly.inference.influence.CorrectionParts.total` did.  A hand-built
        check carries no corrections and gets the both-guards wording, which is what every
        such object meant before a partial guard was expressible.
        """
        if self.corrections is None or not self.corrections.rows:
            return "D = D* - D*_Q - D*_g"
        subtracted = {row.equation for row in self.corrections.rows if row.solved}
        return "D = D*" + "".join(f" - {term}" for term in ("D*_Q", "D*_g") if term in subtracted)

    def summary(self) -> str:
        """Return a printable summary.

        Returns
        -------
        str
            A printable table, one line per checked score equation.
        """
        if not self.passed:
            verdict = (
                "FAIL: the score equation was not solved -- the "
                f"{spread_name('influence-curve standard errors', self.inference)} this fit "
                "reports do not describe this estimate. See the module docstring for the "
                "usual causes."
            )
            if self.identity_failures:
                # Named as its own thing and *first*, because the other failing rows on
                # such a fit are usually this one's consequence: a curve built from an
                # expression the loop did not solve is not centred, so the correction rows
                # and the per-estimand rows go with it. Which rows failed is in the table
                # above; what a reader cannot get from the table is that iterating longer
                # is not the remedy.
                verdict = "\n".join(
                    [
                        "FAIL: a state identity does not hold -- the score the targeting "
                        "step recorded and the",
                        "term the reported curve carries are not the same functional of "
                        "the state this fit",
                        "returned. That is a defect in the implementation rather than",
                        "a fit that failed to converge, and the "
                        f"{spread_name('standard errors', self.inference)} do not "
                        "describe this estimate.",
                        "Inspect the score rows. For guarded DR-TMLE, inspect "
                        "res.diagnostics.corrections() for the correction identities.",
                    ]
                )
        elif self.declared:
            # The rule's own verdict: the score is inside the threshold the rule set, which is
            # a residual by design and not a root to machine precision.  What the residual
            # costs is the line below the table.
            verdict = (
                f"PASS: the targeting step met the {self.stopping} stopping rule on every "
                "equation it judges by that rule. The score equation is solved to the rule's "
                "threshold and not to machine precision."
            )
            if self.corrected:
                verdict += f"\nValidity is not efficiency: the curve reported is {self._curve()}."
        elif self.corrected:
            # Not "the efficient score equation": this fit solved *its own* equations, and
            # the curve they leave is the estimator's influence function at the nuisance
            # limits. The two coincide when both nuisances are consistent, because the
            # corrections then vanish row by row -- the corrections add cost and no
            # asymptotic gain there. With one nuisance wrong the interval holds only in the
            # cases the contract lists (dr-tmle/theorem.md, the conditioning terms).
            solved = sum(1 for row in self.rows if row.kind == "fluctuation")
            verdict = "\n".join(
                [
                    f"PASS: the targeting step solved all {solved} estimated score equations "
                    "of the doubly-robust estimator.",
                    *(
                        [
                            f"Validity is not efficiency: the curve reported is {self._curve()}. "
                            "Theorem 1 makes its interval valid under its conditions.",
                            "With one wrong nuisance the interval holds only in the cases "
                            "cleverly.estimators.drtmle lists.",
                        ]
                        if supplies_inference(self.inference)
                        else [
                            f"The reported curve is {self._curve()}. This check verifies its "
                            "score equations, not an interval for this fit."
                        ]
                    ),
                ]
            )
        else:
            verdict = "PASS: the targeting step solved the estimated efficient score equation."
        if self.passed and not supplies_inference(self.inference):
            verdict += "\n" + status_record(self.inference).reason
        residual = self.rule_line()
        if residual:
            verdict += "\n" + residual
        return "\n".join(
            [
                "Score-equation check",
                "-" * 20,
                format_table(
                    ["target", "kind", "|score|", "before", "threshold", "ratio", "ok"],
                    [
                        [
                            row.name,
                            row.kind,
                            format_negligible(abs(row.score), row.threshold, digits=3),
                            format_negligible(abs(row.score_initial), row.threshold, digits=3),
                            # A `diagnostic` row is held to no threshold, so it has none to
                            # print. A `nan` beside a `yes` in the `ok` column reads as a
                            # bug rather than as a row that is not being judged.
                            f"{row.threshold:.3e}" if np.isfinite(row.threshold) else "-",
                            format_negligible(row.ratio, 1.0, digits=2),
                            "yes" if row.passed else "NO",
                        ]
                        for row in self.rows
                    ],
                ),
                *self._notes(),
                "",
                verdict,
            ]
        )

    def _notes(self) -> list[str]:
        """Anything the table has no column for: named failures, fold convergence."""
        notes: list[str] = []
        for row in self.rows:
            if np.isfinite(row.residual_se):
                bound = (
                    f"; the rule bounds it at {row.rule_bound_se:.3g}"
                    if np.isfinite(row.rule_bound_se)
                    else ""
                )
                notes.append(
                    f"  {row.name}: residual {row.residual_se:.3g} standard errors, a Wald "
                    f"interval keeps at least {row.coverage_bound:.4f} coverage{bound}"
                )
            if row.failure:
                # A recomputed row's `failure` is a statement about the fit's arithmetic
                # rather than about a solver that gave up, so it is not introduced as one.
                stopped = (
                    ""
                    if row.kind in ("correction", "identity", "diagnostic")
                    else "targeting stopped -- "
                )
                notes.append(f"  {row.name}: {stopped}{row.failure}")
            if row.folds_converged is not None:
                good, total = row.folds_converged
                if good < total:
                    notes.append(f"  {row.name}: {total - good} of {total} folds did not converge")
            if np.isfinite(row.hessian_condition) and row.hessian_condition > 1e8:
                notes.append(
                    f"  {row.name}: Hessian condition {row.hessian_condition:.2e} -- epsilon is "
                    "barely identified, so a solved score may still be fragile"
                )
        return ["", *notes] if notes else []

    def raise_if_failed(self) -> None:
        """Raise when the check failed -- for use in pipelines that must not ship a bad fit."""
        if self.passed:
            return
        names = ", ".join(row.name for row in self.failures)
        raise AssertionError(
            f"the TMLE score equation was not solved for: {names}. {self.summary()}"
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return self.summary()


def score_check(result: TMLEResult, *, tolerance: float = DEFAULT_TOLERANCE) -> ScoreCheck:
    """Check that targeting solved the score equation.

    Two things are verified, and both matter:

    * per targeted family, the fluctuation's own score
      ``mean(w * h * (Y - Q*))`` -- this is the equation the fluctuation is
      *solving*;
    * per estimand, the mean of its influence curve -- this is the equation the
      *estimate* relies on, which for a derived estimand such as the risk ratio is a
      transformation of the first.

    Parameters
    ----------
    result : TMLEResult
        A fitted result.
    tolerance : float
        Score magnitude, in standard errors, a row may reach and pass.

    Returns
    -------
    ScoreCheck
        One row per score equation the fit's method relies on.
    """
    n = result.data.n
    rows: list[ScoreCheckRow] = []

    # ``plugin_std_error`` and not ``std_error``: every use of this number here is a
    # *scale* for a convergence tolerance, which is a diagnostic.  A selector-path
    # collaborative fit refuses ``std_error``, and reading it would make ``summary()``
    # raise on every such fit through ``score_verdict``.
    reference_se = max(
        (
            estimate.plugin_std_error
            for estimate in result.estimates.values()
            if np.isfinite(estimate.plugin_std_error)
        ),
        default=1.0,
    )
    # One call rather than the same product written at each row below, and the same call a
    # study auditing this package against another implementation makes -- see
    # :func:`score_threshold`.
    threshold = score_threshold(
        (estimate.plugin_std_error for estimate in result.estimates.values()),
        n,
        tolerance=tolerance,
    )

    # Every draw's fluctuation, not just the first. Each solved its own score equation,
    # and a draw whose Newton step failed contributes to the reported estimate exactly as
    # the others do -- checking one of R would let that failure through silently. The row
    # name carries the draw index only when there is more than one, so an ordinary fit's
    # report is unchanged.
    rule = _declared_rule(result)
    for index, repeat in enumerate(result.repeats):
        span = repeat.nuisance.scaler.range
        for group, fluctuation in repeat.fluctuations.items():
            score = fluctuation.score_norm
            # The default bar, byte for byte, unless the fit declared a rule.  Under a rule
            # the row is held to the threshold the solver stopped at, in the outcome's units,
            # and it passes when the largest score over its threshold is at most one.
            bar, passes, before, judged_by = (
                float(threshold),
                bool(score <= threshold),
                fluctuation.initial_score_norm,
                "relative",
            )
            if _judged_by_rule(fluctuation):
                score, bar = _rule_bar(fluctuation, span)
                passes = bool(fluctuation.rule_ratio <= 1.0)  # type: ignore[operator]
                before = fluctuation.initial_score_norm * span
                judged_by = fluctuation.stopping
            rows.append(
                ScoreCheckRow(
                    name=group if result.n_repeats == 1 else f"{group}[draw {index}]",
                    kind="fluctuation",
                    score=score,
                    threshold=bar,
                    std_error=float(reference_se),
                    passed=passes,
                    converged=fluctuation.converged,
                    n_iter=fluctuation.n_iter,
                    method=fluctuation.method,
                    score_initial=before,
                    stopping=judged_by,
                    failure=fluctuation.failure or "",
                    hessian_condition=fluctuation.hessian_condition,
                    folds_converged=(
                        (sum(f.converged for f in fluctuation.folds), len(fluctuation.folds))
                        if fluctuation.folds
                        else None
                    ),
                )
            )
            # A linked working model with baseline strata solves its stratum blocks in a
            # nested fluctuation at the stratum coefficients.  The stratum estimates are
            # read from it, so its score is checked on its own row.
            # Under a declared rule that fluctuation is an outcome solve the rule stopped, so
            # its row is judged by its own rule ratio, as the row above is.
            nested = fluctuation.stratified
            if nested is not None:
                score = nested.score_norm
                bar, passes, before, judged_by = (
                    float(threshold),
                    bool(score <= threshold),
                    nested.initial_score_norm,
                    "relative",
                )
                if _judged_by_rule(nested):
                    score, bar = _rule_bar(nested, span)
                    passes = bool(nested.rule_ratio <= 1.0)  # type: ignore[operator]
                    before = nested.initial_score_norm * span
                    judged_by = nested.stopping
                stem = f"{group} (strata)"
                rows.append(
                    ScoreCheckRow(
                        name=stem if result.n_repeats == 1 else f"{stem}[draw {index}]",
                        kind="fluctuation",
                        score=score,
                        threshold=bar,
                        std_error=float(reference_se),
                        passed=passes,
                        converged=nested.converged,
                        n_iter=nested.n_iter,
                        method=nested.method,
                        score_initial=before,
                        stopping=judged_by,
                        failure=nested.failure or "",
                        hessian_condition=nested.hessian_condition,
                    )
                )
            # A group whose parameter is defined through the mechanism solves *two*
            # equations, and the per-estimand rows below check only their sum -- the
            # influence curve contains both terms, so its mean cannot be zero unless
            # both are solved, but a failure there does not say which. This row does.
            if fluctuation.mechanism is not None:
                mechanism = fluctuation.mechanism
                score = float(np.max(np.abs(mechanism.score))) if mechanism.score.size else 0.0
                stem = f"{group} (mechanism)"
                rows.append(
                    ScoreCheckRow(
                        name=stem if result.n_repeats == 1 else f"{stem}[draw {index}]",
                        kind="fluctuation",
                        score=score,
                        threshold=float(threshold),
                        std_error=float(reference_se),
                        passed=bool(score <= threshold),
                        converged=mechanism.converged,
                        n_iter=len(mechanism.trace),
                        method=fluctuation.method,
                        score_initial=(
                            float(np.max(np.abs(mechanism.score_initial)))
                            if mechanism.score_initial.size
                            else float("nan")
                        ),
                        failure=mechanism.failure or "",
                        hessian_condition=(
                            float("nan")
                            if mechanism.hessian_condition is None
                            else mechanism.hessian_condition
                        ),
                    )
                )
            # A doubly-robust fit solves a *third* equation, in a second submodel of the
            # same group -- so the row above is the mechanism half of it and this one is
            # the second outcome half. Reported apart for the same reason: the influence
            # curve's mean is zero only if all three are solved, and says nothing about
            # which one is not.
            # Under a declared rule this equation's columns are stopped by the rule too (the
            # closing joint solve, or the stacked missing-outcome solve), so the row is judged
            # by the record's own rule ratio.
            reduction = fluctuation.reduction
            if reduction is not None and np.asarray(reduction.score).size:
                score = float(np.max(np.abs(reduction.score)))
                initial = (
                    float(np.max(np.abs(reduction.score_initial)))
                    if np.asarray(reduction.score_initial).size
                    else float("nan")
                )
                bar, passes, judged_by = float(threshold), bool(score <= threshold), "relative"
                if rule != "relative" and reduction.rule_ratio is not None:
                    score, bar = _ratio_bar(
                        reduction.score, reduction.thresholds, reduction.rule_ratio, span
                    )
                    passes = bool(reduction.rule_ratio <= 1.0)
                    initial = initial * span
                    judged_by = rule
                stem = f"{group} (reduced)"
                rows.append(
                    ScoreCheckRow(
                        name=stem if result.n_repeats == 1 else f"{stem}[draw {index}]",
                        kind="fluctuation",
                        score=score,
                        threshold=bar,
                        std_error=float(reference_se),
                        passed=passes,
                        converged=reduction.converged,
                        n_iter=reduction.n_outer,
                        method=fluctuation.method,
                        score_initial=initial,
                        stopping=judged_by,
                        failure=reduction.failure or "",
                    )
                )

    # Each arm's corrections, recomputed from the state its draw returned. Two kinds of row
    # and they are two different failures: an ``identity`` row says the solver and the curve
    # are not evaluating one expression, which is a software defect; a ``correction`` row
    # says the term the curve subtracts is not negligible, which is an unsolved equation.
    # A third kind, ``diagnostic``, is neither and cannot fail: it is a term a single-guard
    # fit does not subtract, on the table because it is what says what the guard left out.
    # `cleverly.validation.drtmle` derives all three, and returns nothing at all for a fit
    # that reports no corrections -- so no ordinary report gains a row.
    corrections = correction_check(result, tolerance=tolerance, std_error=float(reference_se))
    rows.extend(
        _correction_rows(corrections, n_repeats=result.n_repeats, ruled=_ruled_reduction(result))
    )

    for name, estimate in result.estimates.items():
        if rule != "relative":
            rows.append(_estimand_diagnostic(result, name, estimate, rule))
            continue
        threshold = tolerance * estimate.plugin_std_error / np.sqrt(n)
        score = abs(estimate.score)
        rows.append(
            ScoreCheckRow(
                name=name,
                kind="influence curve",
                score=float(score),
                threshold=float(threshold),
                std_error=estimate.plugin_std_error,
                passed=bool(score <= threshold),
                converged=True,
                n_iter=0,
                method=estimate.scale,
            )
        )

    return ScoreCheck(
        rows=tuple(rows),
        tolerance=tolerance,
        n=n,
        # Read off the records rather than passed down from the estimator: this function
        # takes a result and a result does not name the class that made it.  A guard of
        # ``()`` leaves no reduction record, so such a fit is *not* corrected here -- which
        # is right, since it is a plain TMLE in every other respect too.
        corrected=any(
            fluctuation.reduction is not None
            for repeat in result.repeats
            for fluctuation in repeat.fluctuations.values()
        ),
        corrections=corrections,
        backend=result.data.backend,
        inference=result.inference_status,
        stopping=rule,
        alpha_sig=result.config.alpha_sig,
    )


def _declared_rule(result: TMLEResult) -> str:
    """The stopping rule the fit declared, ``"relative"`` when it declared none."""
    for repeat in result.repeats:
        for fluctuation in repeat.fluctuations.values():
            if fluctuation.stopping != "relative":
                return fluctuation.stopping
    return "relative"


def _judged_by_rule(fluctuation: Fluctuation) -> bool:
    """Whether this fluctuation's row is held to a declared rule's threshold."""
    return fluctuation.stopping != "relative" and fluctuation.rule_ratio is not None


def _ruled_reduction(result: TMLEResult) -> bool:
    """Whether a declared rule stopped the doubly-robust outcome columns of this fit."""
    return any(
        fluctuation.stopping != "relative"
        and fluctuation.reduction is not None
        and fluctuation.reduction.rule_ratio is not None
        for repeat in result.repeats
        for fluctuation in repeat.fluctuations.values()
    )


def _rule_bar(fluctuation: Fluctuation, span: float) -> tuple[float, float]:
    """The score and the threshold of a row under a declared rule, in the outcome's units.

    The column with the largest score over its threshold, so ``score / threshold`` is the
    fluctuation's ``rule_ratio``.  A fluctuation without per-column thresholds (the pooled
    fold-targeted fit holds the largest ratio of its folds) reports that ratio against a bar
    of one.
    """
    return _ratio_bar(
        fluctuation.score,
        fluctuation.thresholds,
        float(fluctuation.rule_ratio),  # type: ignore[arg-type]
        span,
    )


def _ratio_bar(
    score: FloatArray, thresholds: FloatArray | None, ratio: float, span: float
) -> tuple[float, float]:
    """:func:`_rule_bar` for any record that carries a score, thresholds and a rule ratio.

    When the thresholds do not reproduce ``ratio`` (a ratio taken over more columns than the
    record holds, as on the missing-outcome fit), the row reports the ratio against a bar of
    one, so the row and the record read one number.
    """
    score = np.asarray(score, dtype=float)
    if thresholds is not None and score.size == thresholds.size:
        statistic = np.abs(score) * span
        column = int(np.argmax(statistic / np.maximum(thresholds, 1e-300)))
        recomputed = float(statistic[column] / max(float(thresholds[column]), 1e-300))
        if abs(recomputed - ratio) <= 1e-9 * max(1.0, ratio):
            return float(statistic[column]), float(thresholds[column])
    return ratio, 1.0


def _estimand_diagnostic(
    result: TMLEResult, name: str, estimate: ParameterEstimate, rule: str
) -> ScoreCheckRow:
    """An estimand row of a fit under a declared rule: a diagnostic, never a failure.

    A declared rule leaves a score residual by design, so the mean of an influence curve is
    not held to the default bar.  The row reports the residual in standard errors and the
    coverage a Wald interval keeps at it.  Under ``scaled_var`` it also reports the residual
    the rule implies before the fit is read.
    """
    residual = residual_se(estimate)
    alpha = result.config.alpha_sig
    return ScoreCheckRow(
        name=name,
        kind="diagnostic",
        score=float(abs(estimate.score)),
        threshold=float("nan"),
        std_error=estimate.plugin_std_error,
        passed=True,
        converged=True,
        n_iter=0,
        method=estimate.scale,
        stopping=rule,
        residual_se=residual,
        coverage_bound=wald_coverage_bound(residual, alpha),
        rule_bound_se=_rule_bound_se(result, name, estimate)
        if rule == "scaled_var"
        else float("nan"),
    )


def _column_weights(result: TMLEResult, name: str) -> tuple[str, dict[int, float]] | None:
    """The fluctuation group and the score columns an arm parameter is a combination of.

    ``g`` of the plan: an arm mean is its own column, a difference of two arm means is the
    first column minus the second, and a conditional effect (``att``, ``atc``) is the contrast
    column of its arm.  Columns are found by the names the submodel builders give them
    (``h1``, ``h_att``, ``h_att[2]``), not by position.

    On a stratified fit the group holds one block per stratum, ``I(S = s) H_s / P_n(S = s)``,
    and no marginal column.  A stratum's parameter is that stratum's block, with weight one.
    A marginal parameter is the weighted sum of the blocks: ``P_n(S = s)`` for an arm mean or
    a difference, and ``P_n(S = s) P_n(A = a | S = s) / P_n(A = a)`` for a conditional effect,
    with ``a`` the arm the effect conditions on (the arm for ``att``, the reference for
    ``atc``), because each stratum block is built with its own arm share.

    ``None`` for any other estimand: a ratio, a regime, a shift or policy, an incremental
    tilt, a working model.  ``None`` also when a named column is absent.
    """
    from ..data.causal_data import arm_share
    from ..estimators._strata import stratum_probabilities
    from ..sensitivity._parameters import arm_parameters
    from ..targets.base import stratum_alias

    data = result.data
    parameters = arm_parameters(result)
    parameter = parameters.get(name)
    stratum: int | None = None
    if parameter is None and data.has_strata:
        for marginal, candidate in parameters.items():
            for code in range(data.n_strata):
                if stratum_alias(marginal, data.stratum_label(code)) == name:
                    parameter, stratum = candidate, code
    if parameter is None:
        return None
    fluctuation = result.repeats[0].fluctuations.get(parameter.group)
    if fluctuation is None:
        return None
    if parameter.group == "mean":
        base = {f"h{float(parameter.arm):g}": 1.0}
        if parameter.versus is not None:
            base[f"h{float(parameter.versus):g}"] = -1.0
        conditioning = None
    elif parameter.group in ("att", "atc") and parameter.versus is not None:
        column = f"h_{parameter.group}"
        multi = f"h_{parameter.group}[{float(parameter.arm):g}]"
        stratified_names = {label.split(" | ")[0] for label in fluctuation.names}
        base = {multi if multi in stratified_names else column: 1.0}
        conditioning = float(parameter.arm if parameter.group == "att" else parameter.versus)
    else:
        return None

    names = list(fluctuation.names)
    stratified = any(" | " in label for label in names)
    if not stratified:
        if stratum is not None or any(column not in names for column in base):
            return None
        return parameter.group, {names.index(column): g for column, g in base.items()}

    probabilities = stratum_probabilities(data)
    assert data.strata is not None
    weights: dict[int, float] = {}
    for code in range(data.n_strata):
        if stratum is not None and code != stratum:
            continue
        multiplier = 1.0
        if stratum is None:
            multiplier = probabilities[code]
            if conditioning is not None:
                within = arm_share(
                    data.treatment, data.weights, conditioning, mask=data.strata == code
                )
                multiplier *= within / arm_share(data.treatment, data.weights, conditioning)
        label = data.stratum_label(code)
        for column, g in base.items():
            block = f"{column} | {label}"
            if block not in names:
                return None
            weights[names.index(block)] = g * multiplier
    return parameter.group, weights


def _rule_bound_se(result: TMLEResult, name: str, estimate: ParameterEstimate) -> float:
    """``sum_j |g_j| threshold_j / se`` for an arm parameter of a ``scaled_var`` fit.

    A fit that met its rule has ``|P_n S_j| <= threshold_j`` on every column, and the arm
    parameter's ``P_n IC`` is ``sum_j g_j P_n S_j`` (:func:`_column_weights`), so this sum
    bounds ``|P_n IC| / se`` before the fit is read.  On a unit-weight arm mean of an
    unstratified fit the threshold is ``c * se``, with ``c = 1 / min(log n_u, 10)``, and the
    value is ``c``.  On a stratified fit the thresholds are taken from the score columns
    (declared difference D4), so the value is not ``c``.

    ``nan`` where the bound is not computed:

    * the estimand is not a fixed combination of outcome score columns: a ratio, a regime;
    * an incremental tilt: its curve also carries the treatment-mechanism equation, which the
      rule does not govern;
    * a working-model coefficient: its curve is the score columns times the inverse of the
      working model's normalising matrix, which the result does not carry;
    * a shift or policy row: the result does not record which estimate reads which policy
      column;
    * a fit with several draws, whose reported curve averages several fluctuations;
    * a doubly-robust fit, whose corrected curve carries terms the outcome columns do not.
    """
    mapped = _column_weights(result, name) if len(result.repeats) == 1 else None
    error = estimate.plugin_std_error
    if mapped is None or not (np.isfinite(error) and error > 0.0):
        return float("nan")
    group, weights = mapped
    fluctuation = result.repeats[0].fluctuations.get(group)
    if (
        fluctuation is None
        or fluctuation.reduction is not None
        or fluctuation.thresholds is None
        or max(weights) >= fluctuation.thresholds.size
    ):
        return float("nan")
    total = sum(abs(g) * float(fluctuation.thresholds[column]) for column, g in weights.items())
    return float(total / error)


#: The correction terms that restate the doubly-robust *outcome* columns (equation (10), and
#: the outcome drift block of the missing-outcome fit).  Under a declared rule those columns
#: are judged on the ``(reduced)`` row by the rule ratio, so these rows are diagnostics there.
_RULED_CORRECTIONS = frozenset({"D*_Q", "D*_Y"})


def _correction_rows(
    check: CorrectionCheck, *, n_repeats: int, ruled: bool = False
) -> list[ScoreCheckRow]:
    """One row per arm per equation, and a second where there is an identity to check.

    The draw index goes into the name only when there is more than one, exactly as the
    fluctuation rows do it, so an ordinary doubly-robust report reads as one fit's table.

    An equation the fit's guard does not solve gets kind ``"diagnostic"`` rather than
    ``"correction"``: its term is not in the reported curve, so there is nothing for it to
    invalidate and no threshold to hold it to.  It is on the table anyway because it is
    what says which robustness the guard bought and which it did not.

    Under a declared rule (``ruled``) a ``D*_Q`` or ``D*_Y`` row is a ``"diagnostic"`` too.
    It restates the outcome columns the rule stopped, which the ``(reduced)`` row judges by
    the rule ratio, and a declared rule leaves a residual there by design.  Its identity row
    is still checked, because an identity failure is a software defect under any rule.
    The mechanism rows (``D*_g``, ``D*_A``, ``D*_M``) keep the default bar.
    """
    rows: list[ScoreCheckRow] = []
    for row in check.rows:
        suffix = "" if n_repeats == 1 else f"[draw {row.draw}]"
        by_rule = ruled and row.solved and row.equation in _RULED_CORRECTIONS
        held = row.solved and not by_rule
        rows.append(
            ScoreCheckRow(
                name=f"{row.name}{suffix}",
                kind="correction" if held else "diagnostic",
                score=row.reported,
                threshold=check.threshold if held else float("nan"),
                std_error=check.std_error,
                passed=bool(not held or abs(row.reported) <= check.threshold),
                converged=True,
                n_iter=0,
                method="recomputed",
                failure=(
                    ""
                    if held
                    else "the declared stopping rule judges these outcome columns on the "
                    "(reduced) row, so this restatement is held to nothing"
                    if by_rule
                    else "not among the equations this fit's guard= solves, so the reported "
                    "curve does not subtract it -- held to nothing"
                ),
            )
        )
        if not np.isfinite(row.residual):
            # No stored score to compare against is the *absence* of a check, not a pass,
            # and the `failure` above is what says so.
            continue
        rows.append(
            ScoreCheckRow(
                name=f"{row.name} identity{suffix}",
                kind="identity",
                score=row.residual,
                threshold=check.identity_threshold,
                std_error=check.std_error,
                passed=bool(abs(row.residual) <= check.identity_threshold),
                converged=True,
                n_iter=0,
                method="recomputed",
                failure=(
                    ""
                    if abs(row.residual) <= check.identity_threshold
                    else f"the mechanism truncation binds on {row.clipped} row(s) and "
                    f"absorbs B_clip = {row.clip_bias:.3e} of this equation"
                    if np.isfinite(row.clip_bias)
                    else "the solved score and the reported term are different expressions"
                ),
            )
        )
    return rows
