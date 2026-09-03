r"""What observation weights mean here, and what the reported variance is a variance of.

Weighting a causal estimator is not a single operation with a single meaning.  "Survey
weight", "inverse-probability-of-sampling weight", "frequency weight" and "calibrated
weight" describe different statistical experiments, and an implementation that multiplies
influence-curve values by a weight vector is correct for some of them and wrong for
others.  This module states which experiment this library assumes, what parameter it
therefore estimates, and where the assumption is known not to hold.

The estimand
------------

Let :math:`O = (W, A, \Delta, Y)` be the observed data and let :math:`w = w(O) \ge 0` be
the supplied weight, with :math:`0 < E_P[w] < \infty`.  Define the **tilted law**

.. math::

    dP_w = \frac{w\,dP}{E_P[w]} .

The parameter this library reports under weighting is the same causal functional it
reports without weighting, evaluated at that law:

.. math::

    \Psi_w(P) \;:=\; \Psi(P_w).

This is a statement about the whole procedure, not only about the final average.  Every
nuisance is fitted by weighted loss, so :math:`\hat g` and :math:`\hat{\bar Q}` estimate
:math:`P_w`-conditionals; the targeting step solves
:math:`\sum_i w_i H_i (Y_i - \bar Q_i) = 0`, which is the score under the weighted
empirical measure :math:`P_{n,w}`; and the plug-in is a :math:`w`-weighted average.  The
estimator is therefore *exactly* TMLE run on :math:`P_{n,w}` -- a single coherent object,
not an unweighted fit with weights applied at the end.

Two consequences follow immediately, and both are worth being explicit about:

* :math:`\Psi(P_w)` has a causal interpretation only if the identification assumptions
  (consistency, no interference, no unmeasured confounding, positivity) hold **under**
  :math:`P_w`.
  Weighting does not create identification; it moves the population in which
  identification is being claimed.
* A learner that ignores ``sample_weight`` is fitting the wrong conditional -- not merely
  a less efficient one.  Such learners warn at fit time
  (:func:`cleverly.learners._fitting`); heed the warning.

The influence curve
-------------------

Contaminate :math:`P` towards a point mass, :math:`P_t = (1-t)P + t\delta_o`, and push it
through the tilt.  Writing :math:`\bar w = E_P[w]`,

.. math::

    \left.\frac{d}{dt} P_{t,w}\right|_{t=0}
      = \frac{w(o)}{\bar w}\,\bigl(\delta_o - P_w\bigr),

so the contamination path in :math:`P` maps to the contamination path in :math:`P_w`
*scaled by* :math:`w(o)/\bar w`.  By the chain rule the efficient influence function of
:math:`\Psi_w` in the nonparametric model is

.. math::

    D^*_{\Psi_w}(o) \;=\; \frac{w(o)}{\bar w}\; D^*_{P_w}(o),

where :math:`D^*_{P_w}` is the ordinary efficient influence function of :math:`\Psi`
evaluated at the tilted law.  That is precisely what the library computes: the usual
influence curve, built from the weighted nuisance fits and centred at the weighted
estimate, multiplied row-wise by the normalised weight.  It is verified numerically, to
double precision and against a longhand statement of :math:`\Psi(P_w)`, in
``tests/unit/test_weighted_estimand.py`` -- including for weights that depend on the
outcome, where the tilt changes :math:`\bar Q` itself.

So the variance the library reports,
:math:`\widehat{\operatorname{Var}}(\hat\psi) = \operatorname{Var}_n(\tilde w D^*)/n`,
is the variance of an efficient estimator of :math:`\Psi(P_w)` **in the model where the
rows are i.i.d. draws of** :math:`(O_i, w_i)` **and** :math:`w_i` **is observed rather
than estimated.**  Everything below is a consequence of, or an exception to, that
sentence.

Over time
---------

Nothing above is about one time point, and :class:`~cleverly.longitudinal.LTMLE` reads
``weights=`` on exactly these terms.  With :math:`O = (W, L_1, A_1, C_1, \ldots, Y)` and
:math:`w = w(O)` a function of the observed row, the estimand is the same regimen
parameter at :math:`P_w`; every node's mechanism, every node's censoring factor and every
regression in the backward recursion is fitted by weighted loss; each node's fluctuation
solves :math:`\sum_i w_i h_t(i)\,(Z_t(i) - \bar Q^*_t(i)) = 0`; the plug-in is a weighted
average; and the reported curve is :math:`\tilde w_i D^*(O_i)` for the sequential
:math:`D^*`.  Verified against a longhand :math:`\Psi(P_w)` in
``tests/unit/test_weighted_estimand_longitudinal.py``, including for a weight that reads
the treatment, the censoring indicator, the time-varying confounder and the outcome --
where the tilt moves every one of those nuisances rather than only the covariate marginal.

**A weight is not a factor in the clever covariate**, and this is worth stating because
the refusal that preceded the implementation said it was.  :math:`h_t` divides by the
:math:`2T` treatment and censoring probabilities and by nothing else; the weight tilts the
population the parameter is defined in, and it enters the estimating equation as a
multiplier of each row's score, not as a denominator.  Putting :math:`w` inside
:math:`h_t` would divide the equation by the very tilt it is supposed to apply.

Unlike the point-treatment estimator discussed next, LTMLE has no
``g_bounds="auto"`` procedure.  Its default is the visible fixed cumulative pair
``(0.01, 1.0)``; observation weights do not change it.  A cumulative probability can
naturally fall below that heuristic as follow-up grows even when each factor is moderate,
so the fit reports the raw-versus-bounded share at every node.

Under doubly-robust inference
-----------------------------

:class:`~cleverly.DRTMLE` reads ``weights=`` on the same terms, for **fixed analysis
weights**, and there is one thing more to check than there is anywhere else.  Its extra
score equations are stated in terms of *reduced-dimension regressions* of each nuisance's
residual on the other, and those have to be conditional expectations under :math:`P_w`
rather than :math:`P_0` -- which weighted loss gives -- **and** they have to condition on
and divide by the :math:`P_w`-mechanism rather than :math:`g_0`, which holds because they
are built from the weighted mechanism fit.  Neither is automatic from "every equation
carries :math:`w`", which is why it is checked: ``tests/unit/test_remainder_drtmle.py``
takes the whole second-order expansion at two tilted laws, and keeps the wrong transport as
a test rather than as a caveat.

An **estimated** weight is where that stops.  The argument above for the ordinary
estimator -- that the interval conditions on the weights, which ``weights_estimated=``
declares -- is about :math:`D^*`, and the reduced regressions of a random tilt are not
something anything read here derives.

Normalisation
-------------

Supplied weights are rescaled to mean one, :math:`\tilde w_i = n w_i / \sum_j w_j`.  This
is a convention with two deliberate effects.

* **Scale invariance.**  :math:`w \mapsto c\,w` changes nothing -- not the estimate, not
  the standard error, not a single fold.  Weights only ever matter relatively.
* **The ratio is already linearised.**  The estimator is a Hájek ratio
  :math:`\sum_i w_i f_i / \sum_i w_i`, and the influence curve carries its centring term
  :math:`\tilde w_i (f_i - \hat\psi)` rather than :math:`\tilde w_i f_i`.  That *is* the
  delta-method correction for the random denominator, so no further adjustment is needed
  and none should be added.  A Horvitz--Thompson form (dividing by a known population
  total instead of by :math:`\sum_i w_i`) is a different estimator with a different, and
  usually larger, variance; it is not what this library computes.

Rows with :math:`w_i = 0` are kept, contribute nothing to the estimate, and still count
towards :math:`n`.  Under the i.i.d.-weights model that is correct: drawing a zero weight
is an outcome of the experiment, and it is informative about the tilt.

Which sample size
-----------------

The *variance* needs no help from the effective sample size: normalisation scales the
surviving influence-curve values up by exactly the factor the larger :math:`n` divides
out, so zero-weighting a stratum and deleting it give the same standard error, and the
reported variance is right whatever the design effect.

Everything the estimator *tunes* from the sample size is a different matter, and
``g_bounds="auto"`` is the one that bites.  The rule
:math:`5 / (\sqrt{n}\log n)` is a bias-variance compromise -- truncate hard enough to
control the variance of :math:`1/g`, loosely enough that the truncation bias vanishes --
and both sides of it are governed by the information in the sample, not by the row count.
So it is resolved at Kish's effective sample size
:math:`n_{\text{eff}} = (\sum w)^2/\sum w^2`, which equals :math:`n` exactly when the
weights are constant and is smaller by the design effect otherwise.  At a design effect
of four the row count would set a bound nearly three times too loose.  This is a
deliberate divergence from R's ``tmle``, it applies only to weighted fits, and the fit
summary names it where it takes effect.  An explicit ``g_bounds=`` is never second-guessed.

A design effect above :data:`CONCENTRATED_DESIGN_EFFECT` warns at construction, because
past that point the number governing the truncation, the asymptotics and the width of the
interval is one the user has not been shown.

Which weighting problem is which
--------------------------------

*Known sampling probabilities*, :math:`w = 1/\pi(O)`, with selection depending only on
observed data.  **Supported.**  :math:`P_w` is the population law, so :math:`\Psi(P_w)`
is the population parameter -- the case the whole construction is for.

*Design weights from a complex survey* (strata, PSUs, finite-population corrections).
**Point estimate supported**, variance only partly: it ignores everything about the
design except what ``id=`` captures.  See "complex survey designs" below.

*Outcome-dependent sampling with known sampling fractions* (case-control).
**Supported**, and the tilt identity holds even though :math:`w` depends on :math:`Y` --
but only if the fractions are genuinely known.  They are not estimable from the sample.

*Estimated selection or non-response weights*.  **Variance conditions on**
:math:`\hat w`, and is conservative under the conditions given below.

*Calibration, raking, post-stratification, trimming*.  As above, but with no general
guarantee on the direction of the error.  Bootstrap the weight derivation instead.

*Frequency (count) weights*.  **Rejected**: each row stands for :math:`w_i` units, so the
sample size is :math:`\sum_i w_i` rather than :math:`n`.

*Replicate weights* (BRR, jackknife, bootstrap columns).  **Rejected**: a set of designs,
not one weight vector.

Estimated weights
^^^^^^^^^^^^^^^^^

Pass ``weights_estimated=True`` when the weights came out of a fitted model.  It changes
no number -- it cannot, since the package never sees the model -- but it makes the
reports say what the standard errors condition on, which is the point of the exercise.

The reason the numbers do not change is that there is a genuine choice of target here.
Treat the weights as part of the data and :math:`\Psi(P_{\hat w})` is exactly what is
being estimated, with the reported variance correct for it.  Treat the *full population*
as the target and the two-phase efficient influence function acquires a second term,

.. math::

    D_{\text{2-phase}}(o) = \frac{S}{\pi(V)} D^*_F(o)
        - \left(\frac{S}{\pi(V)} - 1\right) E\bigl[D^*_F \mid V\bigr],

whose role is to credit the information in the estimated :math:`\hat\pi`.  Dropping it,
as the fixed-weight variance does, leaves a valid but *inefficient* estimator when
:math:`\pi` is known, and a **conservative** variance when :math:`\pi` was fitted by
maximum likelihood in a model containing the truth -- the classical result that
estimating known weights improves efficiency.  For calibration, raking or trimmed weights
neither statement is guaranteed, because the weights are not a likelihood-based fit of a
selection mechanism.  There the honest options are a bootstrap that re-derives the
weights inside each replicate (which this library's bootstrap does *not* do: it resamples
rows and renormalises the weights it was given), or a design-based package.

Four things called "weights", and which of them differ
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The names in circulation suggest four distinct modes.  They are not four:

**Fixed analytic weights**, **sampling (design, inverse-probability, calibrated) weights**
and **estimated weights** all define the *same* estimand -- :math:`\Psi(P_w)` on the tilted
law :math:`dP_w = w\,dP/E[w]` -- and are estimated by the same arithmetic.  Where they
differ is only in what the interval conditions on.  For the first two the weight is a known
function of the row, so nothing extra is random and the variance above is complete.  For the
third the weight came out of a fitted model, and the interval is *conditional on that fit*:
the section immediately above says why dropping the :math:`\hat\pi` term is conservative
under a likelihood-based selection model and not guaranteed otherwise.  There is no
separate code path and there should not be one; ``weights_type=`` therefore does not
distinguish them, and ``weights_estimated=`` records the distinction that does exist.

**Frequency (count) weights** are the odd one out, and they are refused -- they change what
:math:`n` means rather than which population is targeted.  See below.

**Replicate weights** (BRR, jackknife) are refused too, for a different reason again: they
are a set of designs rather than one weight vector, so there is no single tilt to estimate.

Complex survey designs
^^^^^^^^^^^^^^^^^^^^^^

The variance above is the with-replacement linearisation variance.  Relative to a
full design-based treatment:

* **Stratification** is ignored, which makes the interval *conservative*.
* **Clustering** must be declared with ``id=`` (the PSU).  If it is not, the interval is
  **anti-conservative** -- this is the one omission that fails in the dangerous
  direction, and it is not detectable from the weights.
* **Finite-population corrections** are not applied, which is conservative.
* **Replicate weights** are a different variance estimator entirely and are not
  supported.  Supplying one replicate column as ``weights`` estimates that replicate's
  tilt, which is not a parameter anyone wants.

Frequency weights
^^^^^^^^^^^^^^^^^

These are rejected rather than approximated.  Under counts the experiment has
:math:`N = \sum_i w_i` independent units, so the variance divides by :math:`N`; passing
them as probability weights divides by the number of rows instead and overstates the
standard error by roughly :math:`\sqrt{\bar w}`.  Getting it right would require every
fold, cluster sum and bootstrap draw in the library to count units rather than rows, so
the supported answer is to expand the rows -- ``frame.loc[frame.index.repeat(counts)]``
-- and let :math:`n` mean what the rest of the library assumes it means.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .._typing import FloatArray
from ..exceptions import DataError, WeightingWarning
from .validate import check_weights

__all__ = [
    "CONCENTRATED_DESIGN_EFFECT",
    "WeightKind",
    "WeightReport",
    "WeightSpec",
    "describe_weights",
    "estimand_lines",
    "resolve_weight_kind",
    "warn_if_concentrated",
    "warn_if_counts",
]

#: Design effect at which the weighting is called out rather than merely reported.
#: Four means the effective sample size is a quarter of the rows -- past that, every
#: sample-size-dependent choice the estimator makes, and the asymptotics behind the
#: interval, are working from a number much smaller than ``n``, and the user should know
#: it without having to ask for a report.
CONCENTRATED_DESIGN_EFFECT = 4.0

#: How the supplied weights are to be read.  ``"probability"`` covers design, sampling,
#: inverse-probability and calibrated weights -- everything whose meaning is a *tilt* of
#: the population.  ``"frequency"`` means counts of identical units and is rejected; it
#: exists as a value so that the rejection is a clear error rather than a silent
#: misinterpretation.
WeightKind = Literal["probability", "frequency"]

_KIND_ALIASES = {
    "probability": "probability",
    "sampling": "probability",
    "survey": "probability",
    "design": "probability",
    "pweight": "probability",
    "frequency": "frequency",
    "count": "frequency",
    "fweight": "frequency",
}

_FREQUENCY_MESSAGE = (
    "frequency (count) weights are not supported. They mean the sample contains "
    "sum(weights) independent units rather than {n} rows, so the variance would have to "
    "divide by that total and every fold, cluster sum and bootstrap draw would have to "
    "count units rather than rows. Passing counts as probability weights gives the same "
    "point estimate with a standard error inflated by roughly sqrt(mean(counts)). "
    "Expand the rows instead -- pandas: frame.loc[frame.index.repeat(counts)]; polars: "
    "frame.select(pl.all().repeat_by(counts).explode()) -- or aggregate to a rate "
    "outcome. See cleverly.data.weighting for the full statement."
)


@dataclass(frozen=True)
class WeightSpec:
    """How to read the weight column, recorded alongside the data.

    Carries no numbers, only the interpretation: the numbers change under
    :meth:`~cleverly.data.CausalData.subset` and the interpretation does not.

    Attributes
    ----------
    kind:
        ``"probability"`` -- the only supported reading.  See the module docstring.
    estimated:
        ``True`` when the weights came from a fitted model (a propensity of selection, a
        non-response adjustment, a calibration).  Purely declarative: it changes no
        number, and makes the reports state that the standard errors condition on the
        estimated weights rather than pretending they were known.
    name:
        Column the weights came from, for the reports.
    scale:
        Mean of the weights as supplied, so the normalisation is recoverable.  The stored
        weights are ``supplied / scale``.
    """

    kind: WeightKind = "probability"
    estimated: bool = False
    name: str | None = None
    scale: float = 1.0

    def rescaled(self, scale: float) -> WeightSpec:
        """A copy recording a new normalisation constant, as :meth:`subset` needs."""
        return WeightSpec(kind=self.kind, estimated=self.estimated, name=self.name, scale=scale)


def resolve_weight_kind(kind: str | None, n: int) -> WeightKind:
    """Normalise a ``weights_type`` argument, rejecting the unsupported readings.

    Accepts the common synonyms (``"sampling"``, ``"survey"``, ``"design"``, Stata's
    ``"pweight"``) so that the argument can be written the way the caller thinks about
    their design.
    """
    if kind is None:
        return "probability"
    key = str(kind).strip().lower()
    resolved = _KIND_ALIASES.get(key)
    if resolved is None:
        raise DataError(
            f"unknown weights_type {kind!r}; expected 'probability' (design, sampling, "
            "inverse-probability or calibrated weights) or 'frequency' (counts)"
        )
    if resolved == "frequency":
        raise DataError(_FREQUENCY_MESSAGE.format(n=n))
    return "probability"


def _prepare_weights(
    weights: Any,
    n: int,
    *,
    weights_type: str | None,
    weights_estimated: bool,
    weights_name: str | None,
) -> tuple[FloatArray, WeightSpec]:
    """Validate a container's weights and retain their declared meaning and original scale."""
    label = weights_name or "weights"
    kind = resolve_weight_kind(weights_type, n)
    obs_weights = check_weights(weights, n, label)
    if weights is None:
        spec = WeightSpec(kind=kind, estimated=weights_estimated)
    else:
        # The private emitters account for this frame, preserving construction warning locations.
        _warn_if_counts(np.asarray(weights, dtype=float), label)
        _warn_if_concentrated(obs_weights, label)
        spec = WeightSpec(
            kind=kind,
            estimated=weights_estimated,
            name=label,
            scale=float(np.mean(np.asarray(weights, dtype=float))),
        )
    return obs_weights, spec


def warn_if_counts(weights: FloatArray, name: str) -> None:
    """Warn when weights supplied as probabilities look like counts.

    A frequency-weight vector passed without ``weights_type`` produces a valid estimate
    of the right parameter with a standard error that is too large by roughly
    ``sqrt(mean(counts))``.  That is a silent inefficiency rather than a crash, so it is
    worth naming when the evidence is strong: all values whole numbers, at least one, and
    averaging clearly above one.
    """
    _warn_if_counts(weights, name)


def _warn_if_counts(weights: FloatArray, name: str) -> None:
    w = np.asarray(weights, dtype=float).reshape(-1)
    if w.size == 0 or np.unique(w).size < 2:
        return
    if not np.all(np.abs(w - np.rint(w)) < 1e-9):
        return
    if float(w.min()) < 1.0 or float(w.mean()) < 1.5:
        return
    warnings.warn(
        f"{name} are whole numbers averaging {w.mean():.3g}, which looks like counts of "
        "identical units. They are being read as probability weights, which estimates the "
        "right parameter but with a standard error too large by roughly "
        f"sqrt({w.mean():.3g}) = {np.sqrt(w.mean()):.3g}. If they are counts, expand the "
        "rows instead; if they are sampling weights, ignore this. See "
        "cleverly.data.weighting.",
        WeightingWarning,
        stacklevel=4,
    )


def effective_sample_size(weights: FloatArray, *, on_degenerate: float | None = None) -> float:
    """Kish's effective sample size, ``(sum w)^2 / sum w^2``.

    The size of the unweighted sample carrying the same information.  It is the sample
    size the estimator's asymptotics are really working from, which is why it -- and not
    the row count -- is what ``g_bounds="auto"`` is evaluated at.

    ``on_degenerate`` says what to answer when the weights are empty or sum to zero, and
    the two kinds of caller want different things.  For a *fit* that state has no
    estimand at all, so the default is to raise.  For a *diagnostic* -- the per-arm and
    per-shift overlap reports, which are handed whatever subset of rows their arm or
    their mask selected -- an empty selection is a describable state and not an error, so
    those pass ``0.0`` and report it.

    This formula was written out six further times before it was one function, with three
    different answers in the degenerate case; the argument is here so the difference stays
    a stated choice rather than whichever copy a caller happened to reach for.
    """
    w = np.asarray(weights, dtype=float).reshape(-1)
    total = float(w.sum())
    if total <= 0:
        if on_degenerate is None:
            raise DataError("weights sum to zero, so there is no effective sample size")
        return on_degenerate
    return float(total**2 / float(np.square(w).sum()))


def warn_if_concentrated(weights: FloatArray, name: str) -> None:
    """Warn when the weights leave the fit resting on a small part of the sample.

    Reported unconditionally by :meth:`WeightReport.summary`; warned about here because
    the consequences are not confined to the report.  Below a quarter of the rows the
    effective sample size is small enough that it changes what the estimator *does* --
    ``g_bounds="auto"`` truncates harder, and the central limit theorem behind the
    interval has that many terms to work with, not ``n``.
    """
    _warn_if_concentrated(weights, name)


def _warn_if_concentrated(weights: FloatArray, name: str) -> None:
    w = np.asarray(weights, dtype=float).reshape(-1)
    if w.size == 0 or float(w.sum()) <= 0:
        return
    ess = effective_sample_size(w)
    design_effect = w.size / ess
    if design_effect <= CONCENTRATED_DESIGN_EFFECT:
        return
    warnings.warn(
        f"{name} are concentrated: the effective sample size is {ess:.0f} of {w.size} rows "
        f"(design effect {design_effect:.1f}). The estimate rests on that much of the "
        "sample, sample-size-dependent settings such as g_bounds='auto' are resolved from "
        "it, and the asymptotics behind the interval have that many terms. Inspect "
        "data.weight_report() before trusting the interval.",
        WeightingWarning,
        stacklevel=4,
    )


@dataclass(frozen=True)
class WeightReport:
    """How much the weighting costs, and what the estimand statement is.

    The headline number is Kish's effective sample size
    :math:`n_{\\text{eff}} = (\\sum w)^2 / \\sum w^2`, the size of the *unweighted* sample
    carrying the same information.  A weighted analysis of 5000 rows with
    :math:`n_{\\text{eff}} = 400` has 400 rows' worth of precision, and the design effect
    :math:`n / n_{\\text{eff}}` is the factor by which the variance is inflated.

    This is the observation-weight counterpart of the effective sample sizes in
    :mod:`cleverly.sensitivity.positivity`, which report the cost of the *clever
    covariate*.  The two multiply: a fit can be comfortable on each separately and thin
    on both together, which is why the positivity report folds the observation weights in
    rather than reporting the clever covariate alone.
    """

    kind: WeightKind
    estimated: bool
    name: str | None
    n: int
    effective_n: float
    design_effect: float
    scale: float
    coefficient_of_variation: float
    ratio: tuple[float, float]
    top_1pct: float
    top_5pct: float
    n_zero: int

    @property
    def is_weighted(self) -> bool:
        """``False`` for a constant weight vector, where nothing above applies."""
        return self.design_effect > 1.0 + 1e-9

    def to_dict(self) -> dict[str, Any]:
        """The report as a flat mapping, for logging or a dataframe row."""
        return {
            "kind": self.kind,
            "estimated": self.estimated,
            "name": self.name,
            "n": self.n,
            "effective_n": self.effective_n,
            "design_effect": self.design_effect,
            "scale": self.scale,
            "cv": self.coefficient_of_variation,
            "min_ratio": self.ratio[0],
            "max_ratio": self.ratio[1],
            "top_1pct": self.top_1pct,
            "top_5pct": self.top_5pct,
            "n_zero": self.n_zero,
        }

    def summary(self) -> str:
        """A printable statement of the estimand and the cost of the weighting."""
        if not self.is_weighted:
            return "Observation weights: constant, so the fit is the unweighted one."
        label = self.name or "weights"
        lines = [
            "Observation weights",
            "=" * 19,
            f"column: {label}; kind: {self.kind} weights, treated as "
            + ("estimated" if self.estimated else "fixed"),
            f"n = {self.n}, effective n = {self.effective_n:.1f} "
            f"(Kish), design effect = {self.design_effect:.2f}",
            f"weight range (relative to the mean) [{self.ratio[0]:.3g}, {self.ratio[1]:.3g}], "
            f"CV = {self.coefficient_of_variation:.2f}",
            f"largest 1% of weights hold {self.top_1pct:.1%} of the mass; "
            f"largest 5% hold {self.top_5pct:.1%}",
        ]
        if self.n_zero:
            lines.append(
                f"{self.n_zero} rows carry zero weight: they do not enter the estimate but "
                "do count towards n, which is correct under the i.i.d.-weights model"
            )
        lines.append("")
        lines.extend(estimand_lines(self))
        if self.design_effect > CONCENTRATED_DESIGN_EFFECT:
            lines.append("")
            lines.append(
                "VERDICT: the weights dominate. Effective sample size is under a quarter of "
                "the rows, so the estimate rests on a small part of the sample and the "
                "asymptotics behind the interval are working from that smaller number."
            )
        return "\n".join(lines)


def estimand_lines(report: WeightReport) -> list[str]:
    """The estimand statement, as lines for a report.

    Short by design -- the derivation lives in this module's docstring.  What has to
    appear next to any weighted number is *which* population it refers to and *what* the
    standard error conditions on.
    """
    if not report.is_weighted:
        return []
    lines = [
        "Estimand: the requested causal parameter in the weight-tilted population "
        "dP_w = w dP / E[w], estimated by TMLE on the weighted empirical measure "
        "(weighted nuisance fits, weighted targeting, weighted plug-in).",
        "Standard errors: influence curve (w / E[w]) * D*(P_w), i.e. the efficient "
        "influence function of that parameter when the weights are observed data.",
    ]
    if report.estimated:
        lines.append(
            "Weights were declared estimated: the interval conditions on the fitted "
            "weights. For weights fitted by maximum likelihood in a correct selection "
            "model this is conservative for the full-population parameter; for "
            "calibrated, raked or trimmed weights there is no general guarantee. Note "
            "that this package's bootstrap (n_bootstrap=) does not close the gap: it "
            "resamples rows and renormalises the weights it was handed, never re-deriving "
            "them, so its intervals condition on the fitted weights too. Closing it needs "
            "the weight model in the resampling loop, outside this package."
        )
    lines.append(
        "Complex designs: stratification and finite-population corrections are ignored "
        "(conservative); clustering is not, so a multi-stage design must declare its PSU "
        "with id=."
    )
    return lines


def describe_weights(
    weights: FloatArray,
    spec: WeightSpec,
) -> WeightReport:
    """Summarise a normalised weight vector against its :class:`WeightSpec`."""
    w = np.asarray(weights, dtype=float).reshape(-1)
    n = int(w.size)
    if n == 0:
        raise DataError("cannot describe an empty weight vector")
    total = float(w.sum())
    ess = effective_sample_size(w) if total > 0 else float("nan")
    mean = total / n
    return WeightReport(
        kind=spec.kind,
        estimated=spec.estimated,
        name=spec.name,
        n=n,
        effective_n=ess,
        design_effect=float(n / ess) if ess > 0 else float("inf"),
        scale=float(spec.scale),
        coefficient_of_variation=float(np.std(w) / mean) if mean > 0 else float("nan"),
        ratio=(float(w.min() / mean), float(w.max() / mean)) if mean > 0 else (float("nan"),) * 2,
        top_1pct=_top_share(w, 0.01),
        top_5pct=_top_share(w, 0.05),
        n_zero=int(np.count_nonzero(w == 0.0)),
    )


def _top_share(weights: FloatArray, fraction: float) -> float:
    """Share of the total weight held by the largest ``fraction`` of rows."""
    w = np.asarray(weights, dtype=float).reshape(-1)
    total = float(w.sum())
    if total <= 0:
        return float("nan")
    count = max(1, int(np.ceil(fraction * w.size)))
    return float(np.sort(w)[-count:].sum() / total)
