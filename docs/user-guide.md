# User guide

One runnable recipe per capability: what to pass, what comes back, and what the fit will
refuse. Every section is self-contained — read the one you need.

The derivations behind these numbers, the influence curves, and the tests that hold each
claim in place live in [the technical appendix](methodology.md). The
[README](../README.md) has the quickstart and the architecture.

## Multi-valued treatment

A treatment with more than two levels is estimated the same way as the binary fit in the
[quickstart](../README.md#quickstart), and reports one counterfactual mean per arm plus a
contrast against a reference arm you choose:

```python
from cleverly import TMLE
from cleverly.datasets import make_multi_arm

frame, truth = make_multi_arm(n=2000, seed=0)  # arms "low", "medium", "high"
res = TMLE(random_state=0, reference="low").fit(frame, outcome="Y", treatment="A").single()
print(res.summary())
```

```
n = 2000; covariates = 3; arm shares: high=0.364, low=0.284, medium=0.351
...
estimand            psi       std_err  95% CI                 p_value
------------------  --------  -------  ---------------------  -------
ate[high vs low]    1.3749    0.04624  [1.2843, 1.4656]       <1e-4
ate[medium vs low]  0.65619   0.05255  [0.55319, 0.75918]     <1e-4
ey[high]            1.3908    0.0414   [1.3097, 1.4719]       <1e-4
ey[low]             0.015852  0.04233  [-0.067105, 0.098808]  0.7080
ey[medium]          0.67204   0.0467   [0.58052, 0.76356]     <1e-4
```

(Population values for this process are `ey[low] = 0`, `ey[medium] = 0.6`,
`ey[high] = 1.44`.)

Parameters are named with your own labels, not the internal codes. Any contrast the
reference did not produce comes from the joint influence curve with no refit:

```python
res.contrast(lambda psi: psi[0] - psi[1], ["ey[high]", "ey[medium]"])
```

Without `reference=`, the reference is the lowest level in sort order — which for string
labels is alphabetical, so `{"low", "medium", "high"}` defaults to `"high"`. Pass
`reference=` whenever the ordering matters, which for an ordered treatment is always.

**The conditional effects generalise too, and are opt-in here.** "The effect among those
who actually received arm `a`" is one parameter per non-reference arm rather than one
number, so `estimands=("att", "atc")` reports `att[medium vs low]` — the effect among the
units that received `medium` — and `atc[medium vs low]`, the same contrast among the
reference arm's units. Every `atc` column conditions on the same population, which is what
"the controls" becomes when there are more than two arms; every `att` column conditions on
its own. The clever covariate is the binary one with `1{A = a}` and the odds `g_a / g_ref`
in place of the treated/untreated pair, so the reference arm's overlap is what these rest
on — `g_bounds="auto"` truncates them harder for that reason, as it always has.

They are **not** in a multi-arm fit's default report, and that is a decision about not
moving existing fits rather than a doubt about the parameter: they are `2(K-1)` further
parameters behind two further score equations, and a default that grew to include them
would change the simultaneous bands of every multi-arm fit that already ran. Ask for them,
or ask for `estimands="all"`. `result.contrast()` is not the alternative — a conditional
effect conditions on `A = a` and so is not a function of the marginal means at all.

**The sensitivity analyses are one per contrast.** Name the parameter and they answer for
the two arms it names:

```python
res.sensitivity.omitted_variable("ate[medium vs low]")
res.sensitivity.robustness_value("att[medium vs low]")
res.sensitivity.evalue("rr[medium vs low]")
res.sensitivity.missingness_tilt()  # every arm's mean, and every contrast
```

That is a wider loop rather than a wider derivation, and it is worth saying why: the
omitted-variable bound's `nu^2` is the second moment of *that parameter's* Riesz
representer, and the representer of `ate[medium vs low]` is two columns of the same
`K`-column clever covariate the fit already targeted. So each contrast gets its own bound,
its own robustness value and its own E-value, and none of them is a summary of the others.
The bare `"ate"` is not a parameter of such a fit, so asking for it lists the contrasts
that are. `res.sensitivity.report()` picks the first reported contrast when you do not
name one.

What is still refused rather than guessed at on a multi-valued treatment, **not written
yet** rather than unsound: `CTMLE` — both searches order candidates by one propensity
margin, and with `K` arms there is no canonical single ordering, which makes it the one
row with no settled answer rather than the one nobody has asked for.

A two-armed fit is unchanged in every respect, including the familiar `ate` / `att` /
`atc` / `ey1` / `ey0` names — the same numbers to the last bit, and `reference=` selects
which contrast a conditional effect reports exactly as it already selected which `ate`.
One exception, and it is a correction rather than a change: the MNAR tilt and the
E-value used to read arms `1` and `0` as constants, so on a two-armed fit that declared
`reference=1` they answered for the *other* contrast — the tilt for `E[Y¹] − E[Y⁰]` where
the fit reports `E[Y⁰] − E[Y¹]`, and the ATT among the treated where the parameter is
among the untreated. They now read the arms off the parameter, as everything else here
does.

## Dynamic and stochastic regimes

"Set `A` to 1 for everybody" is one intervention among many, and until you say otherwise
it is the one every estimand above assumes. `interventions=` says otherwise. A **regime**
is a conditional distribution over the treatment arms, `g*(a | W)`, and three kinds are
supported: a constant arm, a deterministic rule `d(W)`, and a stochastic assignment you
supply. The clever covariate generalises from `1{A = a} / g(a | W)` to the density ratio
`g*(A | W) / g(A | W)`, and the parameter from a mean per arm to a mean per regime.

```python
import numpy as np

from cleverly import TMLE
from cleverly.datasets import make_nonlinear_ate
from cleverly.interventions import Rule, Static, Stochastic

frame, truth = make_nonlinear_ate(n=2000, seed=0)
res = (
    TMLE(
        random_state=0,
        interventions=(
            Static(0, name="treat nobody"),
            Rule(lambda w: (np.asarray(w["W1"]) > 0).astype(int), name="treat if W1 > 0"),
            Stochastic(
                lambda w: np.column_stack([np.full(len(w), 0.7), np.full(len(w), 0.3)]),
                name="treat 30% at random",
            ),
        ),
        reference="treat nobody",
    )
    .fit(frame, outcome="Y", treatment="A")
    .single()
)
print(res.summary())
```

```
estimand                                            psi  std_err  95% CI
-----------------------------------------------  ------  -------  ------------------
ey_regime[treat nobody]                          2.0148  0.04816  [1.9204, 2.1091]
ey_regime[treat if W1 > 0]                       3.0747  0.05535  [2.9662, 3.1832]
ey_regime[treat 30% at random]                   2.4711  0.03870  [2.3952, 2.5469]
ate_regime[treat if W1 > 0 vs treat nobody]      1.0600  0.05220  [0.95766, 1.1623]
ate_regime[treat 30% at random vs treat nobody]  0.4563  0.02037  [0.41637, 0.49624]
```

A `Stochastic` regime's density must be a *known* function of `W` — a fixed design, not
one derived from the estimated mechanism. That restriction is the whole content of the
first refusal below.

A rule is handed the **covariates only**, in the backend you passed in, and returns the
treatment *level* to assign — your own label, not an internal code. Reading `Y` there is
not an intervention and reading `A` is a different object again, so the frame simply does
not carry them. Note that the columns are the *encoded* covariates, so a categorical
column appears as the indicators the data layer expanded it into.

Regimes replace the arm-indexed estimands rather than joining them: `interventions=`
declares what the fit's counterfactuals are, and asking one fit for both `ey1` and
`ey_regime` is refused. With static regimes the numbers are bit-for-bit those of an
ordinary fit, which is what `tests/unit/test_regimes.py` asserts.

Positivity means something different here, and has its own report:

```python
print(res.sensitivity.support().summary())
```

```
regime                       min g   max ratio   effective n  unsupported
-------------------------------------------------------------------------
treat nobody                 0.193        4.31        1022.4            0
treat if W1 > 0             0.3853       2.596        1163.8            0
treat 30% at random         0.1494       3.017        1711.3            0
```

A rule's positivity question is not "is `g` bounded away from zero" but "is it bounded
away from zero at the arm this rule assigns". Two fits with identical marginal overlap can
differ completely on that, and no arm-level table shows it.

One intervention is **not a regime, and has its own keyword**, because of the influence
function and not for want of effort:

| not here | where it went |
| --- | --- |
| incremental propensity-score interventions (`g*_δ = δg / (δg + 1 - g)`) | `g*` is a functional of `P`, so the EIF carries a further term for the pathwise derivative through `g` (Kennedy 2019) and the estimator must fluctuate the mechanism too. That is a parameter axis of its own: `incremental=`, see [Tilting the odds of treatment](#tilting-the-odds-of-treatment). Building one by hand as a `Stochastic` regime reports a standard error for a different functional, and a *smaller* one |

A modified treatment policy — shifting a *continuous* dose — is **not** an intervention in
this sense and does not go in `interventions=`. It reads the treatment a unit actually
received rather than assigning one from `W`, so it has its own keyword and its own score
equation; see [Shifting a continuous dose](#shifting-a-continuous-dose).

Why this is the right number, and how it is checked:
[the density-ratio covariate](methodology.md#regimes-the-density-ratio-covariate).

## Shifting a continuous dose

Every estimand above names an arm, and a dose has none. A **modified treatment policy**
names a change instead: `d(a, w) = a + δ`, held back at a cap `u`. `shifts=` declares one,
which also declares the treatment continuous — so the column keeps its own values rather
than being coded into arms, the mechanism becomes a conditional density `g(a | W)`, and the
clever covariate becomes a density *ratio* rather than an inverse probability:

```
h(a, W) = g(a - δ | W) / g(a | W) · 1{a ≤ u}  +  1{a > u - δ}
```

```python
from cleverly import TMLE
from cleverly.datasets import make_shift_dose
from cleverly.interventions import Shift

frame, truth = make_shift_dose(n=2000, seed=0)

res = (
    TMLE(
        shifts=[Shift(0.0, cap=None), Shift(0.5, cap=5.0)],
        density_bins=40,
        random_state=0,
    )
    .fit(frame, outcome="Y", treatment="A")
    .single()
)
print(res.summary())
```

```
n = 2000; covariates = 3; dose: mean 1.974, range [-2.94, 6.41]

estimand                           psi      std_err  95% CI              p_value
---------------------------------  -------  -------  ------------------  -------
ey_shift[natural course]           3.349    0.06501  [3.2216, 3.4764]    <1e-4
ey_shift[+0.5]                     4.1036   0.07395  [3.9586, 4.2485]    <1e-4
ate_shift[+0.5 vs natural course]  0.75459  0.02486  [0.70587, 0.80331]  <1e-4
```

`Shift(0.0, cap=None)` is the *natural course* — the policy that changes nothing. Its mean
is `E[Y]` exactly, not approximately, and declaring it first makes `ate_shift` read as
*the effect of shifting* rather than as a contrast of two arbitrary policies.

**The cap is declared, never estimated.** `cap=` is required and has no default. Fitting a
support boundary `u(w)` from the data would make the *parameter* depend on the data: the
reported standard error would condition on an estimated boundary, and every bootstrap
replicate would target a slightly different policy. Defaulting it to `max(A)` is worse,
pinning the estimand to an extreme order statistic. `cap=None` is allowed, means no cap,
and warns with the share of rows whose shifted dose leaves the observed range — because
there `Qbar` is being extrapolated.

**The density is a discrete hazard, not a new learner contract.** Adding `predict_density`
would have split `Learner` into two incompatible tiers and invalidated every preset. Instead
the dose is binned and `λ_b(W) = P(bin = b | bin ≥ b, W)` is a conditional probability of a
binary event, so one classifier on a long `(unit, bin)` expansion estimates all of them —
`treatment_learner=` unchanged, and the Super Learner's negative log-likelihood on the long
data *is* the discretised conditional log-likelihood. `density_bins=` sets the resolution,
and what the bins cost is reported rather than hidden: a shift much smaller than a bin
leaves the clever covariate at exactly one for most rows, so the intervention is invisible
rather than merely noisy, and the fit says so.

Positivity is a different question here, so it has a different report. `sensitivity.positivity()`
refuses a continuous fit — there is no per-arm propensity to tabulate — and what matters
instead is whether the *ratio* stays bounded:

```python
for report in res.sensitivity.shift_support().values():
    print(report.summary())
```

```
natural course: min g(A|W)=0.000483, max ratio=1, ESS=2000 (100.0% of n), capped=0.0%, unsupported=0
    ratio quantiles -- 1%: 1, 5%: 1, 50%: 1, 95%: 1, 99%: 1
+0.5: min g(A|W)=0.000483, max ratio=23, ESS=863 (43.1% of n), capped=2.4%, unsupported=0
    ratio quantiles -- 1%: 0.085, 5%: 0.337, 50%: 0.893, 95%: 2.54, 99%: 7.46
```

Why this is the right number, and how it is checked:
[why an MTP is not the regime it induces](methodology.md#shifting-a-continuous-dose-why-an-mtp-is-not-the-regime-it-induces).

### Missing outcomes, an intermediate, and weights on a dose

`delta=`, `intermediate=` and `weights=` all work here, and mean what they mean on an arm.
They were once refused together, on a reason that was wrong for all three — see
[the roadmap](roadmap.md#refusals-worth-lifting).

With `delta=` the clever covariate gains a factor, and *where* it is evaluated is the part
worth knowing:

```
H(a, W) = h(a, W) / π(a, W),     π(a, W) = P(Δ = 1 | A = a, W)
```

The fluctuation updates `Qbar` as a function of the dose, so obtaining `Qbar*(d(A,W), W)`
reads the mechanism **at the dose the policy assigns**, not at the one observed — exactly as
the arm path evaluates `π` at each counterfactual arm, where the `1{A = a}` indicator hides
it. `intermediate=` adds `P(Z = z | A, W)` on the same footing, and the report is then the
controlled direct effect *under the policy*, `E[Y^{d(A,W), z}]`.

```python
# `frame` as above, with a `Delta` column and `Y` missing wherever it is zero.
res = (
    TMLE(shifts=[Shift(0.0, cap=None), Shift(0.5, cap=5.0)], density_bins=40, random_state=0)
    .fit(frame, outcome="Y", treatment="A", delta="Delta")
    .single()
)

for report in res.sensitivity.shift_support().values():
    print(report.summary())
```

```
natural course: min g(A|W)=0.000483, max weight=12.2, min mechanism=0.0823, ESS=1365 (68.2% of n), capped=0.0%, unsupported=0
    weight quantiles -- 1%: 1.04, 5%: 1.09, 50%: 1.55, 95%: 4.71, 99%: 8.85
+0.5: min g(A|W)=0.000483, max weight=116, min mechanism=0.0823, ESS=306 (15.3% of n), capped=2.4%, unsupported=0
    weight quantiles -- 1%: 0.107, 5%: 0.417, 50%: 1.37, 95%: 10.3, 99%: 36.5
```

Note what the natural course now costs. Without `delta=` its ratio is one everywhere and its
effective sample size is exactly `n`; here the missingness alone takes a third of it, which is
the report saying that the two reweightings multiply.

Three consequences to have in mind.

**The double-robustness condition is about a product.** Consistency needs `Qbar` right **or**
the product `h · π` right — not either mechanism on its own. A perfectly estimated density
buys nothing when the missingness model is wrong, and errors in the two can cancel exactly.
`tests/unit/test_remainder_shift_cde.py` measures all three statements.

**The natural course is no longer `mean(Y)`.** Without `delta=` that identity is exact and is
the canary that `h` is one under the identity policy. With it, `ey_shift[natural course]` is
the MAR-identified `E[Y]`, and the mean over recorded rows is the wrong answer.

**`nuisance_bound=` is the only truncation on this axis.** `g_bounds=` does nothing here —
there is no per-arm propensity, and the density ratio is deliberately untruncated — so the
mechanisms are what `nuisance_bound=` protects and
`res.sensitivity.truncation_curve(mechanism=True)` is what sweeps it. `shift_support()` then
reports the whole weight `h / (π · q_z)` rather than the bare ratio, because the two
reweightings multiply.

`res.sensitivity.missingness_tilt()` is still refused on this axis, and says why: the tilt
re-mixes `Qbar` under a moved mechanism, a shift's plug-in is `Qbar` at the assigned dose,
and whether the tilted parameter is still the shift parameter has not been derived here.

`weights=` needs no such care, because a weight is not in the clever covariate at all. It
tilts the *population*: the estimand becomes the shift parameter at `dP_w = w dP / E[w]`,
every nuisance — the density included — is fitted by weighted loss, and the reported curve is
`w` times the whole bracket. Putting `w` in the covariate's denominator would divide the
estimating equation by the very tilt it applies.

One modelling caveat that is easy to miss. `missingness_learner=` falls back to
`treatment_learner=`, and on a continuous fit the missingness design's first column is the raw
dose — so a default fit with `library="glm"` models `logit π` **linear in the dose**. That is
the same limitation the outcome regression has here, now applying to a second nuisance; where
`π` is non-monotone in the dose, pass a flexible learner.

## Tilting the odds of treatment

Every intervention above replaces the treatment decision. An **incremental
propensity-score intervention** leaves it where it was and multiplies its *odds* by `δ`
(Kennedy 2019):

```
q_δ(1 | W) = δ g(W) / (δ g(W) + 1 - g(W)),     D_δ = δ g + 1 - g
```

"Make everyone `δ` times as likely to be treated as they already were." `δ = 1` changes
nothing — `q₁ = g` identically — so it is the natural course and the usual reference, the
way `Shift(0.0, cap=None)` is on the dose axis.

```python
from cleverly import TMLE
from cleverly.datasets import make_nonlinear_ate
from cleverly.interventions import Incremental

frame, truth = make_nonlinear_ate(n=2000, seed=0)
res = (
    TMLE(
        incremental=[Incremental(1.0), Incremental(2.0), Incremental(0.5)],
        random_state=0,
    )
    .fit(frame, outcome="Y", treatment="A")
    .single()
)
print(res.summary())
```

```
estimand                               psi       std_err   95% CI
-------------------------------------  --------  --------  --------------------
ey_ipsi[natural course]                2.8069    0.04286   [2.7229, 2.8909]
ey_ipsi[odds x2]                       3.0877    0.04352   [3.0024, 3.173]
ey_ipsi[odds x0.5]                     2.5217    0.04182   [2.4397, 2.6037]
ate_ipsi[odds x2 vs natural course]    0.28082   0.009157  [0.26287, 0.29877]
ate_ipsi[odds x0.5 vs natural course]  -0.28521  0.008459  [-0.30179, -0.26863]
```

`ey_ipsi[natural course]` is `mean(Y)` — not approximately, and not because the estimator
converged. At `δ = 1` the influence curve collapses to `Y - Ψ` row by row for *any*
nuisances, so the identity holds whatever the learners did. It is the sharpest diagnostic
in the package: it fails if the extra term below is dropped, mis-signed, or left
unsolved, and it costs nothing to check. It holds **absent `delta=`** only: with missing
outcomes `Ψ(1)` is the MAR-identified `E[Y]`, the curve is
`Δ/π·(Y − Q̄(A,W)) + Q̄(A,W) − Ψ`, and the complete-case mean is the wrong answer.

**No positivity assumption.** This is the reason the axis is worth having. Every other
estimand here divides by `g` somewhere and degrades as overlap fails; this one does not.
Its clever covariate is `δ/D` at `A=1` and `1/D` at `A=0`, and since `D` lies between
`min(1, δ)` and `max(1, δ)`, both stay inside

```
[min(δ, 1/δ),  max(δ, 1/δ)]
```

**whatever the mechanism does** — a bound the analyst chose, not one the data granted.
The `g` in numerator and denominator cancels algebraically, so the code never forms the
ratio and never divides by a small propensity. `g_bounds=` is refused rather than ignored,
because `g` is *inside* the estimand here and truncating it would move `Ψ(δ)` rather than
regularise a denominator; `truncation_curve()` is refused for the same reason.

```python
for report in res.sensitivity.incremental_support().values():
    print(report.summary())
```

```
natural course: min g(1|W)=0.0457, covariate in [1, 1] by construction, max=1, ESS=2000 (100.0% of n)
odds x2: min g(1|W)=0.0457, covariate in [0.5, 2] by construction, max=1.79, ESS=1806 (90.3% of n)
odds x0.5: min g(1|W)=0.0457, covariate in [0.5, 2] by construction, max=1.62, ESS=1818 (90.9% of n)
```

A propensity of 0.046 costs this fit 10% of its effective sample size. An `ey1` on the
same data divides by it.

Why this is the right number, and how it is checked:
[two score equations](methodology.md#tilting-the-odds-of-treatment-two-score-equations).

## Summarising the arms: a marginal structural model

Five dose levels and two effect modifiers report ten counterfactual means, which is a
table rather than an answer. `msm=` declares a **working model** `m(a, V; β)` that
summarises them, and makes the fit's parameters its coefficients:

```python
import numpy as np

from cleverly import TMLE
from cleverly.datasets import make_multi_arm
from cleverly.msm import MSM

frame, truth = make_multi_arm(n=2000, seed=0)  # arms "low", "medium", "high"

dose = {"low": 0.0, "medium": 1.0, "high": 2.0}
res = (
    TMLE(
        msm=MSM(
            design=lambda arm, w: np.column_stack([np.ones(len(w)), np.full(len(w), dose[arm])]),
            terms=("(intercept)", "dose"),
        ),
        outcome_learner="glm",  # so the numbers below are quick to reproduce; the
        treatment_learner="glm",  # default library estimates the same parameter
        random_state=0,
    )
    .fit(frame, outcome="Y", treatment="A")
    .single()
)
print(res.summary())
```

```
estimand          psi       std_err  95% CI                 p_value
----------------  --------  -------  ---------------------  -------
msm[(intercept)]  0.004723  0.03997  [-0.073611, 0.083057]  0.9059
msm[dose]         0.68987   0.023    [0.6448, 0.73494]      <1e-4
```

The population values are `β₀ = -0.04` and `β₁ = 0.72`: the least-squares fit of this
process's true arm means `(0, 0.6, 1.44)` on the dose, which is what `β` is *defined* to be.

**The working model does not have to be correct**, and for this process it is not — the
arm means are `0`, `0.6` and `1.44`, which is not a line. `β` is defined as a
*projection*, the minimiser of

```
E[ Σ_a h(a, V) ( E[Y(a) | V] − m(a, V; β) )² ]
```

over a **known** weight function `h`, so it is a well-defined functional whatever the true
dose-response looks like: "the best `m`-shaped summary of the counterfactual means, in the
`h`-weighted least-squares sense" (Neugebauer & van der Laan 2007). Where the model happens
to be right, `β` is the truth. Where it is wrong, the interval is still an honest interval
— for the projection, which is the thing that was estimated, and not for a misspecified
regression's coefficient.

`MSM.linear(modifiers=("W1",))` writes the common case for you — `β₀ + β₁a + β₂W₁ + β₃aW₁`
— and requires numeric arm labels, because a model linear in the arm reads it as a dose to
interpolate between. `{"low", "medium", "high"}` has no such ordering, and the sort order a
coding would fall back on is not one anybody chose, so it is refused rather than guessed;
the example above passes `design=` and says what the doses are.

### A link, and what it makes the coefficients mean

For a binary outcome the identity link above is a *linear-risk* model, and its coefficients
are risk differences — frequently out of range, and not what the applied literature
reports. `link="log"` and `link="logit"` put the linear predictor inside a mean function,
so a coefficient becomes a log risk ratio or a log odds ratio:

```python
from cleverly.datasets import make_binary_outcome

frame, truth = make_binary_outcome(n=2000, seed=0)

res = (
    TMLE(
        msm=MSM.linear(link="logit"),  # m(a) = expit(b0 + b1 a)
        outcome_learner="glm",
        treatment_learner="glm",
        random_state=0,
    )
    .fit(frame, outcome="Y", treatment="A")
    .single()
)
# exp(beta), with the interval exponentiated
print(res.coefficients(scale="ratio")[["estimand", "psi", "ci_lower", "ci_upper", "scale"]])
```

```
           estimand       psi  ci_lower  ci_upper       scale
0  msm[(intercept)]  0.589957  0.518567  0.671176    baseline
1            msm[a]  2.265138  1.904213  2.694474  odds ratio
```

Two arms and two terms is a *saturated* model, so `msm[a]` here is not an approximation of
anything: it is the marginal odds ratio, and `TMLE(estimands=("or",))` on the same data
reports `2.265138` with the same interval to the last digit. That is the check worth
running when a link looks suspicious.

`res.coefficients()` reports `β` on the link scale, which is what the fit estimated;
`scale="ratio"` reports `exp(β)`, with the Wald interval exponentiated from that scale and
the null moved from zero to one. The `scale` column names which ratio it is, because the
two are different numbers: **`exp(β)` is a risk ratio under `log` and an odds ratio under
`logit`**, and the *intercept* is neither — `exp(β₀)` is a baseline mean or a baseline
odds, so it is labelled `baseline` and its p-value tests `β₀ = 0` rather than any absence
of effect. The view is refused on an identity-link fit, where `exp` of a risk difference is
not a quantity.

Why this is the right number, and how it is checked:
[the projection, its matrix and its remainder](methodology.md#the-msm-projection-its-matrix-and-its-remainder).

## Treatment given over time

Everything above gives the treatment once. `LTMLE` gives it repeatedly, and estimates the
mean outcome under a **regimen** `ā = (a₁, …, a_T)` — a plan for every node, not a decision
at one. What makes this a different estimator rather than a wider loop is the covariate
measured *between* the decisions: `L₂` is caused by `A₁` and confounds `A₂`, so it is a
mediator and a confounder at once. Adjust for it and you block the part of `A₁`'s effect
that runs through it; leave it out and the second decision stays confounded. No single
adjustment set is right, which is the whole reason for the module.

```python
from cleverly import LTMLE
from cleverly.datasets import make_longitudinal

frame, truth = make_longitudinal(n=4000, seed=0)  # W1 W2 | A1 C1 | L2 A2 C2 | Y

res = LTMLE(
    {"always": 1, "never": 0},  # a scalar means that arm at every node
    reference="never",
    outcome_learner="glm",  # so the numbers below are quick to reproduce
    treatment_learner="glm",
    n_folds=5,
    random_state=0,
).fit(
    frame,
    outcome="Y",
    treatment=["A1", "A2"],  # the node ordering: one column per time point
    baseline=["W1", "W2"],
    time_varying=[[], ["L2"]],  # measured before that node's treatment
    censoring=["C1", "C2"],  # 1 = still under observation after that node
)
print(res.summary())
```

```
Longitudinal TMLE (2 time points, n = 4000)

parameter                     estimate  std. error  95% CI            p-value
----------------------------  --------  ----------  ----------------  -------
ey_regimen[always]            0.7685    0.0131      [0.7429, 0.7941]  <1e-4
ey_regimen[never]             0.3954    0.0198      [0.3565, 0.4342]  <1e-4
ate_regimen[always vs never]  0.3731    0.0236      [0.3268, 0.4193]  <1e-4

  time points: 2
  outcome family: binomial
  regimens: always=(1/1), never=(0/0)
  reference: never
  cross-fitting: 5 fold(s)
  g_bounds: [0.009532, 0.9905] on the treatment and censoring mechanism at every node
  confidence level: 95%
  ...
  always: 1263 of 4000 units followed it throughout; max weight 12.0, effective n 1096
  never: 822 of 4000 units followed it throughout; max weight 34.6, effective n 630

  simultaneous 95% bands (multiplier bootstrap, critical value 2.329 vs 1.960 pointwise):
    ey_regimen[always]  [0.7381, 0.7989]
    ey_regimen[never]  [0.3492, 0.4415]
    ate_regimen[always vs never]  [0.3182, 0.4280]
```

The truth for the contrast is `0.3616` (`ey_regimen[always]` is `0.7804` and
`ey_regimen[never]` is `0.4189`), known by quadrature rather than by simulation: under
the intervention the mechanism drops out and what is left is a three-dimensional Gaussian
integral of the outcome regression. `tests/unit/test_datasets_longitudinal.py` checks
that quadrature against plain Monte Carlo and against a refinement of its own rule, since
every accuracy claim in this section rests on it.

The bands at the foot are joint across the three reported parameters, as they are on a
point-treatment fit: a fit declaring `R` regimens reports `R` means and `R − 1` contrasts
built from the same influence curves, which is the situation a simultaneous band is for.
Pass `simultaneous=False` to skip them.

**Positivity is the assumption that bites**, and it does so differently here. The clever
covariate divides by a *cumulative* product of `2T` probabilities, so a mechanism that
looks harmless node by node still leaves a handful of units carrying most of the weight —
above, `never` is supported by 822 units whose effective sample size is 630. Each factor is
truncated *before* multiplying rather than the product afterwards, so one near-deterministic
node cannot be rescued by the others; `res.diagnostics()` reports the weight and the
effective `n` per regimen per node, beside that node's `epsilon` and whether it converged.

**Observation weights** are read here exactly as they are at one time point. Pass
`weights="w"` to `.fit()` and the estimand becomes the same regimen parameter in the tilted
population `dP_w = w dP / E[w]`: every node's mechanism, every node's censoring factor and
every regression in the backward recursion is fitted by weighted loss, each node's
fluctuation solves the weighted score `Σ w h_t (Z_t − Q̄*_t) = 0`, the plug-in is a weighted
average, and the reported curve is `(w / E[w]) · D*(P_w)`. A weight is a tilt of the
*population*, so it is **not** a factor in `h_t` — the denominator is still the `2T`
mechanism factors and nothing else. `res.diagnostics()` reports the leverage of `w / ∏g`
rather than of `1/∏g`, since the two reweightings multiply, and `g_bounds="auto"` is
resolved at the effective `n` for the reason
[the weights section](#observation-weights-and-which-population-they-define) gives — which
bites harder here, because that bound reaches all `2T` factors.
`tests/unit/test_weighted_estimand_longitudinal.py` checks the estimand and the curve
against a longhand `Ψ(P_w)` on the exact law, including for a weight reading the treatment,
the censoring indicator, the time-varying confounder and the outcome — where the tilt moves
every one of those nuisances rather than only the covariate marginal.

Nothing here shares the point-treatment estimator's target registry, and that is deliberate:
a `Target` is indexed by an arm, a regime, a shift, a tilt or an MSM coefficient, and a
regimen is none of those. What *is* shared is everything below the estimand — cross-fitting,
the Super Learner, the logistic fluctuation and its failure diagnostics, the influence-curve
variance, the delta method, the multiplier bootstrap and the cluster-robust variance — so
`res.contrast()`, `res.covariance()`, `res.simultaneous` and `id=` work exactly as they do
on a point-treatment fit, and `res.to_frame()` uses the same column names.

`res.sensitivity`, `res.validation` and `res.save()` are the three that do *not*, and each
says so with what it would need rather than an `AttributeError`. For positivity — the
assumption that bites hardest here — `res.diagnostics()` is the answer: it reports the
cumulative weight and effective `n` per regimen per node, which is the leverage the
product of `2T` factors actually produces.

`cleverly.validation.CoverageStudy` does take an `LTMLE`: `make_longitudinal` follows the
`(n, seed) -> (frame, truth)` convention and keys its truth by the names a fit reports, so
a coverage study over regimens needs no adapting.

Why this is the right number, and how it is checked:
[the sequential regression](methodology.md#treatment-given-over-time-the-sequential-regression).

### A regimen that reads the history

A plan may decide a node rather than declare it. Any entry of a plan may be a **rule**
`d_t(H_t)` instead of an arm, so "start everyone, then keep treating only the responders"
is one regimen rather than a special case:

```python
res = LTMLE(
    {
        "always": 1,
        "never": 0,
        # An entry is an arm or a rule; mixing them is the ordinary case. A rule gets the
        # history in the backend you passed in and may return that backend's booleans.
        "continue if L2 > 0": (1, lambda h: h["L2"] > 0),
    },
    reference="never",
    outcome_learner="glm",
    treatment_learner="glm",
    n_folds=5,
    random_state=0,
).fit(
    frame,
    outcome="Y",
    treatment=["A1", "A2"],
    baseline=["W1", "W2"],
    time_varying=[[], ["L2"]],
    censoring=["C1", "C2"],
)
```

```
parameter                                 estimate  std. error  95% CI            p-value
----------------------------------------  --------  ----------  ----------------  -------
ey_regimen[always]                        0.7685    0.0131      [0.7429, 0.7941]  <1e-4
ey_regimen[never]                         0.3954    0.0198      [0.3565, 0.4342]  <1e-4
ey_regimen[continue if L2 > 0]            0.7201    0.0146      [0.6914, 0.7488]  <1e-4
ate_regimen[always vs never]              0.3731    0.0236      [0.3268, 0.4193]  <1e-4
ate_regimen[continue if L2 > 0 vs never]  0.3247    0.0245      [0.2768, 0.3726]  <1e-4

  regimens: always=(1/1), never=(0/0), continue if L2 > 0=(1/d)
    a 'd' is a rule d_t(H_t) read off [W, L_1, ..., L_t]; its followers are a
    covariate-dependent set, so the counts below describe this sample
    assigned arms, continue if L2 > 0: 4e3adffa150fba33
```

A `d` says a rule was declared and not *which* rule, and two rules are two parameters —
so the line under it digests the `(n, T)` arms the rule actually assigned this sample.
That is the only stable fingerprint a closure has, and it is what makes `1{L2 > 0}` and
`1{L2 > 5}` distinguishable in a saved report; a rule written as a `def` is additionally
named, as `d:responders`. A static plan gets no such line, because `1/1` already says
everything there is to say about it.

**A rule is handed `[W, L₁, …, L_t]` and nothing else**, in the backend you passed in.
Not the outcome — reading it is not an intervention — and not the earlier treatments,
because under the regimen those are whatever the rule itself assigned, so a rule reading
the *observed* `A₁` would be reading the treatment of a unit that deviated. It returns one
arm per row and must be a deterministic function of that frame: every rule is called
exactly once, before any nuisance is fitted, so that the follower masks and the designs the
mechanism was evaluated at cannot disagree about what the regimen assigned.

What actually changes is the follower set. A constant plan's followers are a fixed slice;
a rule's are **a different, covariate-dependent set at every node**, so the rows each
sequential regression is fitted on move with the data. That is why the settings report
marks a rule with `d` rather than an arm, and why `res.diagnostics()` carries
`share_assigned_1` — the fraction of the units at risk at that node the regimen would
treat, which for a constant is exactly 0 or 1 and for a rule is the only place the report
says what the rule did to *this* sample:

```
           regimen  time  n_followed  share_assigned_1  max_weight  effective_n
            always     2        1263          1.000000       11.97      1096.17
             never     2         822          0.000000       34.55       630.06
continue if L2 > 0     2        1205          0.810355       14.52       970.26
```

Why this is the right number, and how it is checked:
[what the oracle law checks](methodology.md#dynamic-rules-what-the-oracle-law-checks).

### Summarising the regimens: a marginal structural model

Four plans over two nodes is a table; `2^T` plans over `T` nodes is not a report at all.
`msm=` declares a **working model** `m(ā, V; β)` summarising the regimens and makes the
fit's parameters its coefficients — the same move [`msm=` makes at one
node](#summarising-the-arms-a-marginal-structural-model), and the standard way the applied
literature reports a grid of dynamic rules: a coefficient on the rule's threshold rather
than a mean per plan.

```python
import numpy as np

from cleverly import LTMLE
from cleverly.datasets import make_longitudinal
from cleverly.msm import MSM

frame, truth = make_longitudinal(n=4000, seed=0)

# How long each plan treats for -- the summary the coefficient is per unit of.
months = {"never": 0.0, "late": 1.0, "early": 1.0, "always": 2.0}

res = LTMLE(
    {"never": 0, "late": (0, 1), "early": (1, 0), "always": 1},
    msm=MSM(
        # A design is handed the regimen's label, the horizon, and the baseline
        # covariates -- never a time-varying one, which would condition on a
        # consequence of the first node's arm.
        design=lambda plan, horizon, w: np.column_stack(
            [np.ones(len(w)), np.full(len(w), months[plan])]
        ),
        terms=("(intercept)", "months treated"),
    ),
    outcome_learner="glm",  # so the numbers below are quick to reproduce
    pseudo_learner="glm",
    treatment_learner="glm",
    n_folds=5,
    random_state=0,
).fit(
    frame,
    outcome="Y",
    treatment=["A1", "A2"],
    baseline=["W1", "W2"],
    time_varying=[[], ["L2"]],
    censoring=["C1", "C2"],
)
print(res.summary())
```

```
parameter                    estimate  std. error  95% CI            p-value
---------------------------  --------  ----------  ----------------  -------
msm_regimen[(intercept)]     0.4088    0.0170      [0.3755, 0.4421]  <1e-4
msm_regimen[months treated]  0.1865    0.0118      [0.1634, 0.2097]  <1e-4
```

The population values are `β₀ = 0.4253` and `β₁ = 0.1808`: the least-squares fit of this
process's true regimen means `(0.4189, 0.5811, 0.6441, 0.7804)` on months treated, which is
what `β` is *defined* to be.

**The working model does not have to be correct**, and for this process it is not — `late`
and `early` treat for the same length of time and do not have the same mean. `β` is a
**projection**, the minimiser of

```
E[ Σ_c h(c, V) ( E[Y^c | V] − m(c, V; β) )² ]
```

over a **known** weight function `h`, so it is well defined whatever the true response to
duration looks like (Neugebauer & van der Laan 2007; Orellana, Rotnitzky & Robins 2010).
Where the model happens to be right, `β` is the truth. Where it is wrong, the interval is
still an honest interval — for the projection, which is the thing that was estimated, and
not for a misspecified regression's coefficient.

`V` is a subset of the **baseline** covariates, and that is the estimand's own statement
rather than a convenience: `m(ā, V; β)` summarises `E[Y^ā | V]`, so a design reading `L₂`
would be conditioning on a consequence of `A₁` — a different parameter with a different
identification. The design is simply handed `[W]` and nothing else, the way a dynamic rule
is handed the history and nothing else.

On a **survival** fit the horizon is *inside* the design — `design(label, horizon, W)` —
rather than beside it, so one coefficient vector spans the whole `(regimen, horizon)` grid
and a term in `t` is a trend across horizons. A design saturated in the horizon reproduces
the per-horizon coefficients exactly and adds their joint covariance, so this contains the
per-horizon report rather than replacing it. A **cause** is not a further column: each
cause is its own estimand with its own projection, sharing every nuisance fit exactly as the
per-regimen recursion already shares them.

`reference=` is refused with `msm=`: a working model reports coefficients rather than
contrasts, what an intercept is taken against is whatever the design makes it, and a
difference of two coefficients is `res.contrast()`. `MSM.linear` is refused too — it reads
the label it is handed as a dose to interpolate between, which a treatment arm can be and a
plan cannot.

Why this is the right number, and how it is checked:
[pooling and rank](methodology.md#a-working-model-over-regimens-pooling-and-rank).

### A survival outcome

Everything above has one outcome, at the end. Pass **one event indicator per node**
instead and the outcome joins the time ordering — `W → A₁ → C₁ → Y₁ → L₂ → A₂ → C₂ → Y₂` —
the event is absorbing, and the parameter becomes a **curve**: the cumulative risk
`F_ā(t) = P(event by t under ā)` at every horizon.

```python
from cleverly import LTMLE
from cleverly.datasets import make_longitudinal_survival

frame, truth = make_longitudinal_survival(n=4000, seed=0)  # W1 W2 | A1 C1 Y1 | L2 A2 C2 Y2

res = LTMLE(
    {"always": 1, "never": 0},
    reference="never",
    outcome_learner="glm",
    pseudo_learner="glm",
    treatment_learner="glm",
    n_folds=5,
    random_state=0,
).fit(
    frame,
    outcome=["Y1", "Y2"],  # a list, not a name: that is what declares survival
    treatment=["A1", "A2"],
    baseline=["W1", "W2"],
    time_varying=[[], ["L2"]],
    censoring=["C1", "C2"],
)
print(res.summary())
```

```
parameter                           estimate  std. error  95% CI              p-value
----------------------------------  --------  ----------  ------------------  -------
risk_regimen[always @ t=1]          0.1543    0.0085      [0.1377, 0.1709]    <1e-4
risk_regimen[always @ t=2]          0.3105    0.0120      [0.2870, 0.3340]    <1e-4
risk_regimen[never @ t=1]           0.2440    0.0107      [0.2230, 0.2650]    <1e-4
risk_regimen[never @ t=2]           0.4744    0.0164      [0.4423, 0.5066]    <1e-4
ate_regimen[always vs never @ t=1]  -0.0897   0.0136      [-0.1163, -0.0631]  <1e-4
ate_regimen[always vs never @ t=2]  -0.1639   0.0201      [-0.2032, -0.1246]  <1e-4

  outcome: survival, event indicator at Y1, Y2
  horizons reported: t = 1, 2
  ...
  simultaneous 95% bands (multiplier bootstrap, critical value 2.552 vs 1.960 pointwise):
    risk_regimen[always @ t=1]  [0.1327, 0.1759]
    risk_regimen[always @ t=2]  [0.2799, 0.3412]
    ...
```

The truths are `0.1497` and `0.3172` for `always`, `0.2579` and `0.4552` for `never`,
again by quadrature rather than simulation. The bands at the foot are the point of
reporting a curve rather than `T` separate fits: `R × T` risks and `(R − 1) × T` contrasts
come out of one joint influence-curve matrix, so a band over the whole curve is the
interval a reader of a curve actually wants, and `res.covariance()` and `res.contrast()`
reach the same matrix for anything else — a restricted mean, a ratio of risks, a
difference between two horizons.

One consequence worth stating plainly:

- **Each horizon is its own backward pass**, so a curve costs `T(T+1)/2` regressions per
  regimen rather than `T`. The mechanism is fitted once and shared, which is where the
  cost would otherwise be. At two or three nodes this is not worth thinking about; over a
  monthly panel it is, so `horizons=(6, 12)` names the ones you will report.

Survival is a derived view rather than a second set of rows. `res.curve()` gives the tidy
frame with a `time` column, on either scale:

```python
res.curve(scale="survival")  # S(t) = 1 - F(t), and -(F_a - F_b) for a contrast
```

The two maps are different and only one of them is `1 - x`: applying that to a risk
*difference* would report `1 - RD`, which is not a quantity, with an interval that reads
perfectly plausibly. `curve()` branches on the parameter's scale rather than on the caller
getting it right. `to_frame()` is unchanged and still carries the column names a
point-treatment fit reports — the horizon lives inside `estimand` there, and gets a column
of its own only in `curve()`.

A horizon at which no event was observed among a regimen's followers is refused by name
rather than handed to a classifier with one class in it — reachable on real data at a late
node with a thin risk set.

Why this is the right number, and how it is checked:
[which population each node is fitted on](methodology.md#survival-which-population-each-node-is-fitted-on).

### Competing risks

An event that ends follow-up need not be the only one that can. Pass a **mapping of cause
to its indicator column per node** and each absorbing state gets its own curve — the
cause-specific cumulative incidence `F_j(t) = P(leave through cause j by t)`:

```python
from cleverly import LTMLE
from cleverly.datasets import make_longitudinal_competing

frame, truth = make_longitudinal_competing(n=4000, seed=0)  # W1 W2 | A1 C1 R1 D1 | L2 A2 C2 R2 D2

res = LTMLE(
    {"always": 1, "never": 0},
    reference="never",
    outcome_learner="glm",
    pseudo_learner="glm",
    treatment_learner="glm",
    n_folds=5,
    random_state=0,
).fit(
    frame,
    # A mapping, not a list: that is what declares competing risks.
    outcome={"relapse": ["R1", "R2"], "death": ["D1", "D2"]},
    treatment=["A1", "A2"],
    baseline=["W1", "W2"],
    time_varying=[[], ["L2"]],
    censoring=["C1", "C2"],
)
print(res.summary())
```

```
parameter                                    estimate  std. error  95% CI              p-value
-------------------------------------------  --------  ----------  ------------------  -------
cif_regimen[always, relapse @ t=1]           0.1227    0.0076      [0.1078, 0.1377]    <1e-4
cif_regimen[always, relapse @ t=2]           0.2526    0.0118      [0.2294, 0.2757]    <1e-4
cif_regimen[always, death @ t=1]             0.0315    0.0042      [0.0233, 0.0398]    <1e-4
cif_regimen[always, death @ t=2]             0.0667    0.0064      [0.0541, 0.0793]    <1e-4
cif_regimen[never, relapse @ t=1]            0.1484    0.0090      [0.1307, 0.1662]    <1e-4
...
ate_regimen[always vs never, relapse @ t=2]  -0.0103   0.0185      [-0.0465, 0.0259]   0.5760
ate_regimen[always vs never, death @ t=2]    -0.1096   0.0141      [-0.1373, -0.0819]  <1e-4

  outcome: competing risks, absorbing causes relapse, death at R1, R2, D1, D2
  horizons reported: t = 1, 2
  causes reported: relapse, death
```

The truths are `0.2467` and `0.0705` for `always`, `0.2443` and `0.2109` for `never`, at
`t = 2`. Treatment barely moves the incidence of relapse and roughly thirds the incidence
of death — which is the shape a competing-risks report exists to show, and the one a
single-event fit cannot: pooled into one absorbing event, these two would report a
substantial benefit without saying that all of it is on one cause.

**The causes are reported, not renormalised.** A unit leaves through exactly one cause or
none, so `Σ_j F_j(t) + S(t) = 1` — of the *parameters*. It does not hold of the estimates,
and that is not a bug to fix: each cause is its own backward pass with its own regressions
and its own fluctuation, so nothing constrains the sum and a total above one is possible.
`res.incidence_total()` reports the sum with its standard error and the excess over one,
on the same reasoning that keeps a multi-arm mechanism off the simplex rather than
rescaling it back on. Renormalising would buy a coherent-looking table by moving every
cause's estimate away from the one its own score equation solved, and would hide the thing
worth seeing — a total far from one says the causes disagree about how much risk there was,
which is a statement about the nuisance fits and not about the parameter.

The container checks what the declaration implies rather than assuming it: two causes may
not fire at one node, a unit that has left through one may not later be marked as having
had another, and each cause is absorbing in its own right. `res.curve()` gains a `cause`
column, and a cause with no observed events among a regimen's followers at some horizon is
refused by name — much more reachable for a rare cause than for a pooled event.

Why this is the right number, and how it is checked:
[the cause-specific recursion](methodology.md#competing-risks-the-cause-specific-recursion).

## Collaborative TMLE

A propensity model fitted to predict treatment as well as possible is fitted to the wrong
objective. A covariate that predicts treatment but *not* the outcome — an instrument —
removes no confounding, and putting it in `g` pushes propensity scores towards 0 and 1,
inflating the variance of `1/g` and so of the estimate. `CTMLE` chooses the covariates
entering `g` by cross-validating the loss of the *targeted outcome model* instead, so the
two nuisance fits only have to be right between them (van der Laan & Gruber 2010).

```python
from cleverly import CTMLE
from cleverly.datasets import make_instrument

# W1 confounds; W2 predicts treatment but not the outcome; W3 predicts only the outcome.
frame, truth = make_instrument(n=2000, seed=0)
res = (
    CTMLE(search="ordered", estimands=("ate",), outcome_learner="glm", treatment_learner="glm")
    .fit(frame, outcome="Y", treatment="A")
    .single()
)
print(res.extra["ctmle"].summary())
```

```
search = ordered; target = ate; criterion = cross-validated penalized squared-error loss

k  covariates in g  steps  risk     cv risk
0  (intercept)      1      7.19957  7.22364  <--
1  W1               2      7.20063  7.22640
2  W1, W3           2      7.20060  7.22598
3  W1, W3, W2       3      7.20876  7.24196

selected g: (intercept only)
left out: W1, W2, W3 -- adjusting for these would have cost more variance than the bias
they remove
```

Two things to read off. The instrument is last in the ordering and adding it costs the
largest jump in cross-validated risk of any covariate — that is instrument inflation,
priced. And the selector stops at the intercept: a GLM is correctly specified for the
outcome in this process, so `Qbar` has already done the adjusting and there is no
residual confounding left for `g` to remove. That is collaborative double robustness —
the two nuisance fits only have to be right between them — and it is why C-TMLE's
standard error here is about 25% below a plain TMLE's on the same samples.

`search="greedy"` (the default) builds the ordering by forward selection instead;
`search="ordered"` is the scalable variant (Ju et al. 2019) at `O(p)` propensity fits
rather than `O(p²)`; `search="discrete"` cross-validates an explicit list of candidate
models. One caveat worth knowing: the influence-curve standard error treats the selected
model as given, so it does not include the variability the selection itself contributes,
and is mildly anti-conservative as a result. Pass `n_bootstrap=` for inference that does
— each replicate re-runs the search.

Why this is the right number, and how it is checked:
[how the selection is evidenced](methodology.md#c-tmle-how-the-selection-is-evidenced).

## Doubly-robust inference

> **In progress.** `DRTMLE` is written and tested, and it is not finished. Its influence
> curve is transcribed from R's `drtmle` rather than derived, no number here has been
> compared against that package's, and no study in this repository shows the interval it
> reports is better than a plain TMLE's. The full list is in
> [the roadmap](roadmap.md#what-is-still-open); the short version is at the end of this
> section. Use it where you have a reason to think one nuisance is badly estimated, read
> `res.validation.score_check()` on every fit, and do not treat the interval as settled.

TMLE is **doubly robust for consistency and singly robust for inference**. The estimate stays
consistent if either nuisance is right, because the remainder is a *product* of the two
errors — but the interval needs that product to vanish faster than `1/sqrt(n)`, which takes
*both* of them converging. With one nuisance inconsistent the estimate is still fine and the
confidence interval quietly is not, and it degrades as the sample grows rather than
shrinking.

`DRTMLE` solves two further score equations, built from regressions of each nuisance's
residual on the *other* nuisance (van der Laan 2014; Benkeser, Carone, van der Laan &
Gilbert 2017). Those regressions are univariate however many covariates the fit adjusted
for, so they can be estimated fast enough whether or not the primary nuisances can.

```python
from cleverly import DRTMLE
from cleverly.datasets import make_nonlinear_ate

frame, truth = make_nonlinear_ate(n=2000, seed=0)
res = (
    DRTMLE(estimands=("ate",), outcome_learner="glm", treatment_learner="glm", random_state=0)
    .fit(frame, outcome="Y", treatment="A")
    .single()
)
print(res.validation.score_check())
```

```
Score-equation check
--------------------
target            kind             |score|    before     threshold  ratio     ok
----------------  ---------------  ---------  ---------  ---------  --------  ---
mean              fluctuation      7.237e-18  3.209e-09  1.527e-06  4.74e-12  yes
mean (mechanism)  fluctuation      9.079e-11  8.801e-11  1.527e-06  5.95e-05  yes
mean (reduced)    fluctuation      3.607e-11  4.900e-05  1.527e-06  2.36e-05  yes
ate               influence curve  9.424e-10  -          1.527e-06  6.17e-04  yes

PASS: the targeting step solved the estimated efficient score equation.
```

Two things to read off. There are **three** rows where a plain fit has one: the ordinary
outcome equation, a *mechanism* equation that fluctuates `g`, and a second outcome equation
against the reduced regressions. And what changed is the interval, not the estimate — on this
fit `ate` is 1.5348 against a plain TMLE's 1.5292, a twelfth of a standard error apart, while
the standard error moves from 0.06850 to 0.06828. **Read a `DRTMLE` fit as the same estimate
with an interval entitled to be believed under weaker conditions, not as a better estimate.**

`guard=` says which extra equations to solve, in `drtmle`'s vocabulary, and it is **crossed**:
`"Q"` guards against a misspecified *outcome regression* and adds the equation that fluctuates
`g`; `"g"` guards against a misspecified *mechanism* and adds the one that fluctuates `Qbar`.
Both by default; `guard=()` fits no reduced regressions at all and is bit-for-bit a plain
TMLE. `reduced_outcome_learner=` and `reduced_treatment_learner=` take the reduced
regressions' learners, defaulting to the primary ones.

It costs real time — two further learner fits per arm on every round of an alternation,
refitted *inside* the loop as the source does. One consequence is worth knowing: `retarget`
stops being arithmetic on cached arrays, so a truncation curve on a `DRTMLE` fit costs about
a fit per point rather than a fraction of one, and a result read back from disk cannot
retarget at all.

Scope is what the sources *derive*, which is narrower than what R's `drtmle` accepts: a
binary treatment and the `mean` group. A multi-valued treatment, `att`/`atc`, the other
parameter axes, `delta=`, `intermediate=`, fold-wise targeting,
`reduction="bivariate"` and composition with `CTMLE` are all refused by name.

**What is not visible from the output**, and is why this section opens with a warning. The
influence curve's form is read off `drtmle`'s implementation rather than derived — Theorem 1
of Benkeser et al. (2017) has not been read here, and if the two disagree the theorem wins.
There is no cross-check against `drtmle`'s own numbers. A coverage study on the off-diagonal
of the misspecification grid found *no gap for this variant to close* at the sizes it could
reach: the regime it is for needs an adaptive good nuisance converging more slowly than
`n^(-1/4)`, which is beyond what a nightly budget can simulate. And the alternation does not
reliably converge — equation (10)'s covariate is near-singular on exactly the fits anybody
wants, so some fold draws exit at the outer cap, which is what the score check is for.
[The roadmap](roadmap.md#what-is-still-open) lists these and the rest. Do not read this as a
free improvement over a plain TMLE.

Why this is the right number, and how it is checked:
[what the extra equations remove](methodology.md#doubly-robust-inference-what-the-extra-equations-remove).

## Cross-fitting and CV-TMLE

Three constructions, and it is worth knowing which one you are running:

| setting | estimator |
| --- | --- |
| `cross_fit=True, targeting_scheme="pooled"` (default) | cross-fitted TMLE |
| `targeting_scheme="fold"` | fold-targeted CV-TMLE |
| `targeting_scheme="fold", cv_evaluation=True` | canonical CV-TMLE |

```python
res = TMLE(targeting_scheme="fold").fit(frame, outcome="Y", treatment="A").single()
res.cv_targeting.summary()  # both reports side by side, per-fold psi and epsilon
res.cv_targeting.variance["ate"]
```

Which of the three ran is not left to be reconstructed from the settings —
`res.config.estimator_name` says it in words, and `res.summary()` prints it.

Why this is the right number, and how it is checked:
[what the folds do and do not buy](methodology.md#cross-fitting-what-the-folds-do-and-do-not-buy).

### What the folds guarantee

Two things a cross-fitted estimate assumes, and neither is left to trust. A fold index
outside the declared range, or a fold holding no rows at all, is refused by `Folds` when it
is built. A cluster with rows in more than one fold is refused by a post-condition that
`make_folds` runs on the way out, so it covers every split in the library — the outer folds,
Super Learner's inner folds, C-TMLE's selection folds — without any of them knowing about
it. A third prohibition needs no check: "every row is held out exactly once" has no
counterexample, because a split is one fold index per row and two-fold membership has no
representation.

The fold *policy* is recorded too, and separately from the split it produced:

```python
res.config.crossfit  # CrossFitPlan(n_folds=10, learner_folds=5, scheme=...)
res.config.crossfit.n_folds  # 10 -- what was asked for
res.config.n_folds  # 3  -- what ran, after the cluster count capped it
```

The two agree in the ordinary case. They come apart whenever `resolve_n_folds` capped the
count at the rarest stratum or `make_folds` capped it at the cluster count, and the warning
that said so is emitted at fit time and gone by the time anyone reads the result — so
`summary()` adds a line, and only then.

With a rare outcome, balancing the arms is not enough: eight events across ten
arm-stratified folds leaves at least one with none, and an outcome regression fitted there
is degenerate.

```python
TMLE(n_folds=10, stratify_folds="treatment+outcome")  # caps at the rarest cell, not arm
```

An unobserved outcome is its own stratum rather than being pooled with `Y = 0`, since a
fold with no *observed* outcomes in an arm cannot fit the regression either. The cost is
worth stating: this makes the fold assignment a function of the outcome, and the
cross-fitting argument conditions on the split. That is a statement about which splits are
conditioned on rather than a bias — and what it is weighed against is a fold that cannot fit
the regression at all. Binary outcomes only; a continuous outcome and a continuous dose are
both refused by name.

A single split is one draw from a randomised procedure, and two seeds can move `psi` by an
appreciable fraction of its standard error. `repeats=` draws the split several times and
averages:

```python
TMLE(n_folds=10, repeats=5)  # psi_bar = mean_r psi_r, at five times the cost
```

Every row is out of fold in every draw, so `mean_r psi_r` is the same functional of the
same data with influence curve `mean_r IC_r` — and because the variance, the delta method,
the cluster-robust standard error and the simultaneous bands are all computed *from* the
curve, they stay coherent without a second rule. The aggregation is the mean and only the
mean: the median-of-estimates form common in the DML literature is refused, because the
median of the `psi_r` is not the estimator whose influence curve is the median of the
`IC_r`, so its interval would be describing something other than its point estimate.

What it buys and what it costs were both measured rather than assumed
(`tests/e2e/test_coverage_slow.py`). Holding the data fixed and varying only the fold seed,
five draws cut the spread of `psi` from 0.0132 to 0.0065 — a ratio of 0.49 against the 0.45
that fully independent draws would give, so fold noise behaves very nearly as an independent
component. The cost is that the interval turns mildly conservative: `se_ratio` rises from
about 1.0 to about 1.12, because the influence-curve standard error targets the *sampling*
variance and never included fold noise, while the observed spread it is compared against
did — and averaging is precisely what takes that out. Coverage is unaffected.

A draw redraws *every* split, not only the outer one: the inner cross-validation that
scores Super Learner candidates and C-TMLE's selection folds are drawn from that draw's own
seed, so what is averaged over is the randomised procedure rather than one stage of it.

`result.repeats` holds each draw's nuisance fits, fluctuations and point estimates as a
unit, since a draw's targeted `Q̄` and its mechanism are not interchangeable with another's.
Everything that produces a number follows all of them — the truncation curve, the MNAR tilt,
the omitted-variable bound, the bootstrap, and C-TMLE's propensity selection; the
diagnostics that describe a fitted *mechanism*, where averaging would report a model no
draw fitted, describe the first draw and say which. `result.repeat_spread()` reports
`sd(psi_r)` across the draws beside the standard error, which is how much the fold
assignment moved the answer — a diagnostic of split noise, and neither a substitute for the
influence-curve standard error nor something to add to it.

`cv_evaluation=True` combines with it, with one thing that cannot follow the curve. The
cross-validated variance is defined by a fold partition, and the across-draw average curve
belongs to none of the `R`, so what is reported is the mean of the `R` cross-validated
variances, each computed on its own draw's partition:

```python
TMLE(targeting_scheme="fold", cv_evaluation=True, repeats=5)
```

Each of those is consistent for `Var(D*)/n`, which is what `Var(psi_bar)` converges to as
well, and in finite samples the mean of them errs conservative — `Var(psi_bar) ≤ mean_r
Var(psi_r)` by Cauchy–Schwarz and then Jensen — which is the direction anyone asking for the
cross-validated variance wants. At `R = 1` it is Zheng & van der Laan's construction
unchanged. The alternative that looks more natural, handing the averaged curve to
`cross_validated_variance` under one draw's partition, is not merely arbitrary but vacuous:
at equal fold sizes the fold-averaged second moment equals `mean(IC²)` for *every*
partition, so the fold structure contributes nothing and the result would be the pooled
uncentred second moment wearing a cross-validated name.

Fold-specific targeting is only the first of canonical CV-TMLE's three parts; the others
are fold-wise evaluation of the parameter and the cross-validated variance. By default
this estimator does neither — the fold-targeted predictions are stitched together and the
pooled plug-in and pooled standard error are reported. For `ate`, `ey1` and `ey0` that is
the same number, since they are linear in the targeted predictions. For `rr`, `or`, `att`
and `atc` it is not: a ratio of means is not a mean of ratios, and the pooled conditional
effects weight by the whole sample's arm share rather than each fold's.

```python
res = (
    TMLE(targeting_scheme="fold", cv_evaluation=True)
    .fit(frame, outcome="Y", treatment="A")
    .single()
)
res["rr"].psi  # averaged over folds (on the log scale) rather than pooled
res["rr"].std_error  # the cross-validated standard error
res.cv_targeting.pooled["rr"], res.cv_targeting.canonical["rr"]  # both, always
```

## Observation weights, and which population they define

Passing `weights=` changes the *estimand*, not just its weighting. The nuisances are fitted
by weighted loss, the targeting step solves the weighted score equation, and the plug-in is
a weighted average — the whole fit runs on the weighted empirical measure. So what comes
back is the requested causal parameter evaluated in the tilted population
`dP_w = w dP / E[w]`, and its efficient influence function is `(w / E[w]) * D*(P_w)`, which
is what the reported standard errors are built from.

```python
res = (
    TMLE()
    .fit(
        frame,
        outcome="Y",
        treatment="A",
        weights="sampling_weight",  # 1 / P(selected | observed variables)
        weights_estimated=False,  # True if they came out of a fitted model
        id="psu",  # a multi-stage design must declare its PSU
    )
    .single()
)
print(res.data.weight_report().summary())  # effective n, design effect, estimand statement
```

Two consequences of the weights that are easy to miss. The *variance* needs no special
handling — normalisation scales the surviving influence-curve values up by exactly the
factor the larger `n` divides out, so zero-weighting rows and deleting them give the same
standard error. But `g_bounds="auto"` does: `5 / (sqrt(n) log n)` is resolved at the Kish
effective sample size rather than the row count, since that is the information the
bias-variance compromise is working from (at a design effect of 4 the row count sets a
bound nearly 3x too loose). That is a deliberate divergence from R's `tmle`, applies only
to weighted fits, and is named in the summary where it takes effect. And `n_bootstrap=`
does **not** rescue estimated weights — every replicate inherits and renormalises the
weight column rather than re-deriving it, so those intervals condition on the fitted
weights too; the package says so at fit time rather than letting the mistake pass.

`LTMLE` takes `weights=` on exactly these terms — the same tilted-population estimand, the
same influence function, the same `g_bounds="auto"` divergence — with every node's nuisance
fitted by weighted loss and every node's score equation weighted; see
[treatment over time](#treatment-given-over-time).

That statement is derived and its limits set out in
[`cleverly/data/weighting.py`](../src/cleverly/data/weighting.py), and verified numerically
against a longhand statement of `Psi(P_w)` in `tests/unit/test_weighted_estimand.py` and
`tests/unit/test_weighted_estimand_longitudinal.py` — including for weights that depend on
the outcome, where the tilt changes `Qbar` itself. The short version:

| supplied weights | status |
| --- | --- |
| known sampling probabilities, selection depending only on observed data | supported: `dP_w` is the population law |
| complex survey design (strata, PSUs, FPC) | estimate supported; stratification and FPC are ignored (conservative), clustering is **not** — pass `id=` |
| outcome-dependent sampling with known fractions (case-control) | supported, if the fractions are genuinely known |
| estimated selection or non-response weights | intervals condition on `w`; conservative when `w` is an MLE of a correct selection model |
| calibration, raking, post-stratification, trimming | no general guarantee — bootstrap the weight derivation outside the package |
| frequency (count) weights | **refused**, with instructions: expand the rows |
| replicate weights (BRR, jackknife) | **refused**: a set of designs, not one weight vector |

## Sensitivity

```python
res.sensitivity.positivity()  # overlap, effective sample size, weight mass
res.sensitivity.truncation_curve()  # estimate vs propensity-truncation bound
res.sensitivity.truncation_curve(mechanism=True)  # ... vs the bound on P(Delta=1 | A, W)
res.sensitivity.omitted_variable(cf_y=0.03, cf_d=0.03)
res.sensitivity.robustness_value()  # confounding strength that would null the effect
res.sensitivity.benchmark(["W1", "W2"])  # calibrate cf_y/cf_d against observed covariates
res.sensitivity.evalue()  # VanderWeele-Ding E-value (binary outcomes)
res.sensitivity.missingness_tilt()  # MNAR exponential tilt (needs `delta=`)
```

The three confounding analyses take one parameter at a time, so on a multi-valued
treatment they take its reported name — `omitted_variable("ate[medium vs low]")` — and
answer for the two arms it names. See [multi-valued
treatment](#multi-valued-treatment).

**The MNAR tilt, and whether one gamma fits every arm.** The tilt displaces the unobserved
outcomes on the logit scale by `gamma` and mixes them back in at `1 - P(Delta = 1 | A, W)`,
so `gamma = 0` is the MAR analysis and the curve passes through the reported estimate. By
default one `gamma` moves every arm, which is an assumption and not an accident of the
two-armed case: it says the unobserved outcomes are displaced by the same amount whatever
treatment the unit received. Where that is doubtful — dropout after an ineffective arm need
not mean what dropout after an effective one does — `arm_gamma=` declares a *direction*
instead, and the grid sweeps its magnitude:

```python
# the unobserved outcomes are worse than they look under `low`, better under
# `medium`, and MAR under `high`
direction = {"low": 1.0, "medium": -1.0, "high": 0.0}

res.sensitivity.missingness_tilt([0.0, 0.5, 1.0], arm_gamma=direction)
res.sensitivity.tipping_gamma("ate[medium vs low]", arm_gamma=direction)
```

The tilt at arm `a` is then `arm_gamma[a] * gamma`, and the returned frame carries a
`gamma[<level>]` column per arm saying what each one received. Every arm must be named:
one left out would be tilted by the shared `gamma` after all, which is the assumption the
keyword exists to state rather than inherit. Any per-arm tilt vector is reachable this way
— pass it as the direction with `[1.0]` as the grid — and keeping the sweep
one-dimensional is what keeps `tipping_gamma` a single number: how far along *this*
departure the conclusion survives.

## Validation

```python
res.validation.nuisance()  # CV AUC/Brier/calibration for g, CV R^2/MSE for Q, SL weights
res.validation.score_check()  # did targeting solve mean(EIF) = 0?
res.validation.refute()  # placebo treatment, random common cause, subset stability
```

And a harness that measures the *estimator* rather than a fit, by repeated sampling from a
process whose truth is known:

```python
from cleverly.datasets import nonlinear_dgp
from cleverly.validation import CoverageStudy

study = CoverageStudy(
    dgp=nonlinear_dgp(),
    estimator=lambda: TMLE(estimands=("ate",)),
    n=1000,
    n_replicates=200,
    seed=0,
)
print(study.run().summary())  # bias, sqrt(n) bias, mc se, mean se, coverage, rejection rate
```

Why this is the right number, and how it is checked:
[what the score check proves, and what it does not](methodology.md#what-the-score-check-proves-and-what-it-does-not).

## Options with no section of their own

The settings above each earn a section. These do not, and are collected here rather than
left to be discovered in a signature.

| Capability | Notes |
| --- | --- |
| Outcome types | binary, and bounded continuous via Gruber & van der Laan (2010) scaling |
| Nuisance estimation | any scikit-learn estimator, or the built-in `SuperLearner` (ensemble + discrete). A treatment with more than two arms needs a conditional distribution over them: `SuperLearner` fits one binary ensemble per arm and normalises (one-vs-rest, documented as a modelling choice — nothing constrains `K` independently fit ensembles to sum to one), and any multiclass classifier is used directly |
| Targeting | iterative fluctuation (Newton) or one-step universal least-favorable submodel |
| Fluctuation | logistic or linear; clever covariate or weighted (`target_weights`, R's `target.gwt`) |
| Missing outcomes | `delta=` with its own nuisance model, entering the clever covariate. Assumes MAR given `(A, W)`; the double-robustness condition becomes "`Q̄` right **or** the product `g·π` right" |
| Controlled direct effect | `intermediate=` (R's `Z`) estimates `Ψ_z = E_W[E(Y \| A=a, Z=z, W)]` per level of `Z`, so the returned `TMLEResultSet` holds two results — index the level, `res[0.0]`, rather than calling `.single()`. Needs `Y(a,z) ⊥ Z \| A, W` on top of the usual assumptions — no intermediate confounder affected by `A` — and the DR condition becomes "`Q̄` right **or** the product `g·q_z·π` right". Not a longitudinal estimator and not a natural direct effect; `cleverly.estimators.direct_effect` writes the parameter down, derives its EIF, and states the boundary. Its influence curve is checked against the numerical Gateaux derivative at machine precision, on the same footing as the ATE |
| Clustering | `id=` for cluster-level influence-curve variance and cluster bootstrap |
| Bounds | propensity truncation (`g_bounds`), outcome bounds (`q_bounds`), `alpha` shrinkage. With two arms `g₁` is clipped and the control arm is its complement, so the pair sums to one. With more, each arm is clipped in its own right and the row is **not** renormalised — rescaling back onto the simplex can push a column under the floor and undo the only thing the floor is for. `PositivityReport.simplex_deviation` reports the size of that deliberate inconsistency; it cannot move `Ψ`, which contains no mechanism at all, only the second-order remainder |
| Screening | pre-screening of covariates for the treatment model (`prescreenW.g`, `min_retain`) |
| Inference | IC-based, cluster-robust, targeted bootstrap, multiplier bootstrap, delta method |
| Contrasts | `result.contrast(fn, names)` applies the delta method to the *joint* influence curve, and `result.covariance()` returns the joint covariance matrix. Pass `gradient=` when the derivative is known in closed form: the default central difference is accurate to ~1e-10 relative, which is fine for reporting and not enough to reproduce a closed-form influence curve at 1e-12 |
| Persistence | `result.save(path)` / `cleverly.load(path)` write arrays plus JSON into a versioned `.npz`. No pickle. Everything reached through `retarget` — positivity, truncation curves, the MNAR tilt, the omitted-variable bound, the score check, contrasts, the bootstrap — is bit-for-bit identical after a round trip; `refute()` and `benchmark()` genuinely refit and so need the learners to have been library specifications rather than fitted objects |
| Provenance | every result carries package versions, a fingerprint of the data and a *separate* fingerprint of the realised fold assignment — folds are not recoverable from a seed alone, since they also depend on row order and on the scikit-learn version that made them. Pass `run_id=` for your own identifier; no git commit is collected |
| Targeting diagnostics | `Fluctuation` records the score before as well as after targeting, the Hessian condition number, `epsilon` standard errors, the quasi-log-likelihood, and a named `failure` (`separation_suspected`, `bounds_pinned`, `singular_hessian`, …). Non-convergence in individual CV folds is reported rather than silently averaged away |

## Adding an estimand

Estimands live in a registry rather than in a `Literal`. A `Target` declares which
fluctuation solves its score equation, what scale its inference lives on, what it needs of
the outcome, how to build the estimate — and, as a required field, an `Identification`
record stating its assumptions, the nuisances it consumes and what double robustness buys
for *that* estimand specifically:

```python
from cleverly import Identification, Target, register


def number_needed_to_treat(ctx):
    one, zero = ctx.means[1.0], ctx.means[0.0]  # computed once, shared across the group
    difference = one.psi - zero.psi
    ic = one.influence_curve - zero.influence_curve
    # A *list*: one target is one functional, and a functional may report several
    # numbers. This one reports a single contrast, so the list has one entry.
    return [ctx.finish("nnt", 1 / difference, -ic / difference**2, "difference")]


register(
    Target(
        name="nnt",
        group="mean",  # shares the per-arm mean fluctuation
        scale="difference",
        build=number_needed_to_treat,
        requires_family="binomial",  # see below
        requires_binary_treatment=True,  # `means[1.0]` names an arm; see below
        identification=Identification(
            assumptions=("consistency", "no unmeasured confounding given W", "positivity"),
            required_nuisances=("outcome_regression", "treatment_mechanism"),
            dr_condition="consistent if either Qbar or g is consistent",
        ),
    )
)
```

`ctx.means` is a mapping keyed by **arm**, not a `(psi1, ic1, psi0, ic0)` tuple, because a
treatment may have more than two arms. A target that names particular arms — as this one
does, reaching for `1.0` and `0.0` — must declare `requires_binary_treatment=True`, and a
multi-arm fit then refuses it by name instead of quietly reporting a contrast of two arms
out of five. A target that works for any arm count instead loops `ctx.contrast_arms` and
names each parameter with `ctx.name_for`, which collapses to the bare stem when there are
exactly two arms so the familiar names survive:

```python
def risk_difference(ctx):
    reference = ctx.means[ctx.reference]
    return [
        ctx.finish(
            ctx.name_for("rd", arm, versus=ctx.reference),  # "rd", or "rd[high vs low]"
            ctx.means[arm].psi - reference.psi,
            ctx.means[arm].influence_curve - reference.influence_curve,
            "difference",
        )
        for arm in ctx.contrast_arms
    ]
```

`requires_family="binomial"` is doing real work here rather than decorating. `ctx.means`
are on the *scaled* outcome scale, and `ctx.finish` maps back with the linear rule the
declared scale implies — exact for a functional linear in those means, which is every
built-in estimand, but not for a reciprocal: `1 / (range · x) ≠ range · (1 / x)`. Declaring
the target binary-only pins the scaler to the identity, where the question does not arise.
A nonlinear target on a continuous outcome must unscale the means itself.

The variance, confidence intervals, simultaneous bands, delta method, score diagnostic and
bootstrap then work without further changes.

`parameter_axis=` says what the new target's parameters are indexed *by* — a treatment arm,
a regime declared with `interventions=`, a shift declared with `shifts=`, a tilt of the
mechanism declared with `incremental=`, or a coefficient of a working model declared with
`msm=`. The five partition the registry rather than accumulating, so declaring one axis
makes the other four unavailable to that fit. It is not the same question as `group=`:
`ate` and `att` share the `arm` axis across two different score equations, while `ey_shift`
and `ate_shift` share both.

Registering a target whose reported parameters have no oracle-law branch is a **test
failure** rather than an oversight caught in review; [the oracle-law
gate](methodology.md#the-oracle-law-gate) says why, and what the gate checks in both
directions.

## Adding a fluctuation

`group=` above names a *score equation*, not an estimand — six of the eight built-in
targets share the `mean` fluctuation because they are different functionals of one targeted
distribution. That fluctuation has one column per treatment arm: two for a binary treatment,
`K` for a `K`-armed one. Groups live in their own registry, so a target that needs a score
equation nobody has written yet can supply one:

```python
import numpy as np

from cleverly.fluctuation import Submodel, register_submodel


def treated_only(
    treatment,
    propensity,
    *,
    arms=(0.0, 1.0),
    treated_fraction=None,
    missingness=None,
    intermediate_density=None,
    selection=None,
    regimes=None,
    shifts=None,
    msm=None,
    incremental=None,
):
    """One column, 1{A = 1} / g₁(W) — the Riesz representer of E[Y(1)]."""
    a = np.asarray(treatment, dtype=float).reshape(-1)
    # `propensity` is the (n, K) mechanism, columns in `arms` order.
    g = np.asarray(propensity, dtype=float)
    g1 = g.reshape(-1) if g.ndim == 1 else g[:, arms.index(1.0)]
    inverse = (1.0 / g1).reshape(-1, 1)
    return Submodel(
        a.reshape(-1, 1) * inverse,
        {1.0: inverse, 0.0: np.zeros_like(inverse)},  # covariate per treatment arm
        ("h1",),
        "treated_only",  # must equal the registered name
        {1.0: 0},  # which column targets which arm
    )


register_submodel("treated_only", treated_only)
```

Every builder takes the same keyword-only signature so the registry can dispatch on the
group name alone, ignoring the arguments it has no use for. `arms` joined that signature
when the treatment stopped being binary: a builder cannot key its output by arm without
being told which arms exist, and inferring them from the observed treatment would go wrong
on exactly the subsample that is missing one. `regimes` joined it when an intervention
stopped being a constant arm, and a builder that targets arms accepts and ignores it, as
`mean_submodel` does. A builder written against an older signature gets a `TypeError`
naming the fix rather than a bare "unexpected keyword argument".
`Target.group` is validated against this registry at *registration* time rather than at fit
time.

Both `Submodel` and `InitialFit` key their counterfactual quantities by the treatment level
the arm sets — `arms[1.0]`, `arms[0.0]` — rather than naming two fields, so shrinking,
row-slicing, sign-taking and fluctuating are written once and do not count arms.
`arm_columns` maps an arm to the column of the design whose coefficient targets it, and is
**empty** for a contrast fluctuation like `att`, where a column targets a difference and no
column belongs to one arm — there `contrast_columns` maps the *non-reference* arm to the
column carrying its contrast, and the reference arm loads every column because it is the
arm each of them is taken against. Two mappings rather than one with a wider meaning: they
answer different questions, "which column updates arm `a`" and "which column carries the
parameter `a` is contrasted under", and the second is what lets the conditional effects
index by arm code instead of by the literal `0` a two-armed submodel happens to have. The
estimand layer above keys by arm the same way: `counterfactual_means` returns a mapping of
arm to `(psi, influence_curve)`, and `Target.build` returns one estimate per arm or per
contrast.
