# User guide

One runnable recipe per capability: what to pass, what comes back, and what the fit will
refuse. Each top-level capability establishes its own setup; a subsection that continues an
example uses the setup immediately above it.

The derivations behind these numbers, the influence curves, and the tests that hold each
claim in place live in [the technical appendix](methodology.md). The
[README](../README.md) has the quickstart and the architecture.

## Multi-valued treatment

<!-- doc-section: id=multi-valued-treatment; requires=; paths=src/cleverly/datasets/synthetic.py,src/cleverly/targets/** -->

A treatment with more than two levels is estimated the same way as the binary fit in the
[quickstart](../README.md#quickstart), and reports one counterfactual mean per arm plus a
contrast against a reference arm you choose:

<!-- doc-block: id=multi-arm-fit; tier=fast -->
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

<!-- doc-block: id=multi-arm-contrast; tier=fast -->
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

<!-- doc-block: id=multi-arm-att; tier=fast -->
```python
# `att[...]` is one of the opt-in conditional effects, so the fit has to be asked for it.
res = (
    TMLE(random_state=0, reference="low", estimands="all")
    .fit(frame, outcome="Y", treatment="A")
    .single()
)

res.sensitivity.omitted_variable("ate[medium vs low]")
res.sensitivity.robustness_value("att[medium vs low]")
```

The E-value is defined on the ratio scale, so it wants a binary outcome — the same three
arms, with `family="binomial"`:

<!-- doc-block: id=multi-arm-risk; tier=fast -->
```python
binary, _ = make_multi_arm(n=2000, seed=0, family="binomial")
risk = (
    TMLE(random_state=0, reference="low", estimands=("ate", "rr"))
    .fit(binary, outcome="Y", treatment="A")
    .single()
)

risk.sensitivity.evalue("rr[medium vs low]")
```

The MNAR tilt is one per contrast in the same way — `missingness_tilt()` reports every
arm's mean and every contrast — but it needs a fit with `delta=`, so its example lives
with the others in [Sensitivity](#sensitivity).

That is a wider loop rather than a wider derivation, and it is worth saying why: the
omitted-variable bound's `nu^2` is the second moment of *that parameter's* Riesz
representer, and the representer of `ate[medium vs low]` is two columns of the same
`K`-column clever covariate the fit already targeted. So each contrast gets its own bound,
its own robustness value and its own E-value, and none of them is a summary of the others.
The bare `"ate"` is not a parameter of such a fit, so asking for it lists the contrasts
that are. `res.sensitivity.report()` picks the first reported contrast when you do not
name one.

What is refused rather than guessed at on a multi-valued treatment: one `CTMLE` fit will
not select a different treatment mechanism for each contrast. Every selector builds one
shared categorical propensity path and scores one joint, nonredundant vector — all `K` arm
means, or all `K - 1` contrasts against `reference=`. That is what the fit *is* rather than
a gap in it, and the alternative is refused because a per-contrast mechanism would make the
reported covariance the covariance of estimators that no longer share a nuisance state. If
that is the scientific goal, fit separate estimators and treat their covariance and
selection states separately. [Collaborative TMLE](#collaborative-tmle) names the components
and the settings that produce them.

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

<!-- doc-section: id=dynamic-stochastic-regimes; requires=; paths=src/cleverly/interventions/base.py,src/cleverly/interventions/support.py,src/cleverly/datasets/synthetic.py -->

"Set `A` to 1 for everybody" is one intervention among many, and until you say otherwise
it is the one every estimand above assumes. `interventions=` says otherwise. A **regime**
is a conditional distribution over the treatment arms, `g*(a | W)`, and three kinds are
supported: a constant arm, a deterministic rule `d(W)`, and a stochastic assignment you
supply. The clever covariate generalises from `1{A = a} / g(a | W)` to the density ratio
`g*(A | W) / g(A | W)`, and the parameter from a mean per arm to a mean per regime.

<!-- doc-block: id=dynamic-regimes-fit; tier=fast -->
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

<!-- doc-block: id=dynamic-regimes-support; tier=fast -->
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

<!-- doc-section: id=continuous-dose-shift; requires=; paths=src/cleverly/interventions/shift.py,src/cleverly/datasets/synthetic.py -->

Every estimand above names an arm, and a dose has none. A **modified treatment policy**
names a change instead: `d(a, w) = a + δ`, held back at a cap `u`. `shifts=` declares one,
which also declares the treatment continuous — so the column keeps its own values rather
than being coded into arms, the mechanism becomes a conditional density `g(a | W)`, and the
clever covariate becomes a density *ratio* rather than an inverse probability:

```
h(a, W) = g(a - δ | W) / g(a | W) · 1{a ≤ u}  +  1{a > u - δ}
```

<!-- doc-block: id=continuous-dose-fit; tier=fast -->
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

<!-- doc-block: id=continuous-dose-support; tier=fast -->
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

<!-- doc-section: id=continuous-dose-missing; requires=continuous-dose-fit; paths=src/cleverly/data/**,src/cleverly/interventions/shift.py -->

`delta=`, `intermediate=` and `weights=` all work here, and mean what they mean on an arm.
They share the same missingness construction as the arm path; the technical appendix derives the
composition.

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

<!-- doc-block: id=continuous-dose-missing-fit; tier=fast -->
```python
import numpy as np

# The dose process again, with an outcome that goes missing more often at a high dose.
rng = np.random.default_rng(0)
observed = rng.random(len(frame)) < 1 / (1 + np.exp(-(1.5 - 0.4 * frame["A"])))
missing = frame.assign(Delta=observed.astype(float), Y=frame["Y"].where(observed))

res = (
    TMLE(shifts=[Shift(0.0, cap=None), Shift(0.5, cap=5.0)], density_bins=40, random_state=0)
    .fit(missing, outcome="Y", treatment="A", delta="Delta")
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

<!-- doc-section: id=incremental-intervention; requires=; paths=src/cleverly/interventions/incremental.py,src/cleverly/datasets/synthetic.py -->

Every intervention above replaces the treatment decision. An **incremental
propensity-score intervention** leaves it where it was and multiplies its *odds* by `δ`
(Kennedy 2019):

```
q_δ(1 | W) = δ g(W) / (δ g(W) + 1 - g(W)),     D_δ = δ g + 1 - g
```

"Make everyone `δ` times as likely to be treated as they already were." `δ = 1` changes
nothing — `q₁ = g` identically — so it is the natural course and the usual reference, the
way `Shift(0.0, cap=None)` is on the dose axis.

<!-- doc-block: id=incremental-fit; tier=fast -->
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

<!-- doc-block: id=incremental-support; tier=fast -->
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

## Population intervention measures and baseline strata

The opt-in targets `ey_obs`, `par`, and `paf` add the observed mean, population
attributable risk `E[Y] - E[Y(0)]`, and population attributable fraction
`1 - E[Y(0)] / E[Y]`. They are population-intervention parameters in the sense of Díaz
Muñoz & van der Laan (2012), not synonyms for the ATE. `paf` requires a binary outcome
and positive observed risk. These targets currently require complete outcomes: under
`delta=` the natural-course mean needs an additional MAR score, and complete-case
substitution would answer a different question.

Passing `strata=("V",)` to `fit` adds a parameter for every finite baseline level while
retaining the marginal report. The stratum columns must remain in `covariates=`. The
targeting step jointly solves `I(V=v) H_v / P_n(V=v)` for every level; for ATT and ATC,
`H_v` is rebuilt with `P_n(A=a | V=v)`, rather than multiplying a clever covariate that
was normalised by the marginal arm share. Parameter names carry the level, for example
`ate[V='high']`. Pooled targeting is supported; fold-specific and fold-evaluated strata
are explicitly refused until their fold-local probabilities are implemented.

`variable_importance(...)` repeats a declared target with each candidate column taking
the treatment role. By default the other candidates join the adjustment set; turn off
`adjust_for_other_candidates` when that conditioning is causally inappropriate. Every
row records its actual adjustment set and retains its underlying fit. Tests are
two-sided on the target's native null and adjusted together using Benjamini & Hochberg
(1995); this is exposure-wise causal screening, not a new model-prediction importance
score.

## Summarising the arms: a marginal structural model

<!-- doc-section: id=arm-msm; requires=; paths=src/cleverly/msm.py,src/cleverly/datasets/synthetic.py -->

Five dose levels and two effect modifiers report ten counterfactual means, which is a
table rather than an answer. `msm=` declares a **working model** `m(a, V; β)` that
summarises them, and makes the fit's parameters its coefficients:

<!-- doc-block: id=arm-msm-fit; tier=fast -->
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

<!-- doc-section: id=arm-msm-link; requires=; paths=src/cleverly/msm.py,src/cleverly/datasets/synthetic.py -->

For a binary outcome the identity link above is a *linear-risk* model, and its coefficients
are risk differences — frequently out of range, and not what the applied literature
reports. `link="log"` and `link="logit"` put the linear predictor inside a mean function,
so a coefficient becomes a log risk ratio or a log odds ratio:

<!-- doc-block: id=arm-msm-link-fit; tier=fast -->
```python
from cleverly import TMLE
from cleverly.datasets import make_binary_outcome
from cleverly.msm import MSM

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

### A continuous dose

For a treatment declared with `treatment_kind="continuous"`, `MSM.linear` accepts a
strictly increasing `doses=` grid. The grid is part of the estimand and deterministic
trapezoidal quadrature approximates the Neugebauer & van der Laan (2007) projection.
For example, `MSM.linear(doses=np.linspace(-2, 2, 21))` targets the best line over that
dose interval. The projection uses the quadrature masses, while the outcome-residual
score uses

`h(A,V) phi(A,V) / g(A | W)`.

There is deliberately no quadrature-width factor in that observed-data density ratio.
The outcome regression is evaluated at every declared grid dose, and the conditional
treatment density is estimated by the same cross-fitted hazard-density machinery used
for modified treatment policies. This follows the working-model projection of
Neugebauer & van der Laan (2007) and its targeted estimator in Rosenblum & van der Laan
(2010). It does not copy a random Monte Carlo grid, so repeating or reloading the fit
cannot silently change its target. Missing outcomes and controlled intermediates are
refused on this path until their mechanisms are evaluated at both observed and grid
doses.

## Treatment given over time

<!-- doc-section: id=longitudinal-treatment; requires=; paths=src/cleverly/longitudinal/** -->

Everything above gives the treatment once. `LTMLE` gives it repeatedly, and estimates the
mean outcome under a **regimen** `ā = (a₁, …, a_T)` — a plan for every node, not a decision
at one. What makes this a different estimator rather than a wider loop is the covariate
measured *between* the decisions: `L₂` is caused by `A₁` and confounds `A₂`, so it is a
mediator and a confounder at once. Adjust for it and you block the part of `A₁`'s effect
that runs through it; leave it out and the second decision stays confounded. No single
adjustment set is right, which is the whole reason for the module.

<!-- doc-block: id=longitudinal-fit; tier=fast -->
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
  g_bounds: fixed [0.01, 1] on each cumulative treatment-and-censoring probability
    (package default; R ltmle-compatible heuristic -- inspect truncation diagnostics)
  confidence level: 95%
  ...
  always: 1263 of 4000 units followed it throughout; max weight 12.0, effective n 1096;
    max truncated share 0.0% at t=1
  never: 822 of 4000 units followed it throughout; max weight 34.6, effective n 630;
    max truncated share 0.0% at t=1

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
above, `never` is supported by 822 units whose effective sample size is 630. Following
canonical `ltmle`, the raw treatment-and-censoring factors are multiplied first and each
cumulative prefix is then truncated to `g_bounds`; `res.diagnostics()` reports the
raw-versus-bounded `share_truncated`, resulting weight, and effective `n` per regimen per
node, beside that node's `epsilon` and whether it converged.

The default is the explicit fixed pair `g_bounds=(0.01, 1.0)`, matching R `ltmle`. It is
a heuristic value set, **not** an automatic selection procedure: it does not read `n`, the
effective sample size, the fitted mechanism, or the follow-up depth. A cumulative path
probability can naturally fall below `0.01` even when every individual factor is moderate,
so clipping is not by itself proof of a node-level positivity violation. The fixed floor
also does not vanish with `n` and can make the clever covariate constant when it binds for
every scored row. If `share_truncated` is material, report it with the configured bounds,
maximum weights, and effective sample sizes, then rerun the **complete** fit under
substantively justified alternatives; the earlier pseudo-outcome regressions depend on
later targeting and cannot be left fixed.

The standard errors here are plug-in influence-curve standard errors. Unlike R's default
`variance.method="tmle"`, cleverly does not also compute a recursive robust variance and
take the larger result. R warns that IC-only variance can be anti-conservative under
positivity problems or rare outcomes, so an active bound is a reason to qualify the
interval, not evidence that its uncertainty already includes the truncation choice.

**Observation weights** are read here exactly as they are at one time point. Pass
`weights="w"` to `.fit()` and the estimand becomes the same regimen parameter in the tilted
population `dP_w = w dP / E[w]`: every node's mechanism, every node's censoring factor and
every regression in the backward recursion is fitted by weighted loss, each node's
fluctuation solves the weighted score `Σ w h_t (Z_t − Q̄*_t) = 0`, the plug-in is a weighted
average, and the reported curve is `(w / E[w]) · D*(P_w)`. A weight is a tilt of the
*population*, so it is **not** a factor in `h_t` — the denominator is still the `2T`
mechanism factors and nothing else. `res.diagnostics()` reports the leverage of `w / ∏g`
rather than of `1/∏g`, since the two reweightings multiply. Observation weights do not
change LTMLE's fixed cumulative `g_bounds`; this differs deliberately from the
point-treatment automatic rule described in
[the weights section](#observation-weights-and-which-population-they-define).
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

<!-- doc-section: id=longitudinal-rule; requires=longitudinal-fit; paths=src/cleverly/longitudinal/** -->

A plan may decide a node rather than declare it. Any entry of a plan may be a **rule**
`d_t(H_t)` instead of an arm, so "start everyone, then keep treating only the responders"
is one regimen rather than a special case:

<!-- doc-block: id=longitudinal-rule-fit; tier=fast -->
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

<!-- doc-section: id=longitudinal-msm; requires=; paths=src/cleverly/longitudinal/**,src/cleverly/msm.py -->

Four plans over two nodes is a table; `2^T` plans over `T` nodes is not a report at all.
`msm=` declares a **working model** `m(ā, V; β)` summarising the regimens and makes the
fit's parameters its coefficients — the same move [`msm=` makes at one
node](#summarising-the-arms-a-marginal-structural-model), and the standard way the applied
literature reports a grid of dynamic rules: a coefficient on the rule's threshold rather
than a mean per plan.

<!-- doc-block: id=longitudinal-msm-fit; tier=fast -->
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

<!-- doc-section: id=longitudinal-survival; requires=; paths=src/cleverly/longitudinal/**,src/cleverly/datasets/longitudinal.py -->

Everything above has one outcome, at the end. Pass **one cumulative event indicator per
node** instead — `Y_t = 1` means the event has happened by `t`, so it stays one thereafter —
and the outcome joins the time ordering `W → A₁ → C₁ → Y₁ → L₂ → A₂ → C₂ → Y₂`. The
parameter becomes a **curve**: the cumulative risk
`F_ā(t) = P(event by t under ā)` at every horizon.

<!-- doc-block: id=longitudinal-survival-fit; tier=fast -->
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

<!-- doc-block: id=longitudinal-survival-curve; tier=fast -->
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

<!-- doc-section: id=competing-risks; requires=; paths=src/cleverly/longitudinal/**,src/cleverly/datasets/longitudinal.py -->

An event that ends follow-up need not be the only one that can. Pass a **mapping of cause
to its indicator column per node** and each absorbing state gets its own curve — the
cause-specific cumulative incidence `F_j(t) = P(leave through cause j by t)`:

<!-- doc-block: id=competing-risks-fit; tier=fast -->
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

<!-- doc-section: id=collaborative-tmle; requires=; paths=src/cleverly/estimators/ctmle.py,src/cleverly/datasets/synthetic.py -->

A propensity model fitted to predict treatment as well as possible is fitted to the wrong
objective. A covariate that predicts treatment but *not* the outcome — an instrument —
removes no confounding, and putting it in `g` pushes propensity scores towards 0 and 1,
inflating the variance of `1/g` and so of the estimate. `CTMLE` chooses the covariates
entering `g` by cross-validating the loss of the *targeted outcome model* instead, so the
two nuisance fits only have to be right between them (van der Laan & Gruber 2010).

<!-- doc-block: id=ctmle-fit; tier=fast -->
```python
from cleverly import CTMLE
from cleverly.datasets import make_instrument

# W1 confounds; W2 predicts treatment but not the outcome; W3 predicts only the outcome.
frame, truth = make_instrument(n=2000, seed=0)
res = (
    CTMLE(
        strategy="ordered",
        preorder="logistic",
        estimands=("ate",),
        outcome_learner="glm",
        treatment_learner="glm",
    )
    .fit(frame, outcome="Y", treatment="A")
    .single()
)
print(res.extra["ctmle"].summary())
```

```
strategy = ordered; preorder = logistic; target = ate; criterion = cross-validated penalized squared-error loss

... one row per candidate, with training and cross-validated risks ...

selected g: (intercept only)
left out: W1, W2, W3 -- adjusting for these would have cost more variance than the bias
they remove
```

The selector stops at the intercept: a GLM is correctly specified for the
outcome in this process, so `Qbar` has already done the adjusting and there is no
residual confounding left for `g` to remove. That is collaborative double robustness —
the two nuisance fits only have to be right between them — and it is why C-TMLE's
standard error here is about 25% below a plain TMLE's on the same samples.

`strategy="greedy"` (the default) builds the ordering by forward selection instead;
`strategy="ordered"` is the scalable variant (Ju et al. 2019) at `O(p)` propensity fits
rather than `O(p²)`. Its `preorder="logistic"` default ranks variables by the empirical
loss after one-variable targeting; `preorder="partial_correlation"` ranks the absolute
partial correlation of the initial outcome residual and each covariate conditional on
treatment. `preorder=` is refused for other searches and when `ordering=` supplies an
explicit order, so an inapplicable setting cannot be silently ignored. `strategy="discrete"`
cross-validates an explicit list of candidate models.

The same selectors accept a multi-valued treatment. They build one categorical propensity
model at each path position and optimize one joint, nonredundant vector: all arm means for
`ctmle_estimand="ey"`, or every arm-versus-`reference=` contrast for `"ate"`, `"rr"`, and
`"or"`. The components are visible rather than implicit:

<!-- doc-block: id=ctmle-multi-catalogue; tier=fast -->
```python
multi = CTMLE(
    strategy="discrete",
    candidates=((), ("W1",), ("W1", "W2")),
    ctmle_estimand="ate",
    estimands=("ate",),
    reference="low",
)
# After fitting a three-arm frame:
# result.extra["ctmle"].target_names contains one labelled contrast per nonreference arm.
```

One fit never selects a different `g` for each contrast. If that is the scientific goal,
fit separate estimators and treat their covariance and selection states separately.

**Every strategy replaces the treatment mechanism, so the shared nuisance pass is
outcome-first**: `Qbar` and the missingness model are fitted once, and no ordinary `g(W)`
is fitted and discarded first. One reporting consequence is worth knowing. For the three
selector strategies the propensity row of `fit.validation.nuisance()` still describes the
*selected* mechanism — its calibration, AUC and Brier score are computed from the array the
estimate actually used — but its super-learner weight and risk table is empty, because that
mechanism came off the candidate path rather than from one shared fit, and the candidate
C-TMLE most often selects is the intercept-only model with no learner behind it. Read
`res.extra["ctmle"]` for what the search did instead. `strategy="oat"` does have one shared
fit and does report the full table.

`strategy="oat"` is the outcome-adaptive treatment model from `ctmle3`: it first fits
`Qbar(a, W)` for every treatment level, then fits the categorical treatment mechanism on
that vector of predictions. Unlike the three selector strategies, it has no candidate path
or parameter-specific selection loss. The record under
`res.extra["ctmle"]` shares the practical diagnostic fields `.strategy`,
`.treatment_features`, and `.treatment_risk_selected` across both API paths.

**It is a sharper trade than the other three, and worth making on purpose.** The selector
strategies pick a *subset of your covariates* for `g`, so the full model is always in the
candidate path and collaborative double robustness survives: if the outcome model is wrong,
the search can keep adding until `g` is right. `strategy="oat"` never puts `W` into `g` at
all. Its mechanism is the projection of `A` onto the fitted `Qbar` vector, so it is
consistent for the true propensity only when the true propensity is a function of the
outcome regression — and when `Qbar` is wrong, `g` is generally wrong with it and the
estimator has neither leg to stand on. What you get in exchange is the collaborative
benefit in its purest form: `g` carries only the confounding `Qbar` left behind, so an
instrument cannot reach the denominator. Prefer it when you trust the outcome model and
positivity is the binding problem; prefer a selector strategy, or a plain `TMLE`, when you
would rather keep the second leg. Its treatment design is also a fitted quantity
cross-fitted on the same split as `Qbar`, which `ctmle3` does not do at all (`LF_oat` uses
the full sample) — see the `cleverly.estimators.ctmle` module docstring for what that costs.

Each selection fold cross-fits the predictions used on its training rows and uses one
full-selection-training refit for its validation rows, following the `tmle3`/`sl3`
`fold_fits` plus `full_fit` convention. `selection_inner_folds=2` controls that inner split.
Larger values are available but multiply the candidate-fitting cost; two is the measured
default, not a claim that the inner and outer fold counts should match.

The selected object is the pair `(g_k, Qbar*_k)`, not `g_k` alone. The final pooled
targeting pass continues from that selected outcome state, and save/load and sensitivity
retargeting preserve it. A truncation, missingness-tilt, omitted-variable, or nuisance-bound
sweep therefore starts each perturbed targeting step from the selected candidate's
`Qbar*_k`; it holds selection fixed and does not restart from the original `Qbar0`. Rerun
the fit if the candidate itself should be reselected. Fold-targeted and canonical CV-TMLE
composition is refused until separately derived.

One caveat worth knowing: the influence-curve standard error treats the selected
model as given, so it does not include the variability the selection itself contributes,
and is mildly anti-conservative as a result. Pass `n_bootstrap=` for inference that does
— each replicate re-runs the search.

Why this is the right number, and how it is checked:
[how the selection is evidenced](methodology.md#c-tmle-how-the-selection-is-evidenced).

## Doubly-robust inference

<!-- doc-section: id=doubly-robust-inference; requires=; paths=src/cleverly/estimators/drtmle.py,src/cleverly/validation/drtmle.py,src/cleverly/datasets/synthetic.py -->

> **Conditionally valid, and the condition is on you.** `DRTMLE` computes what Benkeser et
> al.'s Theorem 1 derives — its curve was transcribed from R's `drtmle` rather than derived,
> and has since been checked against Theorem 1's appendices, against the Gateaux derivative
> of the parameter, and against exact finite-support laws, and agrees with all three. The
> interval it reports is valid **conditional on** your obtaining adequate primary *and
> reduced-regression* fits, which are rate conditions on five estimated functions that no fit
> can check for itself — **solving the score equations does not establish them**. A coverage
> study over 6,000 fits shows the interval is much better than a plain TMLE's where one
> nuisance is badly estimated — `0.844` against `0.532` in the cell built for it — and that
> it **attains nominal coverage nowhere in that study**, the best reading being `0.880`.
> [`docs/drtmle.md`](drtmle.md) is the contract in full. Use it where you have a reason to
> think one nuisance is badly estimated, and do not treat the interval as settled.

What Theorem 1 licenses is an interval *conditional on* the score equations being solved to a
negligible order, so a fit's own answer to that question is on the face of its report:
`summary()` ends with the score check whenever the check fails, and `res.score_verdict`
carries the verdict — the same object `res.validation.score_check()` returns — whether it
passed or not. A passing fit says nothing extra. This matters here more than anywhere else in
the package, because such a fit returns a `psi`, an `se` and a confidence interval formatted like
any other. The rate of failure that used to be quoted here — 23 of 24 fits on `weak_overlap_dgp` —
predated a fix to the targeting convention that accounts for it. **The sweep has since been rerun
on the same seeds and the rate is 0 of 24**, with the worst score five orders lower, on draws whose
overlap is exactly as poor as before. So the number to carry is zero, and the reason to keep
reading the verdict is that no argument here proves a fit must solve its equations — only its own
diagnostics say whether it did.

The check goes further on a doubly-robust fit than on any other, because there is more that can
be wrong: it recomputes each arm's corrections from the state the fit returned and compares them
with the scores the targeting step recorded, so a curve built from an expression the fit did not
solve is reported as such rather than inferred from an uncentred estimand. That check is what
caught the convention above, on a fit whose three fluctuation rows all read `1e-11`.

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

<!-- doc-block: id=drtmle-fit; tier=fast -->
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
target                   kind             |score|    before     threshold  ratio     ok
-----------------------  ---------------  ---------  ---------  ---------  --------  ---
mean                     fluctuation      6.661e-19  3.745e-07  1.527e-06  4.36e-13  yes
mean (mechanism)         fluctuation      7.743e-11  7.743e-11  1.527e-06  5.07e-05  yes
mean (reduced)           fluctuation      6.278e-20  4.900e-05  1.527e-06  4.11e-14  yes
mean (D*_g)[0]           correction       1.841e-10  -          1.527e-06  1.21e-04  yes
mean (D*_g)[0] identity  identity         7.977e-20  -          1.533e-11  5.20e-09  yes
mean (D*_Q)[0]           correction       2.908e-21  -          1.527e-06  1.90e-15  yes
mean (D*_Q)[0] identity  identity         1.039e-22  -          1.533e-11  6.78e-12  yes
mean (D*_g)[1]           correction       1.187e-09  -          1.527e-06  7.77e-04  yes
mean (D*_g)[1] identity  identity         2.526e-18  -          1.533e-11  1.65e-07  yes
mean (D*_Q)[1]           correction       1.064e-18  -          1.527e-06  6.97e-13  yes
mean (D*_Q)[1] identity  identity         1.014e-19  -          1.533e-11  6.61e-09  yes
ate                      influence curve  1.003e-09  -          1.527e-06  6.57e-04  yes

PASS: the targeting step solved all 3 estimated score equations of the doubly-robust estimator.
Validity is not efficiency: the curve reported is D = D* - D*_Q - D*_g, entitled to be believed
under weaker conditions than D* rather than efficient under them. See cleverly.estimators.drtmle.
```

Three things to read off. There are **three** `fluctuation` rows where a plain fit has one: the
ordinary outcome equation, a *mechanism* equation that fluctuates `g`, and a second outcome
equation against the reduced regressions.

Then, **per arm**, two more kinds. A `correction` row is the mean of a term the reported curve
actually subtracts, recomputed from the state the fit returned; an `identity` row is that mean's
difference from the score the targeting step recorded for the same equation. The two are different
questions and a failure in each means a different thing — a `correction` row says the fit did not
solve its equations, an `identity` row says the solver and the curve are not evaluating the same
expression, which is a defect that iterating longer will not fix. Per arm and not only on the
`ate`, because errors in the two arms cancel in a difference. `res.validation.correction_check()`
is the recomputation itself, with the clipping bias `B_clip` that explains an identity failure when
the mechanism truncation is what caused it. A third kind, `diagnostic`, appears only on a
single-guard fit and is neither: it is the correction for the equation that fit does **not** solve,
reported because it says what the guard left out, and held to no threshold because nothing
subtracts it.

And what changed is the interval, not the estimate — on this
fit `ate` is 1.5348 against a plain TMLE's 1.5292, a twelfth of a standard error apart, while
the standard error moves from 0.06850 to 0.06828. **Read a `DRTMLE` fit as the same estimate
with an interval entitled to be believed under weaker conditions, not as a better estimate.**

Which is also why the verdict does not say "the estimated efficient score equation" as a plain
fit's does. **Validity is not efficiency.** Under misspecification the efficient influence
function at the true law is still `D*`; the curve this fit reports is
`D = D* - D*_Q - D*_g`, the *estimator's* asymptotic influence function at the limits its
nuisances converge to, and the estimator is generally **not** efficient there. So `DRTMLE`
buys an interval that stays valid where a plain TMLE's stops being valid — **not** a narrower
one, and not an efficient one. When both nuisances are consistent the two corrections vanish
row by row, the curves coincide, and this is the ordinary efficient estimator; that is exactly
the case the variant is not for. Do not read the smaller standard error above as the general
case: it is one draw, and 0.06828 against 0.06850 is well inside what a different seed moves.

`guard=` says which extra equations to solve, in `drtmle`'s vocabulary, and it is **crossed**:
`"Q"` guards against a misspecified *outcome regression* and adds the equation that fluctuates
`g`; `"g"` guards against a misspecified *mechanism* and adds the one that fluctuates `Qbar`.
Both by default; `guard=()` fits no reduced regressions at all and is bit-for-bit a plain
TMLE. `reduced_outcome_learner=` and `reduced_treatment_learner=` take the reduced
regressions' learners, defaulting to the primary ones. The randomized missing-outcome
construction requires both guards because the cited algorithm targets its treatment,
observation, and outcome correction blocks together.

Missing outcomes are supported for the theorem-backed randomized-trial case. With binary
treatment, an observation indicator `Delta`, no cross-fitting, and no weights, fit with
`DRTMLE(randomized=True, cross_fit=False, estimands=("ate",))` and pass `delta="Delta"`.
This estimates the randomization probabilities, as Díaz & van der Laan (2017) recommend for
finite-sample balance. If the design probabilities are known, instead pass them as
`treatment_probabilities=` to `fit`; that bypasses the treatment learner. Prefer the mapping
form, `{"placebo": p0, "active": p1}` — one row-aligned column per arm, keyed by the treatment
level as you wrote it. A bare `(n,)` vector is also accepted, but it binds to the arm whose code
is `1`, which is the *second sorted level*, so in a trial labelled `active`/`placebo` it is
`P(A='placebo'|W)` and not "the probability of treatment".

The paper's five reductions keep treatment and observation separate. Targeting solves distinct
`D_A`, `D_Delta`, and `D_Y` scores, and `res.validation.correction_check()` reports all three.
The ordinary outcome clever covariate still divides by
`P(A=a|W) P(Delta=1|A=a,W)`, so `res.sensitivity.positivity()` reports that derived product,
but it is not a third stored mechanism. `g_bounds` and the ordinary truncation curve apply to
treatment; `nuisance_bound` and `truncation_curve(mechanism=True)` apply to observation.

Observational treatment, cross-fitting, missing treatment, and `treatment_probabilities=` with
`n_bootstrap=` all remain refused; the exact restrictions and derivation are in the
[DR-TMLE contract](drtmle.md#randomized-trials-with-missing-outcomes). Saved fits that used
row-aligned known probabilities retain their estimates and retargeting operations, but cannot
reconstruct an estimator for refit-based analyses because later row identity and order cannot be
verified.

For missing-outcome fits the dedicated Díaz--van der Laan cycle is always used; `update_order=`
controls only the complete-outcome construction. There it is a **diagnostic** keyword rather than a tuning one, and it is here because a
question about this estimator is open rather than because there is a choice to make. The source's
own algorithm states six steps in a particular order; this package's alternation is not a
transcription of them, and the source's termination condition is the three score equations rather
than the route — so the two orders are the same estimator if they land in the same place, and
`update_order="paper"` exists so that "if" can be measured instead of assumed. Leave it at
`"cleverly"` unless you are running that comparison. What is measured so far is two draws, on
which the estimates agree to within a fifth of a standard error and the *standard errors* differ
by up to 2.3%; a systematic comparison remains proposed under the
[DR-TMLE roadmap priority](roadmap.md#1-extend-the-published-dr-tmle-surface).

`reduced_crossfit=` is the **second** diagnostic keyword and is here for the same kind of reason.
The reduced regressions are fitted on the primary cross-fitting split, so fold `k`'s regression
trains on rows whose design *and target* came from models that saw fold `k`. Whether that matters
is discussed under [reduced-regression cross-fitting](drtmle.md#reduced-regression-cross-fitting).
The argument that it does not matter needs
one quantity to vanish, and that quantity is exactly the difference between this construction and
`reduced_crossfit="nested"` — which refits the primary nuisances leaving each outer fold out as
well. So the expensive one exists to be measured against, not to be used: leave it at `"pooled"`
unless you are running that comparison. It costs `n_folds` times the primary nuisance fitting and,
more than that, more alternation rounds — 1.3x to 17x a pooled fit's wall clock on the draws
measured so far.

`evaluation=` is the **third**, and it exists so that a *condition of the theorem* is a
measurement rather than an assumption. Solving the three score equations is necessary; the source
separately assumes the remainder left over is negligible, and checking that needs the population
mean of the fitted doubly-robust curve — a mean over rows the fit never saw, since the empirical
one is the quantity targeting drove to zero. Passing an independent draw here evaluates every
fold's nuisances at it and moves them by the same targeting steps the fitted arrays take, so the
curve is available as a *function*:

<!-- doc-block: id=drtmle-companion; tier=fast -->
```python
holdout, _ = make_nonlinear_ate(n=20_000, seed=99)
res = (
    DRTMLE(
        estimands=("ate",),
        outcome_learner="glm",
        treatment_learner="glm",
        random_state=0,
        evaluation=holdout,
    )
    .fit(frame, outcome="Y", treatment="A")
    .single()
)
companion = res.repeats[0].fluctuations["mean"].reduction.evaluation
```

The draw contributes to no fit, no fold and no score, so a fit that declares one is bit for bit a
fit that does not — and it is refused with `repeats=`, `targeting="one_step"` and
`target_weights=True`, each by name. What it is for is integrating the population mean of the
*fitted* doubly-robust curve against an independent draw, which is what the second-order remainder
diagnostic needs and what `P_n D̂` is refused as a substitute for — see
[the remainder terms](drtmle.md#the-remainder-terms-and-the-rate-conditions). This is a research
instrument rather than something an applied fit needs.

**A single guard reports a shorter curve, and the report says which.** One correction per
equation the guard asks for: `guard=("g",)` solves equation (10) and reports
`D = D* - D*_Q`, and the verdict names that rather than the both-guards curve. The other
equation's correction is still recomputed and printed, as a `diagnostic` row held to no
threshold — it is what says what the guard did not buy — and it cannot fail a check, because
nothing subtracts it. A regression test now ensures the curve subtracts only the corrections
whose equations the selected `guard=` poses; otherwise a single-guard fit would carry a term
whose equation it had never posed; measured at `1.2e-03` on the outcome scale against a
`5.4e-06` bar, with the mechanism truncation binding on no row at all.

What *is* worth care is the statistical reading, which is unchanged: one guard insures
against the one nuisance it names and nothing else. And two guards are not strictly better
— on a law where both conditioning sets are saturated they over-correct, which is arithmetic
rather than a defect and is what `tests/unit/test_remainder_drtmle.py` shows.

It costs real time — two further learner fits per arm on every round of an alternation,
refitted *inside* the loop as the source does. One consequence is worth knowing: `retarget`
stops being arithmetic on cached arrays, so a truncation curve on a `DRTMLE` fit costs about
a fit per point rather than a fraction of one, and a result read back from disk cannot
retarget at all.

Scope is a discrete treatment and the `mean` group, plus the restricted randomized
missing-outcome case above. `att`/`atc`, the other parameter axes, observational missing
outcomes, missing treatment, `intermediate=`, fold-wise targeting,
`reduction="bivariate"`, `treatment_probabilities=` under `n_bootstrap=`, and composition with
`CTMLE` are all refused by name.

**What is not visible from the output**, and is why this section opens with a warning. The
influence curve's form is read off `drtmle`'s implementation rather than derived. It has since
been checked against Theorem 1 — in the 2016 working-paper version, kept at
UCB Biostatistics paper 356 — and it agrees, though the paper's own display of one correction
prints a sign its appendices contradict, so the check took an argument rather than a glance
([the sign of the mechanism correction](drtmle.md#the-sign-of-the-mechanism-correction)).
It has since also been checked against a *perturbation of the law*, which is the check every
other estimand here gets and which this one could not have until the fixture was made wrong on
purpose: with one nuisance consistent, the corrected curve a fit reports is the efficient
influence function row for row, and a flipped sign misses by half a unit or more against a
`1e-12` window.
There is no cross-check against `drtmle`'s own *numbers* and there will not be: both
implementations descend from one source, so agreement would be evidence about the transcription
and blind to exactly the sign above — the
[standing decisions](architecture-invariants.md#validation-and-evidence) give that reasoning. A coverage study
on the off-diagonal of the misspecification grid found *no gap for this variant to close* at the
sizes it could reach: the regime it is for needs an adaptive good nuisance converging more slowly than
`n^(-1/4)`, which is beyond what a routine validation budget can simulate. And the alternation does not
reliably converge — equation (10)'s covariate is near-singular on exactly the fits anybody
wants, so some fold draws exit at the outer cap, which is what the score check is for.
[What the validation programme established](drtmle.md#what-the-validation-programme-established)
lists these and the rest. Do not read this as a free improvement over a plain TMLE.

Why this is the right number, and how it is checked:
[what the extra equations remove](methodology.md#doubly-robust-inference-what-the-extra-equations-remove).

## Cross-fitting and CV-TMLE

<!-- doc-section: id=cross-fitting; requires=; paths=src/cleverly/estimators/recipe.py,src/cleverly/estimators/targeting.py,src/cleverly/datasets/synthetic.py -->

Both literature routes use one common targeting coefficient. The original CV-TMLE usually
minimises the equal average validation-fold loss and evaluates the plug-in inside each
fold. Levy's easy implementation stacks all out-of-fold predictions and then performs an
otherwise ordinary TMLE; that is the package default. The pinned `tmle3` source snapshot
in the references implements the same route through
`tmle3_Update(cvtmle=TRUE)`. Levy's paper is the stable specification here; `tmle3` is an
implementation cross-check. The package also keeps a separate-per-fold epsilon extension:

| setting | estimator |
| --- | --- |
| `cross_fit=True, targeting_scheme="pooled"` (default) | stacked CV-TMLE (Levy 2018): common epsilon, stacked evaluation; matched by the pinned `tmle3` snapshot |
| `targeting_scheme="pooled", cv_evaluation=True` | original fold-evaluated CV-TMLE: common epsilon, fold-wise evaluation and variance |
| `targeting_scheme="fold"` | extension with one epsilon per fold |

<!-- doc-block: id=crossfit-fit; tier=fast -->
```python
from cleverly import TMLE
from cleverly.datasets import make_nonlinear_ate

frame, _ = make_nonlinear_ate(n=2000, seed=0)
res = TMLE(cv_evaluation=True).fit(frame, outcome="Y", treatment="A").single()
res.cv_targeting.summary()  # fold-evaluated and stacked reports, fold estimates, epsilon
res.cv_targeting.variance["ate"]
```

Which of the three ran is not left to be reconstructed from the settings —
`res.config.estimator_name` says it in words, and `res.summary()` prints it.

Validation-fold outcomes fit epsilon in both schemes. What is out of fold is each row's
*initial nuisance prediction*. The difference is whether epsilon is common across the
validation risks (the common update) or separately estimated inside each fold (the
extension). The default stacked route preserves the ordinary row-weighted empirical loss;
the fold-evaluated route normalises weights within fold before its equal `1/V` average.

Why this is the right number, and how it is checked:
[what the folds do and do not buy](methodology.md#cross-fitting-what-the-folds-do-and-do-not-buy).

### What the folds guarantee

<!-- doc-section: id=cross-fitting-guarantees; requires=crossfit-fit; paths=src/cleverly/estimators/recipe.py,src/cleverly/estimators/targeting.py -->

Two things a cross-fitted estimate assumes, and neither is left to trust. A fold index
outside the declared range, or a fold holding no rows at all, is refused by `Folds` when it
is built. A cluster with rows in more than one fold is refused by a post-condition that
`make_folds` runs on the way out, so it covers every split in the library — the outer folds,
Super Learner's inner folds, C-TMLE's selection folds — without any of them knowing about
it. A third prohibition needs no check: "every row is held out exactly once" has no
counterexample, because a split is one fold index per row and two-fold membership has no
representation.

The fold *policy* is recorded too, and separately from the split it produced:

<!-- doc-block: id=crossfit-cluster-plan; tier=fast -->
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

<!-- doc-block: id=crossfit-stratification; tier=fast -->
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

<!-- doc-block: id=crossfit-repeats; tier=fast -->
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

<!-- doc-block: id=crossfit-cv-repeats; tier=fast -->
```python
TMLE(cv_evaluation=True, repeats=5)
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

Original fold evaluation computes the updated parameter inside each validation fold and
averages with weight `1/V`. The common targeting loss and observation weights are likewise
normalised within fold. With unequal folds the variance keeps its exact `n_v^-2` factors,
and the aggregate influence-curve rows are scaled by `n/(V n_v)`; the shortcut
`mean(IC²)/n` is exact only for equal folds.

`ate`, counterfactual levels, regime/shift/incremental levels and contrasts, and ATT/ATC
have supported fold evaluations. ATT/ATC rebuild the empirical arm probability inside
each fold before the common update. `rr`, `or`, and MSM coefficients are deliberately
refused with `cv_evaluation=True`: averaging those nonlinear parameters changes their
gradient by fold, and reusing the ordinary fluctuation leaves a nonzero influence-curve
score. Their valid stacked-validation estimates remain available with the default pooled
evaluation.

<!-- doc-block: id=crossfit-fold-targeting; tier=fast -->
```python
res = (
    TMLE(cv_evaluation=True, estimands=("ate", "att"))
    .fit(frame, outcome="Y", treatment="A")
    .single()
)
res["att"].psi  # averaged over folds rather than pooled
res["att"].std_error  # the cross-validated standard error
res.cv_targeting.pooled["att"], res.cv_targeting.fold_evaluated["att"]  # both, always
```

## Observation weights, and which population they define

<!-- doc-section: id=observation-weights; requires=; paths=src/cleverly/data/**,src/cleverly/inference/**,src/cleverly/datasets/synthetic.py -->

Passing `weights=` changes the *estimand*, not just its weighting. The nuisances are fitted
by weighted loss, the targeting step solves the weighted score equation, and the plug-in is
a weighted average — the whole fit runs on the weighted empirical measure. So what comes
back is the requested causal parameter evaluated in the tilted population
`dP_w = w dP / E[w]`, and its efficient influence function is `(w / E[w]) * D*(P_w)`, which
is what the reported standard errors are built from.

<!-- doc-block: id=observation-weights-fit; tier=fast -->
```python
import numpy as np

from cleverly import TMLE
from cleverly.datasets import make_nonlinear_ate

frame, _ = make_nonlinear_ate(n=2000, seed=0)

# A design: sampling weights that oversample one region, and a PSU per cluster of ten.
surveyed = frame.assign(
    sampling_weight=np.where(frame["W1"] > 0, 2.5, 1.0),
    psu=np.arange(len(frame)) // 10,
)

res = (
    TMLE()
    .fit(
        surveyed,
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

`LTMLE` takes `weights=` on exactly these terms — the same tilted-population estimand and
the same influence function — with every node's nuisance fitted by weighted loss and every
node's score equation weighted. Its cumulative bounds remain the explicit fixed pair
described above rather than using the point-treatment `"auto"` procedure; see
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

<!-- doc-section: id=sensitivity; requires=; paths=src/cleverly/sensitivity/**,src/cleverly/datasets/synthetic.py -->

<!-- catalogue: what the suite offers, listed together. No single fit supports all of it —
     `truncation_curve(mechanism=True)` needs a missingness mechanism, `evalue()` a binary
     outcome, `missingness_tilt()` a `delta=` — so this enumerates rather than demonstrates,
     and the sections below run each one against the fit it needs. -->

<!-- doc-block: id=sensitivity-catalogue; tier=fast -->
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

<!-- doc-block: id=sensitivity-missingness; tier=fast -->
```python
import numpy as np

from cleverly import TMLE
from cleverly.datasets import make_multi_arm

# The three arms again, with outcomes that go missing more often at a high `W1`.
arms, _ = make_multi_arm(n=2000, seed=0)
rng = np.random.default_rng(1)
seen = rng.random(len(arms)) < 1 / (1 + np.exp(-(1.2 + 0.5 * arms["W1"])))
dropout = arms.assign(Delta=seen.astype(float), Y=arms["Y"].where(seen))

tilted = (
    TMLE(random_state=0, reference="low")
    .fit(dropout, outcome="Y", treatment="A", delta="Delta")
    .single()
)

# the unobserved outcomes are worse than they look under `low`, better under
# `medium`, and MAR under `high`
direction = {"low": 1.0, "medium": -1.0, "high": 0.0}

tilted.sensitivity.missingness_tilt([0.0, 0.5, 1.0], arm_gamma=direction)
tilted.sensitivity.tipping_gamma("ate[medium vs low]", arm_gamma=direction)
```

The tilt at arm `a` is then `arm_gamma[a] * gamma`, and the returned frame carries a
`gamma[<level>]` column per arm saying what each one received. Every arm must be named:
one left out would be tilted by the shared `gamma` after all, which is the assumption the
keyword exists to state rather than inherit. Any per-arm tilt vector is reachable this way
— pass it as the direction with `[1.0]` as the grid — and keeping the sweep
one-dimensional is what keeps `tipping_gamma` a single number: how far along *this*
departure the conclusion survives.

## Validation

<!-- doc-section: id=validation; requires=; paths=src/cleverly/validation/**,src/cleverly/datasets/synthetic.py -->

<!-- catalogue: the three reports are the validation surface, shown without repeating a fit;
     the repeated-sampling example below constructs the estimator state it needs. -->
<!-- doc-block: id=validation-catalogue; tier=fast -->
```python
res.validation.nuisance()  # CV AUC/Brier/calibration for g, CV R^2/MSE for Q, SL weights
res.validation.score_check()  # did targeting solve mean(EIF) = 0?
res.validation.refute()  # placebo treatment, random common cause, subset stability
```

And a harness that measures the *estimator* rather than a fit, by repeated sampling from a
process whose truth is known:

<!-- doc-block: id=validation-coverage-study; tier=slow -->
```python
from cleverly import TMLE
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

<!-- doc-section: id=adding-estimand; requires=; paths=src/cleverly/targets/** -->

Estimands live in a registry rather than in a `Literal`. A `Target` declares which
fluctuation solves its score equation, what scale its inference lives on, what it needs of
the outcome, how to build the estimate — and, as a required field, an `Identification`
record stating its assumptions, the nuisances it consumes and what double robustness buys
for *that* estimand specifically:

<!-- doc-block: id=custom-estimand-registration; tier=fast -->
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

<!-- doc-block: id=custom-estimand-multi-arm; tier=fast -->
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

<!-- doc-section: id=adding-fluctuation; requires=; paths=src/cleverly/fluctuation/** -->

`group=` above names a *score equation*, not an estimand — six of the eight built-in
targets share the `mean` fluctuation because they are different functionals of one targeted
distribution. That fluctuation has one column per treatment arm: two for a binary treatment,
`K` for a `K`-armed one. Groups live in their own registry, so a target that needs a score
equation nobody has written yet can supply one:

<!-- doc-block: id=custom-fluctuation-registration; tier=fast -->
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
