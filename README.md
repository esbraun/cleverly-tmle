# cleverly

Targeted maximum likelihood estimation (TMLE) for Python — with sensitivity analysis and
validation diagnostics treated as first-class parts of the estimator, not afterthoughts.

`cleverly` is a Python counterpart to the R targeted-learning ecosystem (`tmle`, `tlverse`/`tmle3`).
It takes **pandas or polars** dataframes interchangeably (via [narwhals](https://narwhals-dev.github.io/narwhals/)),
returns results in whichever backend you handed it, and every estimator ships with:

- **influence-curve based inference**, plus cluster-robust variance, targeted bootstrap, and
  simultaneous (max-t) confidence intervals across estimands;
- **sensitivity analysis** — positivity/overlap diagnostics, truncation curves,
  omitted-variable-bias bounds with robustness values, E-values, and MNAR tilt analysis;
- **validation** — nuisance-model calibration and cross-validated risk, an explicit check that the
  efficient-influence-function score equation was solved, refutation tests, and a reusable
  simulation harness that measures bias and confidence-interval coverage.

## Install

```bash
pip install cleverly              # core: numpy, scipy, scikit-learn, narwhals, joblib
pip install "cleverly[all]"       # + pandas, polars, lightgbm, matplotlib
```

## Quickstart

```python
from cleverly import TMLE
from cleverly.datasets import make_nonlinear_ate

frame, truth = make_nonlinear_ate(n=2000, seed=0, backend="polars")  # or "pandas"

est = TMLE(estimands=("ate", "att", "atc", "ey1", "ey0"))
res = est.fit(
    frame,
    outcome="Y",
    treatment="A",
    covariates=["W1", "W2", "W3", "W4"],
).single()  # fit() returns one result per parameter; an ordinary fit has one

print(res.summary())
print(f"true ATE = {truth['ate']:.4f}")

res.to_frame()  # tidy results, in the backend you passed in
res.estimates["ate"].psi  # point estimate
res.estimates["ate"].ci  # (lower, upper)
res.estimates["ate"].influence_curve
```

```
              psi     std_err      ci_lower    ci_upper     p_value
ate      1.982031    0.061142      1.862195    2.101867    0.000000
att      1.994884    0.078210      1.841595    2.148173    0.000000
...
```

### Multi-valued treatment

A treatment with more than two levels is estimated the same way, and reports one
counterfactual mean per arm plus a contrast against a reference arm you choose:

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

A two-armed fit is unchanged in every respect, including the familiar `ate` / `ey1` /
`ey0` names.

### Dynamic and stochastic regimes

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

One intervention is **refused rather than approximated**, because of the influence
function and not for want of effort:

| refused | what it would need |
| --- | --- |
| incremental propensity-score interventions (`g*_δ = δg / (δg + 1 - g)`) | `g*` is a functional of `P`, so the EIF carries a further term for the pathwise derivative through `g` (Kennedy 2019). Estimating one as a `Stochastic` regime would report a standard error for a different functional |

A modified treatment policy — shifting a *continuous* dose — is **not** an intervention in
this sense and does not go in `interventions=`. It reads the treatment a unit actually
received rather than assigning one from `W`, so it has its own keyword and its own score
equation; see [Shifting a continuous dose](#shifting-a-continuous-dose).

The influence curve is checked on the same footing as the ATE's: against the complex-step
Gateaux derivative of an independently written functional at `1e-12`
(`tests/unit/test_influence_gateaux_regime.py`), and the second-order remainder against its
closed form (`tests/unit/test_remainder_regime.py`), over a static regime, a rule that
depends on `W`, and a stochastic one that is degenerate nowhere.

### Shifting a continuous dose

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

A shift is *not* the stochastic regime that induces the same density, though the temptation
to reuse `regime_means` is real: `d` induces `g^d(b | W) = Σ_{a: d(a,W)=b} g(a | W)`, and the
`Stochastic` regime at that density has the same mean *and* the same clever covariate, entry
for entry. The influence curves differ anyway. A regime's plug-in term averages `Qbar` over
the doses, `Σ_b g^d(b | W) Qbar(b, W)`, a function of `W` alone; a shift's reads the dose the
unit actually received, `Qbar(d(A, W), W)`. The two agree only in conditional expectation
given `W`, and the gap is exactly

```
Var(D_mtp) = Var(D_regime) + Var( Qbar(d(A,W),W) − E[Qbar(d(A,W),W) | W] )
```

so a modified treatment policy is strictly *harder* to estimate than the regime inducing the
same mean. `tests/unit/test_influence_gateaux_shift.py` keeps that as a negative control: it
asserts the two means agree, the two influence curves do not, and the identity above holds —
so a later "simplification" that delegates one to the other fails loudly.

The influence curve is checked on the same footing as the rest: against the complex-step
Gateaux derivative of an independently written functional at `1e-12`, on a law with four
ordered doses and **two** caps (`tests/discrete_law_shift.py`). The second cap is not
redundant. A unit can only have been shifted *to* dose `a` if the shift from `a - δ` was not
itself held back, so `h` carries the further indicator `1{a ≤ u}` — which is invisible
whenever the cap sits at or above the largest dose, the common case. The tight cap is what
caught that term missing.

### Summarising the arms: a marginal structural model

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

The clever covariate is `h(a, V) φ(a, V) / g(a | W)`, one column per term, so the score
equation is one per coefficient rather than one per arm — which is why this is a fourth
parameter axis and why `msm=` cannot be combined with `interventions=` or `shifts=`. The
counterfactuals are still the arms; what changed is the summary. A **saturated** working
model — one indicator per arm — reproduces the per-arm report exactly, point estimate and
influence curve alike, which `tests/e2e/test_msm.py` asserts against a plain fit.

Two things are **refused rather than approximated**, both because of the derivation:

| refused | what it would need |
| --- | --- |
| a non-identity link (`log`, `logit`) | `∂m/∂β` then depends on `β`, so the clever covariate does too, and solving the score needs an outer `(β, ε)` iteration this fluctuation does not run. A one-shot version would report a standard error for an equation that was not solved. For a binary outcome an identity-link MSM is a linear-risk model, and its coefficients are risk differences |
| weights derived from the estimated mechanism (a "stabilised" MSM) | `h` would be a functional of `P`, so the EIF carries a further term for the pathwise derivative through `ĝ` — the same argument that refuses incremental propensity-score interventions |

The influence curve is checked on the same footing as the rest: against the complex-step
Gateaux derivative of an independently written functional at `1e-12`
(`tests/unit/test_influence_gateaux_msm.py`), and the second-order remainder against its
closed form (`tests/unit/test_remainder_msm.py`). The oracle's working model is
deliberately **not** saturated — three coefficients against six `(w, a)` cells — because a
saturated one agrees with the means whatever the projection code does, and its weights are
deliberately **not** uniform: with `h ≡ 1` the design is orthogonal and `β_a` collapses to
the marginal ATE *identically*, so code that reported the ATE under the name `msm[a]` would
pass every check.

### Collaborative TMLE

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

Two more things about how this is evidenced, because they change how the numbers above
should be read. When `Qbar` is correctly specified — as it is in the example process — the
*empty* propensity model is a legitimate MSE-minimising choice, and C-TMLE usually makes
it: 10 seeds out of 10 for the greedy search at `n = 700`. That is right, not a defect, but
it means a favourable comparison against plain TMLE on such a process would also be won by
a selector hard-wired to select nothing, so it is not evidence that the search
discriminates between covariates. The claim that it does is tested where selecting nothing
is *wrong* — with the outcome model reduced to a constant, the search includes the
confounder in every seed and still leaves the instrument out, while a do-nothing selector
is biased by 0.81 against 0.037. Second, there is no cross-language check: R's `ctmle` is
not compared against here or in CI. `cleverly.estimators.ctmle` sets both out in full.

### Cross-fitting and CV-TMLE

Three constructions, and it is worth knowing which one you are running:

| setting | estimator |
| --- | --- |
| `cross_fit=True, targeting_scheme="pooled"` (default) | cross-fitted TMLE |
| `targeting_scheme="fold"` | fold-targeted CV-TMLE |
| `targeting_scheme="fold", cv_evaluation=True` | canonical CV-TMLE |

Cross-fitting the nuisances is what removes the Donsker condition on the nuisance
*estimators*. Pooled targeting on top of that adds an empirical-process term of its own,
because `epsilon` is fit on the rows it fluctuates — controlled, but by a different
argument: *conditional on the training-fold fits* `Qbar` is fixed, and `{Qbar(epsilon)}`
is then indexed by a fixed finite-dimensional coefficient over a compact set (two entries
for the default estimand, one per arm), Lipschitz in it, and so Donsker however complex
`Qbar` is.

That controls the empirical-process term and nothing else. Efficiency still needs the rest:
positivity bounding the clever covariate (the `g_bounds` truncation), the estimated
influence curve converging in `L_2`, the score solved to `o_P(n^-1/2)`, and a second-order
remainder that is `o_P(n^-1/2)` by a *product rate* on `ghat` and `Qbarhat` — a condition
on the learners, which the finite-dimensional fluctuation does not supply. Note too that a
single pooled `epsilon_hat` couples the folds: each row's nuisance prediction is out of
fold, but its *targeted* prediction is not. The two schemes share a first-order limit under
those conditions — but they are not the same estimator, and Zheng & van der Laan prove
their result for the fold-targeted construction specifically. See `targeting_scheme` in the
API docs for the full statement.

```python
res = TMLE(targeting_scheme="fold").fit(frame, outcome="Y", treatment="A").single()
res.cv_targeting.summary()  # both reports side by side, per-fold psi and epsilon
res.cv_targeting.variance["ate"]
```

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

### Observation weights, and which population they define

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

That statement is derived and its limits set out in
[`cleverly/data/weighting.py`](src/cleverly/data/weighting.py), and verified numerically
against a longhand statement of `Psi(P_w)` in `tests/unit/test_weighted_estimand.py` —
including for weights that depend on the outcome, where the tilt changes `Qbar` itself.
The short version:

| supplied weights | status |
| --- | --- |
| known sampling probabilities, selection depending only on observed data | supported: `dP_w` is the population law |
| complex survey design (strata, PSUs, FPC) | estimate supported; stratification and FPC are ignored (conservative), clustering is **not** — pass `id=` |
| outcome-dependent sampling with known fractions (case-control) | supported, if the fractions are genuinely known |
| estimated selection or non-response weights | intervals condition on `w`; conservative when `w` is an MLE of a correct selection model |
| calibration, raking, post-stratification, trimming | no general guarantee — bootstrap the weight derivation outside the package |
| frequency (count) weights | **refused**, with instructions: expand the rows |
| replicate weights (BRR, jackknife) | **refused**: a set of designs, not one weight vector |

### Sensitivity

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

### Validation

```python
res.validation.nuisance()  # CV AUC/Brier/calibration for g, CV R^2/MSE for Q, SL weights
res.validation.score_check()  # did targeting solve mean(EIF) = 0?
res.validation.refute()  # placebo treatment, random common cause, subset stability
```

`score_check()` is necessary, not sufficient: it verifies that the fluctuation reached the
root of the equation the library posed, which a *consistently* wrong clever covariate would
also do. That the equation itself is the right one is a claim about the library rather than
about your fit, and is checked in the test suite — against the numerical Gateaux derivative
of the target parameter (`tests/unit/test_influence_gateaux.py`) and against the
second-order product remainder that double robustness consists of
(`tests/unit/test_remainder.py`), both exactly, on a law with finite support.

Missing outcomes get the same treatment rather than being taken on trust, because the
`1/P(Δ=1 | A, W)` factor is the kind of thing that solves its own score equation whether
or not it is right. `tests/discrete_law_mar.py` carries a finite-support law whose support
*is* the observed-data support — `(W, A, Δ)` always, `Y` only when `Δ=1` — and the two
modules above have `_mar` counterparts checking the influence curve against the numerical
Gateaux derivative at all eighteen points, including the six where nothing was observed and
the residual term must vanish exactly. What double robustness means there is not what it
means without missingness, and the remainder module states it: **consistent if `Q̄` is
right, or if the _product_ `g·π` is right** — a correct propensity buys nothing on its own
when the missingness model is wrong, and errors in the two mechanisms can cancel exactly.

The controlled direct effect now has the same class of proof rather than an argument for one.
`tests/discrete_law_cde.py` carries a law on `(W, A, Z, Δ, ΔY)` whose CDE changes sign between
the two levels of `Z` — so confusing them inverts the answer rather than nudging it — and
whose negative controls are the mistakes this parameter invites: dropping the `q_z` factor
(which quietly estimates a total effect), using the other level's density, substituting the
marginal `P(Z=z)` for the conditional `q_z(a, W)`, and averaging the plug-in over the `Z=z`
stratum instead of over everybody.

One more thing worth knowing about double robustness, because the reassuring form of the
slogan is the one that sticks: **the two halves are not interchangeable when positivity is
strained.** With `Q̄` right the estimand is recovered by integrating a regression over the
covariate distribution, which needs no overlap at all. With only `g` right, everything rests
on `1/g` weights — and on a process with 11% of the population below `g = 0.05` that half
stops delivering, at a measured bias of −0.13 against −0.01 for the outcome half. It is not a
truncation artefact and not a bug; it is the positivity premise failing.
`tests/e2e/test_double_robustness.py` runs both overlap regimes and pins the asymmetry.

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

## How this is validated

Three tiers, and they fail on different mistakes. Worth knowing which claims rest on which,
because a simulation that comes out well is the weakest of the three and the easiest to read
too much into.

**Exact proofs, on laws a sample realises exactly.** `tests/discrete_law*.py` build
finite-support distributions whose every cell probability is a multiple of `1/N`, so an
`N`-row frame *is* the law rather than a draw from it. Handed oracle nuisances the initial
fit is exactly right, `epsilon` is zero, and the influence curve the estimator reports is
the EIF at `P₀` rather than an estimate of it. Against that: the reported curve equals the
complex-step Gateaux derivative of the identification formula — written longhand, sharing no
code with the library — to `1e-12`, for every estimand: the seven binary ones without
missingness (`test_influence_gateaux.py`), under MAR (`..._mar.py`) and for the controlled
direct effect (`..._cde.py`), and the per-arm means and contrasts on a **three-armed** law
(`..._multi.py`, against `tests/discrete_law_multi.py`). The third arm is not decoration:
two arms cannot distinguish code that keys everything by arm from code that has two columns
and calls them 0 and 1, and that law's labels sort into a different order than they were
written in so a helper equating arm code with arm position fails rather than passes. The
regime estimands get the same treatment (`..._regime.py`), over a static regime, a rule
that depends on `W` and a stochastic one that is degenerate nowhere — three kinds because
two static regimes could not distinguish code that mixes over the arms from code that
picks a column. The shift estimands get it too (`..._shift.py`, against
`tests/discrete_law_shift.py`), on a law with four ordered doses and two caps — the tight
one because a cap above the largest dose never exercises the `1{a ≤ u}` factor, and that
law found the factor missing. The second-order remainder is checked against its closed form in the four
matching `test_remainder*.py` modules, which is what double robustness actually consists of.
Every one of these modules carries deliberate-mutation controls: each plausible way of
building the thing wrong is shown to move the answer by more than `1e-2`, four orders past
the window the real assertions use.

**Deterministic invariants and algebraic identities.** Relabelling `A ← 1 - A` has to swap
`EY1`/`EY0`, negate the ATE and turn ATT into `-ATC`; an arm-independent outcome mean has to
give exactly zero however hard the treatment was confounded; weights scale out; the iterative
and one-step solvers agree; the weighted and clever-covariate fluctuations solve the same
equation. These are cheap and they fail on the mistakes statistics is worst at catching — a
swapped sign, a swapped conditioning population.

**Simulation, for the claims that are about repeated sampling and nothing else.** Coverage,
root-n consistency, type I error and the estimator variants live in
`tests/e2e/test_coverage_slow.py` and run nightly. The double-robustness grid runs on every
push, in both a comfortable-overlap and a weak-overlap version — and the two disagree, which
is the point of having both.

Two things this does **not** include, stated plainly because their absence is easy to miss.
There is no comparison against another implementation: not R's `tmle`, not `tmle3`, not
`ctmle`. And `score_check()` passing is not evidence that the equation was the right one —
see below.

## What is implemented

Classic point-treatment TMLE for a binary treatment. The table below covers what R's `tmle`
package covers, plus the pieces that matter from `tmle3` and the literature — but read that
as a statement about *features*, not about numbers: **`cleverly` has never been compared
against R's output.** No cross-language test exists here or in CI, and there are already
known deliberate divergences (`g_bounds="auto"` resolves at Kish's effective sample size on
weighted fits; see below). What the estimates *are* checked against is set out under
[How this is validated](#how-this-is-validated).

| Capability | Notes |
| --- | --- |
| Estimands | `EY1`, `EY0`, `ATE`, `ATT`, `ATC`, `RR`, `OR` for a binary treatment; `EY` (one mean per arm) and `ATE`/`RR`/`OR` against a reference arm for a multi-valued one; `ey_regime` / `ate_regime` when the fit declares `interventions=`; `ey_shift` / `ate_shift` when it declares `shifts=`; `msm[...]`, one per term, when it declares `msm=`. The four sets are exclusive, not cumulative: `interventions=` and `shifts=` declare what "counterfactual" means for the fit and `msm=` declares how the counterfactuals are summarised, and one fluctuation cannot report parameters from two score equations under one heading |
| Multi-valued treatment | any number of arms up to 20. The mechanism becomes a distribution over the arms and the `mean` fluctuation gets one clever-covariate column per arm, so the fit reports `K` counterfactual means with a joint influence-curve matrix and `K-1` contrasts against `reference=`. Every other contrast — a dose-response comparison, a pairwise difference the reference skipped — comes from `result.contrast()` with no refit. Parameters are named with your own labels: `ey[high]`, `ate[high vs low]`. A two-armed fit is unchanged, bit for bit, and keeps the short names. What is refused rather than guessed at: `ATT`/`ATC` (they reweight one arm by the propensity odds), `CTMLE` (both searches order candidates by one propensity margin), the omitted-variable bound and the MNAR tilt |
| Interventions | `interventions=` declares what "counterfactual" means for the fit: a constant arm (`Static`), a deterministic rule `d(W)` (`Rule`), or a known stochastic assignment `g*(a \| W)` (`Stochastic`). All three are one `(n, K)` density over the arms, so one clever covariate `g*(A \| W) / g(A \| W)` covers them and collapses to the familiar indicator form exactly when the regime is static — where the numbers are bit for bit an ordinary fit's. The report becomes `ey_regime[...]` per regime and `ate_regime[... vs ...]` per non-reference regime, and `sensitivity.support()` reports the positivity a regime actually needs. What is refused rather than approximated: incremental propensity-score interventions (their `g*` depends on `P`, so the EIF carries a further term) |
| Continuous treatment | `shifts=` declares a modified treatment policy `d(a, w) = min(a + δ, u)` and with it that the treatment is a dose: no arms, a conditional density `g(a \| W)` in place of the propensity, and a clever covariate that is a density ratio. The density is a discrete hazard fitted by the ordinary `treatment_learner=` on a long `(unit, bin)` expansion, so every preset, screener and thread limit works untouched. The report becomes `ey_shift[...]` and `ate_shift[... vs ...]`, and `sensitivity.shift_support()` reports the ratio's tail and the effective sample size it leaves. `cap=` is required rather than estimated, since a fitted support boundary would make the parameter itself data-dependent. What is refused rather than guessed at: `delta=`, `intermediate=` and estimated weights, each of which puts a further conditional density beside `g` and needs its own derivation |
| Marginal structural model | `msm=` declares a working model `m(a, V; beta)` for `E[Y(a) \| V]` and makes the fit's parameters its coefficients, reported as `msm[a:W1]` under the term names you gave. `beta` is a **projection** under a known weight `h(a, V)`, not the truth of an assumed regression, so the estimand and its interval are well defined whether or not the model is correct (Neugebauer & van der Laan 2007). The clever covariate is `h(a,V) phi(a,V) / g(a \| W)`, one column per term, and the projection is solved by weighted least squares against the *targeted* `Qbar` — which zeroes the second half of the influence curve by construction, so no outer iteration is needed. A saturated working model reproduces the per-arm report exactly. What is refused rather than approximated: a non-identity link (its `dm/dbeta` depends on `beta`) and weights derived from the estimated mechanism (they would make `h` a functional of `P`) |
| Outcome types | binary, and bounded continuous via Gruber & van der Laan (2010) scaling |
| Nuisance estimation | any scikit-learn estimator, or the built-in `SuperLearner` (ensemble + discrete). A treatment with more than two arms needs a conditional distribution over them: `SuperLearner` fits one binary ensemble per arm and normalises (one-vs-rest, documented as a modelling choice — nothing constrains `K` independently fit ensembles to sum to one), and any multiclass classifier is used directly |
| Cross-fitting | out-of-fold nuisance fits; stratified and cluster-respecting folds. Stratification handles a multi-valued treatment natively |
| CV-TMLE | `targeting_scheme="fold"` — an `epsilon` per validation fold, plus per-fold diagnostics; `cv_evaluation=True` adds fold-wise evaluation and the cross-validated variance for the canonical construction |
| C-TMLE | `CTMLE` — greedy, scalable-ordered and discrete collaborative selection of the covariates entering `g` |
| Targeting | iterative fluctuation (Newton) or one-step universal least-favorable submodel |
| Fluctuation | logistic or linear; clever covariate or weighted (`target_weights`, R's `target.gwt`) |
| Missing outcomes | `delta=` with its own nuisance model, entering the clever covariate. Assumes MAR given `(A, W)`; the double-robustness condition becomes "`Q̄` right **or** the product `g·π` right" |
| Controlled direct effect | `intermediate=` (R's `Z`) estimates `Ψ_z = E_W[E(Y \| A=a, Z=z, W)]` per level of `Z`, so the returned `TMLEResultSet` holds two results — index the level, `res[0.0]`, rather than calling `.single()`. Needs `Y(a,z) ⊥ Z \| A, W` on top of the usual assumptions — no intermediate confounder affected by `A` — and the DR condition becomes "`Q̄` right **or** the product `g·q_z·π` right". Not a longitudinal estimator and not a natural direct effect; `cleverly.estimators.direct_effect` writes the parameter down, derives its EIF, and states the boundary. Its influence curve is checked against the numerical Gateaux derivative at machine precision, on the same footing as the ATE |
| Weights | probability/sampling weights, with the tilted-population estimand and its EIF stated and tested; frequency and replicate weights refused |
| Clustering | `id=` for cluster-level influence-curve variance and cluster bootstrap |
| Bounds | propensity truncation (`g_bounds`), outcome bounds (`q_bounds`), `alpha` shrinkage. With two arms `g₁` is clipped and the control arm is its complement, so the pair sums to one. With more, each arm is clipped in its own right and the row is **not** renormalised — rescaling back onto the simplex can push a column under the floor and undo the only thing the floor is for. `PositivityReport.simplex_deviation` reports the size of that deliberate inconsistency; it cannot move `Ψ`, which contains no mechanism at all, only the second-order remainder |
| Screening | pre-screening of covariates for the treatment model (`prescreenW.g`, `min_retain`) |
| Inference | IC-based, cluster-robust, targeted bootstrap, multiplier bootstrap, delta method |
| Contrasts | `result.contrast(fn, names)` applies the delta method to the *joint* influence curve, and `result.covariance()` returns the joint covariance matrix. Pass `gradient=` when the derivative is known in closed form: the default central difference is accurate to ~1e-10 relative, which is fine for reporting and not enough to reproduce a closed-form influence curve at 1e-12 |
| Persistence | `result.save(path)` / `cleverly.load(path)` write arrays plus JSON into a versioned `.npz`. No pickle. Everything reached through `retarget` — positivity, truncation curves, the MNAR tilt, the omitted-variable bound, the score check, contrasts, the bootstrap — is bit-for-bit identical after a round trip; `refute()` and `benchmark()` genuinely refit and so need the learners to have been library specifications rather than fitted objects |
| Provenance | every result carries package versions, a fingerprint of the data and a *separate* fingerprint of the realised fold assignment — folds are not recoverable from a seed alone, since they also depend on row order and on the scikit-learn version that made them. Pass `run_id=` for your own identifier; no git commit is collected |
| Targeting diagnostics | `Fluctuation` records the score before as well as after targeting, the Hessian condition number, `epsilon` standard errors, the quasi-log-likelihood, and a named `failure` (`separation_suspected`, `bounds_pinned`, `singular_hessian`, …). Non-convergence in individual CV folds is reported rather than silently averaged away |

### Adding an estimand

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
a regime declared with `interventions=`, or a shift declared with `shifts=`. The three
partition the registry rather than accumulating, so declaring one axis makes the other two
unavailable to that fit. It is not the same question as `group=`: `ate` and `att` share the
`arm` axis across two different score equations, while `ey_shift` and `ate_shift` share both.

Registering a target whose reported parameters have no branch in the `functional` of an
oracle law — `tests/discrete_law.py` for the arm- and regime-indexed estimands,
`tests/discrete_law_shift.py` for the shift-indexed ones — is a **test failure**, not an
oversight caught in review. The evidence this package offers that
an influence curve is correct is that it agrees, to ~1e-12, with one obtained by complex-step
differentiation of an independently written functional on an exactly representable law. An
estimand without that has no such evidence, and the registry is deliberately not allowed to
make skipping it easy. The gate walks the *parameter* names a target reports rather than the
target name, so a per-arm target needs an oracle for each — and a target intended for more
than two arms needs one on `tests/discrete_law_multi.py`, the three-armed law, since two arms
cannot distinguish code that keys by arm from code that has two columns and calls them 0 and 1.
The gate runs in both directions: an oracle branch no target reports is dead code, so a law
and the registry must cover each other exactly.

### Adding a fluctuation

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
**empty** for a contrast fluctuation like `att`, where the single column targets a
difference and no column belongs to one arm. The estimand layer above keys by arm the same
way: `counterfactual_means` returns a mapping of arm to `(psi, influence_curve)`, and
`Target.build` returns one estimate per arm or per contrast.

## Roadmap

The base classes (`estimators/base.py`, `inference/`, `learners/`, `fluctuation/`) are shared
infrastructure; the following variants plug into them:

- longitudinal TMLE (`ltmle`) for time-varying treatments and censoring
- survival TMLE (`survtmle`) and competing risks
- incremental propensity-score interventions, whose `g*` depends on the estimated
  mechanism and so needs a further influence-function term (Kennedy 2019)
- doubly-robust TMLE with nonparametric inference (`drtmle`)

### On native acceleration

A Rust extension for the numerical kernels was planned. `benchmarks/bench_tmle.py` says it
is not worth building. Profiling a full fit by module (`cProfile`, total time):

| fit | cleverly-authored code | scikit-learn + LightGBM |
| --- | --- | --- |
| n=5,000, `library="default"` | **0.5%** | 44% |
| n=20,000, `library="glm"` | 22% | 17% |

The targeting step is 1.5–1.7% of a `glm` fit and does not appear at all in a `default`
one — it is a 2×2 Newton solve with a closed-form Hessian. Nuisance estimation dominates,
and it already runs in compiled code. Note how much the preset matters: `glm` is the
cheapest library available, so it makes every other line's share look several times larger
than it is. Benchmark with `--library default` before drawing a conclusion.

The 22% figure above is almost entirely *one* function, and profiling it turned up waste
rather than arithmetic — waste that was cheaper to fix than to rewrite:

- **The multiplier bootstrap was 92–95% multiplier *generation* and 2–3% matrix product.**
  It drew a full float64 uniform to produce one Rademacher sign. Generating bits instead
  is ~2.4× faster. Better: for `multiplier_kind="normal"` the max-t law has a closed form
  — `xi @ IC` is a linear map of a Gaussian — so the whole resampling loop collapses to
  one covariance and a draw from an *m*-dimensional normal, which is **80–360× faster**
  and never allocates a `(n_replicates, n)` array.

  That speed is not free, and `multiplier_kind` still defaults to `"rademacher"`. The
  closed form exists *because* the Gaussian max-t law depends on the influence curves only
  through their covariance — so `"normal"` is a plug-in normal approximation rather than a
  resampling scheme, and it cannot see the leverage a `1/g(W)` clever covariate produces
  under weak overlap. Simulated against a brute-force max-t distribution, it is biased
  conservative there (+0.14 on a true 2.16 at n=200, +0.07 at n=2,000), while `"rademacher"`
  stays within 0.02. On well-behaved influence curves all three kinds agree. Use `"normal"`
  when *n* is large, the curves are well behaved, and resampling actually shows up in a
  profile.
- **The cluster bootstrap rebuilt its membership index inside every replicate**, an
  `O(n_clusters × n)` scan per draw. Building it once is **24–160× cheaper** per replicate,
  which a 1000-replicate cluster bootstrap pays back a thousand times over.
- `cluster_sums` used `np.add.at`, which is unbuffered; `np.bincount` is ~2× faster.

None of that needed Rust, and the package stays pure-Python. The other place that mattered
turned out to be thread scheduling rather than arithmetic: nuisance fits run
single-threaded by default so parallelism happens across folds and candidates instead of
inside each fit (see `cleverly.learners.set_thread_limit`).

**When to revisit this.** Native code pays where the nuisance estimator is *not* an
scikit-learn model, and today none of them is. The trigger is **HAL** (highly adaptive
lasso) and its undersmoothed variant: a zero-order spline basis of `n × O(n·d)` binary
indicators that scikit-learn's lasso cannot take, where basis enumeration, sparse assembly
and coordinate descent are a natural fit for a native extension — R's `hal9001` ships a C++
backend for exactly this. The EP-learner benefits *through* HAL rather than on its own; its
other cost is targeting a *k*-dimensional score with *k* = basis size, which is BLAS-bound
and already fine. Longitudinal and survival TMLE are weaker cases: the loop over timepoints
is Python, but each body is a nuisance fit, so they stay scikit-learn-bound.

The measurement is reproducible — rerun the benchmark before revisiting this.

## Development

```bash
uv venv && uv pip install -e ".[dev]"
ruff check . && ruff format --check .
mypy src/cleverly
pytest -m "not slow" -q     # fast tier, ~3 minutes
pytest -m slow -q           # statistical validation tier (nightly in CI, ~1 hour)
python benchmarks/bench_tmle.py
```

`noxfile.py` wraps the same steps (`nox -s lint typecheck tests`), and the fast tier runs on
Python 3.10–3.13 in CI.

## References

- van der Laan & Rubin (2006), *Targeted Maximum Likelihood Learning*.
- Gruber & van der Laan (2010), *A targeted maximum likelihood estimator of a causal effect on a
  bounded continuous outcome*.
- Gruber & van der Laan (2012), *tmle: An R Package for Targeted Maximum Likelihood Estimation*.
- Zheng & van der Laan (2011), *Cross-validated targeted minimum-loss-based estimation*.
- van der Laan & Gruber (2010), *Collaborative double robust targeted maximum likelihood
  estimation*.
- Gruber & van der Laan (2010), *An application of collaborative targeted maximum likelihood
  estimation in causal inference and genomics*.
- Ju, Gruber, Lendle, Chambaz, Franklin, Wyss, Schneeweiss & van der Laan (2019), *Scalable
  collaborative targeted learning for high-dimensional data*.
- van der Laan & Gruber (2016), *One-step targeted minimum loss-based estimation*.
- Chernozhukov, Cinelli, Newey, Sharma & Syrgkanis (2022), *Long story short: omitted variable bias
  in causal machine learning*.
- VanderWeele & Ding (2017), *Sensitivity analysis in observational research: introducing the
  E-value*.
- Scharfstein, Rotnitzky & Robins (1999), *Adjusting for nonignorable drop-out using semiparametric
  nonresponse models*.

## License

MIT
