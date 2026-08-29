# Sensitivity and validation methods

This page accounts for every instrument `cleverly` gives you to review a method. Each entry answers
three questions in the same order. Why do you use it? What does it tell you? How does it tell you
that?

The instruments fall into four layers, and the layers see different mistakes. Read them in order.
A layer does not replace the layer above it.

| layer | the question it answers | what it cannot answer |
| --- | --- | --- |
| [Diagnostics on the fit you have](#diagnostics-on-the-fit-you-have) | did this fit, on this sample, do what the estimator asked of it? | whether the estimator asked for the right thing |
| [Sensitivity to untestable assumptions](#sensitivity-to-untestable-assumptions) | how wrong would an assumption have to be to change the conclusion? | whether the assumption is in fact wrong |
| [Refutation and simulation you run](#refutation-and-simulation-you-run) | does the fitted workflow behave as it must under a known answer? | anything the law you simulated does not contain |
| [How the library certifies itself](#how-the-library-certifies-itself) | is the derivation this library implements the correct one? | how your own data behaves |

## Diagnostics on the fit you have

These read the artifacts a completed fit already holds. They are cheap, and they run without
refitting a nuisance model.

### Positivity and overlap

**Why.** Every clever covariate for an intervention on treatment divides by an estimated density.
The observed-mean estimand is the exception, and it intervenes on nothing. A small denominator makes
one row dominate the estimate, and targeting does not restore missing support.

**What it tells you.** How much of the estimate rests on how few rows, and how much of the
mechanism the truncation bound replaced.

**How.** `result.diagnostics.support()` returns a `PositivityReport` from
[`sensitivity/positivity.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/sensitivity/positivity.py).
It reports five separate quantities, because they fail in different places.

| quantity | how it is computed | what a bad value means |
| --- | --- | --- |
| effective sample size | Kish's $(\sum \omega)^2 / \sum \omega^2$ over the clever-covariate weights, folded with the observation weights | the interval is that of a much smaller study |
| weight concentration | the share of the estimating equation carried by the top 1% of rows | a handful of rows decide the answer |
| truncation load | the count of clipped propensities, and how far each one moved | the estimate is partly the bound rather than the data |
| per-arm overlap | the mechanism's predicted probability distribution, arm by arm | one arm has a region the other never enters |
| maximum clever covariate | the largest absolute covariate value | the leverage of the single worst row |

The report is per arm. A multi-arm fit reads its arms from the parameter's structured index rather
than assuming two.

### Truncation stability

**Why.** A truncation bound is a finite-sample choice. A conclusion that survives only at one bound
is a conclusion about the bound.

**What it tells you.** How far the estimate moves as the bound moves.

**How.** `result.diagnostics.truncation_curve()` sweeps the `g_bounds` level and **retargets** the
cached nuisances at each level through `TMLE.retarget`. It refits no nuisance model, so it is a
retarget operation rather than a refit operation. `LTMLE` refuses it: `g_bounds` enters the
pseudo-outcome of every earlier node through the backward recursion, so changing it changes what
the earlier regressions were fitted to, and the whole pass has to run again.

### Nuisance model quality

**Why.** A nuisance model can predict well and remain miscalibrated. The clever covariate divides by
the predicted probability itself, so a miscalibrated fit moves every weight.

**What it tells you.** Whether each nuisance fit is calibrated out of fold, and which library
candidates the Super Learner actually used.

**How.** `result.diagnostics.nuisance_models()` returns `NuisanceDiagnostics` from
[`validation/nuisance.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/validation/nuisance.py):
out-of-fold propensity AUC, a calibration slope from a logistic recalibration of the out-of-fold
predictions, a calibration table, outcome $R^2$ or Brier score, and the Super Learner candidate
weights.

Read the propensity AUC as a positivity signal and not as a score. A higher AUC means the treatment
is more predictable, which means the arms overlap less. Higher is not better here.

### Score equations

**Why.** TMLE is defined by the equation its fluctuation solves. A fit that stopped early solves it
approximately, and the reported influence curve is then not mean zero.

**What it tells you.** Whether the fluctuation reached the root of the equation the library posed.

**How.** `result.diagnostics.score_equations()` recomputes $P_n \hat{D}^*(O)$ from the fitted
influence curves and compares it against a tolerance scaled to the score's own units
([`validation/score.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/validation/score.py)).
A point-treatment fit compares the score in the outcome's units against
`tolerance * se / sqrt(n)`. A longitudinal fit bounds each node's relative score.

**What it does not tell you.** `score_check` is necessary and not sufficient. A clever covariate
that is wrong in the same way in both the targeting step and the reported curve solves its own
equation exactly. That the equation is the right one is a claim about the library, and
[how the library certifies itself](#how-the-library-certifies-itself) is where that claim is
tested.

### Correction identities

**Why.** `DRTMLE` reports a curve assembled from three separate score equations. The curve is only
mean zero if each correction the curve subtracts is the correction whose equation the fit solved.

**What it tells you.** Two different failures, told apart. An **identity residual** means the
software solved one expression and reported another. A **correction score** means the fit did not
converge.

**How.** `result.diagnostics.corrections()` recomputes each correction's empirical mean from the
exact returned state and compares it with the score the solver recorded
([`validation/drtmle.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/validation/drtmle.py)).
It reports the two residuals and a clipping bias term separately.

This instrument found a real defect. One clipped row in six hundred left the reported curve
uncentred at `2e-04` while all three fluctuation rows reported `1e-11`.

### Intervention support

**Why.** Arm positivity, regime support, shift support, and incremental support are four different
questions. One propensity histogram answers none of them.

**What it tells you.** Whether the declared intervention puts mass where the data has none.

**How.** Each intervention class exposes its own report through
[`interventions/support.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/interventions/support.py).
A shift fit asks whether the density *ratio* stays bounded. A per-arm propensity table has no rows
on a continuous dose, so `diagnostics.support()` dispatches to the question that applies rather
than returning an empty table.

### Design weights

**Why.** Observation weights tilt the population. Their cost to precision is separate from the
clever covariate's cost, and adding the two together hides both.

**What it tells you.** The effective sample size the declared weights leave, before any positivity
cost.

**How.** `WeightReport` in
[`data/weighting.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/data/weighting.py)
reports the weighted effective sample size and warns when the weights concentrate.

### The status contract

Every diagnostic returns one of five states, and the states are part of the contract rather than a
presentation choice.

| status | what it means |
| --- | --- |
| `passed` | the check ran and its condition holds |
| `failed` | the check ran and its condition does not hold |
| `warning` | the check ran and the result needs qualification |
| `not_applicable` | no such analysis exists for this scientific question |
| `unavailable` | the analysis is meaningful, and a derivation or a fitted artifact is missing |

An `unavailable` row says which of two routes it took. A row that a capability declaration already
refuses carries that declaration's reason. A row that had to inspect the fit first is prefixed
`refused on inspection:`. Nothing else is caught. An error that no capability declared propagates,
because a report that renders a defect as `unavailable` states a scientific conclusion nobody
established.

`ASSESSMENT_CAPABILITIES` in
[`assessment.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/assessment.py)
declares, for each operation and each result family, the answer, the required artifacts, the cost,
and the execution class. The two costly classes are disjoint and are named separately. A **refit**
operation fits new nuisance models. A **retarget** operation re-solves the fluctuation against
cached ones. `run_all` excludes both by default, and each skipped row names the flag that runs it.

## Sensitivity to untestable assumptions

Identification rests on assumptions that no diagnostic can test. These instruments do not test them
either. Each one puts a number on how wrong an assumption would have to be before the conclusion
changes.

| instrument | the assumption it stresses | the number it reports | what it assumes to report it |
| --- | --- | --- | --- |
| omitted-variable bounds | no unmeasured confounding | the largest bias an unmeasured confounder of declared strength can produce | the confounder acts through the outcome regression and the treatment mechanism, with declared partial-$R^2$ strength in each |
| robustness value | no unmeasured confounding | the single strength at which the conclusion flips | that the two strengths are equal |
| benchmark | no unmeasured confounding | the strength of a confounder "as strong as" a named observed covariate | that dropping the covariate and refitting calibrates the scale |
| E-value | no unmeasured confounding | the minimum risk-ratio association with both treatment and outcome that explains away the effect | a risk-ratio scale |
| missingness tilt | outcomes missing at random | how the estimate moves as the unobserved outcomes are tilted away from the observed ones | the tilt is a constant on the logit scale |
| tipping gamma | outcomes missing at random | the tilt at which the conclusion changes | as above |

### Omitted-variable bounds, robustness value, benchmark, and contours

**How.** [`sensitivity/omitted_variable.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/sensitivity/omitted_variable.py)
implements Chernozhukov, Cinelli, Newey, Sharma and Syrgkanis (2022). The bound is

$$
|\text{bias}| \le |\rho| \sqrt{\frac{c_D^2}{1-c_D^2}}\; c_Y \sqrt{\sigma^2 \nu^2},
$$

with $\sigma^2 = E[(Y - \bar{Q})^2]$ and $\nu^2$ the second moment of the Riesz representer. The
two primitives are exposed as `elements()`. A median-combined repeated fit refuses this analysis
because the median bound needs its own influence function.
`robustness_value()` inverts the bound for the single strength that flips the conclusion.
`benchmark()` drops each named observed covariate, refits, and calibrates the strength scale
against what that covariate was worth. `contour()` returns the grid a contour plot needs.

`benchmark()` is the only member of this group that refits.

### E-value

**How.** [`sensitivity/evalue.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/sensitivity/evalue.py)
implements VanderWeele and Ding (2017): $E = RR + \sqrt{RR(RR-1)}$. It computes the E-value for the
point estimate and, separately, for the confidence limit, because the second is the one an
adversarial reader asks for.

Two scale conversions are available and both are flagged approximate. An odds ratio converts as
$\sqrt{OR}$, which needs a rare outcome. A standardized mean difference converts as
$\exp(0.91 d)$.

### Missingness tilt and tipping gamma

**How.** [`sensitivity/missingness.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/sensitivity/missingness.py)
implements the Scharfstein, Rotnitzky and Robins (1999) tilt. It sets

$$
\bar{Q}^{\text{miss}}_\gamma = \operatorname{expit}\{\operatorname{logit} \bar{Q}^* + \gamma\}
$$

for the unobserved outcomes, and mixes it with $\bar{Q}^*$ by the estimated missingness
probability. At $\gamma = 0$ it reproduces the missing-at-random estimate by construction, which is
the control that says the tilt is wired in. `arm_gamma=` gives per-arm tilt directions and must
name every arm. `tipping_gamma()` inverts the tilt for the value at which the conclusion changes.

This is a retarget operation and not a refit.

### The scope rule

A point-treatment sensitivity formula is not reused on longitudinal data. `LTMLE` reports these
operations `unavailable` with the reason its own capability row declares, rather than borrowing a
derivation. Stagewise support, scores, and nuisance loss are supported longitudinally, because each
has its own derivation.

## Refutation and simulation you run

These fit new models. They cost what a fit costs, multiplied by the number of draws.

### The four refuters

**Why.** A diagnostic reads the fit you have. A refuter constructs a case whose answer is known and
checks that the workflow returns it.

**What each tells you.** [`validation/refute.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/validation/refute.py)
ships four. Three test the implementation. The fourth tests a design, and the paragraph below it
says what that buys and what it does not.

| refuter | what it does | what must happen | what it tests |
| --- | --- | --- | --- |
| `placebo` | permutes the treatment column and refits | the estimate goes to zero | the pipeline, not the data |
| `random_common_cause` | adds an irrelevant covariate and refits | the estimate does not move | the adjustment set is not sensitive to noise |
| `subset` | refits on random subsamples | the scatter is about one standard error | the reported standard error is the right size |
| `negative_control_outcome` | refits on an outcome the treatment cannot affect | the estimate goes to zero | the design, under the control assumptions the paragraph below states |

A refuter refits the nuisance models once for each replication. The default is five
replications of each of three refuters, so `refute()` costs about 15 fits.
`run_all(include_refits=True)` runs it. `refute()` draws its randomization from the seed of the
fit, unless the caller passes `random_state`. A fit that carries a seed gives the same refutation
on every call. A fit that carries no seed gives a different refutation on every call.

The report records the seed under `random_state`. Pass that value back to `refute()` to obtain
the report again. The seed governs the perturbations and the refits they feed, so it repeats
the report of a fit that carries no seed of its own. The seed applies to a copy of the
estimator, so a refutation never changes the fit it examines.

A negative-control outcome must have no causal path from treatment. It must also share the relevant
confounding structure with the primary outcome. A non-null result flags residual bias or a bad
control, and the refuter cannot tell you which. A null result does not establish that unmeasured
confounding is absent. See [negative controls](../references.md#negative-controls).

### Coverage studies

**Why.** Coverage, bias, and standard-error calibration are claims about repeated sampling. One fit
contains no information about any of them.

**What it tells you.** Three numbers, from
[`validation/simulation.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/validation/simulation.py).

| number | how it is computed | how to read it |
| --- | --- | --- |
| coverage | the share of replications whose interval contains the truth | sustained undercoverage beyond Monte Carlo uncertainty indicates invalid intervals on that law |
| root-n bias | $\sqrt{n}$ times the mean error | bounded values support a negligible first-order bias claim. They do not establish efficiency |
| SE ratio | mean reported standard error over the empirical standard deviation of the estimates | one means the reported uncertainty matches the real spread |

**How.** `CoverageStudy` draws from a generator with a known truth, runs the complete estimator on
each draw, and summarises through `summarize_replications`. A failed draw is retained as a
`ReplicationFailure` record carrying its index, its seed, and its exception. A study that silently
replaced failed draws would report the distribution of the draws that happened to work.

The generators live in
[`datasets/`](https://github.com/esbraun/cleverly-tmle/tree/main/src/cleverly/datasets) and each
one carries an exact `truth`.

**A simulated law is an instrument, and it can be wrong.** A coverage study is evidence only if the
number it calls the truth is the number an adjusted fit estimates. Two shipped clustered generators
once failed that test, and
[evidence.md](evidence.md#a-simulated-law-is-an-instrument-too-and-it-can-be-wrong-the-same-way)
records what went wrong and what each generator now asserts about itself.

### Variable importance

`variable_importance` gives each candidate covariate the treatment role in its own fit, and
reports the target-relevant change with multiplicity-adjusted p-values
([`variable_importance.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/variable_importance.py)).
It is an assessment of the fitted causal workflow. It is not a predictive feature-importance score,
and it introduces no new influence function.

## How the library certifies itself

The three layers above review *your fit*. This layer reviews *the implementation*. It is the
evidence that the equation `score_check` solved is the right equation.

The instruments go blind in different places, and the differences are the reason there are six of
them. [evidence.md](evidence.md) records which instrument covers which registered estimand, in both
directions, and it is a test rather than a note.

| instrument | why it exists | what it tells you | how it tells you | what it cannot see |
| --- | --- | --- | --- | --- |
| **exact oracle law** | an estimator has to recover a parameter that was computed rather than estimated | the reported number is the parameter, exactly | a finite-support law whose every cell probability is a multiple of $1/N$, so an $N$-row frame **is** the law. Handed oracle nuisances, the fit is exactly right and $\epsilon$ is zero | nothing about a term that is zero at the truth |
| **Gateaux comparison** | the influence curve is what every interval is built from | the reported curve is the pathwise derivative of the parameter | complex-step differentiation of an independently written functional, compared at about `1e-14` absolute with `rtol=0` | a sign on any block that vanishes at correct nuisances, and any counterfactual block, because $\epsilon$ is zero there |
| **second-order remainder** | double robustness *is* the remainder carrying both nuisance errors | one wrong nuisance still leaves the remainder second order | the von Mises expansion evaluated at nuisances that are wrong on purpose, against a longhand form of the exact remainder | a first-order error that cancels inside the remainder |
| **exact identity** | some mistakes are algebraic and cheap to catch | a relation that holds by definition holds bit for bit | relabelling the arms, a null outcome model giving zero, weights scaling out, the one-step and iterative solvers agreeing | anything symmetric in whatever the identity is symmetric in |
| **theorem check** | the anchor the others need | the implementation agrees with the source's own theorem | evaluation at values where the quantity does **not** vanish | nothing the theorem does not state |
| **deliberate mutation** | a passing test proves nothing unless a wrong version fails it | each plausible way of building the thing wrong is shown to fail | the component is broken on purpose and the suite is required to go red | a mistake nobody thought to make |

Three supporting rules make the table mean what it says.

- **The oracle laws share no code with the library.** `tests/unit/test_oracle_independence.py`
  asserts that the oracle modules never import `cleverly`. A shared helper would move both sides of
  the comparison equally.
- **A heading is not enough.** `tests/unit/test_registry.py::TestEvidenceManifest` checks the
  evidence table against the target registry in both directions, checks that every module named
  there exists, and checks that the oracle-law column names the law whose functional really has the
  branch.
- **Cross-fitting is checked without a tolerance.**
  `tests/unit/test_crossfit_leakage.py` rigs a law in which one covariate is constant within a
  cluster and the outcome *is* that covariate with no noise. A nearest-neighbour learner then
  reproduces a held-out row bit for bit if and only if a same-cluster row was in its training set.
  The assertions are array equality, so leakage is not a matter of degree.

### The oracle-law gate

Registering a target whose reported parameters have no branch in an oracle law's `functional` is a
test failure rather than an oversight caught in review. The evidence this package offers that an
influence curve is correct is that it agrees with one obtained by complex-step differentiation of
an independently written functional on an exactly representable law. An estimand without that has
no such evidence.

The gate walks the *parameter* names a target reports rather than the target name, so a per-arm
target needs an oracle for each arm. A target intended for more than two arms needs one on the
three-armed law, because two arms cannot distinguish code that keys by arm from code that has two
columns and calls them 0 and 1. The gate runs in both directions. An oracle branch that no target
reports is dead code, so a law and the registry must cover each other exactly.

### Registered repeated-sampling studies

The instruments above ask whether each parameter is implemented correctly. A registered study asks
the complementary question. Apply a complete estimator to samples from a known law, and does its
bias and its uncertainty behave as its source theory predicts?

The design rules are in
[method benchmarking strategy](../development/method-benchmarking.md). The grid is in
[the technical reference index](method-evidence/validation-grid.md). The test-by-test results
are in [the implementation validation studies](method-evidence/index.md).

Three properties of the harness are worth stating here, because they are what make a green study
mean something.

- **A verdict is bounded by a margin declared before the run.** No rule tests whether a discrepancy
  is exactly zero.
  [The verdict rules](method-evidence/how-to-read.md#the-verdict-rules) give the argument and list
  every rule with its own control.
- **Every positive claim carries a control that must fail.** Double robustness carries a
  both-wrong-nuisance control. A type-I error cell carries a power cell, so an inert test cannot
  pass by never firing. An interval-calibration cell carries deliberately invalid inference.
- **A replication is a fixed sample.** Seeds spawn on the study's own record and on the replication
  index, so replication *k* is the same draw whatever the study's budget. A two-replication probe
  redraws exactly the published first two.

Matching a canonical R implementation is a separate and weaker claim. Two implementations
descended from one source share transcription errors, so agreement localises a discrepancy and does
not certify either one. Every study therefore tests each implementation against known truth first,
and separately.
