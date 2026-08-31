# Test tiers and gates

This page explains how the test tiers divide the work, and how to decide which one a change has to
satisfy. `README.md` gives the commands. This page gives the design behind them.

Benchmarking is a separate subject. Read
[method benchmarking strategy](method-benchmarking.md) to design and register a validation study.

## The tiers

| tier | command | what it gates |
| --- | --- | --- |
| fast | `pytest -m "not slow" -q` | every unit, integration and end-to-end test, plus the documentation gates. This is the default handoff gate |
| registered validation studies | part of the fast tier | the committed results of each [implementation validation study](../technical-reference/method-evidence/validation-grid.md). The fast tests recompute every verdict from the artifacts, so the statistical evidence is checked in minutes rather than hours |
| evidence re-execution | `pytest -m slow -q` | re-runs each registered study's property cells from scratch, refits committed replications, and recomputes every resampling bound at the full bootstrap budget |

**The registered studies are the primary statistical evidence.** A registered study runs its
expensive sampling once, commits the per-replication results, and the fast tier then checks every
verdict, every published number, and every negative control against those artifacts. That gives
repeated-sampling evidence at fast-tier cost.

The registered studies supersede the re-execution tier. No gate is contingent on `pytest -m slow`
running, and no handoff waits for it.

## How an implementation study validates a method

A shipped method is validated by its rows in the
[implementation validation grid](../technical-reference/method-evidence/validation-grid.md), and
not by a tier that somebody remembers to run. The mechanism has four parts.

| part | what it does | where it lives |
| --- | --- | --- |
| the declaration | names the scenarios, estimands, sample size, replication count, and every acceptance margin, before the run | a `StudyRecord` in `tests/studies/evidence/registry.py` |
| the run | samples a law with an exact parameter oracle, fits `cleverly`, and fits the digest-pinned reference implementation on the identical rows | the study's own law and fit modules |
| the artifacts | the per-replication estimates, the property cells, the equivalence results, and a hash-complete `manifest.json` | the study's artifact directory |
| the gate | recomputes every verdict, published number, and negative control from those artifacts | the fast tier |

A study answers three separate questions, and the counts are not interchangeable: accuracy against
a known truth, agreement with an independently maintained implementation, and declared
repeated-sampling properties.
[How to read these studies](../technical-reference/method-evidence/how-to-read.md) defines each
one, and states what each cannot establish.

Two properties make the artifacts evidence rather than a record of one run. Each margin is
declared before the run rather than chosen after it. Each positive cell is paired with a control
that must fail the same instrument in the opposite direction, so an inert test cannot pass.

The scientific derivation is checked separately. The exact-law, Gateaux, remainder, identity, and
deliberate-mutation instruments run in the fast tier and are listed per estimand in the
[evidence manifest](../technical-reference/evidence.md). A study measures a complete estimator
under repeated sampling. It does not replace those instruments.

Read [method benchmarking strategy](method-benchmarking.md) to design and register one.

## The deprecated studies

The repeated-sampling studies that predate the registered rows are **deprecated and do not run.**
There are 66 of them, marked `legacy_study`, and pytest skips each one with that reason. They are
skipped rather than deleted, because deleting them would drop a claim without recording that it
had been dropped. Run them with `pytest --run-legacy-studies` while building the row that replaces
one.

**Read the consequence plainly.** Ordinary and cross-fitted weighted longitudinal fits now have
separate registered reporting studies. Controlled direct effects, weighted point-treatment
inference, and clustered point-treatment inference also have registered paired studies. The
controlled direct-effect row uses exact finite-support nuisances. The fast end-to-end test checks
learned-nuisance bias, but no active study checks learned-nuisance interval coverage for every
learner library.

Multi-arm means and selectors have moved into four registered
[multi-arm point-treatment studies](../technical-reference/method-evidence/index.md). Categorical
longitudinal treatment has separate ordinary and cross-fitted rows. Each uncovered family still
runs its exact-law, Gateaux, remainder and mutation tests in the fast tier. Those tests check the
parameter and the influence curve.

Weights remain construction-specific. The registered [fixed-nuisance point-treatment
row](../technical-reference/method-evidence/weighted-point-treatment-tmle.md) isolates weighted
targeting and inference. The separate [learned-nuisance point-treatment
row](../technical-reference/method-evidence/learned-weighted-point-treatment-tmle.md) reaches
ordinary weighted regressions on two continuous covariates. The ordinary and cross-fitted
weighted longitudinal rows add sequential targeting, covariance, and direct learner-weight
controls under a fixed selection law. They do not cover estimated weights or arbitrary flexible
learner libraries.

The registered
[repeated row](../technical-reference/method-evidence/repeated-cross-fitting.md) covers the median
reporting rule at three draws. Its `repeat_stability` property also tests the spread-reduction
rationale. The property compares one and three draws across 400 paired fold seeds on a fixed
binary sample.

The weighted longitudinal rows use reporting policy. Their red cells remain visible in the
committed tables, so registration does not imply that every primary or property gate passed.

A test nobody runs is not evidence. Skipping these says so out loud rather than leaving a green
tier that nothing executes.

## Choosing a fast test

Use `tests.conftest.FAST_KWARGS` and parametric learners for estimator tests. The exception is a
test whose subject *is* flexible learning.

Documentation examples are not statistical evidence. Cover behaviour in the ordinary fast tier or
in a named statistical study. The documentation gates check that an example parses, resolves its
links, and does not raise. They do not check that its numbers mean anything.

## When to run a slow test

Run a slow test only when all three conditions hold.

1. It executes a changed runtime path.
2. Its assertion can observe the possible effect.
3. The evidence depends on repeated sampling, a large sample, or many expensive flexible fits that
   the fast tier cannot supply.

Coverage, type-I error, root-n and large-sample consistency, comparative variance, and
flexible-learner bias are claims a registered study makes. A slow run inspects them. It is not
where they are published.

Inspect the relevant slow test before you run it. A file location is not enough. Neither is a broad
label such as "estimator change".

| a change to | normally justifies a slow run |
| --- | --- |
| an estimand, influence curve, variance or clustering calculation | yes |
| targeting or cross-fitting behaviour, nuisance predictions, randomization | yes |
| weights, bounds, a statistical DGP, or slow-study machinery | yes |
| documentation, formatting, or type-only work | no |
| messages, or serialization and presentation that preserve fitted arrays | no |
| validation that exits before engine construction | no |

The last row is worth an example. Changing a configuration declaration from silent omission to a
pre-construction refusal cannot affect a slow study's fitted sampling distribution. It does not
justify running one.

## The evidence re-execution tier

`pytest -m slow -q` re-executes what the fast tier reads. It is a diagnostic, and it is not a gate.
No handoff, no pull request, and no evidence row is contingent on it running.

One thing it reaches that the fast tier does not is a stale `property-replicates.csv.gz`. The fast
tier recomputes every published verdict from the committed rows and refits two replications per
study. It never re-executes the property fits.

Regeneration closes that gap at the point where it opens. `regenerate.py` produces the property
rows with the current code. A gated study refuses a failed scientific verdict. A reporting study
publishes red verdicts but still refuses schema, provenance, convergence, and replication-accounting
failures. The artifacts a study commits were therefore produced by the code it was committed
against. Re-executing them afterwards confirms determinism rather than freshness.

Reach for this tier when you are diagnosing a specific disagreement between committed rows and
current behaviour. Name the study whose path and assertion can observe the change, rather than
running the whole tier.

## Never run two tiers at the same time

Both tiers size themselves from `tests.parallel.available_cores()`, and each one expects the whole
machine. Running them together oversubscribes every core and takes longer than running them in
sequence. Finish the fast tier before you start a re-execution run or a regeneration.

## Adding a study

To register a new validation study rather than a slow test, follow
[method benchmarking strategy](method-benchmarking.md).
