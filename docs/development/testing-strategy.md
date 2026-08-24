# Testing strategy

This page explains how the test tiers divide the work, and how to decide which one a change has to
satisfy. `README.md` gives the commands. This page gives the design behind them.

## The tiers

| tier | command | what it gates |
| --- | --- | --- |
| fast | `pytest -m "not slow" -q` | every unit, integration and end-to-end test, plus the documentation gates. This is the default handoff gate |
| registered validation studies | part of the fast tier | the committed results of each [implementation validation study](../technical-reference/index.md#implementation-validation-grid). The fast tests recompute every verdict from the artifacts, so the statistical evidence is checked in minutes rather than hours |
| slow | `pytest -m slow -q` | repeated-sampling studies that no registered study covers yet. About one hour |

**The registered studies are becoming the primary statistical evidence.** A registered study runs
its expensive sampling once, commits the per-replication results, and the fast tier then checks
every verdict, every published number, and every negative control against those artifacts. That
gives repeated-sampling evidence at fast-tier cost.

The slow tier is not retired. It still holds 128 tests across six modules, and they still have to
be run correctly when a change reaches them. Folding the remaining studies into registered studies
is separate work. Until it lands, both sets of rules below apply.

A claim leaves the slow tier when a registered cell establishes it more strongly, and not before.
Eleven design families still have no registered row: fold repeats, DR-TMLE, multi-arm means,
multi-arm selectors, clustering, weights, missing outcomes, controlled direct effects, incremental
interventions, weighted longitudinal fits, and competing risks. Each needs its own law, exact
oracle, margins, and paired control before its slow tests can go.

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

Slow claims currently include coverage, type-I error, root-n and large-sample consistency,
comparative variance, and flexible-learner bias.

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

## Which slow tests to run

Run one named slow study when only one statistical family is reachable. Do not run unrelated
studies beside it.

Run the complete `pytest -m slow -q` tier in two cases. The first is a shared estimation or
inference path that can affect several families. The second is an acceptance gate that explicitly
requires the whole tier.

If no slow test can execute the changed path or observe its result, do not run slow tests. State
that path analysis in the handoff. This is an evidence decision and not a time-budget waiver.

## Never run the two tiers at the same time

Both tiers size themselves from `tests.parallel.available_cores()`, and each one expects the whole
machine. Running them together oversubscribes every core and takes longer than running them in
sequence. Finish the fast tier before you start any slow study.

## Adding a study

To register a new validation study rather than a slow test, follow
[method benchmarking strategy](method-benchmarking.md).
