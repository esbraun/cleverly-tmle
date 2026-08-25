# Testing strategy

This page explains how the test tiers divide the work, and how to decide which one a change has to
satisfy. `README.md` gives the commands. This page gives the design behind them.

## The tiers

| tier | command | what it gates |
| --- | --- | --- |
| fast | `pytest -m "not slow" -q` | every unit, integration and end-to-end test, plus the documentation gates. This is the default handoff gate |
| registered validation studies | part of the fast tier | the committed results of each [implementation validation study](../technical-reference/index.md#implementation-validation-grid). The fast tests recompute every verdict from the artifacts, so the statistical evidence is checked in minutes rather than hours |
| evidence re-execution | `pytest -m slow -q` | re-runs each registered study's property cells from scratch, refits committed replications, and recomputes every resampling bound at the full bootstrap budget. Run it when artifacts are rebuilt |

**The registered studies are becoming the primary statistical evidence.** A registered study runs
its expensive sampling once, commits the per-replication results, and the fast tier then checks
every verdict, every published number, and every negative control against those artifacts. That
gives repeated-sampling evidence at fast-tier cost.

## The deprecated studies

The repeated-sampling studies that predate the registered rows are **deprecated and do not run.**
There are 94 of them, marked `legacy_study`, and pytest skips each one with that reason. They are
skipped rather than deleted, because deleting them would drop a claim without recording that it
had been dropped. Run them with `pytest --run-legacy-studies` while building the row that replaces
one.

**Read the consequence plainly.** Eleven design families now have no active repeated-sampling
evidence: fold repeats, DR-TMLE, multi-arm means, multi-arm selectors, clustering, weights,
missing outcomes, controlled direct effects, incremental interventions, weighted longitudinal
fits, and competing risks. Their exact-law, Gateaux, remainder and mutation tests still run in the
fast tier, so the parameter and the influence curve are still checked. What is no longer checked is
whether the interval built from that curve covers under repeated sampling. Each family needs its
own law, exact oracle, margins, and paired control before that claim exists again. Deleting the
deprecated module is the last step of registering its replacement, not the first.

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

## When to run the evidence re-execution tier

Run `pytest -m slow -q` when a registered study's artifacts are rebuilt, and when shared estimation
or inference code changes in a way that could alter what a committed replication produces.

The fast tier recomputes every published verdict from the committed rows and refits two
replications per study. What it cannot see is a `property-replicates.csv.gz` that has gone stale
against the code, because it never re-executes the property fits. That is the gap this tier closes,
and it is the reason it survived the deprecation above: no registered study can replace it, since
it is the thing that checks the registered studies.

A regeneration that has just run is already covered. `regenerate.py` produces the property rows
with the current code and refuses the run on any failed gate, so re-executing them immediately
afterwards confirms determinism rather than freshness.

## Never run two tiers at the same time

Both tiers size themselves from `tests.parallel.available_cores()`, and each one expects the whole
machine. Running them together oversubscribes every core and takes longer than running them in
sequence. Finish the fast tier before you start a re-execution run or a regeneration.

## Adding a study

To register a new validation study rather than a slow test, follow
[method benchmarking strategy](method-benchmarking.md).
