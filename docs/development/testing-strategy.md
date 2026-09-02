# Fast tests and validation studies

This page separates the checks that run before every pull request from statistical studies that
run only when a relevant implementation changes. `README.md` gives the fast command.
[Method benchmarking](method-benchmarking.md) gives the study design and registration rules.

## The two checks

| check | command | what it establishes |
| --- | --- | --- |
| fast tests | `pytest -q -n auto --dist loadgroup` | unit, integration, end-to-end, documentation, provenance, artifact, and published-verdict behavior |
| registered validation study | the affected study directory's documented regeneration command | repeated-sampling, large-sample, flexible-learner, and reference-implementation claims |

Run the fast tests before every pull request. They recompute every published verdict and negative
control from committed study artifacts. They do not refit committed study replications.

Run a registered study only when a result-determining change can affect what that study computes.
The study regeneration draws every declared sample, fits both implementations, regenerates its
property cells, validates every gate, and writes a complete manifest.

The repository has no pytest slow tier. Partial re-execution duplicated study work and could not
replace regeneration. The registered studies also replace the deprecated repeated-sampling tests.

## What belongs in the fast suite

Use a fast test when one compact sample, an exact law, or a deliberate mutation can observe the
behavior. This includes these checks:

- public behavior, validation, serialization, and presentation;
- exact-law, Gateaux, remainder, identity, and mutation instruments;
- targeting, cross-fitting, nuisance routing, and inference formulas on compact data;
- artifact hashes, schemas, provenance, generated claims, and verdict recomputation;
- documentation parsing, links, and executable examples.

Use `tests.conftest.FAST_KWARGS` and explicit parametric learners for estimator tests. Use a
flexible learner only when flexible learning is the subject.

Do not add an expensive pytest marker. A claim that needs repeated sampling, large samples, many
flexible fits, or an external implementation belongs in a registered validation study.

## Choose affected studies

A change affects a study when it can move that study's inputs, fits, or verdicts. Typical examples
include these changes:

| change | action |
| --- | --- |
| an estimand, influence curve, variance, clustering, targeting, cross-fitting, nuisance prediction, or randomization path | regenerate every registered row that evaluates the path |
| a study law, margin, cell, seed, learner, estimator argument, schema, package pin, or reference runner | regenerate that study |
| documentation, formatting, comments, type annotations, or presentation that preserves fitted arrays | run fast tests only |
| validation that exits before estimator construction | run fast tests only |

Use the [implementation validation grid](../technical-reference/method-evidence/validation-grid.md)
to find the rows for a method. Read each row's coverage limits before selecting it. If several rows
evaluate the changed path, regenerate each one.

Record the selected study names and commands in the pull request. For regenerated evidence, also
name the artifacts that moved and those that stayed byte-identical.

## Result-neutral study edits

A study manifest records the Python modules that produced the run. A Python hash difference does
not fail the fast tests because a comment or extracted helper can change the bytes without changing
the result. Regenerate only when the edit changes what the study computes.

Reference Dockerfiles and R sources follow a stricter rule. The fast suite refuses an undeclared
hash difference. Regenerate a result-determining reference edit. Record a result-neutral edit in
`tests/canonical/provenance-revisions.md` without rewriting the recorded manifest hash.

Read [method benchmarking](method-benchmarking.md#what-makes-a-study-stale) for the complete
provenance contract.

## Add a validation study

Follow [method benchmarking](method-benchmarking.md) to register a new study. Pair each positive
claim with a control that must fail the same instrument.
