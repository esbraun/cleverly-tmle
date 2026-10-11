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
Choose a complete rerun or an audited targeted rerun under the conditions below.
Both routes retain every declared replication and recompute each affected summary and verdict.

The repository has no pytest slow tier. Registered studies replace the deprecated repeated-sampling tests.
An unrecorded partial probe cannot replace published study evidence.

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
| a study law, margin, cell, seed, learner, estimator argument, schema, package pin, or reference runner | regenerate the affected evidence; use a targeted rerun only when its eligibility conditions hold |
| documentation, formatting, comments, type annotations, or presentation that preserves fitted arrays | run fast tests only |
| validation that exits before estimator construction | run fast tests only |

Use the [implementation validation grid](../technical-reference/method-evidence/validation-grid.md)
to find the rows for a method. Read each row's coverage limits before selecting it. If several rows
evaluate the changed path, regenerate each one.

Record the selected study names and commands in the pull request. For regenerated evidence, also
name the artifacts that moved and those that stayed byte-identical.

## Targeted reruns

Use a targeted rerun when an audit identifies every affected replication before fitting replacement samples.
A seed collision repair can meet this condition without refitting samples whose seeds stay unchanged.
When the conditions below hold, the study can reuse unchanged fits.

| condition | required evidence |
| --- | --- |
| bounded change | identify affected replication keys from inputs or code, before reading replacement outcomes |
| fixed design | preserve the laws, learners, sample sizes, budgets, margins, pairing, and reading rules |
| reusable evidence | pin the original artifact commit and verify every inherited artifact hash |
| unchanged path | show that inherited samples and fitted quantities retain their result-determining inputs and code |
| independent fits | construct fresh estimators and learners for each replication; exclude shared training state and execution-order dependence |
| runtime | match the original runtime for replacement fits; independently investigate any unexplained replay difference |
| complete accounting | replace every affected key exactly once; retain every unaffected row and all declared counts |
| dependent fits | rerun both sides of an affected paired comparison and every dependent fit |
| analysis | recompute all affected summaries, bootstrap intervals, controls, and verdicts from the complete combined rows |
| provenance | distinguish inherited evidence from replacement fits and recomputed analyses in the manifest |
| verification | independently check the selection, preserved evidence, replacement fits, complete analysis, and manifest |

Select replacement seeds from the declared allocation rule. Do not select samples from their fitted results.

Keep an unchanged replay witness when it can test the claimed reuse boundary.
A replay witness supports the code audit. It does not establish that every inherited fit is unchanged.
Declare any numerical replay tolerance before fitting. Require exact agreement for keys, counts, and discrete decisions.
Passing a tolerance alone does not explain a runtime or source difference.

Retain primary and reference files as exact bytes when the change cannot affect them.
Retain unaffected replication rows exactly, even when recompressing their complete file changes its hash.
Publish failed or unresolved verdicts under the original acceptance rules.
Do not repair a failed fit by dropping its row or drawing a more favorable sample.

Use a complete rerun when the audit cannot bound the change or verify the inherited evidence.
Estimator changes that affect every fit usually require this route.
Changing a runtime, learner, law, or reference can also invalidate the claimed reuse boundary.

[Method benchmarking](method-benchmarking.md#targeted-regeneration-record) defines the provenance record.
Record the targeted command, selected keys, baseline, preserved artifacts, and validation in the pull request.

## Result-neutral study edits

A study manifest records the Python modules that produced the run. A Python hash difference does
not fail the fast tests because a comment or extracted helper can change the bytes without changing
the result. Regenerate only when the edit changes what the study computes.

Reference Dockerfiles and R sources follow a stricter rule. The fast suite refuses an undeclared
hash difference. Regenerate a result-determining reference edit. Record a result-neutral edit in
`tests/canonical/provenance-revisions.md` without rewriting the recorded manifest hash.

Read [method benchmarking](method-benchmarking.md#what-makes-a-study-stale) for the complete
provenance contract.

## Resume a study run

`tests/canonical/regenerate.py` keeps the scratch of each run outside the repository. An R
reference stores the result of each group as a checkpoint. A rerun of the same command reuses the
Python phase, the calibration and each checkpoint. The rerun prints what it reused. Every error
stays fatal: the first error stops the phase with its message.

A run shares its cache with a rerun only when the two have one *resume key*. The key holds the
declared study without its output path, `HEAD` and the tree state. It also holds the reference
sources, the R harness, the content of the samples and the truth, n, the replicate count and the
Python versions. A change to any of these starts a new cache.

| item | location or command |
| --- | --- |
| host scratch | `%LOCALAPPDATA%/cleverly/runs/<slug>/<key>/` on Windows, `~/.cache/cleverly/runs/<slug>/<key>/` elsewhere. `--cache DIR` replaces it |
| R checkpoints | `/cache/<slug>/<key>/<runner>__<output>/` in the Docker volume `cleverly-cache` |
| discard the cache of this key | add `--fresh` |
| host lock | `_lock` in the host scratch. A second driver on one key refuses and names the holder. The driver removes the lock of a dead process |
| container lock | the container name `cleverly-<slug>-<key>-<leaf>`. Docker refuses a second live container of one name |
| the RM39 smoke | `python -m tests.canonical.harness_fixture.smoke` |

A run that completes deletes its key directory in the volume and its default host scratch. It does
not delete a `--cache` directory. A declared run keeps both until its pull request merges, for
audit. The run prints both paths, and the implementer records them in the row's progress file.
Delete them after the merge with these commands:

```text
docker run --rm -v cleverly-cache:/cache alpine rm -rf /cache/<slug>/<key>
rmdir /s /q "%LOCALAPPDATA%\cleverly\runs\<slug>\<key>"
```

The driver sizes the R workers from measured memory. A calibration run fits the first group of
each scenario in a fresh fork. It measures the peak and the private memory of each fork.
The harness then sets the workers to 0.85 of the available memory over 1.25 times the per-worker
cost.

A runner can declare `Reference(worker_memory_mb=...)` as a floor, or
`Reference(memory_override=(factor, share, reason))`. At its start, a phase logs the memory, the
calibration, the plan and its binding reason. Each retry pass logs `MemAvailable` and the workers.

| event | what the run does |
| --- | --- |
| a killed worker | the next pass halves the workers and fits the group again, alone |
| a second kill of one group | the phase stops with exit 3 and names the group. A rerun resumes |
| a pass that makes no progress | the phase stops with exit 3. A rerun resumes |
| a container killed whole (exit 137) | the driver runs the container once more with `CLEVERLY_R_WORKER_CAP` at half the last workers |
| a second container kill | the run stops. A rerun resumes |

Give the study machine a memory limit in `.wslconfig`, so a memory overrun kills a container and
not the host. The log line `MemTotal ... MemAvailable ... cgroup limit ...` shows what the
container saw.

Run the RM39 smoke after an edit to `tests/canonical/study_harness.R`, to a fixture runner, or to
the `tmle3` Dockerfile. The smoke runs in Docker, because CI installs no R. It commits its results
and `tests/canonical/harness_fixture/fixture-manifest.json`. The fast suite fails until the
manifest names the current bytes of each of those files.

## Notebook artifacts

A committed notebook stores the outputs of a run rather than recomputing them. The fast suite
checks that its stamp still matches its code cells and stored outputs. This detects an edit after
stamping, but the unkeyed hashes do not prove that one payload produced the other. Re-execute with
`python scripts/execute_notebook.py <path>`. The command needs network access.

The stamp it writes has two halves. The gated half covers the notebook alone, and the fast suite
asserts it equal. The recorded half fingerprints the repository context of the run. The fast suite
asserts only that it is present and well formed. A library edit therefore fails no notebook check.
`tests/notebooks.py` gives the reason, and it is the reason a study manifest does not gate its
Python module hashes either.

A hash comparison cannot see a library change that moved a published number while every cell kept
its bytes. Run `python scripts/execute_notebook.py <path> --check` to find one. The command
re-executes the notebook, reports each cell whose printed text moved, and writes nothing.

## Add a validation study

Follow [method benchmarking](method-benchmarking.md) to register a new study. Pair each positive
claim with a control that must fail the same instrument.
