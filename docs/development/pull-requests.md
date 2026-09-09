# Pull requests

This page gives the commit style, the pull request body, what CI checks, and what CI does not
check. Read [contributing](contributing.md) first for the setup and the commands.

## Before you open one

Branch from `main`, and run the checks your change needs. The
[definition of done](../roadmap.md#definition-of-done) in the roadmap lists the conditions an item
has to meet. Its last condition governs this page: every relevant check has run locally, and GitHub
Actions is green. CI is the final merge signal, not a substitute for the local validation record.

## Write the commit message

The subject is one imperative sentence in sentence case. It carries no trailing full stop, no
conventional-commit prefix, and no scope tag. Keep it near 50 characters.

These are real subjects from the history.

```text
Add MSM validation studies
Correct four wrong claims the DR-TMLE study made about the code
Share the necessity rule, and stop publishing a false one
Deprecate the repeated-sampling studies the registered rows replace
```

The body carries the argument, and it is often long. Write it to answer four questions.

| question | what to write |
| --- | --- |
| what was wrong | the defect, and the mechanism that produced it |
| what changed | the fix, and the reason this fix rather than another |
| what moved | the measured effect, with the numbers and the tolerance |
| what did not move | the values, files, or behaviour that stayed identical |

End the body with the gate you ran, in this form.

```text
pytest: 5081 passed, 47 skipped.
```

Add a `Co-Authored-By:` trailer when a tool wrote part of the change.

## Shape the commits inside a branch

A branch normally follows one rhythm.

| commit | subject form | what it touches |
| --- | --- | --- |
| first | `Plan <topic>` | [the roadmap](../roadmap.md) alone |
| second | `Add <topic>` | the work itself |
| later | `Fix review findings on <topic>` | one review finding each |

Shape each commit with care. A pull request lands as a merge commit, so every commit on the branch
survives in the history of `main`.

## Write the pull request body

State the claim the change makes, and the evidence that supports it. Name the checks you ran. Tell
the reviewer which file to read first.

For an evidence pull request, name the artifacts that moved and the artifacts that stayed
byte-identical. A regeneration that changes nothing is a result, and it belongs in the body.

For an implementation pull request, name every registered study that evaluates the changed path.
Give the regeneration command for each selected study. State why no study applies when the change
cannot affect a fitted result.

## What CI runs on your pull request

`.github/workflows/ci.yml` runs on every pull request. It has six jobs.

| job | Python | command |
| --- | --- | --- |
| `lint` | 3.12 | `ruff check .`, and then `ruff format --check .` |
| `typecheck` | 3.12 | `mypy` |
| `docs` | 3.12 | `sphinx-build -W --keep-going -b html docs docs/_build/html` |
| `tests` | 3.11, 3.12, 3.13 | `pytest -q -n auto --dist loadgroup` |
| `minimal-install` | 3.11 | `python scripts/smoke_backend.py pandas`, and the same for `polars` |
| `package` | 3.12 | build, strict metadata and archive checks, and two clean artifact installs |

`nox` with no argument runs the `lint`, `typecheck`, `docs`, and `tests` sessions, which mirror the
first four jobs. `minimal-install` has no session, because it installs one dataframe backend and
nothing else. The package job builds both distribution formats before it creates clean smoke-test
environments.

A release-bearing pull request changes `src/cleverly/_version.py` to the next unused `0.1.N`
version. Read [releases](releases.md) for the version and tag rules.

The `docs` job builds the site that `.github/workflows/pages.yml` deploys, and it treats every
Sphinx warning as an error. It covers what no fast test can see: numpydoc validation of each
rendered docstring, a document that no toctree references, and a cross-reference Sphinx cannot
resolve. Run `nox -s docs` before you push, because the whole build takes several minutes on a
runner.

## What CI does not check

Prose is not linted in CI. `tests/prose.py` reports on it, and the fast tier gates the report. A
finding with no recorded judgment fails. The prose itself never does.

No scientific claim is checked by the pull request beyond the verdicts each registered study
recomputes from its committed artifacts. Read
[method benchmarking strategy](method-benchmarking.md) before you make a claim, because the
evidence has to be designed before it is run.

## Evidence pull requests

A pull request that regenerates a registered study carries three extra obligations.

1. Regenerate every published table with `python -m tests.studies.evidence.document`. Quote no
   measured number by hand.
2. Check that the manifest hashes match the artifacts you committed. The fast tier recomputes each
   published verdict from those artifacts, so the hashes are what tie the two together.
3. Write the artifacts with line-feed endings. `.gitattributes` sets `* text=auto eol=lf` and marks
   `*.gz` as binary, because a hashed artifact written with carriage returns passes on Windows and
   fails on Linux CI.

Read [method benchmarking strategy](method-benchmarking.md) for the study design rules, and
[fast tests and validation studies](testing-strategy.md) for how the checks divide the work.

## Review and merge

One person reviews. Expect findings, and answer each one in its own commit rather than in a
comment alone.

A pull request lands as a GitHub merge commit, so the branch commits stay in the history. The
remote holds `main` alone, so delete your branch after the merge.
