# Pull requests

This page gives the commit style, the pull request body, what CI checks, and what CI does not
check. Read [contributing](contributing.md) first for the setup and the commands.

## Before you open one

Branch from `main`, and run the checks your change needs. The
[definition of done](../roadmap.md#definition-of-done) in the roadmap lists the conditions an item
has to meet. Its last condition governs this page: every relevant check has run locally, and GitHub
Actions is green. CI is the final merge signal, not a substitute for the local validation record.

## Bring the docs up to date

Every pull request reviews the pages its change can make stale, and fixes them, before it opens.
`tests/docs_map.py` says which pages each part of the code affects, and
`python -m tests.docs_impact` reads your change against it. Fetch `main` first, write your draft
pull request body to a file, and run the checker on your working tree.

```text
git fetch origin main
python -m tests.docs_impact --worktree --body-file body.md
```

It lists four kinds of finding. A **routed** page belongs to an area of the map that your change
touches. A **removed** finding is a page that still quotes a public name the change removed (a
function, class, method, field, parameter or `Literal` option that no module offers any more). A page calls an
old value the **default** of a parameter whose default changed. An **added** public name of a
published module is mentioned on no page. The report also lists pages worth a look that never
fail, such as the tutorials of a method: their printed numbers are already gated by the
notebook freshness test.

For each finding, read the page against your diff and rewrite what the change made false. Follow
[writing for readers](writing-for-readers.md), and for a fact that a test ties to the code, fix
the owning page and link it rather than copy it
([facts a reader page borrows](writing-for-readers.md#facts-a-reader-page-borrows)). Then run the
[documentation procedure](contributing.md#write-documentation). A routed page is satisfied by an
edit beyond whitespace, a quoted removed name or old default by the page no longer quoting it,
and an addition by some page mentioning it.

When a finding needs no change, put a waiver line in the pull request body that says why:

```text
Docs-waiver: docs/user-guide/settings.md: the new field is private and no fit reads it
Docs-waiver: cluster_floor: set only by the study harness, never by a user
Docs-waiver: all: a pure refactor; every public output and message is byte-identical
```

A reason needs at least four words, and the reviewer reads it. `all` covers routed pages only,
never a removed name, an old default or an addition. A waiver that names nothing the change
affects fails, so a waiver goes stale when a later commit removes its cause; delete it then. The
check reads paths, names and reasons, never prose: an edited page passes it, and only the review
shows the edit is right.

A change to a file the map does not cover fails `tests/unit/test_docs_map.py` in the fast suite.
Add the file to the area whose pages it can make stale, or, for a private module that changes
nothing a reader sees, to `INTERNAL` with the reason. The check judges a change by the union of
the base branch's map and the changed map, so narrowing the map does not excuse the change that
narrows it.

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

`.github/workflows/ci.yml` runs on every pull request. It has eight jobs.

| job | Python | command |
| --- | --- | --- |
| `lint` | 3.12 | `ruff check .`, and then `ruff format --check .` |
| `typecheck` | 3.12 | `mypy` |
| `docs` | 3.12 | `sphinx-build -W --keep-going -b html docs docs/_build/html` |
| `tests` | 3.11 and 3.13 in 6 shards each, 3.12 in 8 | `pytest -q -n auto --dist loadgroup` with `CLEVERLY_TEST_SHARD=i/N`. On 3.11 and 3.13 it adds `--ignore=tests/unit/test_method_evidence.py`. The `tests` check reports every shard together |
| `slow-tests` | 3.12 | `pytest -q -n auto --dist loadgroup -m slow`: each tutorial notebook with its refit cells |
| `property-differential` | 3.12, in 6 parts | `pytest -q -n auto --dist loadgroup tests/canonical/property_differential.py` with `PROPERTY_DIFFERENTIAL_PART=i/N`, when the change touches a study, `tests/canonical/`, `src/cleverly`, the dependency pins, or the CI files |
| `minimal-install` | 3.11 | `python scripts/smoke_backend.py pandas`, and the same for `polars` |
| `package` | 3.12 | build, strict metadata and archive checks, and two clean artifact installs |

`.github/workflows/docs-impact.yml` runs one more check, `docs-impact`, the
[docs check](#bring-the-docs-up-to-date) above. It is its own workflow because it runs again when
you edit the pull request body, where the waivers live, and that must not restart the tests. It
installs nothing and takes about a minute.

The study evidence checks run on 3.12 only. They recompute the published verdicts from committed
files with test code, and they took about half of the 2-hour 3.12 run before it was sharded.
A shard keeps each `xdist_group` whole (`tests/conftest.py`). Set `CLEVERLY_TEST_SHARD=i/N` to run one locally.
3.12 has more shards because it runs those checks too. Each shard and each part prints the
runner's memory, swap and disk before its tests, and the least headroom it had at the end. A
runner that runs out of either dies with no log, so a shrinking margin in a job that passed is
the warning.

The `tests` and `property-differential` checks are summary jobs that report their shards or parts
together. They run when a shard fails, but not when the run is cancelled. Otherwise a cancelled
run stays open until its summary job finds a runner, and holds back the next push to the same pull
request. A run cancelled on the head commit therefore shows those two checks as skipped,
which branch protection counts as passing. Re-run it before you merge.

`nox` with no argument runs the `lint`, `typecheck`, `docs`, `tests`, and `slow` sessions, which
mirror the first five jobs. `minimal-install` has no session, because it installs one dataframe
backend and nothing else. The package job builds both distribution formats before it creates clean smoke-test
environments.

A release-bearing pull request changes `src/cleverly/_version.py` to the next unused `0.2.N`
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
   `*.gz` and `*.parquet` as binary, because a hashed artifact written with carriage returns passes on Windows and
   fails on Linux CI.

Read [method benchmarking strategy](method-benchmarking.md) for the study design rules, and
[fast tests and validation studies](testing-strategy.md) for how the checks divide the work.

## Review and merge

One person reviews. Expect findings, and answer each one in its own commit rather than in a
comment alone.

A pull request lands as a GitHub merge commit, so the branch commits stay in the history.
GitHub deletes the branch when the pull request merges.

## Branches

Push a branch whenever the work needs one. A weekly job deletes the branches nobody needs, so do
not clean up by hand. It runs every Monday at 06:30 UTC. It deletes a branch that no open pull
request uses and no keep list names, once its work is done or it has been idle for 14 days. Done means its tip is in `main`, a closed pull request ended at its tip, or every file it
changed reads the same on `main`, as when its results were copied into another pull request. A
done branch waits 3 days after its last push, and idle means no commit or push.

Before it deletes a branch, the job saves the tip as `refs/archive/branches/<branch>/<date>`. That
ref is not a branch, so clones and the branch list do not show it, but it keeps the commits. To
see the saved copies, run `git ls-remote origin 'refs/archive/branches/*'`. To bring one back,
fetch its ref with `git fetch origin refs/archive/branches/<branch>/<date>`, then run
`git push origin FETCH_HEAD:refs/heads/<branch>`.

To keep a branch that will sit idle, such as a held result or a long run's checkpoint branch, add
a line `<branch or glob> <reason>` to `.github/branch-keep.txt` through a pull request. Delete the
line when the branch is no longer needed. An open pull request also protects its branch.
