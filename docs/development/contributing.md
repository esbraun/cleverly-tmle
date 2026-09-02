# Contributing

This page tells you how to set up `cleverly`, which checks your change has to pass, and where each
working rule is written down. Read [pull requests](pull-requests.md) before you open one. That page
gives the commit style, the pull request body, and what each CI job checks.

## What to expect

One person maintains `cleverly`, so a review can take several days. Open an issue before you start
a large change. An early conversation is cheaper than a rejected branch.

`cleverly` is alpha software. The public API can change between `0.1.N` releases.

The [roadmap](../roadmap.md) is the single planning contract. It puts source-backed work in one
binding sequence. It keeps work without published theory in a separate future grid. Read its
[eligibility rule](../roadmap.md#eligibility) before you propose a new method. `cleverly`
implements established statistical methods. It does not use a package feature as the place to
invent one.

## Set up your environment

```bash
git clone https://github.com/esbraun/cleverly-tmle.git
cd cleverly-tmle
uv venv
uv pip install -e ".[dev,docs]"
```

The `dev` extra pins Ruff and mypy to exact versions, and `uv.lock` resolves them. CI and `nox`
install the same extra. Do not restate a tool version anywhere else.

The `docs` extra adds Sphinx. Install it so you can run the documentation build directly, which is
faster than the isolated `nox` session.

If an editable install points at another worktree, the test suite refuses to run and names both
paths. Reinstall the current checkout to fix that.

## Pick a branch name

Branch from `main`. Use a lowercase prefix and a hyphenated topic.

| prefix | use it for | example |
| --- | --- | --- |
| `agent/` | a change to the library or to the tests | `agent/ltmle-crossfit-evidence` |
| `docs/` | a change to documentation only | `docs/simplify-navigation-and-structure` |
| `evidence/` | a new registered validation study | `evidence/msm-studies` |

The remote holds `main` alone. Delete your branch after it merges.

## Run the checks

| command | what it gates |
| --- | --- |
| `ruff check .` | the lint rules in `pyproject.toml` |
| `ruff format --check .` | formatting, including every Python fence in a Markdown file |
| `mypy` | types in `src/cleverly` and in `scripts` |
| `pytest -q -n auto --dist loadgroup` | the fast suite, which is the default handoff gate |
| `python -m tests.prose` | a report on the reader-facing prose. It changes nothing and fails nothing |
| `nox -s docs` | the documentation build, with every Sphinx warning as an error |
| `python -m build`, then package checks | distribution metadata, contents, and clean installs |

`nox` with no argument runs the `lint`, `typecheck`, `docs`, and `tests` sessions. Those sessions
mirror the CI jobs, so a green `nox` run predicts a green pull request.

Do not run a study regeneration beside the fast suite. Each process expects the machine's cores.
[Fast tests and validation studies](testing-strategy.md) explains the design behind the commands.

Read [releases](releases.md) before you change `src/cleverly/_version.py` or create a tag.

## Choose the checks your change needs

| your change | run |
| --- | --- |
| documentation, or a docstring | `ruff format --check .`, `python -m tests.prose`, `pytest -q -n auto --dist loadgroup`, `nox -s docs` |
| library code, or a test | `ruff check .`, `ruff format --check .`, `mypy`, `pytest -q -n auto --dist loadgroup` |
| a regenerated evidence artifact | the row above. The fast suite recomputes each study's verdicts from the artifacts you commit |

The fast suite is the handoff gate for every row. It recomputes registered-study verdicts from
committed artifacts, so the statistical evidence is checked in minutes rather than hours.

Regenerate every registered study that evaluates a result-determining implementation change. Read
[fast tests and validation studies](testing-strategy.md) to select the affected rows.

## Write documentation

The root `README.md` and every reader-facing document under `docs/` align with Issue 9 of
ASD-STE100 Simplified Technical English. `CLAUDE.md` states the rules in full. The short form is
one idea per sentence, the active voice, the present tense, a named actor, and a table wherever the
content is parallel.

`docs/README.md` is the source map. It states which document holds which record. It also states the
examples contract: a new method entry needs a new tutorial, and the two link to each other in both
directions.

Follow these steps when you change a document.

1. Rewrite the text your change touches. Do not sweep unrelated pages.
2. Run `python -m tests.prose` and read every finding it reports.
3. Fix the sentence, or run `python -m tests.prose --update` and write `accepted: <reason>` in the
   last column of `tests/prose-report.md`.

The fast tier fails on a finding that carries no recorded judgment. It never fails on the prose
itself. A mechanical edit that satisfies a rule and breaks a sentence is the failure the report
exists to prevent, so the reason you record is the point of the exercise.

## Write docstrings

Docstrings are numpydoc, and `sphinx.ext.napoleon` is not installed. Write `name : type` always,
and give one entry for each parameter. The documentation job builds with `-W`, so a malformed
docstring breaks the published site.

An example runs in the fast tier. Pass explicit learners to every fit, because the default learner
library costs 30 to 120 seconds per fit. `CLAUDE.md` gives the complete docstring rules.

## Add a method, or a validation study

Read [method benchmarking strategy](method-benchmarking.md) first. Its governing rule is that a row
is not written. It is earned by registering a study.

| step | where |
| --- | --- |
| confirm the method is eligible | [eligibility](../roadmap.md#eligibility) in the roadmap |
| register the study, with margins declared before the run | `tests/studies/evidence/registry.py` |
| name every key the study reports | `tests/studies/evidence/descriptions.py` |
| generate the published tables | `python -m tests.studies.evidence.document` |

Pair each positive claim with a control that fails. A double-robustness claim needs a both-wrong
nuisance control. A type-I error claim needs a nonzero-effect power control.

Quote no measured number by hand. The generator writes each table between its generated sentinels,
and a fast test checks the result against the committed artifacts.

## Add an optional dependency

Put a new extra inside the `dev` extra, or give it a CI job that installs it and runs its tests. An
extra that no session installs makes its tests skip, and a skipped correctness check reads exactly
like a passing one.

## AI-assisted contributions

AI-assisted work is welcome, and this project uses it. You own the correctness of what you submit,
and you answer each review finding yourself.

- Add a `Co-Authored-By:` trailer when a tool wrote part of the change. That is the convention here.
- Check every citation against the source, and every measured number against the artifact that
  produced it.
- A generated statistical claim clears the same evidence gate as any other claim.
- Add a nonzero witness or a deliberate-mutation control wherever a sign, mask, guard, or
  counterfactual block can vanish at the truth. An exact-law check is blind to a term that
  disappears at the truth.

`CLAUDE.md` holds the working agreements a coding agent needs, and `AGENTS.md` points at it.

## Report a bug

The project has no issue templates yet. Include these items in the issue.

| item | why it is needed |
| --- | --- |
| the `cleverly` version and commit hash | alpha releases can differ from later Git snapshots |
| your Python version | the supported versions are 3.11, 3.12, and 3.13 |
| your dataframe backend | some code paths differ between pandas and polars |
| the learners you passed | estimator behaviour depends on the learner, not on `cleverly` alone |
| a runnable script, with compact data | a reproducer that runs in seconds gets a faster answer |
| the full traceback | the message alone rarely names the failing module |

## License

`cleverly` is under the MIT License. Your contribution to the project is licensed under the
same terms.

One directory differs. `tests/canonical/` is under the GNU General Public License v3.0,
because its R runners call reference packages in the same process. A contribution to that
directory is licensed under the GPL. No published distribution carries it.
