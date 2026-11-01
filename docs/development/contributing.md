# Contributing

This page tells you how to set up `cleverly`, which checks your change has to pass, and where each
working rule is written down. Read [pull requests](pull-requests.md) before you open one. That page
gives the commit style, the pull request body, and what each CI job checks.

## What to expect

One person maintains `cleverly`, so a review can take several days. Open an issue before you start
a large change. An early conversation is cheaper than a rejected branch.

`cleverly` is beta software. The public API can still change between releases. A result that one
release saves loads in another release with a `VersionMismatchWarning`, and the load does not
migrate it. Do not add compatibility code for an earlier release. The "Compatibility" section of
`CLAUDE.md` states the rule.

The [roadmap](../roadmap.md) is the single planning contract. It orders open work by beta
milestone. It keeps work that needs new fundamental theory in a separate future grid. Read its
[eligibility rule](../roadmap.md#eligibility) before you propose a new method. `cleverly`
accepts established methods and natural extensions of established theory.

The eligibility rule names three acceptance routes: an established method, an acceptable
implementation comparison, and a natural extension. A missing comparator never blocks a method,
as [theory and comparators](../architecture-invariants.md#theory-and-comparators) states. A
comparator run that has not finished does not block a method either. The method merges when its
Python studies pass, and a later change registers the comparator results.

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
| `pytest -q -n auto --dist loadgroup -m slow` | the slow tier: each tutorial notebook with its refit cells |
| `python -m tests.prose` | a report on the reader-facing prose. It changes nothing and fails nothing |
| `nox -s docs` | the documentation build, with every Sphinx warning as an error |
| `python -m build`, then package checks | distribution metadata, contents, and clean installs |

`nox` with no argument runs the `lint`, `typecheck`, `docs`, `tests`, and `slow` sessions. Those
sessions mirror the CI jobs, so a green `nox` run predicts a green pull request.

Do not run a study regeneration beside the fast suite. Each process expects the machine's cores.
[Fast tests and validation studies](testing-strategy.md) explains the design behind the commands.

Read [releases](releases.md) before you change `src/cleverly/_version.py` or create a tag.

## Choose the checks your change needs

| your change | run |
| --- | --- |
| documentation, or a docstring | `ruff format --check .`, `python -m tests.prose`, `pytest -q -n auto --dist loadgroup`, `nox -s docs` |
| library code, or a test | `ruff check .`, `ruff format --check .`, `mypy`, `pytest -q -n auto --dist loadgroup` |
| a regenerated evidence artifact | the row above. The fast suite recomputes each study's verdicts from the artifacts you commit |
| a notebook artifact, or a tutorial callback | `python scripts/execute_notebook.py <path>`, then the documentation row above, then `pytest -q tests/unit/test_documentation_runtime.py -k <stem> -m slow` |

The fast suite is the handoff gate for every row. It recomputes registered-study verdicts from
committed artifacts, so the statistical evidence is checked in minutes rather than hours.

The notebook command executes every cell and replaces the stored outputs. The TWINS notebook needs
network access, because it downloads a pinned dataset. The tutorial notebooks run offline. The
command then writes a stamp with two halves, and only one half can fail the fast suite.
[Example notebooks](example-notebooks.md) gives the page outline, the rules, and the procedure for
a tutorial notebook.

| stamp half | covers | the fast suite |
| --- | --- | --- |
| gated | the notebook's own code cells and stored outputs, its schema, and the repair command | asserts it equal |
| recorded | the shipped package source, the dependency lock, and the generator | asserts it present and well formed |
| recorded | the commit, the version, and the generator's file list | asserts the file list matches this checkout |

Edit a code cell and the gated half fails until you run the command again. Rename a notebook and
the recorded command fails for the same reason. Edit library code and no notebook gate fails. The
recorded half fingerprints the run's repository context. It does not reconstruct a dirty tree or
constrain the tree you work in. `tests/notebooks.py` gives the reason for the split.

A hash comparison cannot see a library change that moved a published number while every cell kept
its bytes. Run `python scripts/execute_notebook.py <path> --check` to find one. The command
re-executes the notebook and reports each cell whose printed text moved. It writes nothing.

Regenerate every registered study that evaluates a result-determining implementation change. Read
[fast tests and validation studies](testing-strategy.md) to select the affected rows.

## Write documentation

[Writing for readers](writing-for-readers.md) holds the writing rules. It names the reader of each
documentation area: a data scientist new to causal inference for the examples, the getting-started
pages, and the user guide; a statistician for the technical reference; and contributors for these
development pages. Write for that reader. On a data-scientist page, lead with the answer, and define
each technical term in plain words where it first appears. Use a table only when the reader compares
items across the same columns. When you rewrite a whole data-scientist page, or write a new one, run
the guide's reader review before you open the pull request.

`docs/README.md` is the source map. It states which document holds which record. It also states the
examples contract: a new method entry needs a new tutorial, and the two link to each other in both
directions.

Follow these steps when you change a document.

1. Rewrite the text your change touches. Do not sweep unrelated pages.
2. Run `python -m tests.prose` and read every finding it reports. Use `--path <file>` for a report
   on one file.
3. Fix each sentence where the change improves its meaning or clarity.
4. After edits, run `python -m tests.prose --update` to refresh the whole ledger. Do not combine
   `--update` with `--path`.
5. For each retained finding, record `accepted: <reason>` in the last column of
   `tests/prose-report.md`. Keep an existing reason if it still applies.

The fast tier rejects findings without an accepted reason and stale ledger rows. Refresh the
ledger even when you fix every finding. A mechanical edit that breaks a sentence defeats the
report's purpose. Keep the sentence whole and record why you accept the finding.

## Write docstrings

Docstrings are numpydoc, and `sphinx.ext.napoleon` is not installed. Write `name : type` always,
and give one entry for each parameter. The documentation job builds with `-W`, so a malformed
docstring breaks the published site.

An example runs in the fast tier. Pass explicit learners to every fit, because the default learner
library costs 30 to 120 seconds per fit. `CLAUDE.md` gives the complete docstring rules.

## Write a summary

A printed summary is for the person who ran the analysis. Model it on statsmodels: a title, a
header of facts in two columns, the estimates table, numbered warnings and notes, and longer
sections only at `detail="full"`. Build it from `cleverly._summary`, never with hand-made
strings.

1. Take every label, note text, and plain name from `cleverly._summary.vocabulary`. A new note
   gets a code in `NOTE_CODES` and a template in `NOTE_TEXTS`. A number in a note comes through a
   placeholder from the constant that sets it, never written into the template.
2. Build the estimates table with `cleverly._summary._estimates.estimates_table`. It is the one
   place a summary applies the inference-status rule of
   [the invariants](../architecture-invariants.md), so a spread that is not a standard error
   never prints under that name.
3. Leave library bookkeeping out at both details: hashes, fingerprints, versions, time stamps,
   seeds, run ids, registry or roadmap ids, cache keys, and the internals of the targeting step.
   They stay on the object.
4. Say what a condition means for the reader's number, in plain words, and keep every condition
   the statement needs. A note that only restates a warning replaces it
   (`RESTATED_WARNINGS`).
5. Keep a configuration object's generated repr. A fitted object's one-line repr comes from
   `short_repr`, and an object with a rounded repr names its content through `_cache_token`, so
   two different objects never share an assessment cache key.
6. Add fixtures to a module in `tests/unit/summary_fixtures/`, compact and with explicit
   learners, so `tests/unit/test_summary_contract.py` checks fidelity against `to_frame()`, width,
   escaping, and the absence of internals for every one. Each new note code fires on a fixture,
   or on a literal check when no compact fit reaches it.

## Add a method, or a validation study

Read [method benchmarking strategy](method-benchmarking.md) first. Its governing rule is that a row
is not written. It is earned by registering a study.

| step | where |
| --- | --- |
| confirm the method is eligible | [eligibility](../roadmap.md#eligibility) in the roadmap, and [theory and comparators](../architecture-invariants.md#theory-and-comparators) |
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
| the `cleverly` version and commit hash | a release can differ from later Git snapshots |
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
