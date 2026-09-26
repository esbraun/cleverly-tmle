# Working on cleverly

`cleverly` is alpha software under heavy development. Treat the current architecture as
provisional: inspect the code and tests before changing it, and do not preserve an implementation
shape solely because this file, a plan, or an investigation note once described it. Code and tests
determine current behavior.

This file collects working agreements for coding agents. Related guidance is routed below:

| for | read |
| --- | --- |
| setup, development commands, the public overview | `README.md` |
| making a change, and the checks it needs | `docs/development/contributing.md` |
| opening a pull request, and the commit style | `docs/development/pull-requests.md` |
| where technical documentation lives | `docs/README.md` |
| cross-module constraints not derivable from one implementation | `docs/architecture-invariants.md` |
| fast tests and selective validation studies | `docs/development/testing-strategy.md` |
| designing and registering a validation study | `docs/development/method-benchmarking.md` |
| which instrument covers which estimand | `docs/technical-reference/evidence.md` |
| what each shipped method was validated against | `docs/technical-reference/index.md` |

## Compatibility

`cleverly` is alpha, and it keeps no backward compatibility: not for saved results, pickled
objects, or `CausalStudy` objects, and not for the public API.

- Do not write migration code, pickle backfills, `__setstate__` field renames, re-stamps of saved
  artifacts, deprecated aliases, or refusals that exist only to name a removed keyword. Remove or
  rename the old thing, and update its callers, tests, and docs in the same change.
- `result.save()` and `cleverly.estimators.serialize.dumps()` record the version that wrote an
  artifact. On a different version, `cleverly.load()` and `loads()` emit
  `VersionMismatchWarning` and load the artifact as saved. That warning is the whole
  cross-version contract (`cleverly._saved_version`).
- Do not add a test that loads an old-format artifact, and do not open a roadmap item for one,
  other than the tests of the warning itself in `tests/unit/test_serialization.py`.
- Cache keys, study provenance hashes, and branches on external library versions are not
  compatibility code. Keep them.

## Scientific changes

- Exact-law checks are blind to terms that vanish at the truth. When a sign, mask, guard, or
  counterfactual block can disappear, add a nonzero witness or a deliberate-mutation control that
  fails when that component is wrong.
- Refuse unsupported but well-posed compositions explicitly and explain what is missing. Do not
  silently return a convenient approximation to a different estimand.

## Tests and tooling

- Read `docs/development/testing-strategy.md` before choosing checks. The fast suite
  (`pytest -q -n auto --dist loadgroup`) is the default handoff gate. A shipped method is
  validated by its rows in the implementation validation grid. The fast suite recomputes their
  verdicts from committed artifacts. Regenerate only the studies whose results a
  result-determining change can move.
- A refactor is not a reason to regenerate a study. The Python module hashes in a study's
  `manifest.json` record the run; no test gates them, so cleaning shared code under
  `tests/studies/evidence/` is free. `tests/unit/test_study_provenance.py` does check that the
  module list is complete. The container and R-runner hashes *are* gated. Declare a
  result-neutral edit in `tests/canonical/provenance-revisions.md` rather than rewriting a
  recorded hash, which would leave the manifest describing bytes that never ran.
  `docs/development/method-benchmarking.md` says how to tell the two kinds of change apart.
- Ruff and mypy are pinned once in `pyproject.toml`'s `dev` extra, which resolves to
  `cleverly[all]` plus tooling. For a new extra, follow
  [Add an optional dependency](docs/development/contributing.md#add-an-optional-dependency).
- Ruff *formats* the Python examples in Markdown, so run it over the whole tree. Its linter does
  not read Markdown at all, and the formatter skips any block it cannot parse. Neither one sees a
  syntax error in an example.
- Follow `docs/development/pull-requests.md` when you prepare a handoff. The `docs` CI job builds
  the site with `-W` on every pull request, so run `nox -s docs` before you hand off.

## Documentation writing

The root `README.md` and reader-facing documents under `docs/` align with Issue 9 of
ASD-STE100 Simplified Technical English. This project does not claim certified compliance.

- Write one idea per sentence. Keep sentences to 20 words in procedures and 25 in descriptions.
- Keep paragraphs to six sentences. Prefer three.
- Use the active voice, the present tense, and a named actor.
- Give the instruction first and the reason second.
- Use one word for one meaning, and do not use the same word as two parts of speech.
- Keep articles. Do not build a noun cluster longer than three words.
- Do not join clauses with an em dash or `--`. Use a full stop, or a table.
- When the content is parallel, write a table. In the technical reference the table is the primary
  communication device and the prose exists to define its terms.
- Statistical terms of art are exempt from the vocabulary restriction: influence curve, nuisance,
  targeting, remainder, estimand names, and any API identifier.
- Give evidence for each material claim. Cite the source, name the test or artifact, or state the
  applicable condition. Remove adjectives and transitions that add no verifiable information.

When you change a reader-facing document, follow the
[documentation procedure](docs/development/contributing.md#write-documentation).
Run `python -m tests.prose` and review every finding. `--path <file>` is for reports only.
After edits, refresh the whole ledger with `python -m tests.prose --update`.
Record `accepted: <reason>` for each retained finding in `tests/prose-report.md`, keeping existing
reasons if they still apply. The fast tier rejects unjudged findings and stale ledger rows.
Check scientific claims against code, tests, artifacts, and sources. No tool here
certifies STE compliance or verifies a scientific claim.

Scope is `README.md` and every reader-facing Markdown, RST, or notebook source under `docs/`.
Generated API pages and `docs/_build/` are not source. Rewrite the text a change touches. Do not
sweep unrelated pages unless the user requests a broad documentation review.

## Docstrings

Docstrings are numpydoc, and `sphinx.ext.napoleon` is not installed, so the loose `name:` form is a
build error (`docs/conf.py` lists the enabled checks). Write `name : type` always, one entry per
parameter. Two names on one line become one parameter with a comma in its name.

- Document a frozen dataclass's fields under `Parameters`. numpydoc reads the generated signature,
  so a field described only under `Attributes` reads as undocumented. Reserve `Attributes` for
  derived properties.
- Give a property-backed attribute its name and type and no description. numpydoc renders the
  property's own docstring over anything written there.
- Where a synthetic signature produces a finding nobody can fix, use the inline
  `# numpydoc ignore=PR01` form on the definition line. Do not use `numpydoc_validation_exclude`,
  which drops the object from every check rather than one.
- `Examples` and `See Also` are required on the task spine only, which
  `tests/unit/test_documentation_api.py:EXAMPLE_TARGETS` declares. Every See Also entry carries a
  description.
- Show the smallest normal use of the documented object or method. Start with the common case.
  Include every import, use compact data, and check deterministic output. Do not use an example
  to assert a statistical guarantee from one sample.
- An example must run in the fast tier without `doctest.SKIP`. Pass explicit learners to fits;
  the default learner library costs 30 to 120 seconds per fit. Put expensive studies and extended
  comparisons in narrative documentation and test them separately.
