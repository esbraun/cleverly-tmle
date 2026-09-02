# Working on cleverly

`cleverly` is alpha software under heavy development. Treat the current architecture as
provisional: inspect the code and tests before changing it, and do not preserve an implementation
shape solely because this file once described it.

This file holds only the working agreements that no other file states and no test enforces.
Everything else is routed:

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

Current behavior is determined by code and tests, not by historical plans or investigation notes.

## Scientific changes

- Exact-law checks are blind to terms that vanish at the truth. When a sign, mask, guard, or
  counterfactual block can disappear, add a nonzero witness or a deliberate-mutation control that
  fails when that component is wrong.
- Refuse unsupported but well-posed compositions explicitly and explain what is missing. Do not
  silently return a convenient approximation to a different estimand.

## Tests and tooling

- Read `docs/development/testing-strategy.md` before choosing checks. The fast suite is the default
  handoff gate. A shipped method is validated by its rows in the implementation validation grid.
  The fast suite recomputes their verdicts from committed artifacts. Regenerate only the studies
  whose results a result-determining change can move.
- A refactor is not a reason to regenerate a study. The Python module hashes in a study's
  `manifest.json` record the run; no test gates them, so cleaning shared code under
  `tests/studies/evidence/` is free. The container and R-runner hashes *are* gated. Declare a
  result-neutral edit in `tests/canonical/provenance-revisions.md` rather than rewriting a
  recorded hash, which would leave the manifest describing bytes that never ran.
  `docs/development/method-benchmarking.md` says how to tell the two kinds of change apart.
- Ruff and mypy are pinned once in `pyproject.toml`'s `dev` extra, which resolves to
  `cleverly[all]` plus tooling. An optional extra kept out of `dev` *and* out of a dedicated CI job
  is installed by no session, so its tests can only skip, and a skipped correctness check reads
  exactly like a passing one. Put a new extra in `dev`, or give it a job that installs and runs its
  tests.
- Ruff *formats* the Python examples in Markdown, so run it over the whole tree. Its linter does
  not read Markdown at all, and the formatter skips any block it cannot parse. Neither one sees a
  syntax error in an example.
- Follow `docs/development/pull-requests.md` when you prepare a handoff. It gives the commit
  subject and body style, the evidence line the body carries, and what each CI job checks. The
  `docs` job builds the site with `-W` on every pull request, so a docstring that numpydoc rejects
  now fails the request rather than the deploy. Run `nox -s docs` before you hand off.

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

When you change a reader-facing document, run `python -m tests.prose`, review every finding it
reports, and plan a fix that keeps the sentence whole. Where the standard should not apply, record
`accepted: <reason>` against that finding in `tests/prose-report.md`; that is a passing outcome and
the reason is the point. The fast tier fails on a finding nobody has judged, never on the prose
itself, because a mechanical edit that satisfies a rule and breaks a sentence is the failure this
report exists to prevent. No tool here certifies STE compliance or verifies a scientific claim.
Check those against the code, tests, artifacts, and sources.

Scope is `README.md` and every reader-facing Markdown, RST, or notebook source under `docs/`.
Generated API pages and `docs/_build/` are not source. Rewrite the text a change touches. Do not
sweep unrelated pages unless the user requests a broad documentation review.

## Docstrings

Docstrings are numpydoc, and `sphinx.ext.napoleon` is not installed. The loose `name:` form that
napoleon accepted is now a build error, because `pages.yml` builds with `-W` and `docs/conf.py`
enables `GL06, GL07, PR01, PR02, PR04, PR10, RT01`. Write `name : type` always, one entry per
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
  `tests/unit/test_documentation_api.py:EXAMPLE_TARGETS` declares. The targets include core
  methods so a direct method anchor explains its own call. Every See Also entry carries a
  description.
- Show the smallest normal use of the documented object or method. Start with the common case.
  Include every import, use compact data, and check deterministic output. Do not use an example
  to assert a statistical guarantee from one sample.
- An example must run in the fast tier without `doctest.SKIP`. Pass explicit learners to fits;
  the default learner library costs 30 to 120 seconds per fit. Put expensive studies and extended
  comparisons in narrative documentation and test them separately.
