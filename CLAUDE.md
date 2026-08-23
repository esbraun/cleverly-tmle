# Working on cleverly

`cleverly` is alpha software under heavy development. Treat the current architecture as
provisional: inspect the code and tests before changing it, and do not preserve an implementation
shape solely because this file once described it.

This file holds only the working agreements that no other file states and no test enforces.
Everything else is routed:

| for | read |
| --- | --- |
| setup, development commands, the public overview | `README.md` |
| where technical documentation lives | `docs/README.md` |
| cross-module constraints not derivable from one implementation | `docs/architecture-invariants.md` |
| test tiers, and when a slow study is justified | `docs/development/testing-strategy.md` |
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

- Read `docs/development/testing-strategy.md` before choosing a tier. The fast tier is the default
  handoff gate. A slow run is an evidence decision, not a time-budget waiver.
- Ruff and mypy are pinned once in `pyproject.toml`'s `dev` extra, which resolves to
  `cleverly[all]` plus tooling. An optional extra kept out of `dev` *and* out of a dedicated CI job
  is installed by no session, so its tests can only skip, and a skipped correctness check reads
  exactly like a passing one. Put a new extra in `dev`, or give it a job that installs and runs its
  tests.
- Ruff *formats* the Python examples in Markdown, so run it over the whole tree. Its linter does
  not read Markdown at all, and the formatter skips any block it cannot parse. Neither one sees a
  syntax error in an example.

## Documentation writing

Reader-facing Markdown under `docs/` follows ASD-STE100 Simplified Technical English.

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

Scope is every `docs/**/*.md`. Rewrite the text a change touches. Do not sweep unrelated pages.
No test enforces this.
