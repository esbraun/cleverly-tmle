# Example notebooks

This page tells you how to write a tutorial under `docs/examples/` as an executed notebook. Each
step of a tutorial notebook shows its code, its output, and a plain reading of that output. The
reader is a practitioner who is new to TMLE and new to `cleverly`.

Use [point-treatment TMLE](../examples/point-treatment-tmle.ipynb) as the reference. It follows
every rule on this page. The [TWINS notebook](../examples/twins-causal-inference.ipynb) is the
other published notebook, and it is not a program tutorial.

## What a tutorial notebook consists of

A tutorial has one stem, such as `cross-fitting`. The stem names both files you own.

| file | what it holds |
| --- | --- |
| `docs/examples/<stem>.ipynb` | the notebook, with its stored outputs and the execution stamp |
| `tests/unit/tutorial_semantics/<module>.py` | the reviewed callback. `<module>` is the stem with each hyphen replaced by an underscore |

The callback module defines `check(namespace)`. The fast tier runs every code cell of the notebook,
in order, in one namespace, and passes that namespace to `check`. The module can also define
`UNPRINTED_DECIMALS`, a mapping from a decimal the prose writes to the reason no output prints it.
The package docstring in `tests/unit/tutorial_semantics/__init__.py` states the convention.

These fast tests read your notebook. Each one takes the notebook path as its test id, so the
selection `-k <stem>` runs all of them.

| test in `tests/unit/test_documentation_runtime.py` | what fails it |
| --- | --- |
| `test_every_notebook_has_an_internally_consistent_execution_artifact` | a code cell or an output that changed after the stamp, an error output, or non-contiguous execution counts |
| `test_every_notebook_records_its_repository_context` | a missing or malformed recorded half of the stamp |
| `test_every_published_figure_has_a_non_image_companion` | a figure with no printed values beside it |
| `test_tutorial_semantics_at_documented_size` | a cell that raises offline, or a claim in `check` that no longer holds |
| `test_every_narrated_decimal_matches_a_stored_output` | a decimal in the prose that no stored output prints at that precision |

The link test `tests/unit/test_documentation_links.py` also reads the markdown cells. Every relative
link and every heading anchor in them must resolve.

## The cell outline

Every tutorial notebook follows this order. Use the cell ids in the table where the step applies.
Each code cell is followed by a markdown cell that starts with `**What this output tells you.**`.

| order | cell id | type | contents | the output it shows |
| --- | --- | --- | --- | --- |
| 1 | `title` | markdown | the title as the only level-one heading, the applied question, who asks it, and a "What you will learn" table | none |
| 2 | `plan` | markdown | "Why this method", and a table of the terms the page uses most | none |
| 3 | `setup` | code | the imports and the installed `cleverly` version | the version |
| 4 | `data` | code | the data, with the columns renamed to the program's names | the shape, the first rows, and the known truth of the synthetic law |
| 5 | `association` | code | an unadjusted comparison, where confounding is part of the lesson | the naive contrast beside the truth |
| 6 | `protocol` | code | the `StudyProtocol` for the question | `protocol.summary_lines()`, including the fingerprint |
| 7 | `identify` | code | the `CausalStudy`, the design, and `study.identify(...)` | `effect.summary()`: the formula, the nuisances, and the assumptions |
| 8 | one id per fit | code | each fit, with its method written out in full | the result summary and the numbers the prose quotes |
| 9 | `failure-mode` | code | the failure mode the [examples index](../examples/index.md#the-program) names for this tutorial | the numbers that show the failure |
| 10 | `assessment` | code | `result.assess()` and the retained reports the page reads | the assessment summary and each report summary |
| 11 | `sensitivity` | code | the sensitivity analysis that applies, or the refusal that explains why none does | the robustness value, the bounds, or the refusal message |
| 12 | `trust` | markdown | "How far to trust this": the layer table and the registered study that covers the construction | none |
| 13 | `where-next` | markdown | the next tutorial and the related reference entries | none |

A tutorial can add steps between rows 8 and 11, such as a second population or a truncation curve.
Give each added step a heading, a code cell, and its reading. Keep every scientific claim of the
Markdown tutorial. A claim the output cannot show moves into the reading as a stated condition.

### The protocol step

`StudyProtocol` in `src/cleverly/protocol.py` records the scientific design. Every result fitted
from a study that carries it stores its fingerprint. Fill each field for the tutorial's question.

| field | what to write |
| --- | --- |
| `target_population` | the population to which the causal question applies |
| `eligibility` | the inclusion and exclusion criteria, one entry each |
| `time_zero` | the event that aligns eligibility, treatment assignment, and follow-up |
| `treatment_strategies` | the strategies under comparison, one entry each |
| `treatment_versions` | the version of each strategy, in the same order |
| `outcome` | the outcome definition and how it is measured |
| `horizon` | the follow-up horizon |
| `intercurrent_event_handling` | the rule for each event after time zero that changes the outcome's meaning |
| `interference_unit` | the unit within which one assignment can affect another outcome |
| `assumption_rationale` | why the design supports each identification assumption |

The protocol does not store the contrast or the analysis. The typed estimand owns the contrast, and
the method owns the analysis configuration. The
[shared study design](../examples/index.md#the-shared-study-design) gives the program values.

## Rules for code cells

| rule | reason |
| --- | --- |
| Give every code cell visible output. Print text rather than rely on a bare last expression | the reading needs an output to interpret, and `--check` compares text |
| Print each number the prose quotes, rounded the way the prose writes it | the narration test compares prose decimals with stored outputs |
| Pass explicit, cheap learners to every fit. Prefer linear and logistic models where the lesson allows | the default learner library costs 30 to 120 seconds per fit |
| Set every seed: the generator seed, `Runtime(random_state=...)`, and each learner's `random_state` | a rerun must reproduce the stored outputs |
| Set `n_jobs=1` in `Runtime` and in every learner | several notebooks execute on one machine at once |
| Keep the whole notebook under 60 seconds with `scripts/execute_notebook.py` | the fast tier runs every tutorial. The reference notebook takes 12 seconds |
| Give each code cell a short, stable, kebab-case id. Never rename an id that a callback reads | `stored_output(path, cell_id)` finds a cell by its id |
| Pair every figure with printed values | an image cannot enter the `--check` comparison |
| Use no network, and write files only inside a `TemporaryDirectory` | the offline gate refuses a connection, and the kernel runs from the repository root |
| Write plain Python. Do not use IPython magics or shell escapes. `display(...)` is allowed | the fast tier runs the cells without a kernel |
| Run `ruff format` and `ruff check` on the notebook before you execute it | the formatter rewrites code cells, and a rewritten cell breaks the stamp |

## Rules for markdown cells

The prose rules in `CLAUDE.md` apply to every markdown cell. Write one idea per sentence, and use a
table where the content is parallel. `python -m tests.prose --path <notebook>` reports the findings.

Start each reading with `**What this output tells you.**`. Then name the part of the output you
read, give its value, and say what it means for the program. Quote a number only when a stored
output prints it.

Introduce each TMLE concept the first time the page uses it. Write one plain sentence, then link
to the reference. The table gives a starting sentence and a link for the common concepts.

| concept | one plain sentence | link |
| --- | --- | --- |
| estimand | the number the question asks for, written before any model is chosen | [estimands](../user-guide/estimands.md) |
| nuisance | a model the estimate needs but the question does not ask about, such as Q or g | [point-treatment TMLE](../technical-reference/point-treatment-tmle.md) |
| targeting | a small update to the outcome model, weighted by the treatment model, that removes first-order bias | [targeting and bounds](../user-guide/methods-learners.md#targeting-and-bounds) |
| influence curve | how much each row moves the estimate. Its variance gives the standard error | [inference](../technical-reference/inference.md) |
| cross-fitting | each row's nuisance prediction comes from models fit without that row | [CV-TMLE](../technical-reference/cv-tmle.md) |
| positivity | every kind of patient has some chance of each arm | [diagnostics](../user-guide/results-assessment.md#diagnostics) |
| double robustness | the point estimate stays consistent when either nuisance model is consistent | [point-treatment TMLE](../technical-reference/point-treatment-tmle.md) |
| sensitivity analysis | how strong an unmeasured confounder would need to be to change the conclusion | [sensitivity analysis](../user-guide/results-assessment.md#sensitivity-analysis) |

Link to a sibling tutorial by its notebook file name, such as `cross-fitting.ipynb`. A heading
anchor in a notebook includes its step number, such as `#step-9-reuse-the-same-outer-split`.

## Write the callback

Start from the tutorial's existing module in `tests/unit/tutorial_semantics/`. Keep each assertion
that still describes the page. Rename a namespace key only when the notebook renames the variable.

Add an assertion for each seeded relation the new readings narrate. An example is "the interval
contains the true value". A relation is a claim about this draw, so it certifies no method. Use
`stored_output(NOTEBOOK, "<cell id>")` to check a stored output that must stay present, such as the
protocol fingerprint.

Do not loosen a tolerance to make a relation pass. A relation that moves under a supported change is
a sampling claim, and it belongs in a registered study.

## Convert a tutorial

Do these steps in order. Run each command from the repository root.

1. Read the Markdown tutorial, its callback module, and the reference notebook.
2. Write `docs/examples/<stem>.ipynb` with the cell outline above. Leave the readings short for now.
3. Format and lint the code cells.

   ```bash
   ruff format docs/examples/<stem>.ipynb
   ruff check docs/examples/<stem>.ipynb
   ```

4. Execute the notebook and write its stamp. Send the log to a file, and check the exit status.
   Do not pipe the run into `tail` or `head`, because the pipe hides the exit status.

   ```bash
   python scripts/execute_notebook.py docs/examples/<stem>.ipynb > execute.log 2>&1
   echo "exit status $?"
   ```

5. Read every stored output. Write each reading from those outputs.
6. Delete the Markdown source with `git rm docs/examples/<stem>.md`.
7. Update the callback module, as the previous section describes.
8. Confirm that a fresh run reproduces the stored outputs. The command writes nothing.

   ```bash
   python scripts/execute_notebook.py docs/examples/<stem>.ipynb --check > check.log 2>&1
   echo "exit status $?"
   ```

9. Confirm that the notebook has line-feed endings. The command prints `False`.

   ```bash
   python -c "import sys; print(bytes([13, 10]) in open(sys.argv[1], 'rb').read())" docs/examples/<stem>.ipynb
   ```

10. Run the targeted checks, and read every finding.

    ```bash
    ruff format --check docs/examples/<stem>.ipynb
    python -m tests.prose --path docs/examples/<stem>.ipynb
    pytest -q -p no:cacheprovider tests/unit/test_documentation_runtime.py -k <stem>
    pytest -q -p no:cacheprovider tests/unit/test_documentation_links.py -k <stem>
    ```

11. Report to the orchestrator. The report is in the last section.

A markdown edit does not change the stamp, so it needs no new execution. A code edit does. After a
code edit, go back to step 3. A new execution can move a number, and the narration test then names
each reading to correct.

Remove `execute.log` and `check.log` before you hand off.

## Share one machine

Several authors execute notebooks at the same time on one machine. Follow these limits.

| limit | reason |
| --- | --- |
| Run `pytest` without `-n`, and never with `-n auto` | a worker pool per author makes the authors contend for the same cores |
| Run only the targeted checks above. Do not run the full suite or `nox -s docs` | the orchestrator runs both once, after the conversions |
| Execute one notebook at a time | each execution starts its own kernel |
| Set `n_jobs=1` everywhere | a parallel learner competes with the other authors |

## Do not edit shared files

Edit only your notebook, your callback module, and the deletion of your Markdown source. Report any
other change the conversion needs. The orchestrator applies shared edits one at a time, so parallel
authors do not collide on one file.

| shared file | what to report instead |
| --- | --- |
| `tests/prose-report.md` | the finding id, the sentence, and the `accepted: <reason>` you propose |
| `docs/examples/index.md`, another tutorial, or any page that links to your `.md` file | each link to rewrite. `git grep -n "<stem>.md"` lists them |
| `tests/unit/test_documentation_runtime.py`, `tests/unit/tutorial_semantics/__init__.py`, `tests/documents.py`, `tests/notebooks.py`, `tests/conftest.py` | the change and the failure it repairs |
| `scripts/execute_notebook.py`, `docs/conf.py`, `pyproject.toml`, `uv.lock` | the change and the failure it repairs |

The report also gives the notebook's execution time, the exit status of each command in step 10,
and every claim of the Markdown tutorial that the notebook changed or removed.
