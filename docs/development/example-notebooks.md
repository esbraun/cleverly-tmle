# Example notebooks

This page tells you how to write a tutorial under `docs/examples/` as an executed notebook. A
tutorial shows its code, its output, and a plain reading of each output. The reader is the data
scientist that [Writing for readers](writing-for-readers.md#the-readers) describes: someone who
knows Python and regression, but not causal inference or `cleverly`, and who wants to run the
analysis and explain the result. That guide holds the writing rules for every markdown cell. This
page adds the rules that are special to a notebook: its outline, its code cells, its tests, and
how to execute it.

Roadmap item X88 rewrites the notebooks to this page, one notebook at a time. Until a notebook's
rewrite merges, use it as an example of code-cell practice, not of prose or headings. The
[TWINS notebook](../examples/twins-causal-inference.ipynb) analyzes public data, and it is not a
program tutorial.

## What a tutorial notebook consists of

A tutorial has one stem, such as `cross-fitting`. The stem names both files you own.

| file | what it holds |
| --- | --- |
| `docs/examples/<stem>.ipynb` | the notebook, with its stored outputs and the execution stamp |
| `tests/unit/tutorial_semantics/<module>.py` | the reviewed callback. `<module>` is the stem with each hyphen replaced by an underscore |

The callback module defines `check(namespace)`. The fast tier runs the code cells of the notebook,
in order, in one namespace, and passes that namespace to `check`. The fast tier does not run a
refit cell. The next section defines a refit cell. The module can also define
`UNPRINTED_DECIMALS`, a mapping from a decimal the prose writes to the reason no output prints it.
The package docstring in `tests/unit/tutorial_semantics/__init__.py` states the convention.

The TWINS notebook downloads its data, so the offline fast tier does not execute it. Its module
`tests/unit/tutorial_semantics/twins_causal_inference.py` defines no `check`. It defines
`UNPRINTED_DECIMALS` and `check_stored()`, which read committed outputs only.

### Refit cells and the slow tier

A refit cell calls `refute`, `simulated_confounding`, or `benchmark`. A cell that reads a name
that a refit cell binds is also a refit cell. The function `refit_cells()` in
`tests/unit/tutorial_semantics/__init__.py` derives the set from the code. You do not tag a cell.

| tier | what it does with a refit cell | which callback it calls |
| --- | --- | --- |
| fast, `pytest -q -n auto --dist loadgroup` | compiles the cell and does not execute it | `check(namespace)` |
| slow, `pytest -q -n auto --dist loadgroup -m slow` | executes every cell in order | `check(namespace)`, then `check_refits(namespace)` |

Put each assertion on a refit result in `check_refits(namespace)`. Keep assertions on stored
outputs and probe logs in `check`, because the fast tier can read them. When `check` reads a
refit-bound name, the fast tier fails with a `KeyError` that names the refit cell.

`test_the_derived_refit_cells_are_the_reviewed_ones` lists the refit cells of each notebook. When
your notebook gains or loses a refit cell, update `EXPECTED_REFIT_CELLS` in the same change.

These fast tests read your notebook. Each one takes the notebook path as its test id, so the
selection `-k <stem>` runs all of them.

| test in `tests/unit/test_documentation_runtime.py` | what fails it |
| --- | --- |
| `test_every_notebook_has_an_internally_consistent_execution_artifact` | a code cell or an output that changed after the stamp, an error output, or non-contiguous execution counts |
| `test_every_notebook_records_its_repository_context` | a missing or malformed recorded half of the stamp |
| `test_every_published_figure_has_a_non_image_companion` | a figure with no printed values beside it |
| `test_tutorial_semantics_at_documented_size` | a cell that raises offline, or a claim in `check` that no longer holds. The test skips refit cells |
| `test_the_derived_refit_cells_are_the_reviewed_ones` | a refit cell that `EXPECTED_REFIT_CELLS` does not list, or a listed cell that is no longer a refit cell |
| `test_tutorial_refits_at_documented_size` | slow tier only: a refit cell that raises offline, or a claim in `check` or `check_refits` that no longer holds |
| `test_every_narrated_decimal_matches_a_stored_output` | a decimal in the prose that no stored output prints at that precision. This test also reads the TWINS notebook |
| `test_every_static_only_notebook_keeps_its_pinned_outputs` | TWINS only: a stored output that `check_stored()` requires |

The link test `tests/unit/test_documentation_links.py` also reads the markdown cells. Every relative
link and every heading anchor in them must resolve.

## The page outline

Every tutorial follows this outline. The landmark column names the role of each section. Write
the heading itself as a landmark the reader recognizes, in words that fit the page, as rule R13 of
the guide asks. The title is the only level-one heading. "At a glance", each main-path landmark,
"Go deeper", and "Where to go next" are level two, and each lesson under "Go deeper" is level
three.

| landmark | contents | cell ids |
| --- | --- | --- |
| At a glance | the question in one sentence, the answer with its interval in plain words, and the one caveat that matters most, stated with its finding. Use the printed numbers of the estimate cell | `title`, which also holds the level-one title. The title names the data situation first, then the question |
| The question | the decision, who it applies to, and what is compared, in the program's words. The quantity to estimate, in one plain sentence. A causal graph only where it informs, as [drawing a causal graph](#drawing-a-causal-graph) describes | `plan` |
| The data | the imports and the installed `cleverly` version, then what one row is, the columns, and the known answer of the simulated data | `setup`, `data` |
| Why a simple comparison misleads | the naive difference beside the known answer, and why it is off | `association` |
| Set up the analysis | the study description, the roles of the columns, and the conditions the answer needs. Mark each condition "the data can check this" or "this needs judgment" | `protocol`, `identify` |
| Estimate the effect | each fit, with each setting explained in a short list | one id for each fit |
| Check the fit | the overlap check and the model checks, read in plain words | `assessment` |
| Try to break the result | the refutation checks | the refutation cell |
| What if a confounder is missing? | the sensitivity analysis that [the sensitivity table](#sensitivity-analysis-for-each-kind-of-fit) names for the fit | `sensitivity` |
| What we can and can't conclude | the answer for the manager, the conditions it rests on, and the limits. Say here, once, that the page shows one simulated draw | `interpret` |
| How we know the method works | the simulation study that checks each method, and what it does not cover, as [the method-evidence section](#how-we-know-the-method-works) describes. It may open "Go deeper" when the main path needs its words | `method-evidence` |
| Go deeper | theory, comparisons between estimators, and checks that print nothing to act on for this fit, such as a weight-cap curve when no patient is capped | the comparison cells, and `failure-mode` only when it is not the point of the page |
| Where to go next | the next tutorial and the related reference entries | `where-next` |

A lesson stays on the main path, before the conclusion, when it is the point of the page. An
example is the failure mode that the [examples index](../examples/index.md) names for the page,
such as "what happens when a model is wrong". Put it after the estimate and give it its own
level-two landmark. A tutorial can add other landmarks, such as a second population or a cap on
extreme weights. Give each one a heading, a code cell, and its reading.

Keep each cell id that a callback reads, and each `text` graph fence. If content moves to another
cell, update the callback in the same change. Keep every scientific claim of the page, in plain
words, in a Details dropdown, or on a linked page, as rule R7 of the guide asks. Drop a claim only
when it is redundant or no longer true, and record the reason in the claim ledger.

A tutorial notebook is an end-to-end example that works. Do not show a refusal as a lesson.

| step | what to do |
| --- | --- |
| a step that would print a `CapabilityError`, a `DataError`, or a refusal message | change it to an analysis that `cleverly` supports, or remove it |
| a step that prints a capability list with unavailable rows | print only the rows that run |
| a statistical failure mode that runs and prints numbers, such as an interval that misses | keep it, as the failure-mode lesson |
| a limit that the reading must name | write the requirement in one sentence, such as "Declare `q_bounds` for a cross-fitted continuous outcome" |

### Drawing a causal graph

Draw a causal graph only when it informs identification, data construction, the estimator, or the
reading. Every node and every arrow must do one of these jobs. Under the graph, say in prose what
each arrow and each missing arrow assumes. Use a table only when there are four or more arrows to
describe.

Name each confounder, mediator, competing event, censoring variable, and effect modifier.
State which assumptions the graph encodes and which it leaves open. A `text` fence renders the
graph without an extra build dependency.

### Comparisons

Keep a comparison only when it teaches a real distinction. Do not add an estimator only as a
reference value. A comparison goes under "Go deeper", unless it is the point of the page.

| for each comparison | what to write |
| --- | --- |
| the estimands | the quantity each method estimates. Compare numbers only for methods that estimate the same quantity, or say that the difference is the lesson |
| the assumptions | the assumptions that each method adds |
| the mechanism | the feature of the simulated data or the problem that causes the difference, named before the output |
| the evidence | the diagnostic, population quantity, or registered study that shows the mechanism |
| the reading | what the difference shows. One simulated sample does not show a general property |

### The data step

Take the program's column names from one place, so every page uses the same names.

| generator | what to write |
| --- | --- |
| `make_nonlinear_bounded` | `frame, truth = navigation_data(n=..., seed=...)`, imported from `cleverly.datasets`. Do not rename its columns in the notebook |
| another generator | `frame.rename(columns=...)` with that generator's mapping. Each generator gives its `W` columns other meanings, so no shared mapping fits them |

Print the column names in the data step, such as `print(list(frame.columns))`. The printed names
show the reader which columns the page uses. Name the generator in the reading.

### The protocol step

`StudyProtocol` in `src/cleverly/protocol.py` records the scientific design. Every result fitted
from a study that carries it stores its fingerprint. Fill each field for the tutorial's question.
Then check each field and each unit name against the structural equations of the generator. For
example, do not call an outcome "standardized" when the generator draws it on its own scale.

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

Do not copy the program values into a notebook. Start from the helper in `cleverly.datasets` that
matches the design.

| design | helper |
| --- | --- |
| one navigation decision at the discharge-home order | `navigation_protocol()` |
| navigation decisions at discharge and later, with time zero at discharge | `longitudinal_navigation_protocol()` |

Pass the helper's result to `dataclasses.replace`, and give only the fields your question changes.
Reuse an unchanged entry of a sequence field by index, such as `*program.assumption_rationale[1:]`.
To add a strategy, such as a dynamic rule, append it and its version:
`treatment_strategies=(*program.treatment_strategies, "<the rule>")`. Write a new `StudyProtocol`
only when neither helper fits the design.

Print each protocol record once on the page, and read it in plain words as the study description
(see the guide's library words). The table gives where each notebook prints it.

| record | where the notebook prints it | later summaries of its study |
| --- | --- | --- |
| the record of the protocol step | the protocol step, with `"\n".join(protocol.summary_lines())` | pass `protocol="fingerprint"`, which prints only the fingerprint line |
| a record that a later step declares | the first summary of its study, with the default `protocol="full"`. A step can instead print the fingerprint and the fields that it changed | pass `protocol="fingerprint"` |
| no record | nothing to print. The summaries state `causal study protocol: absent` | keep the default |

Every protocol that a notebook derives with `dataclasses.replace` needs two records. The reading
names the changed fields. The callback asserts the exact set with `changed_fields(protocol, base)`,
where `base` is the protocol that `replace` received.

## Rules for code cells

| rule | reason |
| --- | --- |
| Give every code cell visible output. Print text rather than rely on a bare last expression | the reading needs an output to interpret, and `--check` compares text |
| Print each number the prose quotes, rounded the way the prose writes it | the narration test compares prose decimals with stored outputs |
| Pass explicit, cheap learners to every fit. Prefer linear and logistic models where the lesson allows | the default learner library costs 30 to 120 seconds per fit |
| Do not pass a bare `HistGradientBoostingClassifier` as a treatment learner below 10,000 rows. Use a regularized or early-stopped booster, or a flexible parametric model, and check its calibration | scikit-learn stops the booster early only above 10,000 rows. Below that size, its default settings overfit the treatment model. The fitted propensities are then too extreme, so the standard error is too large and the default $\nu^2$ estimate can be negative |
| Print a score check as pass or fail and the rounded ratio to its threshold. Never print a raw residual below 1e-8 | a residual below 1e-8 is machine noise. It changes between machines, and `--check` then fails |
| Set every effective seed: the generator seed, `Runtime(random_state=...)`, and each learner's active `random_state` | a rerun must reproduce the stored outputs |
| Set `n_jobs=1` in `Runtime` and where a learner still uses that parameter. Do not pass a deprecated parameter that the installed learner ignores | several notebooks execute on one machine at once, and deprecated settings add warning noise without limiting work |
| Fit every learner through `cleverly`, and do not call `set_thread_limit`. Wrap any other fit in `with thread_limit():` from `cleverly.learners` | `HistGradientBoosting*` has no `n_jobs`, and its OpenMP pool uses every core. `cleverly` fits each learner under `thread_limit()`, which holds OpenMP and BLAS to one thread by default |
| Keep the cells other than refit cells under 60 seconds with `scripts/execute_notebook.py` | the fast tier runs those cells for every tutorial. The stored cell timestamps of `point-treatment-tmle.ipynb` span about 11 seconds |
| Keep the refit cells of a notebook under 5 minutes in total | the slow tier runs every notebook in one CI job with a 45-minute limit. On 2026-10-23 the longest notebook took 3 minutes |
| Give each code cell a short, stable, kebab-case id. Never rename an id that a callback reads | `stored_output(path, cell_id)` finds a cell by its id |
| Pair every figure with printed values | an image cannot enter the `--check` comparison |
| Use no network, and write files only inside a `TemporaryDirectory` | the offline gate refuses a connection, and the kernel runs from the repository root |
| Write plain Python. Do not use IPython magics or shell escapes. `display(...)` is allowed | the fast tier runs the cells without a kernel |
| Print only the part of a library summary that the reading needs, with plain labels. Show a full `summary()` once on the page, and say which lines matter (rule R12) | 40 lines of output bury the two lines that the reading uses |
| Give every printed library word a plain meaning: relabel the print, or say in the reading what the word means, such as `required nuisances` or a `PASS` line | the reader sees the output before the prose, and an unexplained library word in an output blocks them as much as one in the prose |
| Mark each line that a reader changes for their own data with a comment in the cell, such as `# For your own data, change the outcome and treatment names` | the comment sits where the reader edits, and it costs no prose words |
| Run `ruff format` and `ruff check` on the notebook before you execute it | the formatter rewrites code cells, and a rewritten cell breaks the stamp |

## Rules for markdown cells

The rules of [Writing for readers](writing-for-readers.md) apply to every markdown cell.
`python -m tests.prose --path <notebook>` reports the dash, transition, intensifier, and
paragraph findings. `scripts/render_for_reader_review.py` reports the rest of the mechanical
measures.

After each output, start with the finding. Say what to look at and what it means, in two to four
sentences, and quote a number only when a stored output prints it (rules R5 and R6). These rules
keep each reading accurate.

| rule | reason |
| --- | --- |
| Say once, in the conclusion, that the page shows one simulated draw. Do not write that a seeded draw shows or establishes a property of the method; cite a registered study for the property. Where a sentence states a relation that holds on fewer than 90% of swept draws, qualify it in plain words, such as "in this sample", or quote the printed numbers without a multiplier such as "about three times" | one draw is one sample. The callback pins it as a relation, and a relation certifies no method. A multiplier reads as a property of the method |
| Before you call an interval that misses the true value sampling variation, run a seed sweep of the shown configuration with at least 60 draws. Use that explanation only when the sweep covers near the nominal level. Otherwise name the measured bias or under-coverage and the cause that a probe shows | a 95% interval misses on about one draw in 20 only when the configuration covers at the nominal level. A configuration with a bias can miss on most draws |
| Name the source of each repeated-draw number that a reading quotes: a committed probe script or a registered study. List the number in the callback's `UNPRINTED_DECIMALS` with that source. In `check`, assert the number against the committed file that holds it, such as the probe log or the study summary | no stored output prints a sweep number, so the narration test needs the entry. The narration test checks only that a reason exists. The assertion keeps the number equal to its source |
| An overlap or calibration reading names the fitted model that it comes from. It also says what that model cannot see | an overlap check of a wrong treatment model describes that model, not the population. An in-sample calibration slope near 1 does not show that a model is correct |
| Compare two estimates from one fit with `result.contrast(...)`, and read its interval. Do not compare their difference with the larger standard error | the two estimates share rows. `contrast` accounts for their correlation |
| Read a sensitivity output as robustness only when the technical reference derives it for the fitted estimator. Run the analysis that [the sensitivity table](#sensitivity-analysis-for-each-kind-of-fit) names for the fit | an underived bound can be too narrow |
| In a plain reading of an omitted-variable bound, name the side each strength measures (the outcome's remaining variation, or the change in the weights), and quote the strength at which the conclusion tips, computed by the cell, rather than "stronger than" a covariate. Read the robustness value as a floor: a confounder weaker than it on both sides cannot erase the effect. Do not read it as the strength a confounder needs, since one strong on a single side can erase the effect with the other side below it | a covariate that is the strongest confounder on one side can be weak on the other, and "stronger than" can understate the margin many times over |
| Say that a quantity does not exist only when a source shows it | "not implemented" and "not defined" are two claims |
| Put a status word, a label, or a column name in backticks only when the reader must find it in a stored output (rule R15). The assessment summary prints `Returned results`, not `completed` | the reader searches the output for the quoted token |

Define each technical term in plain words where it first appears, and link to the reference page.
The guide's [library words](writing-for-readers.md#library-words-and-their-plain-replacements)
table gives a plain replacement for each `cleverly` term.

Link to a sibling tutorial by its notebook file name, such as `cross-fitting.ipynb`, and name the
section in words (rule R14). Do not link to a heading anchor in another notebook: its headings
change when it is rewritten, and the link test resolves a fragment only against a heading.

### Sensitivity analysis for each kind of fit

Run the sensitivity analysis that the table gives for the fit of the page. Each analysis runs on
that fit and returns numbers. The
[sensitivity section](../technical-reference/validation-methods.md#sensitivity-to-untestable-assumptions)
of the technical reference describes each one and says which of them is a bound.

| kind of fit | the sensitivity analysis to run | how to read it |
| --- | --- | --- |
| point-treatment TMLE or CV-TMLE, on a law whose true propensity stays away from 0 and 1 | `robustness_value()`, `benchmark()`, and `omitted_confounding()`, with the default $\nu^2$ estimator. Add `evalue()` and `simulated_confounding()` | a bound for the first three. $\nu^2$ is the second moment of the Riesz representer, and it is finite only when $1/g$ and $1/(1 - g)$ both have finite means. `evalue()` reads the risk ratio, or a standardized-difference approximation for a continuous outcome, and `simulated_confounding()` is a stress surface that is not a bound |
| C-TMLE | `robustness_value()`, `omitted_confounding()`, and `simulated_confounding()` on the selecting fit. A collaborative fit other than `strategy="derivative"` refuses the bound unless the call declares `assume_consistent_mechanism=True` | a point bound under that declaration, with the declaration stated in the reading. A per-arm outcome-adaptive fit (`oat_design="per_arm"`) needs a declared `mechanism_learner=` with the declaration, because its own treatment learner reads one outcome prediction. State the learner and the assumption that it is consistent. Without the declaration, read the bound of the ordinary TMLE fit of the declared adjustment set. It does not cover the collaborative selection |
| MSM projection | `robustness_value(estimand="ate[medium vs low]")` and its sibling on the multi-arm `ATE(reference="low")` fit, whose contrasts share the influence curves of the saturated MSM coefficients. Add the bound on the slope and `simulated_confounding()` on the slope | a bound on each arm contrast. The slope bound runs on an identity-link MSM with a categorical treatment. A continuous-treatment MSM or a non-identity link refuses it, so state that limit in one sentence |
| a fit with missing outcomes | `missingness()` and `tipping_gamma()`, `evalue()` with the missing-at-random plug-in of the outcome SD, and the omitted-variable bound | the tilt of the unobserved outcomes at which the estimate reaches its null. Do not run the outcome-axis `simulated_confounding()` when the response mechanism reads the arm |
| longitudinal TMLE, with or without a time-to-event outcome | refit without one recorded covariate at a time, and print each move in standard errors. Run `simulated_confounding()` only on an uncensored end-of-study fit with no time-varying covariate after the first treatment node | a benchmark, not a bound. Each move is signed and specific to the dropped covariate |
| a regime, or a shift whose observed and shifted doses are all declared `dose_atom` values | `omitted_confounding()` on the contrast ([X82](../roadmap.md#x82-refutation-and-sensitivity-checks-for-every-estimator) part G2c) and `simulated_confounding()` where the intervention has a known density. A fit that declares `q_bounds` admits only a treatment-only grid, such as `ConfounderStrengthGrid(treatment=(0.0, 0.1), outcome=(0.0,))`. One flip strength is enough | the bound reads the contrast. A treatment-only cell moves the estimate by misclassification of the treatment, not by confounding. It is not a bound. Say so in the reading, or omit the step |
| a continuous-dose shift | `simulated_confounding()` only | the bound refuses a continuous-dose shift. State that limit in one sentence |
| an incremental intervention | `simulated_confounding()` only | the bound refuses the `ipsi` axis, because the tilt reads the treatment mechanism. State that limit in one sentence |

Run the checks below on the fit of the page as well. The tutorial runs each one that the fit
supports and states each refused one as a limit.

| check | what to run | how to read it |
| --- | --- | --- |
| refutation | `refute()`: `placebo`, `subset`, and `random_common_cause` at the default `R`. Add `negative_control_outcome` where the data hold a pre-treatment measure | a verdict prints only where a `VERDICT_EVIDENCE` row covers the fit. Otherwise the numbers print with `passed=None`. Say in plain words whether a simulation study backs a pass or fail for this fit |
| generated outcomes | `dummy_outcome`, `simulated_outcome`, and `plasmode` at a small `R` | a level-1 reading. The level-2 bands need `R` = 400, so run that version once as a review probe outside the notebook |
| tilt | `missingness()` and `tipping_gamma()` | the tilt at which the estimate reaches its null |

A tutorial cell runs offline at its documented size, in the fast tier or in the slow tier, and
no tier shrinks a refit cell. Keep `R` at its default or at a small count in every executed cell.

### How we know the method works

The `method-evidence` cell sits under the landmark "How we know the method works". It links the
simulation study that checks each method on the page, in a short paragraph or a table. Do not copy study
results, cell names, or citations into the notebook. The study page holds them, and a copy goes
stale when the study is regenerated.

| part of the cell | what to write |
| --- | --- |
| the table | one row for each fit: the section, a link to the registered study page in `docs/technical-reference/method-evidence/`, and whether the study uses the law of the page or another law |
| the gap | one sentence for each part of the page that no registered study covers, such as a learner, a fold count, or a fitted mechanism |
| the page's own simulations | one sentence that says a simulation run for this page (a script under `reviews/notebook-review/probes/`) checks a reading of the page, and is not a simulation study that validates the method. Quote its result in the reading that uses it |
| the technical entry | a link to the "Validation issues special to this method" section of the method's technical entry |

## Write the callback

Start from the tutorial's existing module in `tests/unit/tutorial_semantics/`. Keep each assertion
that still describes the page. Rename a namespace key only when the notebook renames the variable. Keep the markdown cell ids and
the `text` graph fence that the callback reads.

Add an assertion for each seeded relation the new readings narrate. An example is "the interval
contains the true value". A relation is a claim about this draw, so it certifies no method. Use
`stored_output(NOTEBOOK, "<cell id>")` to check a stored output that must stay present.

Use the shared helpers in `tests/unit/tutorial_semantics/__init__.py`. Do not define a local copy.

| helper | what it checks |
| --- | --- |
| `covers(estimate, value)` | the estimate's interval, or a `(low, high)` pair, contains the value |
| `assert_protocol_recorded(NOTEBOOK, "<cell id>", protocol, *results)` | the cell prints the protocol fingerprint, and each result carries it |
| `changed_fields(protocol, base)` | the set of fields in which `protocol` differs from the protocol it was derived from. Assert the exact set the reading names |

Do not loosen a tolerance to make a relation pass. A relation that moves under a supported change is
a sampling claim, and it belongs in a registered study.

## Convert a tutorial

Do these steps in order. Run each command from the repository root.

1. Read the Markdown tutorial, its callback module, and a merged notebook for code-cell practice.
2. Write `docs/examples/<stem>.ipynb` with the page outline above. Leave the readings short for now.
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
    pytest -q -p no:cacheprovider tests/unit/test_documentation_runtime.py -k <stem> -m slow
    pytest -q -p no:cacheprovider tests/unit/test_documentation_links.py -k <stem>
    ```

Use the notebook stem for ``<stem>``, such as `point-treatment-tmle`. Each test id holds the
notebook path, so the stem selects the tests of that notebook.

11. For a new page or a whole-page rewrite, run the
    [reader review](writing-for-readers.md#the-reader-review) to its pass. For a rewrite, also
    compare the stored outputs with the notebook before the rewrite: only the prints you meant to
    change may differ, and no estimate may move.
12. Report to the orchestrator. The report is in the last section.

A markdown edit does not change the stamp, so it needs no new execution. A code edit does. After a
code edit, go back to step 3. A new execution can move a number, and the narration test then names
each reading to correct.

Remove `execute.log` and `check.log` before you hand off.

## Share one machine

On a local machine, several authors can execute notebooks at the same time. Follow these limits.

| limit | reason |
| --- | --- |
| Run `pytest` without `-n`, and never with `-n auto` | a worker pool per author makes the authors contend for the same cores |
| Run only the targeted checks above. Do not run the full suite or `nox -s docs` | the orchestrator runs both once, after the conversions |
| Execute one notebook at a time | each execution starts its own kernel |
| Set `n_jobs=1` everywhere | a parallel learner competes with the other authors |

In a cloud session, each author has a machine of its own. Run `nox -s docs` once before you open
the pull request.

## Own one notebook

Each notebook has exactly one owner at a time: the thread or author whose branch edits it. Edit
only your notebook, its callback module, and the files your change needs. Two rules apply to other
files.

- When you rename headings, you may retarget the links to them in other notebooks' markdown cells,
  so that they become file links. A markdown edit does not change the execution stamp. Bring in
  `main` first, and change nothing else in those notebooks.
- Each branch that changes a document makes the `tests/prose-report.md` rows of other branches
  stale. Rerun `python -m tests.prose --update` after you bring in `main`, just before you merge.

Report a change that shared infrastructure needs, and do not make it yourself. The shared files are
`tests/unit/test_documentation_runtime.py`, `tests/unit/tutorial_semantics/__init__.py`,
`tests/documents.py`, `tests/notebooks.py`, `tests/conftest.py`, `scripts/execute_notebook.py`,
`docs/conf.py`, `pyproject.toml`, and `uv.lock`. Send the change and the failure it repairs to
the orchestrator.

The report also gives the notebook's execution time, the exit status of each targeted check, and
every claim of the page that the change moved or removed.
