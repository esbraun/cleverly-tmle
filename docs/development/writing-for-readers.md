# Writing for readers

These rules apply to every reader-facing page of `cleverly`: the root `README.md` and every
Markdown, RST, or notebook source under `docs/`. Each documentation area has one reader, and a
rewritten data-scientist page passes a reader review before it merges.

`CLAUDE.md` gives the short form of these rules. [Contributing](contributing.md#write-documentation)
gives the steps that every documentation change follows, including the prose report.
[Example notebooks](example-notebooks.md) adds the rules that are special to a tutorial notebook.

## The readers

Each documentation area has one reader. Write for that reader, and apply the rules the table names.

| area | reader | which rules apply |
| --- | --- | --- |
| `README.md`, `docs/examples/`, `docs/getting-started/`, `docs/user-guide/`, `docs/index.md`, `docs/workflow.md` | the data scientist described below | every rule on this page |
| `docs/technical-reference/` | a statistician who knows causal inference and wants the exact estimand, the conditions, and the evidence | the machine-writing tells, and rules R5, R10, R13, and R14. Terms of art and formulas stay. The table stays the main device where the reader compares rows |
| `docs/development/` and the other pages under `docs/` (`docs/README.md`, `architecture-invariants.md`, `roadmap.md`, `references.md`) | contributors | the machine-writing tells. These pages get no reader review |

The data scientist:

- knows Python, pandas, and scikit-learn, and fits and validates predictive models with train/test
  splits and cross-validation;
- knows regression, confidence intervals, p-values, and standard errors;
- knows that correlation is not causation, and roughly what a confounder is;
- does not know causal inference theory: potential outcomes, identification, and efficiency
  theory;
- has not met TMLE, influence curves, double robustness, or positivity and estimands as terms;
- does not know `cleverly`'s objects or its vocabulary.

Their goal is to run this kind of analysis on their own data this afternoon, and then explain the
result to a manager who asks "so did it work, and how sure are we?". They skim. They read the code
and the first sentence after each output, and they return to the details only when something looks
wrong.

## Rules

| # | rule | why |
| --- | --- | --- |
| R1 | Lead with the answer. Open each example with an "At a glance" section: the question in one sentence, the answer with its interval in plain words, and the one caveat that matters most. Skip a list of the sections, which the sidebar already shows | the reader decides in 30 seconds whether this page is theirs |
| R2 | Put the main path first: the data, the naive comparison, setting up the analysis, the estimate, the checks, and the conclusion. Move theory and comparisons between estimators into a "Go deeper" section after the conclusion, unless that lesson is the point of the page | the reader wants to run the analysis before learning why it works |
| R3 | Define each unavoidable technical term in plain words the first time it appears, in the sentence that uses it. Link to the reference page for the precise version | a separate table of terms is read once and forgotten |
| R4 | Do not lean on `cleverly`'s own vocabulary. Say what the code does in ordinary words, then give the API name in backticks: "describe the study (`StudyProtocol`)". [Library words](#library-words-and-their-plain-replacements) lists the replacements | library terms are the hardest jargon, because the reader cannot look them up anywhere else |
| R5 | After each output, say what to look at and what it means, in two to four sentences. Point at the line or the column. Do not restate every number | the reader needs the reading, not a transcript |
| R6 | Quote a number at the precision that a decision needs: an effect to two or three significant figures. The narration test accepts a prose decimal within half a unit of its last written place, so rounding in prose needs no new print. Do not round so far that the check stops meaning anything, such as "0.3" for 0.2649 | 0.0021809 tells the reader nothing that -0.002 does not |
| R7 | Keep every statistical claim accurate. Simplify by choosing what to say, never by saying something false. A condition that a claim needs stays in the sentence, in plain words. Formulas, derivations, exact conditions, and probe numbers go in a collapsed `:::{dropdown} Details` block or on the linked reference page | accuracy is not negotiable; density is |
| R8 | Write paragraphs, not tables, unless the reader compares several items across the same columns. A table with one or two rows, or a table of reasons, is a paragraph. Keep a paragraph to six sentences, which `python -m tests.prose` reports | a table for each idea is a strong machine tell |
| R9 | Vary sentence length and join related ideas, but do not let any connective become a habit. Keep a sentence under about 30 words, and most well under | a run of one-idea sentences reads as a list with full stops, and a repeated "because ... so ..." is a new template |
| R10 | State a caveat once, where it matters, in its strongest true form. Say once, in the conclusion, that the page shows one simulated draw | repeated caveats teach the reader to skip them |
| R11 | Use the program's own words: "patients", "the navigation offer", "the experience score", "customers", "price". Use symbols such as $A$, $W$, and $Y$ only beside a formula or inside a Details block | the reader thinks in terms of the business problem |
| R12 | Print only the part of a library summary that the reader needs, with plain labels, unless the page teaches the summary itself. Show a full `summary()` once on a page and say which lines matter | 40 lines of output bury the two lines that the reading uses |
| R13 | Write headings as landmarks that a newcomer recognizes: a question, a task, or a plain noun phrase in sentence case. Do not number steps ("Step 5:") and do not put theory terms in headings. A method's plain name, such as "collaborative TMLE", is allowed where the page compares methods, if the first sentence under the heading defines it. Refer to another section by its name, not by a number | the sidebar and the headings are the reader's map |
| R14 | Link to another page's file, not to its heading anchors, and name the section in words. The link test resolves a fragment only against a heading, so the only safe anchor is a heading that is meant to stay | headings change when a page is rewritten |
| R15 | Use backticks only for code, column names, and an output token that the reader must find in the output. Do not put every printed word in backticks | backticks everywhere read as machine output |

These rules stay from the earlier house style: the active voice with a named actor, one word for
one meaning, no em dash and no spaced double hyphen, evidence for each material claim (a source, a
test, a stored output, or a stated condition), a check of each scientific claim against the code,
the tests, the artifacts, and the sources, and no empty transitions or intensifiers.

R6 rests on `tests/unit/tutorial_semantics/__init__.py`, which compares each prose decimal with
the stored outputs at the precision the prose writes. R14 rests on
`tests/unit/test_documentation_links.py`, which resolves a fragment only against a heading of the
target page.

## Library words and their plain replacements

On a data-scientist page, the prose must not lean on the words in the left column. The right
column says the same thing accurately in plain words. The API name may follow in backticks.

| term | say instead |
| --- | --- |
| estimand | "the quantity we want to estimate". Define "estimand" once if the page uses the word more than twice |
| protocol, `StudyProtocol`, fingerprint | "the study description": who is in the study, when follow-up starts, what each option means, and what the outcome is. The fingerprint is "an ID that ties each result to the description it came from"; mention it at most once |
| design, identify, identification | "tell `cleverly` which column is the treatment, which is the outcome, and which variables to adjust for"; "write down what must be true for the answer to be causal, and mark which of those conditions the data can check" |
| nuisance models, Q, g | "the outcome model", which predicts the outcome from the treatment and the covariates, and "the treatment model", which predicts who gets the treatment. Where present: "the response model" (who answers the survey) and "the censoring model" (who stays in follow-up) |
| targeting, fluctuation, clever covariate | "a correction step that uses the treatment model to remove the bias that the outcome model leaves" |
| influence curve | avoid it. If it is needed: "the standard error comes from how much each row moves the estimate" |
| positivity, overlap | "overlap". In general: "the data hold people like each patient under the option that the question asks about". For treat-all against treat-none: "every kind of patient had a real chance of either option". For a rule: "every patient had a real chance of the option that the rule assigns". For an odds change (an incremental intervention): it needs no overlap condition, so say so. For a dose shift: "the raised dose falls where similar patients actually received doses". Over time: "at each step, and for staying in follow-up" |
| support report | "the overlap check". Never write "support" alone, because it collides with the support labels below |
| double robustness | "the point estimate stays on target in large samples if either the outcome model or the treatment model is right. The interval needs both to be reasonably accurate" |
| cross-fitting, CV-TMLE, folds | "each row's predictions come from models trained without that row, like out-of-fold predictions in cross-validation" |
| truncation, `q_bounds` | "a cap on extreme weights"; "the known range of the outcome (0 to 1 here)" |
| refute, placebo, `random_common_cause`, subset, plasmode, generated outcome | "try to break the result": "shuffle the treatment, so the estimate should be about zero"; "add a column of pure noise, so the estimate should barely move"; "refit on random 70% samples, so the spread should match what the standard error predicts for samples of that size"; "replace the outcome with one where we know the answer" |
| rung, `L0`, `L1`, `L2`, verdict, registry, `VERDICT_EVIDENCE`, `Covered:` | "`cleverly` gives a pass or fail only when a simulation study backs that check for this kind of fit. Here no study does, so it shows the numbers and you judge them". Where an `L1` reading prints, say what that level reads on that page |
| sensitivity analysis, robustness value, `cf_y`, `cf_d`, $\nu^2$, Riesz representer | "how strong would a missing confounder have to be to erase the effect?" Put the symbols in a Details block |
| benchmark | for the omitted-variable bound: "a confounder as strong as `discharge_risk` could move the estimate at most this far, under the bound's assumptions". For a refit that drops a covariate: "dropping this variable moves the estimate this far; that is a comparison, not a bound" |
| E-value | "how strongly a missing confounder would need to be linked to both the treatment and the outcome to explain the effect away" |
| registered study, review probe, method evidence | "a simulation study that checks this method"; "a simulation we ran for this page (script ...)" |
| support label (`supported`, `experimental`) | "`cleverly` marks a result experimental when no validation study covers its setup" |
| ATE, ATT, ATC | "the average effect if everyone got the offer compared with no one" (ATE); "the effect among those who actually got it" (ATT). Keep the acronym after the first definition |
| MSM, LTMLE, C-TMLE, OAT, MTP, IPSI | spell each out once in plain words, such as "a trend line across the three schedules" or "a method for treatments that repeat over time". Keep the acronym only where the API name needs it |

## Machine-writing tells

These habits mark text as unreviewed machine output. Remove them on every page, including the
technical reference and the contributor pages. The examples come from the tutorial notebooks
before their rewrite.

| tell | example | what to do |
| --- | --- | --- |
| a fixed boilerplate opener | `**What this output tells you.**` before every output, 9 to 18 times on a page | start with the finding |
| bold lead-in labels | "**Placebo.**" and "**Subset.**" at the head of bullets | write the sentence |
| a table where a sentence fits | a "Why this method" table of features and choices; a one-row "caution / why" table | write the paragraph (R8) |
| narration of the page | "The table gives the answer and its limits." "Name the mechanism before reading the numbers." | delete it, and let the heading and the code say it |
| output restated digit by digit | "The t statistics are -0.35605 (p = 0.726) and -0.36553 (p = 0.719) on 19 degrees of freedom" | give the conclusion and the one number that carries it (R5, R6) |
| repeated defensive caveats | "One draw is not a coverage result", "One draw shows a pattern, not a property", and "on this draw" up to six times on a page | say it once (R10) |
| stacked qualifiers | "asymptotically valid under positivity, the declared independence of the rows, and a rate condition on the nuisances" in the main reading | in the main text, "valid in large samples when both models are reasonably accurate"; the exact condition goes in a Details block |
| the "X, not Y" contrast as a habit | 14 instances on one page | keep one where the contrast is the point; otherwise state X |
| runs of one-idea sentences | "The arms differ before the offer. The mean `discharge_risk` is 0.226 among offered patients and -0.219 among the others." | join related ideas (R9) |
| generic signposts and summary closers | "The analysis supports one statement." "Each choice below follows from a feature of the problem." | delete them, or replace them with the statement itself |
| checklists of what is not concluded | a five-bullet list headed "The analysis does not support these conclusions" | two or three sentences on the limits that a reader would actually reach for |
| abstract vocabulary | "the mechanism", "the reading", "the configuration", "the law" | name the thing: "the simulated data", "this fit", "the treatment model" |
| backticks on every output token | 528 backticks on one page | R15 |
| the same template on every page | identical heading sets, and an "At a glance" section in one fixed mold on every page | fit the wording to the page |
| em dashes, spaced double hyphens, and empty transitions | an em dash between clauses; a sentence that opens with `Notably,` or `it is worth noting` | remove them; `python -m tests.prose` reports them |

## The reader review

Run the reader review when you rewrite a whole data-scientist page, or write a new one. It has four
parts: a mechanical report, an answer key with a claim ledger, a persona review, and an accuracy
review. Run all of them on your branch, before you open the pull request.

### The mechanical report

Render the page for the reviewers and measure it:

```bash
python scripts/render_for_reader_review.py docs/examples/<stem>.ipynb --output <directory>
```

The script reads a notebook or a Markdown page. It writes two files into the output directory.
`<stem>.reader.md` is the reader's view: every markdown cell, code cell, and stored output in
order, with each cell id shown and each Details dropdown marked. `<stem>.report.md` holds the
measures below. The exit status is 1 when a gated measure fails.

| measure | pass |
| --- | --- |
| `**What this output tells you.**` even once, and any other bold opener used twice or more | none (gated) |
| step numbers in headings or prose, such as "Step 7" or a heading such as `## 1. Setup` | none (gated) |
| words of prose on the main path: everything before a "Go deeper" heading, excluding dropdowns and code | at most 2,000 (gated). The budget is 2,600 for `interventions`, `sequential-offers`, and `sequential-pricing`, which carry two or three analyses. `--budget` sets another limit |
| words inside each dropdown | reported. The accuracy reviewer confirms that each dropdown holds only formulas, exact conditions, derivations, and probe numbers, and no main-path explanation |
| the first use of each term in [library words](#library-words-and-their-plain-replacements), in prose outside code and dropdowns | listed with its cell id. The persona and accuracy reviewers confirm a plain definition at or before it |
| tables in prose | listed with their row and column counts. A table with fewer than three rows needs a reason |
| "on this draw" and the "X, not Y" pattern | counted. The persona reviewer judges them as tells |

### The answer key and the claim ledger

Before the first persona round, write an answer key for the page. Keep it with the review record of
the work item, outside the documentation tree. The key holds three things:

- Q7, a question that only this page answers, such as "Why does the page not adjust for gestational
  age?".
- The correct answer to each quiz question below, with the cells that support it.
- The claim ledger: every scientific claim of the page before the rewrite, by cell id and sentence,
  with its fate. The fate is one of: kept; simplified (give the new sentence); moved to a Details
  block; moved to a linked page (name it); or dropped (give the reason). Drop a claim only when it
  is redundant or no longer true.

The accuracy reviewer approves the key and Q7 before the first persona round, so the author does
not set an easy question.

### The persona review

A fresh reviewer gets only the reader's view, the persona card, the
[tells](#machine-writing-tells) table, and the prompt below. It has no repository access, so it
cannot fill a gap from the source. A reviewer who saw an earlier version reads a fix as already
understood, so every round uses a new reviewer.

The data-scientist card:

```text
You are a data scientist with five years of experience. You use Python, pandas and scikit-learn
daily, you know regression, cross-validation, confidence intervals and p-values, and you know
that correlation is not causation. You have never studied causal inference formally and have
never used cleverly. You were asked to estimate whether a program worked, using observational
data, and to explain the answer to your manager by tomorrow. Read the page the way you would:
skim, read the code, read the first lines after each output. Flag every place where you do not
know what a word means, why a step is there, what an output is telling you, or what to do next.
You may know some of these terms from elsewhere; flag a term anyway if the page uses it without
saying what it means here.
```

The statistician card, for a later review of the technical reference (that review needs the
maintainer's go):

```text
You are a biostatistician who knows potential outcomes, TMLE and semiparametric efficiency. You
want the exact estimand, the conditions, and the evidence for each claim, without reading around
filler. Flag every claim whose estimand, condition, or evidence you cannot find on the page or
through its links, every definition that is imprecise, and every sentence that adds nothing.
```

The persona prompt:

```text
Read the attached page as the person on your card. Return three parts.

1. Flags. For each flag give the cell id, the quoted text, and one class:
   - blocking: a term you need and are not given, even if you know it from elsewhere; a step
     whose purpose is unclear; an output with no reading, or a reading that does not say what
     to look at; a conclusion not stated in plain words; library vocabulary with no plain
     meaning; code you would change for your own data with no hint which lines.
   - tell: an instance of a row of the attached tells table.
   - minor: wording that slows you down.
2. Answers to the quiz below. Cite the cell ids each answer relies on. An answer with no
   supporting cell, or one that uses knowledge the page does not give, counts as wrong.
   Q1. What decision or question does this analysis answer?
   Q2. What is the answer, with its uncertainty, in words a manager understands?
   Q3. Why is the simple comparison wrong here?
   Q4. What must be true for the answer to be trusted, and which of those conditions can the
       data not check?
   Q5. Which lines would you change to run this on your own data?
   Q6. What should you not conclude from this page?
   Q7. <the page-specific question from the key>
3. The page's structure as you would describe it from the headings alone, every heading that
   did not predict its content, and whether "At a glance" gave you the answer before any code.
```

### The accuracy review

A second reviewer, with full repository access, checks the rewrite. Give it this prompt:

```text
Check every claim in the rewritten prose of <page> against its stored output, the code, the
technical reference, and the registered study it cites. Compare the claim ledger in <key> with
the rewrite: no claim lost without a reason, no simplification that makes a claim false, and
every condition a claim needs still present, in plain words or in a Details dropdown. Confirm
that no main-path explanation sits in a dropdown to meet the word budget. Grade the persona's
quiz answers against the key, using the cells the persona cited. Return each finding with the
cell id, the quoted text, what is wrong, and the evidence.
```

### Pass criteria

| criterion | pass |
| --- | --- |
| mechanical report | every gated measure passes |
| blocking flags | none open |
| tells | none open. A retained instance needs a one-line reason in the pull request body |
| quiz | Q1 to Q7 correct, and each supported by the cells the persona cited |
| At a glance | the persona found the answer there |
| accuracy review | no finding open |
| claim ledger | every claim has a fate, and every dropped claim has a reason |
| headings | the persona's description of the structure matches the headings |
| tests | the fast suite, the link test, and `nox -s docs` pass |

After three persona rounds without a pass, stop and report the open flags to the maintainer with a
recommendation.
