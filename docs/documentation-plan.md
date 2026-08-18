# Production documentation plan

This plan defines the production documentation package for `cleverly`. It was written before the
site implementation so the information architecture, scientific coverage, and acceptance gates
can be reviewed independently of the finished prose.

## Objective and audience

The documentation must let an applied causal-inference user move from a research question to a
defensible `cleverly` result, while also letting a methodologist trace each implemented estimator
to its statistical parameter, assumptions, literature, source implementation, and correctness
evidence. It must serve three readers without mixing their paths:

1. a new user who needs installation and one successful fit;
2. an analyst who needs design, estimand, learner, inference, and diagnostic guidance;
3. a reviewer or contributor who needs equations, citations, implementation provenance, API
   contracts, and validation evidence.

The package remains alpha software. The docs must distinguish shipped behavior from proposals and
must not turn roadmap items into apparent capabilities.

## Reference review

Two documentation sets provide the structural models.

- [DoubleML](https://docs.doubleml.org/stable/) makes the applied path visible at the top level:
  Getting Started, Workflow, User Guide, Examples, and API. Its workflow starts with problem
  formulation and causal assumptions before data objects, learners, and fitting. `cleverly` will
  use that reader journey, adapted to its typed design -> estimand -> identification -> method ->
  result model.
- [diff-diff](https://diff-diff.readthedocs.io/) groups a large API by user-recognizable
  capabilities, generates object pages from source docstrings, and gives implementations their own
  methodology and reference pages. `cleverly` will use that separation between narrative guidance,
  implementation reference, and generated Python API.

The resulting site will not copy either project’s content or visual identity. It will use their
information-architecture strengths and make `cleverly`'s scientific evidence unusually explicit.

## Site architecture

The site will use Sphinx with MyST Markdown and the PyData Sphinx Theme. This choice keeps the
repository's existing Markdown readable on GitHub, supports math and cross-references, and provides
stable `autodoc` / `autosummary` generation from the installed package.

The primary navigation will be:

1. **Getting started** — support status, installation, first point-treatment fit, reading a result,
   and the next documents to visit.
2. **Workflow** — formulate the question, declare the observed-data design, choose a typed
   estimand, inspect identification, configure estimation, assess the fit, and communicate the
   result.
3. **User guide** — data backends and study designs; estimands and interventions; learners and
   cross-fitting; estimation methods; inference and results; diagnostics and sensitivity;
   longitudinal designs; persistence; capability refusals.
4. **Technical reference** — a coverage matrix and one focused account for every implemented
   statistical family and estimator variant. Each entry must state the parameter or algorithm,
   identification conditions, estimator / influence-function construction, primary citation,
   existing implementation used as provenance where applicable, local source location, and local
   validation evidence.
5. **Examples** — complete applied workflows for point treatment, interventions, longitudinal
   treatment, and post-fit assessment. Examples are explanatory; existing fast tests remain the
   behavioral evidence, while documentation tests compile Python fences.
6. **Python API** — generated reference for the supported root API, grouped by study and design,
   estimands, methods, results and assessment, learners, interventions, datasets, sensitivity,
   validation, and exceptions.
7. **Development reference** — the existing architecture invariants, evidence manifest, migration
   guide, roadmap, benchmark reports, and accepted implementation plans. These remain available but
   do not interrupt the applied reading path.

The root `README.md` will become the repository landing page: concise value proposition, maturity
warning, installation, one canonical quickstart, capability summary, documentation routes,
development gate, citation, and license. Detail belongs on the site rather than in an ever-growing
README.

## Technical-reference coverage contract

The implementation reference will cover these shipped families explicitly:

| family | required coverage |
| --- | --- |
| Point-treatment TMLE | counterfactual means; ATE, ATT, ATC; risk and odds ratios; multi-arm contrasts; missing outcomes; observation weights |
| Population interventions | natural-course mean, population-attributable risk, and population-attributable fraction |
| Known regimes | deterministic, dynamic, and stochastic regime means and contrasts |
| Modified treatment policies | continuous-dose shift means and contrasts, density-ratio clever covariate, support conditions |
| Incremental propensity interventions | odds tilts, treatment-mechanism targeting, and the non-double-robust remainder condition |
| Marginal structural models | point and longitudinal projections, working-model interpretation, pooled targeting, rank conditions |
| Controlled direct effects | intermediate mechanism, missingness composition, supported point-treatment contract |
| Longitudinal TMLE | end-of-study outcomes, sequential regression, time-varying confounding, censoring, regimen contrasts |
| Survival and competing risks | risk-set masks, cause-specific recursion, horizons, and structured parameter keys |
| Collaborative TMLE | candidate truncation path, cross-validated selector, refit, supported target families |
| DR-TMLE | reduced regressions, compatible targeting, theorem-backed supported targets, and explicit refusals |
| Cross-fitting and inference | outer / learner folds, repeated cross-fitting, CV-TMLE, influence-curve variance, clustering, simultaneous intervals, bootstrap |
| Assessment | positivity and score diagnostics, sensitivity analyses, validation reports, variable importance, and persistence / replayability |

For every row, the site must link to a primary citation in `references.md`, identify any external
implementation used as provenance (for example `tmle3`, `ltmle`, or `drtmle`) without treating
parity as proof, link to the relevant module under `src/cleverly`, and link to the evidence manifest
or named tests that can detect an incorrect implementation.

## Build and publishing design

The implementation will add:

- a `docs` optional dependency containing pinned or bounded Sphinx, MyST, theme, and copy-button
  dependencies;
- `docs/conf.py`, a root `docs/index.md`, section indexes, API autosummary pages, static styling, and
  a Sphinx build command;
- a GitHub Pages workflow that builds the default branch and deploys generated HTML as an artifact,
  without committing it to the source branch;
- a local `nox -s docs` session that installs the package and fails on Sphinx warnings;
- link, navigation, and API-surface tests that make new docs part of the existing documentation
  contract.

Generated `_build/` and autosummary output will remain uncommitted. Published pages should be built
from the default branch so source and API signatures stay aligned. The canonical site is
`https://esbraun.github.io/cleverly-tmle/`; the alpha release publishes one current version without
a version switcher.

## Work packages

### 1. Infrastructure and navigation

Add Sphinx / MyST configuration, theme styling, dependency declarations, the GitHub Pages workflow,
the docs build session, and the complete toctree. Prove that a clean local build finishes with
warnings treated as errors.

### 2. Landing pages and onboarding

Rewrite the root README and add the site landing, installation, getting-started, and workflow pages.
Use one canonical beginner example and consistent terminology across all four entry points.

### 3. Applied user guide and examples

Restructure the current recipes into task-oriented pages. Cover data inputs, every typed estimand
family, method configuration, learner selection, point and longitudinal paths, results, inference,
assessment, persistence, and refusals. Add complete examples that link back to the concepts they
exercise.

### 4. Technical reference

Build the coverage matrix and focused method pages from current code, tests, `methodology.md`,
`drtmle.md`, `evidence.md`, and `references.md`. Resolve discrepancies in favor of code and tests;
do not preserve a historical claim merely because existing prose contains it.

### 5. Python API

Generate categorized autosummary pages from supported public objects. Ensure every symbol exported
by `cleverly.__all__` is either present in the API navigation or deliberately accounted for by a
test-enforced exclusion. Fix only documentation defects necessary for a warning-free build; do not
expand the runtime public surface as part of this package.

### 6. Verification and release

Run, in order:

1. documentation link and Python-fence tests while iterating;
2. `ruff check .` and `ruff format --check .`;
3. `mypy src/cleverly`;
4. the warning-as-error Sphinx build;
5. the full fast test tier, without any slow tier because documentation cannot execute a changed
   statistical path or alter a sampling claim.

Finally, inspect the rendered site at desktop and narrow viewport widths, audit every objective
against files and build output, commit the complete package as one intentionally scoped change,
push `agent/production-docs`, and open one draft pull request against `main` with the local
validation record. Hosted GitHub Actions are not a correctness signal for this repository.

## Acceptance criteria

The package is complete only when all of the following are true:

- the README is a useful standalone landing page and prominently routes to the canonical hosted
  site;
- Getting Started, Workflow, User Guide, Technical Reference, Examples, and Python API are all
  first-class site sections reachable from the main navigation;
- the technical reference accounts for every family in the coverage contract with theory,
  citation, external provenance where applicable, local implementation, and evidence;
- every root public symbol is covered by generated API documentation or an explicit exclusion;
- a clean warning-as-error site build succeeds;
- all relative links and all Python fences pass the repository's documentation tests;
- lint, formatting, type checking, and the full fast test tier pass locally;
- generated HTML is visually checked and no build artifact is committed;
- the Pages workflow builds the warning-as-error site and deploys its artifact from `main`;
- one branch, one coherent commit, and one draft PR contain the completed documentation package.
