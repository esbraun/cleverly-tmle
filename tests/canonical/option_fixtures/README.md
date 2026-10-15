# Option fixtures: the R side

This directory holds the R side of four comparisons for the targeting and fitting options.
Each fixture is one R script, one registry entry and, after the R run, one JSON file.
Every fixture is **pending**: the cloud session that wrote the scripts has no R.
The Python exact tests of each option gate the change. A pending comparator blocks nothing.

Reader-facing rows that cite a pending fixture carry the label "exact tests, with comparator runs pending".

## Fixtures

| fixture | option parts | R call | tolerance |
| --- | --- | --- | --- |
| `tmle3_updater.json` | a, b, c | tmle_TSM_all() with tmle3_Update: sample_size, scaled_var and constrain_step | step counts and best step equal; cumulative epsilon and final estimates 1e-6; thresholds 1e-9 relative; final ED 1e-8 |
| `drtmle_stratify.json` | f | drtmle(stratify = TRUE) and drtmle(stratify = FALSE), guard = NULL | `Qn` and propensity 1e-8; TMLE estimate 1e-6 |
| `ctmle_patience.json` | g | ctmleDiscrete(preOrder = TRUE, patience = 2) and patience = FALSE | selected index equal; cross-validated score 1e-6 relative; estimate 1e-6 |
| `ctmle_wg.json` | h | ctmleDiscrete(Wg = W[, c(1, 3)], preOrder = TRUE) and Wg = W | selected index equal; cross-validated score 1e-6 relative; estimate 1e-6 |

## Images

The images are the pinned images of the registered studies. The base image is pinned by digest.

| image | Dockerfile directory | base image | package commit |
| --- | --- | --- | --- |
| `cleverly-tmle3-reference:ed72f8a` | `tests/canonical/tmle3` | `rocker/r-ver:4.5.2@sha256:fd4ccdd3a4a6f7ef805e2daeee2a0fe3bf126bc231f36351223baecf5a595a4c` | `ed72f8a20e64c914ab25ffe015d865f7a9963d27` |
| `cleverly-drtmle-reference:538a3a2` | `tests/canonical/drtmle` | `rocker/r-ver:4.5.2@sha256:fd4ccdd3a4a6f7ef805e2daeee2a0fe3bf126bc231f36351223baecf5a595a4c` | `538a3a264c1ca984b6d88978ca7f96165f43152c` |
| `cleverly-ctmle-reference:18de559` | `tests/canonical/ctmle_selector` | `rocker/r-ver:4.5.2@sha256:fd4ccdd3a4a6f7ef805e2daeee2a0fe3bf126bc231f36351223baecf5a595a4c` | `18de559f47dc1286617350a0668391e80e1dbf7c` |

The built image id enters each JSON file through `CLEVERLY_IMAGE_ID`. No image was built in the cloud session.

## Commands

Run every command from the repository root. Build each image once.

```bash
docker build -t cleverly-tmle3-reference:ed72f8a tests/canonical/tmle3
docker build -t cleverly-drtmle-reference:538a3a2 tests/canonical/drtmle
docker build -t cleverly-ctmle-reference:18de559 tests/canonical/ctmle_selector
```

The local Study queue runs one command for each fixture.
The container mounts this directory, so each script writes its JSON file beside itself.
The `ctmle` image sets its own entry point, so every command sets `--entrypoint Rscript`.

### tmle3_updater

```bash
docker run --rm --entrypoint Rscript -e CLEVERLY_IMAGE_ID="$(docker image inspect --format '{{.Id}}' cleverly-tmle3-reference:ed72f8a)" -e CLEVERLY_SCRIPT_SHA256="$(sha256sum $PWD/tests/canonical/option_fixtures/tmle3_updater.R | cut -d" " -f1)" -v "$PWD/tests/canonical/option_fixtures:/fixture" cleverly-tmle3-reference:ed72f8a /fixture/tmle3_updater.R /fixture/tmle3_updater.json
```

### drtmle_stratify

```bash
docker run --rm --entrypoint Rscript -e CLEVERLY_IMAGE_ID="$(docker image inspect --format '{{.Id}}' cleverly-drtmle-reference:538a3a2)" -e CLEVERLY_SCRIPT_SHA256="$(sha256sum $PWD/tests/canonical/option_fixtures/drtmle_stratify.R | cut -d" " -f1)" -v "$PWD/tests/canonical/option_fixtures:/fixture" cleverly-drtmle-reference:538a3a2 /fixture/drtmle_stratify.R /fixture/drtmle_stratify.json
```

### ctmle_patience

```bash
docker run --rm --entrypoint Rscript -e CLEVERLY_IMAGE_ID="$(docker image inspect --format '{{.Id}}' cleverly-ctmle-reference:18de559)" -e CLEVERLY_SCRIPT_SHA256="$(sha256sum $PWD/tests/canonical/option_fixtures/ctmle_patience.R | cut -d" " -f1)" -v "$PWD/tests/canonical/option_fixtures:/fixture" cleverly-ctmle-reference:18de559 /fixture/ctmle_patience.R /fixture/ctmle_patience.json
```

### ctmle_wg

```bash
docker run --rm --entrypoint Rscript -e CLEVERLY_IMAGE_ID="$(docker image inspect --format '{{.Id}}' cleverly-ctmle-reference:18de559)" -e CLEVERLY_SCRIPT_SHA256="$(sha256sum $PWD/tests/canonical/option_fixtures/ctmle_wg.R | cut -d" " -f1)" -v "$PWD/tests/canonical/option_fixtures:/fixture" cleverly-ctmle-reference:18de559 /fixture/ctmle_wg.R /fixture/ctmle_wg.json
```

## After the run

1. Check that the file ends in LF and that `provenance.script_sha256` equals the registry checksum.
2. Set `status="present"` and `reason=""` on the entry in `registry.py`.
3. Run `pytest tests/unit/test_option_fixtures.py`. The parity test of the entry now compares the file.
4. Fix every finding in `cleverly` or in the script. A change to a script changes its checksum.
5. Print the checksums with `python -m tests.canonical.option_fixtures.registry` and update the registry.
6. Remove the pending label from the reader-facing rows that cite the file.

## What each script exports

| fixture | content |
| --- | --- |
| `tmle3_updater.json` | the sample; the initial outcome regression at the observed treatment and at each arm; the propensity; for ten updater runs the per-step epsilons and EDs, the best step, the final arrays, estimates and thresholds, and a replay of the walk |
| `drtmle_stratify.json` | the sample; for `stratify = TRUE` and `FALSE` the initial outcome regression and propensity at both arms and the estimates |
| `ctmle_patience.json` | the sample, the selection folds, the initial outcome regression; for `patience = FALSE` and `2` the selected index, the estimate and the score of each candidate |
| `ctmle_wg.json` | the same for `Wg = W[, c(1, 3)]` and `Wg = W` |

Every file holds `fixture`, `schema`, `provenance` (script, checksum, image id, R version, package versions), the seed and `n`.

## Declared differences and unchecked calls

The scripts were written against the package sources, not run. The first run may find a wrong call.

| item | note |
| --- | --- |
| first step | `tmle3_Update$update` takes a step before it first tests the rule. `solve_fluctuation` tests first. The comparison covers a fit whose initial score meets the rule with one forced step. |
| unconstrained updater | `tmle3` takes one maximum-likelihood step. The comparison uses the Newton solution of `solve_fluctuation`, not a walk. |
| pooled formula | `drtmle(stratify = FALSE)` runs with `glm_Q = "A + W1 + W2"`. The formula `"W1 + W2"` would drop the treatment from the pooled fit. |
| `ctmle` ties | `ctmle` takes the last tied minimum (D7), and scores the rest of its batch after the counter fires (D7b). With `preOrder = TRUE` a batch is one covariate, so neither shows in these fixtures. |
| initial outcome regression | The R scripts fit `glm` and pass the arm predictions as `Q`. `cleverly` fits a binomial `GLM` on the same columns (`q_covariates`) and the comparison checks the two agree to 1e-8. |
| `tools::sha256sum` | Used for the script checksum. Believed to be new in R 4.5.0; the call was not run here. `CLEVERLY_SCRIPT_SHA256` is the fallback. |

