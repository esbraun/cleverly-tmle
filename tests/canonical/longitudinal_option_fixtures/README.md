# Longitudinal option fixtures: the R side

This directory holds the R side of three comparisons for the X33 fitting options. Each fixture
is one R script, one registry entry (`registry.py`) and, after the R run, one JSON file.
Every fixture is **pending** until the local Study queue runs its script in the pinned image
(TRACKER row 46). The Python exact tests of each option gate the option; a pending comparator
blocks nothing.

## Fixtures

| fixture | option part | R call | tolerance |
| --- | --- | --- | --- |
| `lmtp_trim.json` | d, `ratio_trim=` | `lmtp:::trim` on a fixed matrix with zeros and ties; `lmtp_tmle(control = lmtp_control(.trim = 1))` and `.trim = 0.9` on one sample | threshold and trimmed matrix 1e-15; zero pattern of the per-node ratios equal |
| `lmtp_history.json` | e, `markov_order=` | `lmtp:::LmtpVars$new(..., k)$history("A", t)` and `$history("L", t + 1)`, tau = 4, k in 0, 1, 2, Inf | column sets equal, with the declared conventions in `parity.py` |
| `survtmle_pooled.json` | j, `hazard_pooling=` | `survtmle(method = "mean")` with a per-visit `glm.ctime` and with `trt + W1 + W2 + t` | retention at every at-risk row 1e-6 |

`stand_ins/` holds the same scripts' output from a cloud image built on the pinned base image
and CRAN snapshot. It is not a fixture: the tests prove each comparison on it and on a copy
with one value changed.

## Images

The images are the pinned images of the registered studies.

| image | Dockerfile directory | used by |
| --- | --- | --- |
| `cleverly-lmtp-crossfit:1.5.4` | `tests/canonical/lmtp_crossfit` | `lmtp_trim.R`, `lmtp_history.R` |
| `cleverly-survtmle:1.1.1` | `tests/canonical/survtmle` | `survtmle_pooled.R` |

## Commands

Run every command from the repository root. Build each image once.

```bash
docker build -t cleverly-lmtp-crossfit:1.5.4 tests/canonical/lmtp_crossfit
docker build -t cleverly-survtmle:1.1.1 tests/canonical/survtmle
```

For each fixture (`lmtp_trim`, `lmtp_history` with the lmtp image, `survtmle_pooled` with the
survtmle image), the queue runs the command that `python -m
tests.canonical.longitudinal_option_fixtures.registry` checks and `registry.run_command` prints:

```bash
docker run --rm --entrypoint Rscript -e CLEVERLY_IMAGE_ID="$(docker image inspect --format '{{.Id}}' IMAGE)" -e CLEVERLY_SCRIPT_SHA256="$(sha256sum $PWD/tests/canonical/longitudinal_option_fixtures/NAME.R | cut -d" " -f1)" -v "$PWD/tests/canonical/longitudinal_option_fixtures:/fixture" IMAGE /fixture/NAME.R /fixture/NAME.json
```

Then `pytest -q tests/unit/test_longitudinal_option_fixtures.py` reads nothing new until the
registry entry is set `present`, which the follow-up change does.
