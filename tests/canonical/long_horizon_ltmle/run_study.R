# Comparator runner of the long-horizon ordinary LTMLE study: a declared stub.
#
# The R ltmle 1.3-0 comparator runs of this study are pending. Roadmap item X59 owns them: it
# replaces this file with the K-node runner, moves the record's pending_reference to
# reference, and regenerates the study. Until then the regeneration driver runs no reference
# process, and the manifest records the bytes of this file as a declared source.
#
# published_row() states the row the X59 runner publishes, in the column order of
# tests/studies/evidence/schema.py (REPLICATE_COLUMNS).

published_row <- function(row) {
  data.frame(
    implementation = row$implementation,
    scenario = row$scenario,
    replicate = row$replicate,
    n = row$n,
    estimand = row$estimand,
    truth = row$truth,
    estimate = row$estimate,
    inference_estimate = row$inference_estimate,
    std_error = row$std_error,
    ci_lower = row$ci_lower,
    ci_upper = row$ci_upper,
    inference_scale = row$inference_scale,
    covered = row$covered,
    initial_estimate = row$initial_estimate,
    stringsAsFactors = FALSE
  )
}

stop(
  "the R ltmle 1.3-0 comparator of this study is pending under roadmap item X59; ",
  "this runner is a declared stub and fits no replication"
)
