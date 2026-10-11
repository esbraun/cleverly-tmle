# The RM39 fixture runner with one random draw inside each fit.
#
# usage: Rscript fixture_random_runner.R SAMPLES.csv.gz TRUTH.csv OUTPUT.csv
#
# The resume smoke uses it: a run stopped part way and resumed must write the bytes of the
# uninterrupted run.  That holds only if each fit draws from the seed the driver declared for
# its replicate, so an unseeded draw, or a seed that depends on the worker, fails the smoke.
fixture_random <- TRUE
source("/fixture/harness_fixture/fixture_runner.R")
