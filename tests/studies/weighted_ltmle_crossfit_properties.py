"""Properties for cross-fitted weighted end-of-study longitudinal TMLE."""

from tests.studies.canonical_weighted_ltmle_crossfit import STUDY
from tests.studies.weighted_longitudinal_properties_common import *  # noqa: F403
from tests.studies.weighted_longitudinal_properties_common import (
    generate_property_rows as _generate,
)
from tests.studies.weighted_longitudinal_properties_common import (
    summarize_properties as _summarize,
)


def generate_property_rows(*, n_jobs: int = STUDY_JOBS):  # noqa: F405
    return _generate(STUDY, cross_fit=True, n_jobs=n_jobs)


def summarize_properties(rows):
    return _summarize(rows, STUDY)
