"""Regenerate the cross-fitted clustered LTMLE evidence on a t reference at few clusters."""

from pathlib import Path

from tests.canonical.regenerate import main
from tests.studies import few_cluster_crossfit_ltmle, few_cluster_crossfit_properties

if __name__ == "__main__":
    main(
        few_cluster_crossfit_ltmle,
        few_cluster_crossfit_properties,
        here=Path(__file__).resolve().parent,
    )
