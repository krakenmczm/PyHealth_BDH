from pyhealth.datasets import MIMIC4EHRDataset
from pyhealth.tasks.in_icu_mortality_emdot_mimic4 import (
    EMDOT_CONFIG,
    EMDOT_TABLES,
    InICUMortalityEMDOTMIMIC4,
    cohort_summary,
)

# Test comment

# Folder that holds hosp/, icu/ and derived/ (change to yours)
ROOT = "/Users/karen/mimic-iv-3.1"

if __name__ == "__main__":
    dataset = MIMIC4EHRDataset(
        root=ROOT,
        tables=EMDOT_TABLES,
        config_path=EMDOT_CONFIG,
        dev=True,  # small sample first; remove for the full run
    )
    records = [s["record"] for s in dataset.set_task(InICUMortalityEMDOTMIMIC4())]
    selection, by_year = cohort_summary(records)
    print(selection)
    print(by_year)