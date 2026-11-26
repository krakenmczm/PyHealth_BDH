from typing import Any, List, Dict
from pyhealth.datasets import MIMIC4EHRDataset, SampleDataset, split_by_patient, create_sample_dataset
from pyhealth.datasets.utils import get_dataloader
from pyhealth.models import LogisticRegression
from pyhealth.trainer import Trainer
from pyhealth.tasks.in_icu_mortality_emdot_mimic4 import (
    EMDOT_CONFIG,
    EMDOT_TABLES,
    InICUMortalityEMDOTMIMIC4,
    cohort_summary,
    select_features,
    vectorize
)

# Folder that holds hosp/, icu/ and derived/ (change to yours)
ROOT = "/Users/karen/mimic-iv-3.1"

# For train/valid/test splits
train_frac = 0.5
valid_frac = 0.25
inperiod_test_frac = 0.25

def flatten_records(sample_ds: SampleDataset) -> List[Dict[str, Any]]:
    '''Further process resulting SampleDataset from set_task: 
        Remove excluded samples and organize numeric vs binary features.
        Remove nesting level for patient record.

    Args:
        sample_ds: SampleDataset produced from setting task on base MIMIC4EHRDataset with EMDOT config

    Returns:
        flat_records: Further pre-processed list of samples
    '''
    flat_records = []

    for pt in sample_ds:
        pt_rec_dict = {}
        pt_rec = pt['record']

        pt_rec_dict['patient_id'] = pt_rec['patient_id']
        pt_rec_dict['hadm_id'] = pt_rec['hadm_id']
        pt_rec_dict['stay_id'] = pt_rec['stay_id']
        pt_rec_dict['est_year'] = pt_rec['est_year']

        if pt_rec['excluded_at'] is None:
            num, bin = vectorize(pt_rec, spec)
            pt_rec_dict['numeric'] = num
            pt_rec_dict['binary'] = bin

            pt_rec_dict['mortality'] = pt['mortality'].item()

            flat_records.append(pt_rec_dict)

    return flat_records

if __name__ == "__main__":
    dataset = MIMIC4EHRDataset(
        root=ROOT,
        tables=EMDOT_TABLES,
        config_path=EMDOT_CONFIG,
        dev=True,  # small sample first; remove for the full run
    )

    # First set_task: To get feature_spec and select_features dictionary
    ds_sd = dataset.set_task(InICUMortalityEMDOTMIMIC4())
    records = [s["record"] for s in ds_sd]
    spec, sel_dict = select_features(records)

    # Dataset summary
    selection, by_year = cohort_summary(records)
    print(selection)
    print(by_year)

    # Obtain flattened records
    flattened_records = flatten_records(ds_sd)

    # Create sample dataset from flattened records using create_sample_dataset, in preparation for splitting into train/validation/test sets. Pass RawProcessors to avoid early normalization before splitting
    all_pts_sd_presplit = create_sample_dataset(
        samples = flattened_records,
        input_schema = {"numeric": 'raw', 
                        "binary": 'raw'},
        output_schema = {'mortality': 'raw'},
        in_memory = False
    )

    # Use split_by_patient to split into train/valid/test sets
    allperiod_train, allperiod_valid, allperiod_test = split_by_patient(dataset = all_pts_sd_presplit, ratios = (train_frac, valid_frac, inperiod_test_frac), seed = 6250)

    # For train set only: generate copy of train set but this time with actual type-specific processors (e.g. TensorProcessor with normalization & impute_strategy, etc.)
    allperiod_train_processed = create_sample_dataset(
        samples = allperiod_train,
        input_schema = {"numeric": ('tensor', {'normalization': "zscore", "impute_strategy": "zero"}), 
                        "binary": 'tensor'},
        output_schema = {'mortality': 'binary'},
        in_memory = False
    )

    # Fitted training set processors
    allperiod_train_input_processors = allperiod_train_processed.input_processors
    allperiod_train_output_processors = allperiod_train_processed.output_processors

    # For valid, in-period test (& out-of-period test, eventually): apply fitted training set processors
    allperiod_valid_processed = create_sample_dataset(
        samples = allperiod_valid,
        input_schema = allperiod_train_input_processors,
        output_schema = allperiod_train_output_processors
    )

    allperiod_test_processed = create_sample_dataset(
        samples = allperiod_test,
        input_schema = allperiod_train_input_processors,
        output_schema = allperiod_train_output_processors
    )

    # Create dataloaders for train/valid/test sets
    allperiod_train_dl = get_dataloader(allperiod_train_processed, batch_size = len(dataset.unique_patient_ids))
    allperiod_valid_dl = get_dataloader(allperiod_valid_processed, batch_size = len(dataset.unique_patient_ids))
    allperiod_test_dl = get_dataloader(allperiod_test_processed, batch_size = len(dataset.unique_patient_ids))

    # Get number of features
    num_feats = len(allperiod_train_processed[0]['numeric']) + len(allperiod_train_processed[0]['binary'])

    # Create model object
    phlr_model = LogisticRegression(dataset = allperiod_train_processed, embedding_dim = num_feats)

    # Create Trainer object
    ph_trainer = Trainer(model = phlr_model, device = 'cpu', metrics = ['roc_auc'])

    # Create training environment
    ph_trainer.train(train_dataloader = allperiod_train_dl, 
                 val_dataloader = allperiod_valid_dl,
                 test_dataloader = allperiod_test_dl,
                 epochs = 50,
                 patience = 10,
                 load_best_model_at_last = True)

    # Report test set metric(s)
    ph_trainer.evaluate(dataloader = allperiod_test_dl)