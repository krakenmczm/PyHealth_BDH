from typing import Any, List, Dict
import random
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

def split_sd_aftertask(allperiod: List[Dict[str, Any]],
                       threshold_yr: int | None,
                       window: int | None,
                       historical_subsampling: bool | None,
                       train_frac: float,
                       valid_frac: float,
                       inperiod_test_frac: float,
                       seed: int) -> tuple[SampleDataset | dict]:
    '''Splits in-period records into training, validation, and test sets. Also splits out-of-period records into separate test sets by year.
        For numeric features, applies z-score normalization and zero-imputation for missing values on the *training* set, then extracts the fitted processors to apply to the validation, in-period test, and out-of-period test sets.

    Args:
        allperiod: Flattened records for patients in all years. 
        threshold_yr: Cutoff year, before which patient data is included for model training. Includes the cutoff year itself. If None, this becomes all-period model training, which Zhou et al. use as a baseline benchmark. They remark that this time-agnostic 
        window: Width of sliding window, if evaluating model drift using sliding window approach. 
            All-historical evaluation: window = None, historical_subsampling = False, and threshold_yr != None
            All-historical subsampled evaluation: window != None, historical_subsampling = True, and threshold_yr != None
            All-period evaluation: window = None, threshold_yr = None
        historical_subsampling: If all-historical evaluation, should the in-period set be subsampled to match the same size of sliding window.
            If historical_subsampling = True: window should not be None.
        train_frac: Proportion of in-period set to be used for model training.
        valid_frac: Proportion of in-period set to be used for model validation.
        inperiod_test_frac: Proportion of in-period set to be used for model test.
        seed: Random seed for splitting in-period set.
    
    Returns:
        inperiod_train_proc: SampleDataset, training set. Numeric features z-score normalized, and missing values zero-imputed. 
        inperiod_valid_proc: SampleDataset, validation set. Training set feature processor applied. 
        inperiod_test_proc: SampleDataset, in-period test set. Training set feature processor applied.
        outperiod_test_proc: List of SampleDatasets, one per out-of-period year. Empty if all-period evaluation (threshold_yr = None), or if threshold_yr is the most recent year represented in the dataset. Training set feature processor applied.
    '''
    yr_list = sorted(list(set([rec['est_year'] for rec in allperiod])))
    min_year_extent = min(yr_list)
    max_year_extent = max(yr_list)

    ### Input checking ###
    if threshold_yr is None:
        # All-period
        if window is None:
            print('Inferring all-period evaluation.')
            inperiod = allperiod
        else:
            raise ValueError('Sliding window width provided without threshold year.')
    else:
        # All-historical
        if window is None:
            if historical_subsampling is False:
                print('Inferring all-historical evaluation of model drift.')
                inperiod = [rec for rec in allperiod if rec['est_year'] <= threshold_yr]
            else: # historical_subsampling == True
                raise ValueError('Missing window width to be able to perform subsampling.')
        else: # window != None
            if (threshold_yr > max_year_extent) or (threshold_yr < min_year_extent):
                raise ValueError('Threshold year out of timespan covered by dataset.')
            if threshold_yr == max_year_extent:
                print('Threshold year is the latest year covered by dataset - this is equivalent to all-period evaluation.')
            if window <= 0:
                raise ValueError('Sliding window width must be a positive integer.')
            inperiod_sw = [rec for rec in allperiod if (rec['est_year'] <= threshold_yr) and (rec['est_year'] >= threshold_yr - window + 1)]
            if historical_subsampling is False:
                # Sliding window. If dataset small, classes imbalanced, and/or window small, this or the later splitting step may throw a ValueError reflecting detection of 1 class label though 2 class labels expected.    
                print('Inferring sliding window evaluation of model drift.')
                inperiod = inperiod_sw         
            # All-historical subsampling          
            else: 
                print('Inferring all-historical subsampled evaluation of model drift. Subsampling will create an in-period dataset of the same size as a sliding window of width ' + str(window) + ' preceding threshold year ' + str(threshold_yr) + '.')
                inperiod = random.sample([rec for rec in allperiod if rec['est_year'] <= threshold_yr], 
                                         k = len(inperiod_sw))
                # print(set([rec['est_year'] for rec in inperiod]))
                # print(set([rec['est_year'] for rec in inperiod_sw]))

    if (train_frac + valid_frac + inperiod_test_frac) != 1:
        raise ValueError('Sum of in-period training, validation, and test set proportions != 1.')

    ### In-period dataset ###

    # Convert inperiod to SampleDataset using create_sample_dataset, in preparation for splitting into train/validation/test sets. Pass RawProcessors to avoid early normalization before splitting.
    inperiod_presplit = create_sample_dataset(
        samples = inperiod,
        input_schema = {"numeric": 'raw', 
                        "binary": 'raw'},
        output_schema = {'mortality': 'raw'},
        in_memory = False
    )

    # Use split_by_patient to split inperiod into train/valid/test sets
    inperiod_train, inperiod_valid, inperiod_test = split_by_patient(dataset = inperiod_presplit, ratios = (train_frac, valid_frac, inperiod_test_frac), seed = seed)

    # For in-period training set only: generate copy of training set but this time with actual type-specific processors (e.g. TensorProcessor with normalization & impute_strategy, etc.)
    INPUT_SCHEMA = {
        "numeric": ("tensor", {"normalization": "zscore", "impute_strategy": "zero"}),
        "binary": "tensor"
        }
    OUTPUT_SCHEMA = {"mortality": "binary"}

    inperiod_train_proc = create_sample_dataset(
        samples = inperiod_train,
        input_schema = INPUT_SCHEMA,
        output_schema = OUTPUT_SCHEMA,
        in_memory = False
    )

    # Fitted train processors
    train_input_processors = inperiod_train_proc.input_processors
    train_output_processors = inperiod_train_proc.output_processors

    # For valid, inperiod test (& outperiod test, eventually): apply fitted train processors
    inperiod_valid_proc = create_sample_dataset(
        samples = inperiod_valid,
        input_schema = INPUT_SCHEMA,
        output_schema = OUTPUT_SCHEMA,
        input_processors = train_input_processors,
        output_processors = train_output_processors
    )

    inperiod_test_proc = create_sample_dataset(
        samples = inperiod_test,
        input_schema = INPUT_SCHEMA,
        output_schema = OUTPUT_SCHEMA,
        input_processors = train_input_processors,
        output_processors = train_output_processors
    )

    ### Out-of-period dataset ###
    
    # All-period
    if ((threshold_yr is None) and (window is None)) or (threshold_yr == max_year_extent):
        outperiod_test_proc = {}
    # All-historical +/- subsampling, sliding window
    else: 
        outperiod = [rec for rec in allperiod if rec['est_year'] > threshold_yr]
        out_yrs = sorted(list(set([rec['est_year'] for rec in outperiod])))
        outperiod_test_sets = {yr: [] for yr in out_yrs}
        outperiod_test_proc = {}

        for pt in outperiod:
            outperiod_test_sets[pt['est_year']].append(pt)

        for yr in sorted(outperiod_test_sets.keys()):
            yr_set = create_sample_dataset(
                samples = outperiod_test_sets[yr],
                input_schema = INPUT_SCHEMA,
                output_schema = INPUT_SCHEMA,
                input_processors = train_input_processors,
                output_processors = train_output_processors
            )
            outperiod_test_proc[yr] = yr_set

    return inperiod_train_proc, inperiod_valid_proc, inperiod_test_proc, outperiod_test_proc

def emdot_lr_modeling(inperiod_train: SampleDataset, 
                      inperiod_valid: SampleDataset, 
                      inperiod_test: SampleDataset, 
                      outperiod_test_proc: dict[int, SampleDataset] | None,
                      num_epochs: int = 50,
                      patience: int = 10) -> dict[str, Any]:

    num_features = len(inperiod_train[0]['numeric']) + len(inperiod_train[0]['binary'])
    aurocs = {'inperiod': {}, 'outperiod': {}}
    inperiod_train_dl = get_dataloader(inperiod_train, batch_size = len(dataset.unique_patient_ids))
    inperiod_valid_dl = get_dataloader(inperiod_valid, batch_size = len(dataset.unique_patient_ids))

    # All-period would not have any out-of-period test sets
    if not outperiod_test_proc:
        allperiod_phlr_model = LogisticRegression(dataset = inperiod_train, embedding_dim = num_features)

        allperiod_ph_trainer = Trainer(model = allperiod_phlr_model, 
                                       device = 'cpu', 
                                       metrics = ['roc_auc'])

        allperiod_ph_trainer.train(train_dataloader = inperiod_train_dl, 
                                val_dataloader = inperiod_valid_dl,
                                epochs = num_epochs,
                                patience = patience,
                                load_best_model_at_last = True)
        
        # Partition "in-period" test set by year, then report by-year test set AUROC for plotting
        test_set_yrs = sorted(list(set([rec['est_year'] for rec in inperiod_test])))
        test_set_partitioned = {yr: [] for yr in test_set_yrs}

        for pt in inperiod_test:
            test_set_partitioned[pt['est_year']].append(pt)

        for yr in test_set_partitioned.keys():
            inperiod_test_dl = get_dataloader(test_set_partitioned[yr], batch_size = len(dataset.unique_patient_ids))
            yr_scores = allperiod_ph_trainer.evaluate(dataloader = inperiod_test_dl)
            aurocs['inperiod'][yr] = yr_scores['roc_auc']
        
    else:
        inperiod_test_dl = get_dataloader(inperiod_test, batch_size = len(dataset.unique_patient_ids))

        phlr_model = LogisticRegression(dataset = inperiod_train, embedding_dim = num_features)

        ph_trainer = Trainer(model = phlr_model, 
                             device = 'cpu', 
                             metrics = ['roc_auc'])

        ph_trainer.train(train_dataloader = inperiod_train_dl, 
                    val_dataloader = inperiod_valid_dl,
                    epochs = num_epochs,
                    patience = patience,
                    load_best_model_at_last = True)

        # In-period test performance
        inperiod_scores = ph_trainer.evaluate(dataloader = inperiod_test_dl)
        aurocs['inperiod']['test_auroc'] = inperiod_scores['roc_auc']

        # Out-of-period test performance
        for yr, testset in outperiod_test_proc.items():
            outperiod_test_dl = get_dataloader(testset, batch_size = len(dataset.unique_patient_ids))
            outperiod_scores = ph_trainer.evaluate(dataloader = outperiod_test_dl)
            aurocs['outperiod'][yr] = outperiod_scores['roc_auc']

    return aurocs

def compute_results_by_yr(flat_records: List[Dict[str, Any]],
                          window: int | None,
                          historical_subsampling: bool,
                          seed: int) -> dict[int, Any]:
    results_by_yr = {}
    yr_list = sorted(list(set([rec['est_year'] for rec in flat_records])))

    for yr in yr_list:
        train, valid, test, outperiod_test = split_sd_aftertask(
            allperiod = flat_records, 
            threshold_yr = yr, 
            window = window, 
            historical_subsampling = historical_subsampling, 
            train_frac = train_frac, 
            valid_frac = valid_frac, 
            inperiod_test_frac = inperiod_test_frac, 
            seed = seed)
        yr_aurocs = emdot_lr_modeling(
            inperiod_train = train,
            inperiod_valid = valid,
            inperiod_test = test,
            outperiod_test_proc = outperiod_test,
            num_epochs = epochs,
            patience = patience
        )
        results_by_yr[yr] = yr_aurocs

    return results_by_yr

def get_emdot_results(sd_posttask: SampleDataset,
                      seed: int):

    flattened_records = flatten_records(sd_posttask)

    # All-period - no need to call compute_results_by_yr; the emdot_lr_modeling function will output the AUROCs for all-period if input arguments correct
    allperiod_train, allperiod_valid, allperiod_test, outperiod_test = split_sd_aftertask(
        allperiod = flattened_records, 
        threshold_yr = None, 
        window = None, 
        historical_subsampling = False, 
        train_frac = train_frac, 
        valid_frac = valid_frac, 
        inperiod_test_frac = inperiod_test_frac, 
        seed = run_seed
    )
    allperiod_res = emdot_lr_modeling(
        inperiod_train = allperiod_train,
        inperiod_valid = allperiod_valid,
        inperiod_test = allperiod_test,
        outperiod_test_proc = outperiod_test, # should be {}
        num_epochs = epochs,
        patience = patience
    )

    # All-historical
    allhist_res = compute_results_by_yr(
        flat_records = flattened_records,
        window = None,
        historical_subsampling = False,
        seed = seed
    )

    # All-historical, subsampling
    allhistsub_res = compute_results_by_yr(
        flat_records = flattened_records,
        window = window_size,
        historical_subsampling = True,
        seed = seed
    )

    # Sliding window
    sw_res = compute_results_by_yr(
        flat_records = flattened_records,
        window = window_size,
        historical_subsampling = False,
        seed = seed
    )

    return allperiod_res, allhist_res, allhistsub_res, sw_res

if __name__ == "__main__":
    
    # For train/valid/test splits
    train_frac = 0.5
    valid_frac = 0.25
    inperiod_test_frac = 0.25

    # Dataset splitting and modeling config
    window_size = 4
    run_seed = 6250
    epochs = 50
    patience = 10

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

    # Run the EMDOT pipeline
    allperiod_res, allhist_res, allhistsub_res, sw_res = get_emdot_results(
        sd_posttask = ds_sd, 
        seed = run_seed
    )