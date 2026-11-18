"""In-ICU mortality 24 hours after ICU admission, for EMDOT on MIMIC-IV.

Contributors: <Your Name>, <Partner Name>
Paper: Evaluating Model Performance in Medical Datasets Over Time
Paper link: https://proceedings.mlr.press/v219/zhou23a.html
Description: in-ICU mortality prediction task with an estimated calendar
year per sample, for evaluating models over time (EMDOT) on MIMIC-IV.

Task for a replication of the MIMIC-IV experiment of Zhou, H., Chen, Y. and
Lipton, Z. C., "Evaluating Model Performance in Medical Datasets Over Time",
Machine Learning for Healthcare (MLHC) 2023 (code:
https://github.com/acmi-lab/EvaluationOverTime).

**Data.** The original MIMIC-IV tables (``hosp/``, ``icu/``) plus four
tables of the MIT-LCP mimic-code concepts (``mimiciv_derived``):
``first_day_vitalsign``, ``first_day_lab``, ``first_day_sofa`` and
``first_day_rrt``, saved as Parquet in ``derived/``. See
``pyhealth/datasets/configs/mimic4_emdot.yaml``.

**Cohort** (one sample per patient, in this order; each step can be changed):

1. First ICU stay per patient (earliest ``intime``).
2. Admitted to the hospital (``hadm_id`` recorded).
3. ICU discharge time recorded.
4. Anchor year group starting in ``min_anchor_start`` or later (2014).
5. Estimated year of the stay at most ``max_year`` (2022).
6. ICU stay of at least ``min_icu_hours`` (24).
7. Clinical data in the first 24 h: at least one first-day vital sign or lab.

**Labels** (all three are kept on each sample; ``label`` picks the target):

* ``died_icu_outtime_dod``: date of death equals the ICU discharge date.
* ``died_in_icu``: date of death between the ICU admission and discharge dates.
* ``died_in_icu_deathtime``: in-hospital time of death within the ICU stay.

**Features** (known 24 h after ICU admission):

* Demographics: age at ICU admission, sex, hospital service at ICU admission
  (``services``), first care unit.
* ``adm_``: admission type and location, insurance, language, marital status,
  race, emergency department visit (``any_ed``). Discharge information is
  never used.
* ``vit_``, ``fdlab_``, ``sofa_``: every column of the first-day vital sign,
  lab and SOFA tables. ``rrt_rrt``: 1 if dialysis was present or active.
* ``px_``: procedure codes of the admission done by the end of the first
  24 h (day level: ``chartdate``).
* ``med_`` and ``rx_``: medications given in the first 24 h (``emar`` events
  matched to their prescription), by eMAR medication name and by
  prescription drug name.
* ``medclass_``: drug classes of those medications, from a GSN/name lookup
  table (optional, ``medication_class_map``).
* ``dxprior_``: diagnosis codes of hospital stays that ended before this ICU
  stay. Current-admission diagnoses (``dx_``) leak the outcome and are off
  unless ``use_diagnoses=True`` (drift analysis only).

For each code group: ``has_record``, ``n_distinct`` and one yes/no column per
code among the ``top_n`` (50) most common in the cohort. Numeric features
missing for more than ``max_missing_rate`` (10%) of the cohort are removed.

Choosing the top codes and categories and removing mostly-missing features
needs the whole cohort, so the task runs twice:

    >>> from pyhealth.datasets import MIMIC4EHRDataset
    >>> from pyhealth.tasks.in_icu_mortality_emdot_mimic4 import (
    ...     EMDOT_CONFIG, EMDOT_TABLES, InICUMortalityEMDOTMIMIC4,
    ...     cohort_summary, select_features)
    >>> dataset = MIMIC4EHRDataset(  # doctest: +SKIP
    ...     root="/path/to/mimiciv/3.1", tables=EMDOT_TABLES,
    ...     config_path=EMDOT_CONFIG)
    >>> records = [s["record"] for s in dataset.set_task(  # doctest: +SKIP
    ...     InICUMortalityEMDOTMIMIC4())]                   # 1. one record per patient
    >>> selection, by_year = cohort_summary(records)        # doctest: +SKIP
    >>> spec, dictionary = select_features(records)         # doctest: +SKIP
    >>> samples = dataset.set_task(                         # 2. model inputs
    ...     InICUMortalityEMDOTMIMIC4(feature_spec=spec))   # doctest: +SKIP
"""

import ast
import math
import os
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd
import polars as pl

from .base_task import BaseTask

__all__ = [
    "EMDOT_CONFIG",
    "EMDOT_TABLES",
    "LABELS",
    "InICUMortalityEMDOTMIMIC4",
    "cohort_summary",
    "estimate_year",
    "load_medication_class_map",
    "safe_name",
    "select_features",
    "vectorize",
]

EMDOT_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "datasets",
    "configs",
    "mimic4_emdot.yaml",
)
"""Dataset config with the tables and columns this task reads."""

EMDOT_TABLES = [
    "services",
    "diagnoses_icd",
    "procedures_icd",
    "emar",
    "first_day_vitalsign",
    "first_day_lab",
    "first_day_sofa",
    "first_day_rrt",
]
"""Tables to load (``patients``, ``admissions`` and ``icustays`` always are)."""

LABELS = ("died_icu_outtime_dod", "died_in_icu", "died_in_icu_deathtime")
STEP_FIRST_STAY = "First ICU stay per patient"
STEP_HADM = "Admitted to the hospital (hadm_id not null)"
STEP_OUTTIME = "ICU discharge time recorded"
STEP_CLINICAL = "Clinical data in the first 24 h (vitals or labs)"

# Columns of the mimic-code first-day tables (mimiciv_derived)
VITAL_COLUMNS = [
    f"{name}_{stat}"
    for name in (
        "heart_rate",
        "sbp",
        "dbp",
        "mbp",
        "resp_rate",
        "temperature",
        "spo2",
        "glucose",
    )
    for stat in ("min", "max", "mean")
]
LAB_COLUMNS = [
    f"{name}_{stat}"
    for name in (
        "hematocrit", "hemoglobin", "platelets", "wbc", "albumin", "globulin",
        "total_protein", "aniongap", "bicarbonate", "bun", "calcium", "chloride",
        "creatinine", "glucose", "sodium", "potassium", "abs_basophils",
        "abs_eosinophils", "abs_lymphocytes", "abs_monocytes", "abs_neutrophils",
        "atyps", "bands", "imm_granulocytes", "metas", "nrbc", "d_dimer",
        "fibrinogen", "thrombin", "inr", "pt", "ptt", "alt", "alp", "ast",
        "amylase", "bilirubin_total", "bilirubin_direct", "bilirubin_indirect",
        "ck_cpk", "ck_mb", "ggt", "ld_ldh",
    )
    for stat in ("min", "max")
]  # fmt: skip
SOFA_COLUMNS = [
    "sofa", "respiration", "coagulation", "liver", "cardiovascular", "cns", "renal",
]  # fmt: skip
RRT_COLUMNS = ["dialysis_present", "dialysis_active"]
DERIVED_TABLES = {
    "first_day_vitalsign": ("vit_", VITAL_COLUMNS),
    "first_day_lab": ("fdlab_", LAB_COLUMNS),
    "first_day_sofa": ("sofa_", SOFA_COLUMNS),
}

CATEGORICAL = (
    "gender",
    "curr_service",
    "first_careunit",
    "adm_admission_type",
    "adm_admission_location",
    "adm_insurance",
    "adm_language",
    "adm_marital_status",
    "adm_race",
)
MISSING = "MISSING"
CODE_GROUPS = {
    "px": "px_", "med": "med_", "rx": "rx_", "dx": "dx_", "dxprior": "dxprior_",
}  # fmt: skip
CODE_SOURCES = {
    "px": "procedures_icd",
    "med": "emar",
    "rx": "prescriptions",
    "dx": "diagnoses_icd",
    "dxprior": "diagnoses_prior",
}
VALUE_SOURCES = {
    "age": "patients",
    "adm": "admissions",
    "sofa": "first_day_sofa",
    "vit": "first_day_vitalsign",
    "fdlab": "first_day_lab",
    "rrt": "first_day_rrt",
}
BINARY_VALUES = ("adm_any_ed", "rrt_rrt")

# eMAR events that count as "given" 
EMAR_GIVEN = "(administered|applied|confirmed|started|rate change)"
EMAR_NOT_GIVEN = r"\b(not|stopped)\b"
_EMAR_GIVEN = re.compile(EMAR_GIVEN)
_EMAR_NOT_GIVEN = re.compile(EMAR_NOT_GIVEN)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _to_datetime(value: Any) -> datetime | None:
    """Parses a MIMIC time or date (string, date or datetime); None if empty."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        # MIMIC times have no time zone
        return datetime.fromisoformat(value.isoformat())
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    if text == "" or text.lower() in ("none", "nan", "nat", "null", "<na>"):
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError as error:
        raise ValueError(f"Unrecognized date or time: {value!r}") from error


def _text(value: Any) -> str | None:
    """Text value, or None when empty."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    empty = ("none", "nan", "<na>", "null")
    return None if text == "" or text.lower() in empty else text


def _id(value: Any) -> str | None:
    """An ID as text, the same whether it was read as int, float or text."""
    text = _text(value)
    return text[:-2] if text is not None and text.endswith(".0") else text


def _float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def _attr(event: Any, name: str) -> Any:
    return None if event is None else getattr(event, name, None)


def safe_name(text: Any) -> str:
    """Column-name form of a code, as in build_features.py.

    Examples:
        >>> safe_name("0.9% Sodium Chloride")
        '0_9_sodium_chloride'
    """
    return re.sub(r"[^0-9a-zA-Z]+", "_", str(text)).strip("_").lower()


def estimate_year(anchor_year: int, anchor_year_group: str, intime: datetime) -> int:
    """Estimated real year of an ICU stay (sql/01_icustays_cohort.sql).

    First year of the anchor year group + 1 (its middle year) + the years
    between the anchor year and the (shifted) ICU admission.

    Examples:
        >>> estimate_year(2180, "2014 - 2016", datetime(2180, 5, 1))
        2015
        >>> estimate_year(2180, "2014 - 2016", datetime(2182, 1, 9))
        2017
    """
    first_year = int(str(anchor_year_group).strip()[:4])
    return first_year + 1 + (intime.year - int(anchor_year))


def _derived_row(patient: Any, table: str, stay_id: str | None) -> dict[str, Any]:
    """The stay's row of a mimiciv_derived table, as a dict ({} if none)."""
    df = patient.get_events(event_type=table, return_df=True)
    if df.height == 0 or f"{table}/stay_id" not in df.columns:
        return {}
    for row in df.iter_rows(named=True):
        if _id(row[f"{table}/stay_id"]) == stay_id:
            prefix = f"{table}/"
            return {k[len(prefix) :]: v for k, v in row.items() if k.startswith(prefix)}
    return {}


def load_medication_class_map(
    path: str,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Reads the drug-class lookup table (extracts/gsn_class_map.csv format).

    Tab-separated, with columns gsn, medication and class_allpass (a list of
    classes). Returns the classes by GSN code and by lower-case medication
    name, as build_features.py did.

    Examples:
        >>> by_gsn, by_name = load_medication_class_map(  # doctest: +SKIP
        ...     "extracts/gsn_class_map.csv")
        >>> by_gsn["002689"]  # doctest: +SKIP
        ['TN420', 'GA105']
    """
    table = pd.read_csv(path, sep="\t", na_values="['']", dtype={"gsn": str})
    table["class_allpass"] = table["class_allpass"].apply(
        lambda x: ast.literal_eval(x) if pd.notna(x) else x
    )
    table = table.dropna(subset=["class_allpass"])
    by_gsn = (
        table.dropna(subset=["gsn"])
        .assign(gsn=lambda d: d["gsn"].str.strip())
        .drop_duplicates(subset=["gsn"], keep="last")
        .set_index("gsn")["class_allpass"]
        .to_dict()
    )
    by_med = (
        table.dropna(subset=["medication"])
        .assign(
            medication=lambda d: d["medication"].astype(str).str.strip().str.lower()
        )
        .drop_duplicates(subset=["medication"], keep="last")
        .set_index("medication")["class_allpass"]
        .to_dict()
    )
    return (
        {k: list(v) for k, v in by_gsn.items()},
        {k: list(v) for k, v in by_med.items()},
    )


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------
class InICUMortalityEMDOTMIMIC4(BaseTask):
    """In-ICU mortality 24 h after ICU admission, with an estimated year.

    Works with :class:`~pyhealth.datasets.MIMIC4EHRDataset` loaded with
    ``tables=EMDOT_TABLES`` and ``config_path=EMDOT_CONFIG``. See the module
    docstring for the data, cohort, labels and features.

    Without ``feature_spec`` (step 1), the task returns one sample for every
    patient with an ICU stay. Its ``record`` holds the cohort step the patient
    was excluded at (``excluded_at``, None if kept), the labels and all the
    features. Pass the records to :func:`cohort_summary` and
    :func:`select_features`. With the ``feature_spec`` that
    :func:`select_features` returns (step 2), the task returns one sample per
    patient in the cohort, with two model inputs:

    * ``numeric``: continuous features. The processor z-scores them with the
      training mean and standard deviation and sets missing values to the
      training mean.
    * ``binary``: yes/no features (code columns, flags, and one column per
      category of each text feature, named ``"column=category"``).

    Every sample also carries ``stay_id``, ``hadm_id``, ``est_year``,
    ``anchor_year_group`` and the three labels.

    Args:
        feature_spec: Output of :func:`select_features`; None for step 1.
        min_anchor_start: Keep anchor year groups starting in this year or
            later; None keeps all. Default 2014.
        max_year: Last estimated year kept; None keeps all. Default 2022
            (2020 = the paper's range).
        min_icu_hours: Minimum ICU stay length; 0 keeps all. Default 24.
        require_clinical_data: Require a first-day vital sign or lab value.
            Default True.
        label: Target, one of :data:`LABELS`. Default ``"died_icu_outtime_dod"``.
        prediction_hours: Prediction time after ICU admission; medications and
            procedures are counted up to it. Default 24.
        use_diagnoses: Also collect current-admission diagnoses (``dx_``).
            They leak the outcome: drift analysis only. Default False.
        medication_class_map: Path of the drug-class lookup table
            (``extracts/gsn_class_map.csv`` format); None skips ``medclass_``.

    Examples:
        >>> InICUMortalityEMDOTMIMIC4().input_schema
        {'record': 'raw'}
    """

    task_name: str = "InICUMortalityEMDOTMIMIC4"
    input_schema: dict[str, Any] = {"record": "raw"}  # noqa: RUF012
    output_schema: dict[str, str] = {"mortality": "binary"}  # noqa: RUF012

    def __init__(
        self,
        feature_spec: dict[str, Any] | None = None,
        min_anchor_start: int | None = 2014,
        max_year: int | None = 2022,
        min_icu_hours: float = 24,
        require_clinical_data: bool = True,
        label: str = "died_icu_outtime_dod",
        prediction_hours: float = 24,
        use_diagnoses: bool = False,
        medication_class_map: str | None = None,
    ) -> None:
        super().__init__()
        if label not in LABELS:
            raise ValueError(f"label must be one of {LABELS}, got {label!r}")
        self.feature_spec = feature_spec
        self.min_anchor_start = min_anchor_start
        self.max_year = max_year
        self.min_icu_hours = min_icu_hours
        self.require_clinical_data = require_clinical_data
        self.label = label
        self.prediction_hours = prediction_hours
        self.use_diagnoses = use_diagnoses
        self.medication_class_map = medication_class_map
        self._classes_by_gsn: dict[str, list[str]] = {}
        self._classes_by_med: dict[str, list[str]] = {}
        if medication_class_map is not None:
            self._classes_by_gsn, self._classes_by_med = load_medication_class_map(
                medication_class_map
            )
        if feature_spec is not None:
            self.input_schema = {
                "numeric": (
                    "tensor",
                    {"normalization": "zscore", "impute_strategy": "zero"},
                ),
                "binary": "tensor",
            }

    def pre_filter(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Keeps only the eMAR events that count as given (the largest table)."""
        if "emar/event_txt" not in set(df.collect_schema().names()):
            return df
        event = (
            pl.col("emar/event_txt").cast(pl.String).fill_null("").str.to_lowercase()
        )
        given = event.str.contains(EMAR_GIVEN) & ~event.str.contains(EMAR_NOT_GIVEN)
        return df.filter((pl.col("event_type") != "emar") | given)

    def build_record(self, patient: Any) -> dict[str, Any] | None:
        """Cohort step, labels and every feature of the patient's first ICU stay."""
        stays = [
            s
            for s in patient.get_events(event_type="icustays")
            if s.timestamp is not None
        ]
        people = patient.get_events(event_type="patients")
        if not stays or not people:
            return None
        person = people[0]
        stay = min(stays, key=lambda s: s.timestamp)  # 1. first ICU stay
        intime = stay.timestamp
        outtime = _to_datetime(_attr(stay, "outtime"))
        hadm_id, stay_id = _id(_attr(stay, "hadm_id")), _id(_attr(stay, "stay_id"))
        anchor_year = int(_float(person.anchor_year))
        group = str(person.anchor_year_group)
        est_year = estimate_year(anchor_year, group, intime)
        record: dict[str, Any] = {
            "patient_id": str(patient.patient_id),
            "stay_id": stay_id,
            "hadm_id": hadm_id,
            "est_year": est_year,
            "anchor_year_group": group,
            "excluded_at": None,
        }

        # Labels (three ways to die in the ICU, all kept)
        admission = next(
            (
                a
                for a in patient.get_events(event_type="admissions")
                if _id(_attr(a, "hadm_id")) == hadm_id
            ),
            None,
        )
        dod = _to_datetime(_attr(person, "dod"))
        deathtime = _to_datetime(_attr(admission, "deathtime"))
        has_out = outtime is not None
        record["died_icu_outtime_dod"] = int(
            dod is not None and has_out and dod.date() == outtime.date()
        )
        record["died_in_icu"] = int(
            dod is not None
            and has_out
            and intime.date() <= dod.date() <= outtime.date()
        )
        record["died_in_icu_deathtime"] = int(
            deathtime is not None and has_out and intime <= deathtime <= outtime
        )

        # 2-6. Cohort steps 
        steps = [
            (hadm_id is not None and admission is not None, STEP_HADM),
            (has_out, STEP_OUTTIME),
        ]
        if self.min_anchor_start is not None:
            steps.append(
                (
                    int(group.strip()[:4]) >= self.min_anchor_start,
                    f"Anchor year group starting {self.min_anchor_start} or later",
                )
            )
        if self.max_year is not None:
            steps.append(
                (
                    est_year <= self.max_year,
                    f"Estimated year {self.max_year} or earlier",
                )
            )
        if self.min_icu_hours > 0:
            hours = (outtime - intime).total_seconds() / 3600 if has_out else -1.0
            steps.append(
                (
                    hours >= self.min_icu_hours,
                    f"First ICU stay of {self.min_icu_hours:g} h or more",
                )
            )
        for passed, step in steps:
            if not passed:
                record["excluded_at"] = step
                return record

        # First-day tables 
        values: dict[str, float | None] = {}
        derived = {}
        for table, (prefix, columns) in DERIVED_TABLES.items():
            row = _derived_row(patient, table, stay_id)
            derived[table] = row
            values.update({f"{prefix}{c}": _float(row.get(c)) for c in columns})
        # 7. Clinical data in the first 24 h (first-day vitals or labs)
        clinical = [
            _float(derived[t].get(c))
            for t, columns in (
                ("first_day_vitalsign", VITAL_COLUMNS),
                ("first_day_lab", LAB_COLUMNS),
            )
            for c in columns
        ]
        if self.require_clinical_data and all(v is None for v in clinical):
            record["excluded_at"] = STEP_CLINICAL
            return record
        rrt = _derived_row(patient, "first_day_rrt", stay_id)
        # sql/11_first_day_rrt.sql: 1 if present or active, otherwise 0
        values["rrt_rrt"] = float(
            _float(rrt.get("dialysis_present")) == 1
            or _float(rrt.get("dialysis_active")) == 1
        )

        # Other numeric features
        values["age_at_icu"] = _float(person.anchor_age) + (intime.year - anchor_year)
        edreg = _to_datetime(_attr(admission, "edregtime"))
        edout = _to_datetime(_attr(admission, "edouttime"))
        # sql/09_admissions.sql: DATETIME_DIFF(edouttime, edregtime, HOUR) > 0,
        # which counts the hour boundaries crossed
        values["adm_any_ed"] = float(
            edreg is not None
            and edout is not None
            and edout.replace(minute=0, second=0, microsecond=0)
            > edreg.replace(minute=0, second=0, microsecond=0)
        )
        record["values"] = values

        # Text features: service at ICU admission and others
        service = None
        services = patient.get_events(event_type="services", end=intime)
        for s in sorted(services, key=lambda e: e.timestamp):
            if _id(_attr(s, "hadm_id")) == hadm_id:
                service = _text(_attr(s, "curr_service"))
        admission_columns = (
            "admission_type",
            "admission_location",
            "insurance",
            "language",
            "marital_status",
            "race",
        )
        record["categories"] = {
            "gender": _text(_attr(person, "gender")),
            "curr_service": service,
            "first_careunit": _text(_attr(stay, "first_careunit")),
            **{f"adm_{c}": _text(_attr(admission, c)) for c in admission_columns},
        }

        # Codes (build_features.py "codes" tables)
        end = intime + timedelta(hours=self.prediction_hours)
        emar_rows = self._given_medications(patient, hadm_id, intime, end)
        record["codes"] = self._codes(patient, hadm_id, intime, end, emar_rows)
        record["has_emar"] = int(bool(emar_rows))
        record["medclass"] = self._medication_classes(emar_rows)
        return record

    def _codes(
        self,
        patient: Any,
        hadm_id: str | None,
        intime: datetime,
        end: datetime,
        emar_rows: list[Any],
    ) -> dict[str, list[str]]:
        """Code lists of the stay, by group."""

        def icd(event: Any) -> str:
            version = _id(_attr(event, "icd_version"))
            return f"{version}_{_text(_attr(event, 'icd_code'))}"

        codes: dict[str, list[str]] = {}
        # sql/06: procedures of the admission done by the prediction day
        codes["px"] = sorted(
            {
                icd(p)
                for p in patient.get_events(event_type="procedures_icd")
                if _id(_attr(p, "hadm_id")) == hadm_id
                and _text(_attr(p, "icd_code"))
                and p.timestamp is not None
                and p.timestamp.date() <= end.date()
            }
        )
        medications = (_text(_attr(e, "medication")) for e in emar_rows)
        codes["med"] = sorted({m for m in medications if m})
        drugs = (_text(_attr(e, "drug")) for e in emar_rows)
        codes["rx"] = sorted({d for d in drugs if d})
        diagnoses = patient.get_events(event_type="diagnoses_icd")
        # sql/04b: diagnoses of stays discharged before this ICU stay
        codes["dxprior"] = sorted(
            {
                icd(d)
                for d in diagnoses
                if d.timestamp is not None
                and d.timestamp < intime
                and _text(_attr(d, "icd_code"))
            }
        )
        if self.use_diagnoses:  # current admission (leaks the outcome)
            codes["dx"] = sorted(
                {
                    icd(d)
                    for d in diagnoses
                    if _id(_attr(d, "hadm_id")) == hadm_id
                    and _text(_attr(d, "icd_code"))
                }
            )
        return codes

    @staticmethod
    def _given_medications(
        patient: Any, hadm_id: str | None, intime: datetime, end: datetime
    ) -> list[Any]:
        """eMAR events of the admission given in the first 24 h (sql/05_emar.sql)."""
        rows = []
        for e in patient.get_events(event_type="emar", start=intime, end=end):
            event = (_text(_attr(e, "event_txt")) or "").lower()
            if (
                _id(_attr(e, "hadm_id")) == hadm_id
                and _EMAR_GIVEN.search(event)
                and not _EMAR_NOT_GIVEN.search(event)
            ):
                rows.append(e)
        return rows

    def _medication_classes(self, emar_rows: list[Any]) -> list[str]:
        """Drug classes of the given medications: by GSN code, then by name."""
        classes: set[str] = set()
        if not (self._classes_by_gsn or self._classes_by_med):
            return []
        for e in emar_rows:
            gsn = (_text(_attr(e, "gsn")) or "").split()
            found = self._classes_by_gsn.get(gsn[0]) if gsn else None
            if found is None:
                name = (_text(_attr(e, "medication")) or "").lower()
                found = self._classes_by_med.get(name)
            classes.update(str(c) for c in (found or []) if c is not None)
        return sorted(classes)

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        record = self.build_record(patient)
        if record is None:
            return []
        if self.feature_spec is None:
            return [
                {
                    "patient_id": patient.patient_id,
                    "record": record,
                    "mortality": record[self.label],
                }
            ]
        if record["excluded_at"] is not None:
            return []
        numeric, binary = vectorize(record, self.feature_spec)
        ids = ("stay_id", "hadm_id", "est_year", "anchor_year_group")
        return [
            {
                "patient_id": patient.patient_id,
                **{k: record[k] for k in ids},
                **{k: record[k] for k in LABELS},
                "numeric": numeric,
                "binary": binary,
                "mortality": record[self.label],
            }
        ]


# ---------------------------------------------------------------------------
# Cohort-level steps (build_cohort.py summaries, build_features.py selection)
# ---------------------------------------------------------------------------
def _flat_features(record: dict[str, Any]) -> dict[str, Any]:
    """Numeric values plus has_record / n_distinct of each code group."""
    row: dict[str, Any] = dict(record["values"])
    for group, codes in record["codes"].items():
        prefix = CODE_GROUPS[group]
        row[f"{prefix}has_record"] = int(bool(codes))
        row[f"{prefix}n_distinct"] = len(codes)
    return row


def cohort_summary(
    records: Iterable[dict[str, Any] | None], label: str = "died_icu_outtime_dod"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Patients left after each cohort step, and stays and deaths by year.

    The two tables build_cohort.py saved (results/cohort_selection.csv and
    results/cohort_by_year.csv). Counts only: safe to share.

    Examples:
        >>> selection, by_year = cohort_summary(records)  # doctest: +SKIP
        >>> selection.to_csv("cohort_selection.csv", index=False)  # doctest: +SKIP
    """
    records = [r for r in records if r is not None]
    excluded = Counter(r["excluded_at"] for r in records if r["excluded_at"])

    def position(step: str) -> tuple[int, str]:
        order = ("Admitted", "ICU discharge", "Anchor", "Estimated", "First ICU stay")
        for i, start in enumerate(order):
            if step.startswith(start):
                return i, step
        return len(order), step

    n = len(records)
    steps = [(STEP_FIRST_STAY, n)]
    for step in sorted(excluded, key=position):
        n -= excluded[step]
        steps.append((step, n))
    selection = pd.DataFrame(steps, columns=["step", "patients"])
    kept = pd.DataFrame(
        [
            {
                "est_year": r["est_year"],
                "has_emar": r.get("has_emar", 0),
                **{k: r[k] for k in LABELS},
            }
            for r in records
            if r["excluded_at"] is None
        ]
    )
    if kept.empty:
        return selection, pd.DataFrame()
    by_year = kept.groupby("est_year").agg(
        stays=("est_year", "size"), **{f"deaths_{c}": (c, "sum") for c in LABELS}
    )
    by_year["death_rate"] = (by_year[f"deaths_{label}"] / by_year["stays"]).round(3)
    by_year["emar_coverage"] = kept.groupby("est_year")["has_emar"].mean().round(2)
    return selection, by_year


def select_features(
    records: Iterable[dict[str, Any] | None],
    top_n: int = 50,
    max_missing_rate: float | None = 0.10,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Chooses the feature columns on the whole cohort, as build_features.py did.

    * Code groups: ``has_record``, ``n_distinct`` and the ``top_n`` codes
      found in the most stays (ties broken by code).
    * Drug classes: every class found in the cohort.
    * Text features: every category found, plus ``"MISSING"``.
    * Numeric features missing for more than ``max_missing_rate`` of the
      cohort are removed; text features are exempt.

    Args:
        records: The ``record`` of each step-1 sample (excluded ones are skipped).
        top_n: Codes kept per group. Default 50.
        max_missing_rate: Missing-rate limit; None keeps all. Default 0.10.

    Returns:
        The ``feature_spec`` for the task, and a feature dictionary (column,
        source, role, missing_rate), like results/feature_dictionary.csv.

    Examples:
        >>> spec, dictionary = select_features(records)  # doctest: +SKIP
        >>> task = InICUMortalityEMDOTMIMIC4(feature_spec=spec)  # doctest: +SKIP
    """
    kept = [r for r in records if r is not None and r["excluded_at"] is None]
    if not kept:
        raise ValueError("no patient in the cohort")
    missing = pd.DataFrame([_flat_features(r) for r in kept]).isna().mean()

    code_counts: dict[str, Counter] = defaultdict(Counter)
    class_counts: Counter = Counter()
    categories: dict[str, set] = {c: set() for c in CATEGORICAL}
    for r in kept:
        for group, codes in r["codes"].items():
            code_counts[group].update(set(codes))
        class_counts.update(set(r.get("medclass", [])))
        for c in CATEGORICAL:
            categories[c].add(r["categories"].get(c) or MISSING)
    top_codes = {
        g: [code for code, _ in sorted(c.items(), key=lambda x: (-x[1], x[0]))[:top_n]]
        for g, c in code_counts.items()
    }

    dictionary: list[dict[str, Any]] = []
    numeric: list[str] = []
    binary: list[str] = []

    def add(column: str, source: str, rate: float, kind: str) -> None:
        dropped = max_missing_rate is not None and rate > max_missing_rate
        role = f"dropped (missing > {max_missing_rate:.0%})" if dropped else "feature"
        dictionary.append(
            {
                "column": column,
                "source": source,
                "missing_rate": round(float(rate), 4),
                "role": role,
            }
        )
        if not dropped:
            (binary if kind == "binary" else numeric).append(column)

    for column in kept[0]["values"]:
        source = VALUE_SOURCES.get(column.split("_")[0], "patients")
        kind = "binary" if column in BINARY_VALUES else "numeric"
        add(column, source, float(missing.get(column, 1.0)), kind)
    for group, prefix in CODE_GROUPS.items():
        if group not in top_codes and group not in kept[0]["codes"]:
            continue
        source = CODE_SOURCES[group]
        add(f"{prefix}has_record", source, 0.0, "binary")
        add(f"{prefix}n_distinct", source, 0.0, "numeric")
        for code in top_codes.get(group, []):
            add(f"{prefix}{safe_name(code)}", source, 0.0, "binary")
    classes = sorted(class_counts)
    for cls in classes:
        add(f"medclass_{safe_name(cls)}", "medication_classes", 0.0, "binary")
    category_lists = {c: sorted(v) for c, v in categories.items()}
    for c, values in category_lists.items():
        rate = sum(1 for r in kept if not r["categories"].get(c)) / len(kept)
        dictionary.append(
            {
                "column": c,
                "source": "categories",
                "missing_rate": round(rate, 4),
                "role": "feature",
            }
        )
        binary.extend(f"{c}={v}" for v in values)

    spec = {
        "numeric": numeric,
        "binary": binary,
        "top_codes": top_codes,
        "medclass": classes,
        "categories": category_lists,
    }
    return spec, pd.DataFrame(dictionary)


def vectorize(
    record: dict[str, Any], spec: dict[str, Any]
) -> tuple[list[float], list[float]]:
    """The ``numeric`` and ``binary`` inputs of one record, in ``spec`` order.

    Examples:
        >>> numeric, binary = vectorize(record, spec)  # doctest: +SKIP
        >>> len(numeric) == len(spec["numeric"])  # doctest: +SKIP
        True
    """
    flat = _flat_features(record)
    for group, codes in record["codes"].items():
        prefix, present = CODE_GROUPS[group], set(codes)
        for code in spec["top_codes"].get(group, []):
            flat[f"{prefix}{safe_name(code)}"] = int(code in present)
    present_classes = set(record.get("medclass", []))
    for cls in spec["medclass"]:
        flat[f"medclass_{safe_name(cls)}"] = int(cls in present_classes)
    for c, values in spec["categories"].items():
        value = record["categories"].get(c) or MISSING
        for v in values:
            flat[f"{c}={v}"] = int(value == v)

    def number(x: Any) -> float:
        return float("nan") if x is None else float(x)

    return (
        [number(flat.get(c)) for c in spec["numeric"]],
        [number(flat.get(c, 0)) for c in spec["binary"]],
    )
