"""Tests for InICUMortalityEMDOTMIMIC4, on small synthetic patients.

No MIMIC data is used: every patient is built by hand, including the rows of
the first-day (mimiciv_derived) tables.
"""

import unittest
from datetime import datetime, timedelta

import polars as pl
import torch

from pyhealth.data import Patient
from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import LogisticRegression
from pyhealth.tasks import InICUMortalityEMDOTMIMIC4
from pyhealth.tasks.in_icu_mortality_emdot_mimic4 import (
    cohort_summary,
    estimate_year,
    select_features,
)

T0 = datetime(2180, 3, 1, 10, 30, 0)  # shifted ICU admission time
FMT = "%Y-%m-%d %H:%M:%S"
TASK = InICUMortalityEMDOTMIMIC4


def h(hours):
    return T0 + timedelta(hours=hours)


def ev(table, timestamp, **attrs):
    row = {"event_type": table, "timestamp": timestamp}
    row.update(
        {f"{table}/{k}": None if v is None else str(v) for k, v in attrs.items()}
    )
    return row


def stay(intime=T0, hours=48.0, stay_id=1, hadm_id=10):
    outtime = (intime + timedelta(hours=hours)).strftime(FMT)
    return ev(
        "icustays",
        intime,
        stay_id=stay_id,
        hadm_id=hadm_id,
        first_careunit="MICU",
        outtime=outtime,
    )


def admission(hadm_id=10, deathtime=None, edreg=None, edout=None, admittime=None):
    return ev(
        "admissions",
        admittime or h(-3),
        hadm_id=hadm_id,
        admission_type="EW EMER.",
        insurance="Medicare",
        race="WHITE",
        deathtime=deathtime and deathtime.strftime(FMT),
        edregtime=edreg and edreg.strftime(FMT),
        edouttime=edout and edout.strftime(FMT),
    )


def vitals(stay_id=1, **values):
    return ev("first_day_vitalsign", None, stay_id=stay_id, **values)


def labs(stay_id=1, **values):
    return ev("first_day_lab", None, stay_id=stay_id, **values)


def patient(*events, group="2014 - 2016", dod=None, age=60, pid="p1"):
    person = ev(
        "patients",
        None,
        gender="F",
        anchor_age=age,
        anchor_year=2180,
        anchor_year_group=group,
        dod=dod and dod.strftime("%Y-%m-%d"),
    )
    df = pl.from_dicts([person, *events], infer_schema_length=None)
    df = df.with_columns(
        pl.lit(pid).alias("patient_id"),
        pl.col("timestamp").cast(pl.Datetime("ms")),
        *(pl.col(c).cast(pl.String) for c in df.columns if "/" in c),
    )
    return Patient(patient_id=pid, data_source=df)


def basic(*extra, **kwargs):
    return patient(stay(), admission(), vitals(heart_rate_min=80), *extra, **kwargs)


class TestDerivedTables(unittest.TestCase):
    def test_values_come_from_the_stay_row(self):
        p = patient(
            stay(),
            admission(),
            vitals(heart_rate_min=70, heart_rate_max=110),
            vitals(stay_id=2, heart_rate_min=40),  # another stay: ignored
            labs(creatinine_max=1.4),
            ev("first_day_sofa", None, stay_id=1, sofa=5, renal=1),
        )
        values = TASK().build_record(p)["values"]
        self.assertEqual(values["vit_heart_rate_min"], 70)
        self.assertEqual(values["vit_heart_rate_max"], 110)
        self.assertEqual(values["fdlab_creatinine_max"], 1.4)
        self.assertEqual((values["sofa_sofa"], values["sofa_renal"]), (5, 1))
        self.assertIsNone(values["sofa_liver"])
        self.assertIsNone(values["vit_spo2_mean"])

    def test_rrt_present_or_active(self):
        def rrt(**flags):
            row = ev("first_day_rrt", None, stay_id=1, **flags)
            return TASK().build_record(basic(row))["values"]["rrt_rrt"]

        self.assertEqual(rrt(dialysis_present=1, dialysis_active=0), 1)
        self.assertEqual(rrt(dialysis_present=0, dialysis_active=1), 1)
        self.assertEqual(rrt(dialysis_present=0, dialysis_active=0), 0)
        self.assertEqual(rrt(dialysis_present=None, dialysis_active=None), 0)
        self.assertEqual(TASK().build_record(basic())["values"]["rrt_rrt"], 0)


class TestCohort(unittest.TestCase):
    def excluded(self, p, **kwargs):
        return TASK(**kwargs).build_record(p)["excluded_at"]

    def test_kept_with_default_rules(self):
        self.assertIsNone(self.excluded(basic()))

    def test_each_rule(self):
        self.assertIsNone(self.excluded(basic(group="2008 - 2010")))  # all years
        self.assertEqual(
            self.excluded(basic(group="2011 - 2013"), min_anchor_start=2014),
            "Anchor year group starting 2014 or later",
        )
        self.assertEqual(
            self.excluded(basic(group="2020 - 2022"), max_year=2020),
            "Estimated year 2020 or earlier",
        )
        short = patient(stay(hours=12), admission(), vitals(heart_rate_min=80))
        self.assertEqual(self.excluded(short), "First ICU stay of 24 h or more")
        self.assertIsNone(self.excluded(short, min_icu_hours=0))
        self.assertEqual(
            self.excluded(patient(stay(), admission(), vitals())),
            "Clinical data in the first 24 h (vitals or labs)",
        )
        self.assertIsNone(self.excluded(patient(stay(), admission(), labs(bun_max=20))))
        no_out = stay()
        no_out["icustays/outtime"] = None
        self.assertEqual(
            self.excluded(patient(no_out, admission(), vitals(heart_rate_min=80))),
            "ICU discharge time recorded",
        )

    def test_first_stay_only_and_year(self):
        later = T0 + timedelta(days=400)
        p = patient(
            stay(intime=later, stay_id=2, hadm_id=11),
            stay(),
            admission(),
            admission(hadm_id=11, admittime=later),
            vitals(heart_rate_min=80),
        )
        record = TASK().build_record(p)
        self.assertEqual((record["stay_id"], record["est_year"]), ("1", 2015))


class TestLabels(unittest.TestCase):
    def labels(self, deathtime=None, dod=None):
        p = patient(
            stay(hours=48),
            admission(deathtime=deathtime),
            vitals(heart_rate_min=80),
            dod=dod,
        )
        r = TASK().build_record(p)
        return r["died_icu_outtime_dod"], r["died_in_icu"], r["died_in_icu_deathtime"]

    def test_three_rules(self):
        self.assertEqual(self.labels(), (0, 0, 0))
        self.assertEqual(self.labels(h(48), h(48)), (1, 1, 1))  # at ICU discharge
        self.assertEqual(self.labels(h(20), h(20)), (0, 1, 1))  # during the stay
        self.assertEqual(self.labels(h(100), h(100)), (0, 0, 0))  # after the ICU

    def test_invalid_label(self):
        with self.assertRaises(ValueError):
            TASK(label="hospital")


class TestFeatures(unittest.TestCase):
    def test_codes_and_text_features(self):
        midnight = T0.replace(hour=0, minute=0)
        p = basic(
            ev("procedures_icd", midnight, hadm_id=10, icd_code="5491", icd_version=9),
            ev("procedures_icd", h(48), hadm_id=10, icd_code="0BH17EZ", icd_version=10),
            ev(
                "emar",
                h(2),
                hadm_id=10,
                medication="Heparin",
                drug="Heparin Sodium",
                event_txt="Administered",
            ),
            ev(
                "emar",
                h(3),
                hadm_id=10,
                medication="Insulin",
                drug="Insulin",
                event_txt="Not Given",
            ),
            ev(
                "emar",
                h(4),
                hadm_id=10,
                medication="Propofol",
                drug="Propofol",
                event_txt="Stopped",
            ),
            ev(
                "emar",
                h(-1),
                hadm_id=10,
                medication="Aspirin",
                drug="Aspirin",
                event_txt="Administered",
            ),
            ev("diagnoses_icd", h(-200), hadm_id=9, icd_code="4280", icd_version=9),
            ev("diagnoses_icd", h(100), hadm_id=10, icd_code="A419", icd_version=10),
            ev("services", h(-2), hadm_id=10, curr_service="MED"),
            ev("services", h(5), hadm_id=10, curr_service="SURG"),
        )
        default = TASK().build_record(p)  # medications off by default
        self.assertNotIn("med", default["codes"])
        self.assertNotIn("rx", default["codes"])
        self.assertEqual(default["has_emar"], 1)  # coverage flag kept
        r = TASK(use_medications=True).build_record(p)
        self.assertEqual(r["codes"]["px"], ["9_5491"])
        self.assertEqual(r["codes"]["med"], ["Heparin"])
        self.assertEqual(r["codes"]["rx"], ["Heparin Sodium"])
        self.assertEqual(r["codes"]["dxprior"], ["9_4280"])
        self.assertNotIn("dx", r["codes"])
        self.assertEqual(
            TASK(use_diagnoses=True).build_record(p)["codes"]["dx"], ["10_A419"]
        )
        self.assertEqual(r["categories"]["curr_service"], "MED")
        self.assertEqual(r["values"]["age_at_icu"], 60)
        self.assertEqual(r["has_emar"], 1)

    def test_any_ed_counts_hour_boundaries(self):
        def any_ed(reg, out):
            p = patient(
                stay(), admission(edreg=reg, edout=out), vitals(heart_rate_min=80)
            )
            return TASK().build_record(p)["values"]["adm_any_ed"]

        day = T0.replace(hour=0, minute=0)
        self.assertEqual(any_ed(day.replace(hour=8, minute=50), day.replace(hour=9)), 1)
        self.assertEqual(
            any_ed(day.replace(hour=8, minute=5), day.replace(minute=55, hour=8)), 0
        )
        self.assertEqual(any_ed(None, None), 0)


class TestCohortLevel(unittest.TestCase):
    def records(self):
        out = []
        for i in range(10):
            died = h(30) if i % 3 == 0 else None
            events = [
                stay(hours=12 if i == 9 else 48),
                admission(deathtime=died),
                vitals(heart_rate_min=70 + i, heart_rate_max=100 + i),
            ]
            if i < 8:
                events.append(labs(creatinine_max=1.0 + i / 10))
            if i % 2:
                events.append(
                    ev(
                        "emar",
                        h(2),
                        hadm_id=10,
                        medication=f"Drug{i % 4}",
                        drug=f"Drug{i % 4}",
                        event_txt="Administered",
                    )
                )
            task = TASK(label="died_in_icu_deathtime", use_medications=True)
            out.append(task.build_record(patient(*events, dod=died, pid=f"p{i}")))
        return out

    def test_summary_and_selection(self):
        records = self.records()
        selection, by_year = cohort_summary(records, label="died_in_icu_deathtime")
        self.assertEqual(selection["patients"].tolist(), [10, 9])
        self.assertEqual(int(by_year["stays"].sum()), 9)
        self.assertEqual(int(by_year["deaths_died_in_icu_deathtime"].sum()), 3)
        spec, dictionary = select_features(records, top_n=1, max_missing_rate=0.10)
        self.assertIn("vit_heart_rate_min", spec["numeric"])
        self.assertNotIn("fdlab_creatinine_max", spec["numeric"])  # 1 of 9 missing
        self.assertEqual(len(spec["top_codes"]["med"]), 1)
        self.assertIn("gender=F", spec["binary"])
        self.assertIn("curr_service=MISSING", spec["binary"])
        self.assertTrue(dictionary["role"].str.startswith("dropped").any())

    def test_model_inputs_train_pyhealth_lr(self):
        spec, _ = select_features(self.records(), top_n=2)
        task = TASK(feature_spec=spec, label="died_in_icu_deathtime")
        samples = []
        for i in range(9):
            events = [
                stay(),
                admission(deathtime=h(30) if i % 3 == 0 else None),
                vitals(heart_rate_min=70 + i),
            ]
            samples += task(patient(*events, pid=f"p{i}"))
        self.assertEqual(len(samples[0]["numeric"]), len(spec["numeric"]))
        dataset = create_sample_dataset(
            samples=samples,
            input_schema=task.input_schema,
            output_schema=task.output_schema,
            dataset_name="toy",
        )
        batch = next(iter(get_dataloader(dataset, batch_size=9, shuffle=False)))
        self.assertFalse(torch.isnan(batch["numeric"]).any())  # filled with the mean
        out = LogisticRegression(dataset=dataset)(**batch)
        self.assertTrue(torch.isfinite(out["loss"]))
        out["loss"].backward()


class TestPreFilter(unittest.TestCase):
    def test_keeps_only_given_emar_events(self):
        df = pl.DataFrame(
            {
                "event_type": ["emar", "emar", "emar", "icustays"],
                "emar/event_txt": [
                    "Administered",
                    "Not Given",
                    "Stopped - Unscheduled",
                    None,
                ],
            }
        ).lazy()
        out = TASK().pre_filter(df).collect()
        self.assertEqual(out["event_type"].to_list(), ["emar", "icustays"])


class TestEstimateYear(unittest.TestCase):
    def test_formula(self):
        self.assertEqual(estimate_year(2180, "2014 - 2016", datetime(2180, 1, 1)), 2015)
        self.assertEqual(estimate_year(2180, "2008 - 2010", datetime(2182, 6, 1)), 2011)


if __name__ == "__main__":
    unittest.main()
