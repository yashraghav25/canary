"""
Dataset download manifest for the vehicle-health-ai project.

This module only DESCRIBES where each dataset lives and how large it is.
It does not download anything on import. Run `download.py` explicitly,
one dataset at a time, after confirming disk space / bandwidth.

Every entry mirrors a row in vehicle-health-ai-plan.md Section 3.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class DatasetEntry:
    key: str
    pool: Literal["training", "held_out"]
    modality: str
    source: str          # "kaggle", "http", "git", "s3"
    location: str        # kaggle slug, URL, or git URL
    approx_size: str      # human-readable, for permission prompts before downloading
    notes: str = ""


MANIFEST = [
    DatasetEntry(
        key="cwru",
        pool="training",
        modality="vibration",
        source="http",
        location="https://engineering.case.edu/bearingdatacenter/download-data-file",
        approx_size="~200MB (subset varies by fault/load selection)",
        notes="Official page lists individual .mat files per fault/load; "
              "GitHub mirror https://github.com/srigas/CWRU_Bearing_NumPy has a pre-packaged .npz version.",
    ),
    DatasetEntry(
        key="ims",
        pool="training",
        modality="vibration",
        source="s3",
        location="https://phm-datasets.s3.amazonaws.com/NASA/4.+Bearings.zip",
        approx_size="~1.5GB zipped",
        notes="Kaggle mirror: vinayak123tyagi/bearing-dataset, if the S3 link is unavailable.",
    ),
    DatasetEntry(
        key="femto",
        pool="training",
        modality="vibration_temperature",
        source="git",
        location="https://github.com/wkzs111/phm-ieee-2012-data-challenge-dataset",
        approx_size="~2.5GB (Learning_set + Test_set + Full_Test_Set)",
    ),
    DatasetEntry(
        key="paderborn",
        pool="training",
        modality="vibration_current",
        source="http",
        location="https://mb.uni-paderborn.de/kat/forschung/bearing-datacenter/data-sets-and-download",
        approx_size="~15GB for the full 32-bearing set; can subset to a handful of .mat files first",
        notes="CC BY-NC 4.0 — cite Lessmeier et al. Confirmed live via direct browse.",
    ),
    DatasetEntry(
        key="mafaulda",
        pool="training",
        modality="vibration_audio_tacho",
        source="http",
        location="https://www02.smt.ufrj.br/~offshore/mfs/page_01.html",
        approx_size="~13GB full; Kaggle mirror vuxuancu/mafaulda-full for partial pulls",
        notes="THE bridge dataset — prioritize this even if others are subset.",
    ),
    DatasetEntry(
        key="subf",
        pool="training",
        modality="audio",
        source="kaggle",
        location="sumairaziz/subf-v2-0-dataset-bearing-faults-sound-data",
        approx_size="~634MB zipped / ~11GB unzipped (6480 CSVs, 100000 rows each) — Kaggle's "
                    "listed size is the zip, not the extracted footprint",
    ),
    DatasetEntry(
        key="ai_mechanic",
        pool="held_out",
        modality="audio",
        source="kaggle",
        location="eoinedge/ai-mechanic-engine-condition-audio-fault-finding",
        approx_size="small, 41 files",
        notes="PRIMARY held-out evaluation set. Real BMW M54B25, real induced faults.",
    ),
    DatasetEntry(
        key="car_diagnostics",
        pool="held_out",
        modality="audio",
        source="kaggle",
        location="malakragaie/car-diagnostics-dataset",
        approx_size="unverified, check on download",
    ),
    DatasetEntry(
        key="engine_acoustic_emissions",
        pool="held_out",
        modality="audio",
        source="kaggle",
        location="julienjta/engine-acoustic-emissions",
        approx_size="~1MB, 4 files",
        notes="REJECTED after inspection: its .mat file keys (normal/inner/roller/outer) "
              "exactly mirror the CWRU bearing-fault taxonomy — a relabeled bearing-rig "
              "simulator, not real engine audio as advertised. Not used in any result; "
              "kept on disk and in sample_data/ as evidence of this check.",
    ),
    DatasetEntry(
        key="vehicle_interior_sound",
        pool="held_out",
        modality="audio",
        source="http",
        location="https://zenodo.org/records/5606504",
        approx_size="~1.2GB, 5980 clips",
        notes="Normal-class diversity only — no fault labels.",
    ),
    DatasetEntry(
        key="mathworks_bearing",
        pool="held_out",
        modality="vibration",
        source="git",
        location="https://github.com/mathworks/RollingElementBearingFaultDiagnosis-Data",
        approx_size="~43MB, 20 .mat files (baseline/inner-race/outer-race)",
        notes="Originally collected by Eric Bechhoefer; MathWorks obtained permission to "
              "redistribute for their Predictive Maintenance Toolbox example (contact "
              "Bechhoefer directly for other commercial uses, per the repo's own README). "
              "Previously used in this project's evaluation without a registry entry here — "
              "added for provenance/traceability.",
    ),
    DatasetEntry(
        key="engine_journal_bearings",
        pool="held_out",
        modality="vibration",
        source="http",
        location="https://data.mendeley.com/datasets/3fcrrdjjvk/5",
        approx_size="~63MB uncompressed, 134 files (90 faulty / 44 healthy) — the smaller of two "
                    "zips on this page; the full ~250MB primary zip was not downloaded",
        notes="THE FIRST REAL-VEHICLE VIBRATION held-out dataset in this project — tri-axial "
              "accelerometer on a real automobile engine's main journal bearing housing, healthy vs. "
              "faulty, across varying RPM/humidity/temperature (climatic chamber, not on-road). "
              "CC BY 4.0, Riaz et al., NUST Islamabad.",
    ),
]


def print_manifest() -> None:
    total_training = [e for e in MANIFEST if e.pool == "training"]
    total_held_out = [e for e in MANIFEST if e.pool == "held_out"]
    print("=== Training pool ===")
    for e in total_training:
        print(f"  {e.key:28s} {e.approx_size:40s} {e.location}")
    print("=== Held-out pool ===")
    for e in total_held_out:
        print(f"  {e.key:28s} {e.approx_size:40s} {e.location}")


if __name__ == "__main__":
    print_manifest()
