"""
Exports one Parquet manifest per dataset (training pool + held-out pool),
plus a companion schema description — the "make it usable... open format,
Parquet preferred" requirement this project had not actually done anywhere
(everything else is loose .npz/CSV caches). This does NOT re-export the raw
signals themselves (that's still whatever format each dataset shipped in,
per download_manifest.py) — it exports the row-level structure that ties a
file to its dataset, modality, label, and role, which is what a stranger
needs to understand and audit what was actually used, without re-running
any code.

Run after the datasets in download_manifest.py are on disk:
    python export_dataset_manifest.py
Writes to sample_data/manifest/*.parquet plus sample_data/manifest/SCHEMA.md
— unlike data/raw/ and data/processed/, sample_data/ is NOT gitignored, since
this manifest (row-level structure, not raw signals) is small enough and
important enough to commit directly (see README Section 2.6).
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from pathlib import Path

import pandas as pd

from data.readers import (
    ai_mechanic_label_fn, ai_mechanic_reader,
    car_diagnostics_label_fn,
    cwru_label_fn_4class,
    engine_journal_bearings_label_fn,
    femto_label_fn_binary,
    ims_label_fn_binary,
    mafaulda_label_fn_multiclass,
    paderborn_label_fn_binary,
    subf_label_fn_3class,
)

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
OUT_DIR = Path(__file__).resolve().parents[2] / "sample_data" / "manifest"

SCHEMA_MD = """\
# Dataset manifest schema

Each row is one raw source file actually used somewhere in this project's
pipeline. Columns:

| Column | Type | Meaning |
|---|---|---|
| `dataset` | string | Which dataset (see `src/data/download_manifest.py` for the full registry) this file belongs to |
| `pool` | string | `"training"` (industrial pretraining pool), `"held_out"` (real-vehicle evaluation pool, never trained on), or `"rejected"` (acquired, inspected, and explicitly excluded — see `label_scheme` for why) |
| `modality` | string | `"vibration"`, `"audio"`, or `"vibration+audio"` (MaFaulDa, the one paired dataset) |
| `relative_path` | string | File path relative to `data/raw/<dataset>/` |
| `label` | int | This project's derived label for the file (dataset-specific meaning — see `label_scheme` column). `-1` means excluded (ambiguous, per that dataset's reader) |
| `label_scheme` | string | Which label function produced `label` (e.g. `cwru_label_fn_4class`), so the integer's meaning is traceable back to code |
| `is_heuristic_label` | bool | True for IMS/FEMTO — those labels are an early/late run-to-failure PROXY, not a verified ground-truth fault boundary (see `check_run_to_failure_label_sensitivity.py`) |

One Parquet file per dataset, all with this same schema, under
`data/processed/manifest/<dataset>.parquet`. A combined
`data/processed/manifest/all_datasets.parquet` concatenates all of them
with the same columns.

This manifest records STRUCTURE (which file, which label, which pool), not
the raw signal content itself — the raw files stay in their original format
(WAV/CSV/.mat/.npz) exactly as downloaded, per `download_manifest.py`.
"""


def _rows_for(dataset: str, pool: str, modality: str, files: list[Path], root: Path,
              label_fn, label_scheme: str, is_heuristic: bool = False) -> pd.DataFrame:
    records = []
    for f in files:
        try:
            rel = str(f.relative_to(root)).replace("\\", "/")
        except ValueError:
            rel = f.name
        try:
            label = label_fn(f)
        except Exception as e:
            label = -2  # distinguishable from the reader's own "-1 excluded" convention
            print(f"  [warn] label_fn raised on {rel}: {e}")
        records.append({
            "dataset": dataset, "pool": pool, "modality": modality,
            "relative_path": rel, "label": label, "label_scheme": label_scheme,
            "is_heuristic_label": is_heuristic,
        })
    return pd.DataFrame.from_records(records)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = []

    ai_root = RAW_DIR / "ai_mechanic" / "ai-mechanic-export"
    if ai_root.exists():
        files = sorted(ai_root.rglob("*.wav"))
        df = _rows_for("ai_mechanic", "held_out", "audio", files, ai_root,
                        ai_mechanic_label_fn, "ai_mechanic_label_fn")
        frames.append(df)

    cd_root = RAW_DIR / "car_diagnostics" / "car diagnostics dataset"
    if cd_root.exists():
        files = sorted(cd_root.rglob("*.wav"))
        df = _rows_for("car_diagnostics", "held_out", "audio", files, cd_root,
                        car_diagnostics_label_fn, "car_diagnostics_label_fn")
        frames.append(df)

    cwru_root = RAW_DIR / "cwru" / "Data"
    if cwru_root.exists():
        files = sorted(cwru_root.rglob("*.npz"))
        df = _rows_for("cwru", "training", "vibration", files, cwru_root,
                        cwru_label_fn_4class, "cwru_label_fn_4class")
        frames.append(df)

    ims_root = RAW_DIR / "ims"
    if ims_root.exists():
        files = [f for f in sorted(ims_root.rglob("*")) if f.is_file() and f.name not in ("IMS.7z", "2nd_test.rar")]
        df = _rows_for("ims", "training", "vibration", files, ims_root,
                        ims_label_fn_binary, "ims_label_fn_binary", is_heuristic=True)
        frames.append(df)

    femto_root = RAW_DIR / "femto" / "Learning_set"
    if femto_root.exists():
        files = sorted(femto_root.rglob("acc_*.csv"))
        df = _rows_for("femto", "training", "vibration", files, femto_root,
                        femto_label_fn_binary, "femto_label_fn_binary", is_heuristic=True)
        frames.append(df)

    pb_root = RAW_DIR / "paderborn"
    if pb_root.exists():
        files = sorted(pb_root.rglob("*.mat"))
        df = _rows_for("paderborn", "training", "vibration", files, pb_root,
                        paderborn_label_fn_binary, "paderborn_label_fn_binary")
        frames.append(df)

    maf_root = RAW_DIR / "mafaulda"
    if maf_root.exists():
        files = sorted(maf_root.rglob("*.csv"))
        df = _rows_for("mafaulda", "training", "vibration+audio", files, maf_root,
                        mafaulda_label_fn_multiclass, "mafaulda_label_fn_multiclass")
        frames.append(df)

    subf_root = RAW_DIR / "subf" / "Dataset"
    if subf_root.exists():
        files = sorted(subf_root.rglob("*.csv"))
        df = _rows_for("subf", "training", "audio", files, subf_root,
                        subf_label_fn_3class, "subf_label_fn_3class")
        frames.append(df)

    ejb_root = RAW_DIR / "engine_journal_bearings"
    if ejb_root.exists():
        files = sorted(ejb_root.rglob("*.csv"))
        df = _rows_for("engine_journal_bearings", "held_out", "vibration", files, ejb_root,
                        engine_journal_bearings_label_fn, "engine_journal_bearings_label_fn")
        frames.append(df)

    mw_root = RAW_DIR / "mathworks_data"
    if mw_root.exists():
        files = sorted(mw_root.rglob("*.mat"))
        mathworks_label_fn = lambda f: 0 if f.name.startswith("baseline_") else 1
        df = _rows_for("mathworks_bearing", "held_out", "vibration", files, mw_root,
                        mathworks_label_fn, "mathworks_label_fn (baseline_*=0, else=1)")
        frames.append(df)

    eae_root = RAW_DIR / "engine_acoustic_emissions"
    if eae_root.exists():
        files = sorted(eae_root.rglob("*.mat"))
        df = _rows_for("engine_acoustic_emissions", "rejected", "audio", files, eae_root,
                        lambda f: -3, "rejected_mislabeled_bearing_rig_simulator")
        frames.append(df)

    for df in frames:
        name = df["dataset"].iloc[0]
        path = OUT_DIR / f"{name}.parquet"
        df.to_parquet(path, index=False)
        print(f"  wrote {len(df):6d} rows -> {path}")

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        combined.to_parquet(OUT_DIR / "all_datasets.parquet", index=False)
        print(f"  wrote {len(combined):6d} rows -> {OUT_DIR / 'all_datasets.parquet'}")

    (OUT_DIR / "SCHEMA.md").write_text(SCHEMA_MD, encoding="utf-8")
    print(f"  wrote schema description -> {OUT_DIR / 'SCHEMA.md'}")
    print("\nDone.")


if __name__ == "__main__":
    main()
