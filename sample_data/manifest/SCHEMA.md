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
