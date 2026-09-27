# Evaluator data sample

A real, representative slice of every dataset this project uses — **~426MB**,
against a full `data/raw/` of ~26GB — committed directly so an evaluator can
see actual physical-sensor data without downloading anything. Every file here
is exactly what the source dataset shipped (WAV/CSV/.mat/.npz), never
synthetic or altered, produced by
[`scripts/utils/build_evaluator_data_sample.py`](../scripts/utils/build_evaluator_data_sample.py),
which stratifies the selection across every class/condition present in each
dataset rather than just taking the first N files alphabetically.

| Folder | Files | Size | What it is |
|---|---|---|---|
| `cwru/` | 12 | 64MB | Bearing vibration, 3 files each from Normal/Inner-race/Ball/Outer-race classes |
| `ims/` | 22 | 12MB | Run-to-failure bearing vibration, early/mid/late snapshots |
| `femto/` | 267 | 22MB | Run-to-failure bearing vibration, full lifecycle of both bearings on disk |
| `paderborn/` | 6 | 52MB | Vibration+current+force+torque, one file per bearing condition (healthy + damaged) |
| `subf/` | 24 | 43MB | Bearing-fault audio, 8 files each from Normal/Inner/Outer classes |
| `mafaulda/` | 3 | 52MB | Paired vibration+audio, normal + 2 fault classes |
| `ai_mechanic/` | 39 | 38MB | Real BMW engine audio — the entire held-out dataset |
| `car_diagnostics/` | 360 | 50MB | Real car audio, ~40 files from every state/fault subfolder |
| `engine_journal_bearings/` | 134 | 63MB | Real automobile engine vibration — the entire held-out dataset |
| `mathworks_data/` | 20 | 43MB | Controlled bearing vibration — the entire held-out dataset |
| `engine_acoustic_emissions/` | 1 | 4MB | The dataset this project inspected and **rejected** (see main README, Section 1.4) — kept as evidence |
| `manifest/` | 11 Parquet + `SCHEMA.md` | <1MB | Full-coverage structural manifest — **every** file this project uses across every dataset (13,900+ rows), not just the sample above: which dataset, which pool (training/held-out/rejected), which label, and why |

`manifest/` is the one directory here that covers 100% of the data, not a
sample of it — it's metadata (file path, label, pool) rather than the signals
themselves, which is why it can be exhaustive at only a few hundred KB.
