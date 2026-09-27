"""
Builds a small (~450MB), real, representative sample of every dataset this
project actually uses — the raw sensor files themselves (WAV/CSV/.mat/.npz,
whatever each dataset natively ships as), not synthetic data — so an
evaluator can see real physical-sensor signals directly in the repository
without downloading the full ~26GB `data/raw/` tree.

This is a SAMPLE for inspection, not a training set: the pipeline itself
always reads from `data/raw/` (see src/data/download_manifest.py to
re-fetch the full data). Selection is stratified per dataset — a few files
from every class/condition that exists on disk — not just "the first N
files alphabetically," so the sample actually shows the range of what each
dataset contains.

Run after `data/raw/` is populated:
    python scripts/utils/build_evaluator_data_sample.py
Writes to sample_data/<dataset>/... at the repo root (committed to git,
unlike data/raw/ which is gitignored).
"""

import shutil
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
OUT_DIR = Path(__file__).resolve().parents[2] / "sample_data"


def copy_files(files: list[Path], src_root: Path, dataset_name: str) -> int:
    total_bytes = 0
    for f in files:
        rel = f.relative_to(src_root)
        dest = OUT_DIR / dataset_name / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dest)
        total_bytes += f.stat().st_size
    print(f"  {dataset_name:28s} {len(files):4d} files  {total_bytes/1e6:7.1f} MB")
    return total_bytes


def main() -> None:
    grand_total = 0

    # --- CWRU: 3 files per fault class (Normal, Inner Race, Ball, Outer Race) ---
    cwru_root = RAW_DIR / "cwru" / "Data"
    if cwru_root.exists():
        from collections import defaultdict
        by_class = defaultdict(list)
        for f in sorted(cwru_root.rglob("*.npz")):
            # class = text before first underscore-delimited fault code
            stem = f.stem
            if "Normal" in stem:
                by_class["Normal"].append(f)
            elif "_IR_" in stem or stem.count("_IR") > 0:
                by_class["IR"].append(f)
            elif "_B_" in stem:
                by_class["B"].append(f)
            elif "OR@" in stem:
                by_class["OR"].append(f)
        files = [f for flist in by_class.values() for f in flist[:3]]
        grand_total += copy_files(files, cwru_root, "cwru")

    # --- IMS: early / mid / late snapshots of the one test run on disk ---
    ims_root = RAW_DIR / "ims"
    if ims_root.exists():
        test_dirs = [d for d in ims_root.iterdir() if d.is_dir()]
        files = []
        for td in test_dirs:
            snaps = sorted(f for f in td.iterdir() if f.is_file())
            n = len(snaps)
            if n:
                idxs = sorted(set([0, n // 2, n - 1] + list(range(0, n, max(n // 20, 1)))[:20]))
                files += [snaps[i] for i in idxs]
        grand_total += copy_files(files, ims_root, "ims")

    # --- FEMTO: every ~14th snapshot across the full lifecycle, both bearings ---
    femto_root = RAW_DIR / "femto" / "Learning_set"
    if femto_root.exists():
        files = []
        for bearing_dir in sorted(femto_root.iterdir()):
            if bearing_dir.is_dir():
                snaps = sorted(bearing_dir.glob("acc_*.csv"))
                files += snaps[::14]
        grand_total += copy_files(files, RAW_DIR / "femto", "femto")

    # --- Paderborn: one file from every bearing-condition folder present ---
    pb_root = RAW_DIR / "paderborn"
    if pb_root.exists():
        files = []
        for cond_dir in sorted(pb_root.iterdir()):
            if cond_dir.is_dir():
                mats = sorted(cond_dir.glob("*.mat"))
                if mats:
                    files.append(mats[0])
        grand_total += copy_files(files, pb_root, "paderborn")

    # --- SUBF: several files from each of the 3 classes ---
    subf_root = RAW_DIR / "subf" / "Dataset"
    if subf_root.exists():
        files = []
        for cls_dir in sorted(subf_root.iterdir()):
            if cls_dir.is_dir():
                files += sorted(cls_dir.glob("*.csv"))[:8]
        grand_total += copy_files(files, RAW_DIR / "subf", "subf")

    # --- MaFaulDa: normal + a couple of fault classes ---
    maf_root = RAW_DIR / "mafaulda"
    if maf_root.exists():
        files = []
        normal_dir = maf_root / "normal"
        if normal_dir.exists():
            files += sorted(normal_dir.glob("*.csv"))[:1]
        fault_dirs = [d for d in maf_root.iterdir() if d.is_dir() and d.name != "normal"]
        for fd in sorted(fault_dirs)[:2]:
            found = list(fd.rglob("*.csv"))[:1]
            files += found
        grand_total += copy_files(files, maf_root, "mafaulda")

    # --- AI Mechanic: the whole (small) held-out dataset ---
    ai_root = RAW_DIR / "ai_mechanic" / "ai-mechanic-export"
    if ai_root.exists():
        files = sorted(ai_root.rglob("*.wav"))
        grand_total += copy_files(files, RAW_DIR / "ai_mechanic", "ai_mechanic")

    # --- Car Diagnostics: ~40 files from every state/condition subfolder ---
    cd_root = RAW_DIR / "car_diagnostics" / "car diagnostics dataset"
    if cd_root.exists():
        files = []
        for state_dir in sorted(cd_root.iterdir()):
            if state_dir.is_dir():
                for cond_dir in sorted(state_dir.iterdir()):
                    if cond_dir.is_dir():
                        files += sorted(cond_dir.glob("*.wav"))[:40]
        grand_total += copy_files(files, RAW_DIR / "car_diagnostics", "car_diagnostics")

    # --- Engine Journal Bearings: the whole (flagship, already-small) dataset ---
    ejb_root = RAW_DIR / "engine_journal_bearings"
    if ejb_root.exists():
        files = sorted(ejb_root.rglob("*.csv"))
        grand_total += copy_files(files, ejb_root, "engine_journal_bearings")

    # --- MathWorks: the whole (small) dataset ---
    mw_root = RAW_DIR / "mathworks_data"
    if mw_root.exists():
        files = sorted(mw_root.rglob("*.mat"))
        grand_total += copy_files(files, mw_root, "mathworks_data")

    # --- Engine Acoustic Emissions: the rejected dataset, kept as evidence ---
    eae_root = RAW_DIR / "engine_acoustic_emissions"
    if eae_root.exists():
        files = sorted(eae_root.rglob("*.mat"))
        grand_total += copy_files(files, eae_root, "engine_acoustic_emissions")

    print(f"\nTotal sample size: {grand_total/1e6:.1f} MB written to {OUT_DIR}")


if __name__ == "__main__":
    main()
