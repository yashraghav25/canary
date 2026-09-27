"""
Data-level sensitivity check for the IMS/FEMTO "early 30% = healthy, late
30% = degraded" labeling heuristic (see make_ims_label_fn / make_femto_label_fn
in data/readers.py). This is a proxy label, not a verified fault-onset
ground truth — both datasets are natural run-to-failure recordings with no
documented discrete healthy/faulty boundary.

Retraining the full DANN pipeline once per threshold choice would be the
"complete" sensitivity check, but that's hours of compute per point on this
CPU-only machine. This script does the cheap, honest thing instead: report
how many labeled files/snapshots actually change class (or move to
excluded/ambiguous) as the threshold moves, at the file level, with no
model involved. A threshold choice that reassigns a large fraction of
labels is a real fragility signal even before any model is trained on it;
one that barely changes anything means the heuristic is at least stable to
reasonable variation in where the cutoff is drawn.
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from pathlib import Path

import numpy as np

from data.readers import make_femto_label_fn, make_ims_label_fn

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

THRESHOLDS = [
    ("strict (20/80)", 0.2, 0.8),
    ("default (30/70)", 0.3, 0.7),
    ("loose (40/60)", 0.4, 0.6),
]


def report_dataset(name: str, files: list[Path], label_fn_factory) -> None:
    print(f"\n{'='*70}\n{name}  ({len(files)} files)\n{'='*70}")
    labelings = {}
    for tag, low, high in THRESHOLDS:
        label_fn = label_fn_factory(low=low, high=high)
        labels = np.array([label_fn(f) for f in files])
        n_healthy = (labels == 0).sum()
        n_degraded = (labels == 1).sum()
        n_excluded = (labels == -1).sum()
        print(f"  {tag:18s}: healthy={n_healthy:5d}  degraded={n_degraded:5d}  excluded={n_excluded:5d}")
        labelings[tag] = labels

    base = labelings["default (30/70)"]
    for tag, _, _ in THRESHOLDS:
        if tag == "default (30/70)":
            continue
        other = labelings[tag]
        usable_both = (base != -1) & (other != -1)
        n_usable_both = usable_both.sum()
        n_flipped = (base[usable_both] != other[usable_both]).sum()
        pct = 100.0 * n_flipped / max(n_usable_both, 1)
        print(f"  default vs {tag:18s}: {n_flipped}/{n_usable_both} labeled-in-both files "
              f"({pct:.1f}%) flip class when the threshold changes")


def main() -> None:
    ims_dir = RAW_DIR / "ims"
    if ims_dir.exists():
        ims_files = [f for f in sorted(ims_dir.rglob("*")) if f.is_file() and f.name not in ("IMS.7z", "2nd_test.rar")]
        report_dataset("IMS", ims_files, make_ims_label_fn)
    else:
        print("IMS not found on disk — skipped.")

    femto_dir = RAW_DIR / "femto" / "Learning_set"
    if femto_dir.exists():
        femto_files = sorted(femto_dir.rglob("acc_*.csv"))
        report_dataset("FEMTO", femto_files, make_femto_label_fn)
    else:
        print("FEMTO not found on disk — skipped.")

    print("\nDone. A high flip-rate above means the DANN vibration domains built from these two "
          "datasets are labeling a fault/healthy boundary that swings a lot under a plausible change of "
          "where the cutoff is drawn — a real, honest fragility in the pretraining labels for those "
          "2 of 4 vibration domains, independent of anything the model itself does.")


if __name__ == "__main__":
    main()
