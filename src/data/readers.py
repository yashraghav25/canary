"""
Per-dataset file readers — written directly against what actually landed
on disk after downloading each dataset (never assumed from documentation
alone).

Each reader returns (signal: np.ndarray float32, sample_rate: int) so it
plugs straight into LazySpectrogramDataset's file_reader parameter.
"""

import json
import re
from pathlib import Path

import numpy as np
import soundfile as sf

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

# --------------------------------------------------------------------------
# AI Mechanic  (held-out evaluation pool — never used for training)
# --------------------------------------------------------------------------
_AI_MECHANIC_ROOT = RAW_DIR / "ai_mechanic" / "ai-mechanic-export"

_NORMAL_LABELS = {"normal engine inside cabin", "idling", "background noise"}
_FAULTY_LABELS = {"air leak", "air leak engine inside cabin", "oil cap off engine inside cabin"}
_EXCLUDE_LABELS = {"unknown", "testing"}

_AI_MECHANIC_LABEL_MAP = None


def _load_ai_mechanic_label_map() -> dict[str, str]:
    """Two info.labels files exist on disk with different path conventions —
    only the top-level one (ai-mechanic-export/info.labels) prefixes paths
    with 'training/'/'testing/', matching path.relative_to(_AI_MECHANIC_ROOT)."""
    labels_file = _AI_MECHANIC_ROOT / "info.labels"
    data = json.loads(labels_file.read_text())
    return {entry["path"]: entry["label"]["label"].strip().lower() for entry in data["files"]}


def ai_mechanic_reader(path: Path) -> tuple[np.ndarray, int]:
    signal, sr = sf.read(str(path), dtype="float32")
    if signal.ndim > 1:
        signal = signal.mean(axis=1)
    return signal, sr


def car_diagnostics_label_fn(path: Path) -> int:
    """0 = normal, 1 = fault. Real held-out car-audio dataset with 3 states
    (braking/idle/startup), each with a 'normal_*' folder and several real
    fault-condition folders — confirmed via directory listing, 1386 files."""
    return 0 if "normal" in path.parent.name else 1


def car_diagnostics_reader(path: Path) -> tuple[np.ndarray, int]:
    signal, sr = sf.read(str(path), dtype="float32")
    if signal.ndim > 1:
        signal = signal.mean(axis=1)
    return signal, sr


def ai_mechanic_label_fn(path: Path) -> int:
    """1 = faulty, 0 = normal, -1 = excluded (unknown/testing rows)."""
    global _AI_MECHANIC_LABEL_MAP
    if _AI_MECHANIC_LABEL_MAP is None:
        _AI_MECHANIC_LABEL_MAP = _load_ai_mechanic_label_map()
    rel_path = str(path.relative_to(_AI_MECHANIC_ROOT)).replace("\\", "/")
    label = _AI_MECHANIC_LABEL_MAP.get(rel_path)
    if label in _EXCLUDE_LABELS or label is None:
        return -1
    if label in _FAULTY_LABELS:
        return 1
    if label in _NORMAL_LABELS:
        return 0
    raise ValueError(f"Unrecognized ai_mechanic label '{label}' for {rel_path}")


# --------------------------------------------------------------------------
# SUBF v2.0  (training pool, audio modality)
# --------------------------------------------------------------------------
# CORRECTED, evidence-based value (was: unverified 44100Hz placeholder).
# The source paper documents the rig as a 3-phase AC motor at 1440 RPM
# (Aziz et al., "Bearing Faults Classification Using Novel Log Energy-Based
# EMD and MFCC") -> 24Hz shaft rotation. FFT of a real "Normal" file (no
# sample-rate assumption, analysis done in cycles/sample) shows a dominant
# peak at 0.004990 cycles/sample with a clean second peak at exactly 2x
# that frequency (0.009980) — a rotation-frequency-plus-harmonic signature.
# Solving 24Hz = 0.004990 cycles/sample * sample_rate gives ~4810Hz,
# rounded to the plausible system rate below. The previous 44100Hz
# assumption was off by ~9.2x, which silently corrupted every SUBF
# spectrogram's frequency axis (and likely inflated the audio domain gap
# against MaFaulDa, whose sample rate IS independently confirmed at 50kHz).
SUBF_SAMPLE_RATE_UNVERIFIED = 4800


def subf_reader(path: Path) -> tuple[np.ndarray, int]:
    signal = np.loadtxt(path, dtype=np.float32)
    return signal, SUBF_SAMPLE_RATE_UNVERIFIED


def subf_label_fn(path: Path) -> int:
    """0 = Normal, 1 = fault (binary framing, for cross-dataset consistency)."""
    return 0 if path.parent.name == "Normal" else 1


def subf_label_fn_3class(path: Path) -> int:
    """0 = Normal, 1 = Inner Race Fault, 2 = Outer Race Fault."""
    mapping = {"Normal": 0, "Inner Race Fault": 1, "Outer Race Fault": 2}
    return mapping[path.parent.name]


# --------------------------------------------------------------------------
# CWRU  (training pool, vibration modality)
# --------------------------------------------------------------------------
# Structure found on disk (srigas/CWRU_Bearing_NumPy mirror):
#   Data/<RPM> RPM/<RPM>_<Fault>_<Diameter?>_<Sensor?>.npz
#   Each .npz has keys 'DE' (drive-end) and sometimes 'FE' (fan-end),
#   shape (n_samples, 1), float64 — ONE long continuous recording per file,
#   not many short clips like SUBF/AI-Mechanic.
#
# Sample rate is not stored inside the .npz itself — inferred from the
# filename suffix and independently VERIFIED (not just inferred) for
# suffix-less "Normal" files two ways:
#   1. Exact entry-count match against Contents.md: 1797_Normal.npz has
#      EXACTLY 243938 entries, identical to 1797_IR_7_DE48.npz (243938) and
#      1797_B_21_DE48.npz (243938) at the same RPM — not just "similar",
#      an exact integer match, which is strong evidence these files share
#      the same 48kHz acquisition rate.
#   2. A supplementary FFT check (same physics-based method used to catch
#      the real SUBF sample-rate error): under the 48kHz hypothesis, the
#      1797_Normal.npz spectrum's strongest peaks cluster near clean
#      integer multiples (~12x) of the file's documented 1797 RPM shaft
#      rotation frequency (29.95 Hz). This check is NOT as clean-cut as
#      SUBF's single dominant tone — bearing vibration spectra are
#      broadband with multiple harmonic families (shaft, line frequency,
#      bearing defect frequencies) — so it's supportive, not decisive, on
#      its own. Combined with the exact count match in (1), 48kHz for CWRU
#      Normal files is confirmed, not assumed.
_CWRU_FAULT_RE = re.compile(r"^\d+_(Normal|IR|B|OR@\d+)")


def _cwru_sample_rate(path: Path) -> int:
    name = path.stem
    if name.endswith("DE48"):
        return 48000
    if name.endswith("DE12") or name.endswith("FE"):
        return 12000
    if "Normal" in name:
        return 48000  # inferred from matching sample count with other _DE48 files, see module docstring
    raise ValueError(f"Could not infer CWRU sample rate for {path.name} — check Contents.md")


def cwru_reader(path: Path) -> tuple[np.ndarray, int]:
    data = np.load(path)
    signal = data["DE"].squeeze().astype(np.float32)  # drive-end channel only, for a single consistent input
    return signal, _cwru_sample_rate(path)


def cwru_label_fn_4class(path: Path) -> int:
    """0 = Normal, 1 = Inner Race, 2 = Ball, 3 = Outer Race (OR@3/6/12 collapsed —
    those suffixes are defect clock-position, not a different fault type)."""
    match = _CWRU_FAULT_RE.match(path.stem)
    if not match:
        raise ValueError(f"Unrecognized CWRU filename pattern: {path.name}")
    fault = match.group(1)
    if fault == "Normal":
        return 0
    if fault == "IR":
        return 1
    if fault == "B":
        return 2
    if fault.startswith("OR@"):
        return 3
    raise ValueError(f"Unhandled CWRU fault code '{fault}' in {path.name}")


def cwru_label_fn_binary(path: Path) -> int:
    return 0 if cwru_label_fn_4class(path) == 0 else 1


# --------------------------------------------------------------------------
# IMS  (training pool, vibration — NASA PCoE, natural run-to-failure)
# --------------------------------------------------------------------------
# Format per NASA PCoE documentation (confirmed structure, not yet
# cross-checked against the extracted file at the time this reader was
# written — verify column count against the first parsed file before
# trusting results): whitespace-delimited text, no header, no extension,
# filename is a timestamp e.g. "2003.10.22.12.06.24". Test 1 has 8 columns
# (4 bearings x 2 channels); Tests 2/3 have 4 columns (1 channel/bearing).
# Sample rate: 20kHz, 20480 points/file (1 second snapshots).
IMS_SAMPLE_RATE = 20000


def ims_reader(path: Path) -> tuple[np.ndarray, int]:
    """First column only, for a single consistent channel regardless of
    which test (1/2/3) the file belongs to."""
    data = np.loadtxt(path, dtype=np.float32)
    col = data[:, 0] if data.ndim > 1 else data
    return col, IMS_SAMPLE_RATE


def make_ims_label_fn(low: float = 0.3, high: float = 1.0 - 0.3) -> "callable":
    """Factory instead of a hardcoded 0.3/0.7 split, so the threshold this
    heuristic depends on can be swept (see
    check_run_to_failure_label_sensitivity.py) instead of being silently
    baked in. `ims_label_fn_binary` below is the same default as before —
    nothing changes unless a caller explicitly asks for a different split."""
    def label_fn(path: Path) -> int:
        test_dir = path.parent
        all_snapshots = sorted(test_dir.glob("*"))
        all_snapshots = [f for f in all_snapshots if f.is_file()]
        idx = all_snapshots.index(path)
        frac = idx / len(all_snapshots)
        if frac < low:
            return 0
        if frac > high:
            return 1
        return -1
    return label_fn


def ims_label_fn_binary(path: Path) -> int:
    """Early/late run-to-failure convention: first 30% of a test's
    chronologically-sorted snapshot files = healthy (0), last 30% =
    degraded (1), middle 40% excluded (-1) as an ambiguous transition. This
    is a HEURISTIC PROXY LABEL, not a verified fault-onset ground truth —
    IMS is a natural run-to-failure dataset with no documented discrete
    fault boundary. See check_run_to_failure_label_sensitivity.py for how
    much this specific threshold choice matters."""
    return make_ims_label_fn()(path)


# --------------------------------------------------------------------------
# Paderborn (KAt)  (training pool, vibration + current)
# --------------------------------------------------------------------------
# Structure CONFIRMED against a real extracted .mat file (K001 test):
# top-level struct named after the file, field `Y` is an array of 7
# channel-structs each with .Name/.Data: force, phase_current_1/2, speed,
# temp_2_bearing_module, torque, vibration_1 (256001 samples -> 64kHz, 4s).
PADERBORN_SAMPLE_RATE = 64000


# CONFIRMED by direct inspection (2025 session, real K001 file): the 7
# channels are NOT all the same sample rate. `phase_current_1/2` have the
# EXACT same sample count as `vibration_1` (256823 samples), so they share
# vibration's 64kHz rate — a genuine electrical-modality channel, ready to
# use with no extra resampling assumptions. `force`/`torque`/`speed` instead
# have 16008 samples over the same ~4.01s recording -> ~3990Hz, rounded to
# the documented Paderborn/KAt DAQ convention of 4kHz for these mechanical
# auxiliary channels (Lessmeier et al.'s own description of the rig: 64kHz
# for vibration/current, 4kHz for load/torque/speed) — empirically confirmed
# here via the sample-count ratio, not assumed from the paper alone.
# `temp_2_bearing_module` is only 5 points per file (~1.25Hz) — not a
# waveform, not given a spectrogram encoder; skipped entirely.
PADERBORN_AUX_SAMPLE_RATE = 4000


def _paderborn_channels(path: Path) -> dict:
    import scipy.io as sio
    d = sio.loadmat(str(path), struct_as_record=False, squeeze_me=True)
    top_key = [k for k in d.keys() if not k.startswith("__")][0]
    return {ch.Name: ch.Data for ch in d[top_key].Y}


def paderborn_reader(path: Path) -> tuple[np.ndarray, int]:
    channels = _paderborn_channels(path)
    return channels["vibration_1"].astype(np.float32), PADERBORN_SAMPLE_RATE


def paderborn_reader_current(path: Path) -> tuple[np.ndarray, int]:
    """Electrical modality — phase_current_1, same native rate as vibration_1
    (confirmed by identical sample count), so no extra rate assumption
    needed. This is real motor-current data, downloaded specifically for
    genuine cross-sensor-type multimodality but never used until now."""
    channels = _paderborn_channels(path)
    return channels["phase_current_1"].astype(np.float32), PADERBORN_SAMPLE_RATE


def paderborn_reader_force(path: Path) -> tuple[np.ndarray, int]:
    """Mechanical load-cell channel, ~4kHz (see PADERBORN_AUX_SAMPLE_RATE)."""
    channels = _paderborn_channels(path)
    return channels["force"].astype(np.float32), PADERBORN_AUX_SAMPLE_RATE


def paderborn_reader_torque(path: Path) -> tuple[np.ndarray, int]:
    """Mechanical torque-sensor channel, ~4kHz (see PADERBORN_AUX_SAMPLE_RATE)."""
    channels = _paderborn_channels(path)
    return channels["torque"].astype(np.float32), PADERBORN_AUX_SAMPLE_RATE


def paderborn_label_fn_binary(path: Path) -> int:
    """0 = healthy (K0xx codes), 1 = any damage (KA/KB/KI codes) — based on
    the bearing code embedded in the filename, per Paderborn's own naming
    convention (see mb.uni-paderborn.de bearing-damage documentation)."""
    stem = path.stem
    bearing_code = stem.split("_")[3] if "_" in stem else stem
    return 0 if bearing_code.startswith("K0") else 1


# --------------------------------------------------------------------------
# MaFaulDa  (training pool — THE bridge dataset: real paired vibration+audio)
# --------------------------------------------------------------------------
# Structure confirmed directly against downloaded files (only "normal" class
# fetched so far — 49 files, 310MB — from the official UFRJ source, not the
# broken Kaggle per-file mirror). Per the UFRJ page and verified against a
# real file: 8 comma-separated columns, 250000 rows (5s @ 50kHz), no header:
#   col 1        tachometer
#   cols 2-4     underhang bearing accelerometer (axial, radial, tangential)
#   cols 5-7     overhang bearing accelerometer (axial, radial, tangential)
#   col 8        microphone
MAFAULDA_SAMPLE_RATE = 50000


def mafaulda_reader_vibration(path: Path) -> tuple[np.ndarray, int]:
    """Single vibration channel (underhang-axial, column index 1) — picked as
    ONE representative channel for the vibration encoder, matching the
    single-channel convention used for CWRU/Paderborn elsewhere in this
    project. The other 5 vibration columns are available in the raw file
    if a richer multi-channel encoder is built later."""
    data = np.loadtxt(path, delimiter=",", dtype=np.float32)
    return data[:, 1], MAFAULDA_SAMPLE_RATE


def mafaulda_reader_audio(path: Path) -> tuple[np.ndarray, int]:
    """Microphone channel (column index 7, last column)."""
    data = np.loadtxt(path, delimiter=",", dtype=np.float32)
    return data[:, 7], MAFAULDA_SAMPLE_RATE


def mafaulda_reader_paired(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """Reads the file ONCE and returns both channels together — avoids
    parsing the same (slow, np.loadtxt) file twice per sample, which the two
    separate readers above would do if called independently on the same path."""
    data = np.loadtxt(path, delimiter=",", dtype=np.float32)
    return data[:, 1], data[:, 7], MAFAULDA_SAMPLE_RATE


def mafaulda_label_fn_binary(path: Path) -> int:
    """0 = normal, 1 = fault. Determined from the folder path: files live
    under mafaulda/<class_name>/<severity>/<file>.csv (fault classes) or
    mafaulda/normal/<file>.csv (no severity subfolder)."""
    return 0 if "normal" in path.parts else 1


def mafaulda_label_fn_multiclass(path: Path) -> int:
    """0=normal 1=horizontal-misalignment 2=vertical-misalignment
    3=imbalance 4=underhang 5=overhang — the 6 real MaFaulDa fault classes."""
    mapping = {
        "normal": 0, "horizontal-misalignment": 1, "vertical-misalignment": 2,
        "imbalance": 3, "underhang": 4, "overhang": 5,
    }
    for part in path.parts:
        if part in mapping:
            return mapping[part]
    raise ValueError(f"Could not determine MaFaulDa class for {path}")


# --------------------------------------------------------------------------
# FEMTO / PRONOSTIA  (training pool, vibration + temperature)
# --------------------------------------------------------------------------
# Structure confirmed against downloaded files (2 of ~17 bearings fetched
# via git sparse-checkout — a representative subset, not the full dataset):
#   Learning_set/Bearing<condition>_<id>/acc_NNNNN.csv — one 0.1s snapshot
#   per file, 2560 rows, 6 columns, no header:
#     hour, minute, second, microsecond, horizontal_accel, vertical_accel
# Snapshots are taken periodically through a run-to-failure lifecycle
# (Bearing1_1 has 2803 snapshot files here). Sample rate: 25.6kHz per the
# PRONOSTIA platform spec (confirmed in the plan's dataset research).
FEMTO_SAMPLE_RATE = 25600


def femto_reader(path: Path) -> tuple[np.ndarray, int]:
    """Horizontal accelerometer channel (column index 4)."""
    data = np.loadtxt(path, delimiter=",", dtype=np.float32)
    return data[:, 4], FEMTO_SAMPLE_RATE


def make_femto_label_fn(low: float = 0.3, high: float = 1.0 - 0.3) -> "callable":
    """Factory instead of a hardcoded 0.3/0.7 split — see make_ims_label_fn
    for why, and check_run_to_failure_label_sensitivity.py for the sweep."""
    def label_fn(path: Path) -> int:
        bearing_dir = path.parent
        all_snapshots = sorted(bearing_dir.glob("acc_*.csv"))
        idx = all_snapshots.index(path)
        n = len(all_snapshots)
        frac = idx / n
        if frac < low:
            return 0
        if frac > high:
            return 1
        return -1
    return label_fn


def femto_label_fn_binary(path: Path) -> int:
    """0 = early-life (first 30% of snapshots in this bearing's run) =
    'healthy', 1 = late-life (last 30%) = 'degraded'. Middle 40% is
    ambiguous transition and excluded (-1) — a run-to-failure dataset
    doesn't have a real fault/no-fault boundary, so this is a deliberate,
    documented simplification for the binary-fault framing used elsewhere
    in this project, not a claim that degradation is actually discrete.
    This is a HEURISTIC PROXY LABEL, not a verified fault-onset ground
    truth. See check_run_to_failure_label_sensitivity.py for how much this
    specific threshold choice matters."""
    return make_femto_label_fn()(path)


# --------------------------------------------------------------------------
# Engine Journal Bearings Dataset (Mendeley, NUST Islamabad)
# --------------------------------------------------------------------------
# A REAL held-out dataset with what AI Mechanic/Car Diagnostics never had:
# genuine VIBRATION (not audio) from a REAL automobile internal combustion
# engine — tri-axial accelerometer mounted on the main journal bearing
# housing, healthy vs. faulty, across varying RPM/humidity/temperature
# conditions (MIL-STD-810G climatic chamber, not on-road, but a real engine,
# not a bearing test rig). Source: Riaz et al., Mendeley Data, DOI
# 10.17632/3fcrrdjjvk.5, CC BY 4.0. We downloaded only the smaller
# supplementary zip ("Dataset 2", ~63MB uncompressed, 134 files: 90 faulty /
# 44 healthy) as a representative subset, not the full ~250MB primary zip.
#
# Format CONFIRMED against a real extracted file: CSV, header row, columns
# Time, Demand 1, Control 1, Output Drive 1, Channel 1-4 (Channel 4 is ~1e-15
# / noise-floor in every file inspected — treated as a dead/unused channel),
# Channel 1-4 Kurtosis, Rear Input 1-8 (all zero in files inspected). ~596
# rows per file, ~2.08s duration. Channel 1 is used as the single
# representative accelerometer axis (same one-channel-per-file convention as
# CWRU/MaFaulDa/FEMTO elsewhere in this project).
#
# Sample rate is NOT documented and is notably low for vibration analysis
# (~296Hz, computed below from the Time column) — more of a logged
# control-loop rate than a dedicated high-frequency vibration DAQ. This caps
# what fault-frequency content is even physically capturable (Nyquist
# ~148Hz), which is an honest limitation of this dataset, not a reading bug.
# Computed PER FILE from that file's own Time column rather than a single
# hardcoded constant, since exact timing varies slightly file to file.


def engine_journal_bearings_reader(path: Path) -> tuple[np.ndarray, int]:
    import pandas as pd
    df = pd.read_csv(path)
    signal = df["Channel 1"].to_numpy(dtype=np.float32)
    dt = df["Time"].diff().median()
    sample_rate = int(round(1.0 / dt)) if dt and dt > 0 else 300
    return signal, sample_rate


def engine_journal_bearings_label_fn(path: Path) -> int:
    """0 = healthy, 1 = faulty — from the top-level folder name
    ('Healthy Bearings Dataset' / 'Faulty Bearings Dataset')."""
    parts = [p.lower() for p in path.parts]
    if any("healthy" in p for p in parts):
        return 0
    if any("faulty" in p for p in parts):
        return 1
    raise ValueError(f"Could not determine healthy/faulty label for {path}")
