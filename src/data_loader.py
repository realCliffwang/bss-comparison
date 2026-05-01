"""
Data loader for CWRU Bearing Dataset and PHM 2010 Milling Tool Wear Dataset.

CWRU: .mat files from https://zenodo.org/records/10987113
PHM:  CSV files from https://phmsociety.org/phm_competition/2010-phm-society-conference-data-challenge/

TODO for CWRU:
  1. Extract the .mat files into data/cwru/ directory
  2. File naming convention (Zenodo version):
     - Files are named as {x}{y}{z}.mat (3 digits)
     - The exact mapping depends on the multivariate_cwru convention
     - Common files: 122.mat (12k DE inner race 0.007" load0), etc.
  3. Use list_cwru_files() to scan what's available

TODO for PHM 2010:
  1. Extract c1.zip and c4.zip into data/phm2010_milling/c1/ and c4/
  2. Each folder contains ~315 CSV files (one per cut) + 1 wear file
  3. CSV columns: force_x, force_y, force_z, vib_x, vib_y, vib_z, AE_rms
  4. If actual column names differ, update COLUMN_MAP in load_phm_cut()

TODO for stamping die data migration:
  - Replace load_cwru() / load_phm_cut() with your own reader
  - Adjust sampling rate fs, column names, and file naming logic
"""

import os
import numpy as np
import pandas as pd
from scipy.io import loadmat


# ===========================
# CWRU data loader
# ===========================

CWRU_FAULT_MAP = {
    # (fault_type, load_hp) -> likely filename patterns (Zenodo convention)
    # Format: {first_digit}{second_digit}{load}
    # This is a best-guess mapping; verify with your actual filenames.
    ("normal", 0): "109",
    ("normal", 1): "110",
    ("normal", 2): "111",
    ("normal", 3): "112",
    ("inner_race_007", 0): "122",
    ("inner_race_007", 1): "123",
    ("inner_race_007", 2): "124",
    ("inner_race_007", 3): "125",
    ("ball_007", 0): "135",
    ("ball_007", 1): "136",
    ("ball_007", 2): "137",
    ("ball_007", 3): "138",
    ("outer_race_6_007", 0): "148",
    ("outer_race_6_007", 1): "149",
    ("outer_race_6_007", 2): "150",
    ("outer_race_6_007", 3): "151",
    # 48k DE data
    ("inner_race_007_48k", 0): "161",
    ("inner_race_007_48k", 1): "162",
    ("inner_race_007_48k", 2): "163",
    ("inner_race_007_48k", 3): "164",
    ("ball_007_48k", 0): "174",
    ("ball_007_48k", 1): "175",
    ("ball_007_48k", 2): "176",
    ("ball_007_48k", 3): "177",
    ("outer_race_6_007_48k", 0): "189",
    ("outer_race_6_007_48k", 1): "190",
    ("outer_race_6_007_48k", 2): "191",
    ("outer_race_6_007_48k", 3): "192",
}


def list_cwru_files(data_dir="data/cwru"):
    """Scan data directory and list available .mat files with their contents."""
    files = [f for f in os.listdir(data_dir) if f.endswith(".mat")]
    files.sort()
    result = []
    for f in files:
        path = os.path.join(data_dir, f)
        try:
            mat = loadmat(path)
            keys = [k for k in mat.keys() if not k.startswith("__")]
            result.append({"file": f, "keys": keys, "path": path})
        except Exception as e:
            result.append({"file": f, "error": str(e), "path": path})
    return result


def load_cwru(data_dir="data/cwru", fault_type="inner_race_007", load=0,
              channels=None, filepath=None):
    """
    Load CWRU bearing vibration data.

    Parameters
    ----------
    data_dir : str
        Path to the directory containing .mat files.
    fault_type : str
        One of: normal, inner_race_007, ball_007, outer_race_6_007, etc.
        See CWRU_FAULT_MAP for complete list.
    load : int
        Motor load in HP: 0, 1, 2, or 3.
    channels : list or None
        List of channel names to return. E.g. ["DE"] or ["DE", "FE"].
        If None, returns only DE channel.
    filepath : str or None
        Direct path to a .mat file. If provided, fault_type and load are ignored.

    Returns
    -------
    signals : ndarray (n_channels, n_samples)
    fs : float
        Sampling rate in Hz.
    rpm : int or None
        Approximate motor RPM.
    """
    if filepath is not None:
        path = filepath
    else:
        key = (fault_type, load)
        filename = CWRU_FAULT_MAP.get(key)
        if filename is None:
            available = list_cwru_files(data_dir)
            raise ValueError(
                f"No file mapping for fault_type={fault_type}, load={load}. "
                f"Available files: {[a['file'] for a in available]}. "
                f"You can pass filepath= directly."
            )
        path = os.path.join(data_dir, f"{filename}.mat")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Expected file not found: {path}. "
                f"Please download and place the .mat files in {data_dir}/"
            )

    mat = loadmat(path)
    # Find DE_time and FE_time keys
    de_key = _find_key(mat, "DE_time")
    fe_key = _find_key(mat, "FE_time")
    rpm_key = _find_key(mat, "RPM")

    if de_key is None:
        raise KeyError(f"No DE_time variable found in {path}. Keys: {list(mat.keys())}")

    if channels is None:
        channels = ["DE"]

    signals_list = []
    for ch in channels:
        if ch == "DE":
            key = de_key
        elif ch == "FE":
            key = fe_key
        else:
            raise ValueError(f"Unknown channel: {ch}")
        if key is None:
            raise KeyError(f"Channel {ch} not found in file.")
        signals_list.append(mat[key].flatten())

    # Determine sampling rate from filename/key context
    # 48k files: digits > 160 in Zenodo convention (e.g. 161-192)
    filename_stem = os.path.splitext(os.path.basename(path))[0]
    if filename_stem.isdigit():
        file_num = int(filename_stem)
        if 160 <= file_num < 200 or 260 <= file_num < 300:
            fs = 48000
        else:
            fs = 12000
    else:
        # Default conservative assumption
        fs = 12000

    rpm = int(mat[rpm_key][0][0]) if rpm_key is not None else None

    signals = np.array(signals_list)
    return signals, fs, rpm


def _find_key(mat, partial_key):
    """Find a mat dict key containing partial_key (case-insensitive)."""
    for k in mat.keys():
        if not k.startswith("__") and partial_key.lower() in k.lower():
            return k
    return None


# ===========================
# PHM 2010 Milling data loader
# ===========================

PHM_COLUMN_MAP = {
    "force_x": 0,
    "force_y": 1,
    "force_z": 2,
    "vib_x": 3,
    "vib_y": 4,
    "vib_z": 5,
    "AE": 6,
}

PHM_SAMPLING_RATE = 50000  # Hz per challenge documentation
PHM_SPINDLE_SPEED = 10400  # RPM


def load_phm_cut(tool_id=1, cut_no=1, data_dir="data/phm2010_milling",
                 sensor_types=None, downsample=1):
    """
    Load a single cut from the PHM 2010 milling dataset.

    Parameters
    ----------
    tool_id : int
        Cutter number: 1 or 4 (training data).
    cut_no : int
        Cut number: 1 to ~315.
    data_dir : str
        Root directory containing c1/ and c4/ subdirectories.
    sensor_types : list or None
        List of sensor keys. E.g. ["vib_x", "vib_y", "vib_z"].
        Available: force_x, force_y, force_z, vib_x, vib_y, vib_z, AE.
        If None, uses ["vib_x", "vib_y", "vib_z"].
    downsample : int
        Downsampling factor (every Nth sample). 1 = no downsampling.

    Returns
    -------
    signals : ndarray (n_channels, n_samples)
    fs : int
        Sampling rate (50000 Hz).
    cut_no : int
        Same as input (for bookkeeping).
    """
    # Build file path
    tool_dir = os.path.join(data_dir, f"c{tool_id}")
    patterns = [
        f"c_{tool_id}_{cut_no:03d}.csv",
        f"{cut_no:03d}.csv",
    ]
    filepath = None
    for pat in patterns:
        candidate = os.path.join(tool_dir, pat)
        if os.path.exists(candidate):
            filepath = candidate
            break

    if filepath is None:
        # List what's available to help debugging
        if os.path.isdir(tool_dir):
            sample_files = os.listdir(tool_dir)[:10]
        else:
            sample_files = ["directory not found"]
        raise FileNotFoundError(
            f"Could not find CSV for tool_id={tool_id}, cut_no={cut_no}. "
            f"Tool dir contents (first 10): {sample_files}. "
            f"TODO: Adjust filename patterns in load_phm_cut()."
        )

    # Read CSV
    # TODO: Verify if header exists. PHM documentation says no header.
    # Columns: force_x, force_y, force_z, vib_x, vib_y, vib_z, AE_rms
    df = pd.read_csv(filepath, header=None)
    data = df.values  # shape: (n_samples, 7)

    if sensor_types is None:
        sensor_types = ["vib_x", "vib_y", "vib_z"]

    signals_list = []
    for st in sensor_types:
        col_idx = PHM_COLUMN_MAP.get(st)
        if col_idx is None:
            raise ValueError(f"Unknown sensor type: {st}")
        sig = data[:, col_idx]
        if downsample > 1:
            sig = sig[::downsample]
        signals_list.append(sig)

    signals = np.array(signals_list)
    fs = PHM_SAMPLING_RATE // downsample
    return signals, fs, cut_no


def load_phm_wear(tool_id=1, data_dir="data/phm2010_milling"):
    """
    Load tool wear labels for a given cutter.

    Parameters
    ----------
    tool_id : int
        Cutter number: 1 or 4.
    data_dir : str
        Root directory.

    Returns
    -------
    wear : ndarray (n_cuts,)
        Wear values in 10^-3 mm (flank wear VB).
    """
    tool_dir = os.path.join(data_dir, f"c{tool_id}")
    # TODO: Verify wear file name. Common patterns:
    patterns = [
        f"c{tool_id}_wear.csv",
        f"c_{tool_id}_wear.csv",
        "wear.csv",
        f"c{tool_id}.wear",
    ]
    filepath = None
    for pat in patterns:
        candidate = os.path.join(tool_dir, pat)
        if os.path.exists(candidate):
            filepath = candidate
            break

    if filepath is None:
        # Try to scan for any file containing "wear"
        if os.path.isdir(tool_dir):
            for f in os.listdir(tool_dir):
                if "wear" in f.lower():
                    filepath = os.path.join(tool_dir, f)
                    break

    if filepath is None:
        raise FileNotFoundError(
            f"Could not find wear file for tool_id={tool_id}. "
            f"TODO: Adjust filename patterns in load_phm_wear()."
        )

    wear_df = pd.read_csv(filepath)
    # The wear file has columns: cut, flute_1, flute_2, flute_3
    # Average across flutes to get mean flank wear per cut
    flute_cols = [c for c in wear_df.columns if "flute" in c.lower()]
    if flute_cols:
        wear = wear_df[flute_cols].mean(axis=1).values
    else:
        wear = wear_df.iloc[:, 1:].mean(axis=1).values
    return wear


def load_phm_all_cuts(tool_id=1, data_dir="data/phm2010_milling",
                      sensor_types=None, max_cuts=None, downsample=1):
    """
    Load all cuts for a PHM tool, stacking them sequentially.

    Returns
    -------
    signals_concat : ndarray (n_channels, total_samples)
        All cuts concatenated along time axis.
    fs : int
    wear : ndarray (n_cuts,)
    cut_starts : list of int
        Sample indices where each cut starts in the concatenated signal.
    """
    wear = load_phm_wear(tool_id, data_dir)
    n_cuts = len(wear)
    if max_cuts is not None:
        n_cuts = min(n_cuts, max_cuts)

    all_signals = []
    cut_starts = [0]
    current_pos = 0

    for cut_no in range(1, n_cuts + 1):
        sig, fs, _ = load_phm_cut(tool_id, cut_no, data_dir,
                                   sensor_types, downsample)
        all_signals.append(sig)
        current_pos += sig.shape[1]
        cut_starts.append(current_pos)

    signals_concat = np.concatenate(all_signals, axis=1)
    return signals_concat, fs, wear[:n_cuts], cut_starts[:-1]


# ===========================
# NASA / UC Berkeley Milling data loader (backup for PHM 2010)
# ===========================

NASA_MILL_SENSOR_MAP = {
    "force_ac": "smcAC",
    "force_dc": "smcDC",
    "vib_table": "vib_table",
    "vib_spindle": "vib_spindle",
    "AE_table": "AE_table",
    "AE_spindle": "AE_spindle",
}

NASA_MILL_FS = 250  # Hz — sampling rate (9k samples in ~36 s per run)


def load_nasa_milling(data_dir="data/phm2010_milling",
                       sensor_types=None, case_filter=None,
                       include_wear_only=False):
    """
    Load NASA / UC Berkeley milling dataset (from mill.mat).

    Parameters
    ----------
    data_dir : str
        Path containing mill.mat.
    sensor_types : list or None
        List of sensor keys. Default: ["vib_table", "vib_spindle", "force_ac"].
    case_filter : list of int or None
        Filter by case number(s). E.g. [1, 2] or None for all.
    include_wear_only : bool
        If True, only return runs that have wear measurements (non-NaN VB).

    Returns
    -------
    all_signals : list of ndarray
        List of (n_channels, n_samples) for each run.
    all_meta : list of dict
        Each dict contains: case, run, VB, DOC, feed, time.
    fs : float
        Sampling rate in Hz.
    """
    mat_path = os.path.join(data_dir, "mill.mat")
    if not os.path.exists(mat_path):
        # Also check nested path
        alt = os.path.join(data_dir, "3. Milling", "mill.mat")
        if os.path.exists(alt):
            mat_path = alt
        else:
            raise FileNotFoundError(
                f"mill.mat not found at {mat_path} or {alt}. "
                f"Download from http://phm-datasets.s3.amazonaws.com/NASA/3.+Milling.zip"
            )

    mat = loadmat(mat_path)
    mill = mat["mill"]  # shape (1, 167), structured array

    if sensor_types is None:
        sensor_types = ["vib_table", "vib_spindle", "force_ac"]

    all_signals = []
    all_meta = []

    for idx in range(mill.shape[1]):
        vb_val = mill["VB"][0, idx].flatten()
        if len(vb_val) > 0:
            vb = float(vb_val[0])
        else:
            vb = float("nan")

        case_val = int(mill["case"][0, idx].flatten()[0])
        run_val = int(mill["run"][0, idx].flatten()[0])

        # Filtering
        if case_filter is not None and case_val not in case_filter:
            continue
        if include_wear_only and np.isnan(vb):
            continue

        doc_val = float(mill["DOC"][0, idx].flatten()[0])
        feed_val = float(mill["feed"][0, idx].flatten()[0])
        time_val = float(mill["time"][0, idx].flatten()[0])

        # Extract sensor signals
        sig_list = []
        for st in sensor_types:
            mat_key = NASA_MILL_SENSOR_MAP.get(st)
            if mat_key is None:
                raise ValueError(f"Unknown sensor: {st}")
            raw = mill[mat_key][0, idx].flatten().astype(np.float64)
            sig_list.append(raw)

        signals = np.array(sig_list)
        all_signals.append(signals)
        all_meta.append({
            "case": case_val,
            "run": run_val,
            "VB": vb,
            "DOC": doc_val,
            "feed": feed_val,
            "time": time_val,
            "index": idx,
        })

    return all_signals, all_meta, NASA_MILL_FS


def load_nasa_milling_single(run_index=0, data_dir="data/phm2010_milling",
                              sensor_types=None, case_filter=None,
                              include_wear_only=False):
    """
    Load a single run from the NASA milling dataset.

    Returns
    -------
    signals : ndarray (n_channels, n_samples)
    meta : dict
    fs : float
    """
    all_signals, all_meta, fs = load_nasa_milling(
        data_dir, sensor_types, case_filter, include_wear_only
    )
    if run_index >= len(all_signals):
        raise IndexError(f"run_index {run_index} out of range ({len(all_signals)} runs)")
    return all_signals[run_index], all_meta[run_index], fs
