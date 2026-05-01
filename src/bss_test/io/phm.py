"""
PHM 2010 铣削数据加载器
"""

import os
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple

from bss_test.utils.logger import get_logger

logger = get_logger(__name__)

# PHM 2010 列映射
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


def load_phm_cut(
    tool_id: int = 1,
    cut_no: int = 1,
    data_dir: str = "data/phm2010_milling",
    sensor_types: Optional[List[str]] = None,
    downsample: int = 1,
) -> Tuple[np.ndarray, int, int]:
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


def load_phm_wear(
    tool_id: int = 1,
    data_dir: str = "data/phm2010_milling",
) -> np.ndarray:
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
    # Try common wear file patterns
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


def load_phm_all_cuts(
    tool_id: int = 1,
    data_dir: str = "data/phm2010_milling",
    sensor_types: Optional[List[str]] = None,
    max_cuts: Optional[int] = None,
    downsample: int = 1,
) -> Tuple[np.ndarray, int, np.ndarray, List[int]]:
    """
    Load all cuts for a PHM tool, stacking them sequentially.

    Parameters
    ----------
    tool_id : int
        Cutter number: 1 or 4.
    data_dir : str
        Root directory.
    sensor_types : list or None
        List of sensor keys.
    max_cuts : int or None
        Maximum number of cuts to load.
    downsample : int
        Downsampling factor.

    Returns
    -------
    signals_concat : ndarray (n_channels, total_samples)
        All cuts concatenated along time axis.
    fs : int
        Sampling rate.
    wear : ndarray (n_cuts,)
        Wear values.
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
