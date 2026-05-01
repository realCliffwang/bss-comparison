"""
CWRU 轴承数据加载器
"""

import os
import numpy as np
from scipy.io import loadmat
from typing import List, Dict, Optional, Tuple

from bss_test.utils.logger import get_logger

logger = get_logger(__name__)

# CWRU 故障映射表
CWRU_FAULT_MAP = {
    # (fault_type, load_hp) -> likely filename patterns (Zenodo convention)
    # Format: {first_digit}{second_digit}{load}
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


def _find_key(mat: dict, partial_key: str) -> Optional[str]:
    """Find a mat dict key containing partial_key (case-insensitive)."""
    for k in mat.keys():
        if not k.startswith("__") and partial_key.lower() in k.lower():
            return k
    return None


def list_cwru_files(data_dir: str = "data/cwru") -> List[Dict]:
    """
    Scan data directory and list available .mat files with their contents.

    Parameters
    ----------
    data_dir : str
        Path to the directory containing .mat files.

    Returns
    -------
    list of dict
        Each dict contains: file, keys, path (or error).
    """
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


def load_cwru(
    data_dir: str = "data/cwru",
    fault_type: str = "inner_race_007",
    load: int = 0,
    channels: Optional[List[str]] = None,
    filepath: Optional[str] = None,
) -> Tuple[np.ndarray, float, Optional[int]]:
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
