"""
NASA / UC Berkeley 铣削数据加载器
"""

import os
import numpy as np
from scipy.io import loadmat
from typing import List, Dict, Optional, Tuple

from bss_test.utils.logger import get_logger

logger = get_logger(__name__)

# NASA 传感器映射
NASA_MILL_SENSOR_MAP = {
    "force_ac": "smcAC",
    "force_dc": "smcDC",
    "vib_table": "vib_table",
    "vib_spindle": "vib_spindle",
    "AE_table": "AE_table",
    "AE_spindle": "AE_spindle",
}

NASA_MILL_FS = 250  # Hz — sampling rate (9k samples in ~36 s per run)


def load_nasa_milling(
    data_dir: str = "data/phm2010_milling",
    sensor_types: Optional[List[str]] = None,
    case_filter: Optional[List[int]] = None,
    include_wear_only: bool = False,
) -> Tuple[List[np.ndarray], List[Dict], float]:
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


def load_nasa_milling_single(
    run_index: int = 0,
    data_dir: str = "data/phm2010_milling",
    sensor_types: Optional[List[str]] = None,
    case_filter: Optional[List[int]] = None,
    include_wear_only: bool = False,
) -> Tuple[np.ndarray, Dict, float]:
    """
    Load a single run from the NASA milling dataset.

    Parameters
    ----------
    run_index : int
        Index of the run to load (0-based).
    data_dir : str
        Path containing mill.mat.
    sensor_types : list or None
        List of sensor keys.
    case_filter : list of int or None
        Filter by case number(s).
    include_wear_only : bool
        If True, only return runs that have wear measurements.

    Returns
    -------
    signals : ndarray (n_channels, n_samples)
    meta : dict
        Contains: case, run, VB, DOC, feed, time, index.
    fs : float
        Sampling rate in Hz.
    """
    all_signals, all_meta, fs = load_nasa_milling(
        data_dir, sensor_types, case_filter, include_wear_only
    )
    if run_index >= len(all_signals):
        raise IndexError(f"run_index {run_index} out of range ({len(all_signals)} runs)")
    return all_signals[run_index], all_meta[run_index], fs
