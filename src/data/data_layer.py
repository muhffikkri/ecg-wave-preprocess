# =====================================================================
# FILE: src/data/data_layer.py
# PURPOSE: Data Access Layer
# =====================================================================

import os
from functools import lru_cache

import numpy as np
import pandas as pd
import scipy.io
import wfdb

from app import config as cfg


# =====================================================================
# RECORD DISCOVERY
# =====================================================================
@lru_cache(maxsize=1)
def get_available_records():
    """
    Scan seluruh dataset yang tersedia.

    Returns
    -------
    dict
        {
            "chapman": [...],
            "ptbxl_100hz": [...],
            "ptbxl_500hz": [...],
            "prosim_simulator": [...]
        }
    """

    records = {
        "chapman": [],
        "ptbxl_100hz": [],
        "ptbxl_500hz": [],
        "prosim_simulator": [],
    }

    # -------------------------------------------------------------
    # Chapman
    # -------------------------------------------------------------
    if os.path.isdir(cfg.CHAPMAN_DIR):
        records["chapman"] = sorted(
            {
                os.path.splitext(f)[0]
                for f in os.listdir(cfg.CHAPMAN_DIR)
                if f.endswith(".mat")
            }
        )

    # -------------------------------------------------------------
    # PTBXL 100Hz
    # -------------------------------------------------------------
    if os.path.isdir(cfg.PTBXL_100HZ_DIR):
        records["ptbxl_100hz"] = sorted(
            {
                os.path.splitext(f)[0]
                for f in os.listdir(cfg.PTBXL_100HZ_DIR)
                if f.endswith(".hea")
            }
        )

    # -------------------------------------------------------------
    # PTBXL 500Hz
    # -------------------------------------------------------------
    if os.path.isdir(cfg.PTBXL_500HZ_DIR):
        records["ptbxl_500hz"] = sorted(
            {
                os.path.splitext(f)[0]
                for f in os.listdir(cfg.PTBXL_500HZ_DIR)
                if f.endswith(".hea")
            }
        )

    # -------------------------------------------------------------
    # ProSim Simulator
    # -------------------------------------------------------------
    prosim_root = cfg.PROSIM_SIMULATOR_DIR

    if os.path.isdir(prosim_root):
        keywords = (
            "bpm",
            "Arr",
            "Sinus",
            "Afib",
            "Afb",
            "Missed",
            "PAC",
        )

        records["prosim_simulator"] = sorted(
            d
            for d in os.listdir(prosim_root)
            if os.path.isdir(
                os.path.join(prosim_root, d)
            )
            and any(k in d for k in keywords)
        )

    return records


# =====================================================================
# LOAD SIGNAL
# =====================================================================
def load_raw_signal(
    dataset_type: str,
    record_id: str,
):
    """
    Memuat sinyal mentah sesuai dataset.

    Returns
    -------
    signal : ndarray
        Shape = (samples, channels)

    fs : float
        Sampling frequency
    """

    if not record_id:
        raise ValueError("record_id kosong.")

    # =============================================================
    # PROSIM
    # =============================================================
    if dataset_type == "prosim_simulator":
        csv_path = os.path.join(
            cfg.PROSIM_SIMULATOR_DIR,
            record_id,
            "data",
            "raw_ecg.csv",
        )

        if not os.path.exists(csv_path):
            raise FileNotFoundError(csv_path)

        df = pd.read_csv(csv_path)
        required = ["ch1", "ch2", "ch3"]

        for col in required:
            if col not in df.columns:
                raise ValueError(
                    f"Kolom '{col}' tidak ditemukan."
                )

        signal = df[
            required
        ].to_numpy(dtype=np.float32)

        fs = 250.0

        return signal, fs

    # =============================================================
    # CHAPMAN
    # =============================================================
    if dataset_type == "chapman":
        mat_path = os.path.join(
            cfg.CHAPMAN_DIR,
            f"{record_id}.mat",
        )

        if not os.path.exists(mat_path):
            raise FileNotFoundError(mat_path)

        mat = scipy.io.loadmat(mat_path)

        if "val" not in mat:
            raise ValueError(
                "'val' tidak ditemukan pada MAT file."
            )

        signal = mat["val"].T.astype(np.float32)
        fs = 500.0
        signal = signal[:, cfg.DEFAULT_LEADS]
        return signal, fs

    # =============================================================
    # PTB-XL
    # =============================================================
    if dataset_type == "ptbxl_100hz":
        folder = cfg.PTBXL_100HZ_DIR

    elif dataset_type == "ptbxl_500hz":
        folder = cfg.PTBXL_500HZ_DIR

    else:
        raise ValueError(
            f"Dataset '{dataset_type}' tidak dikenali."
        )

    record_path = os.path.join(
        folder,
        record_id,
    )

    signal, metadata = wfdb.rdsamp(
        record_path,
    )

    fs = float(
        metadata["fs"]
    )

    signal = signal.astype(np.float32)

    signal = signal[
        :,
        cfg.DEFAULT_LEADS,
    ]

    return signal, fs