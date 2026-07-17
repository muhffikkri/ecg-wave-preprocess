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

    # -------------------------------------------------------------
    # Sensor Records (Dynamic folders)
    # -------------------------------------------------------------
    if os.path.isdir(cfg.SENSOR_RECORD_DIR):
        for entry in os.scandir(cfg.SENSOR_RECORD_DIR):
            if entry.is_dir():
                folder_name = entry.name
                folder_path = os.path.join(cfg.SENSOR_RECORD_DIR, folder_name)
                records[folder_name] = sorted(
                    {
                        os.path.splitext(f)[0]
                        for f in os.listdir(folder_path)
                        if f.endswith(".csv")
                    }
                )

    return records


# =====================================================================
# LOAD SIGNAL
# =====================================================================
import logging
logger = logging.getLogger("ecg_workbench.data_layer")

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
    logger.info(f"Loading record '{record_id}' from dataset '{dataset_type}'...")

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
            raise FileNotFoundError(f"File raw ECG simulator tidak ditemukan: {csv_path}")

        df = pd.read_csv(csv_path)
        required = ["ch1", "ch2", "ch3"]

        for col in required:
            if col not in df.columns:
                raise ValueError(
                    f"Kolom '{col}' tidak ditemukan di CSV simulator."
                )

        signal = df[required].to_numpy(dtype=np.float32)
        fs = 250.0

    # =============================================================
    # CHAPMAN
    # =============================================================
    elif dataset_type == "chapman":
        mat_path = os.path.join(
            cfg.CHAPMAN_DIR,
            f"{record_id}.mat",
        )

        if not os.path.exists(mat_path):
            raise FileNotFoundError(f"File MAT Chapman tidak ditemukan: {mat_path}")

        mat = scipy.io.loadmat(mat_path)

        if "val" not in mat:
            raise ValueError(
                "'val' tidak ditemukan pada MAT file Chapman."
            )

        signal = mat["val"].T.astype(np.float32)
        fs = 500.0
        signal = signal[:, cfg.DEFAULT_LEADS]

    # =============================================================
    # PTB-XL
    # =============================================================
    elif dataset_type in ("ptbxl_100hz", "ptbxl_500hz"):
        if dataset_type == "ptbxl_100hz":
            folder = cfg.PTBXL_100HZ_DIR
        else:
            folder = cfg.PTBXL_500HZ_DIR

        record_path = os.path.join(
            folder,
            record_id,
        )

        # Check if record files exist (.hea and .dat)
        if not os.path.exists(record_path + ".hea"):
            raise FileNotFoundError(f"File metadata WFDB tidak ditemukan: {record_path}.hea")

        signal, metadata = wfdb.rdsamp(
            record_path,
        )

        fs = float(metadata["fs"])
        signal = signal.astype(np.float32)
        signal = signal[:, cfg.DEFAULT_LEADS]

    # =============================================================
    # SENSOR RECORD (DYNAMIC CHANNELS)
    # =============================================================
    elif dataset_type not in ("chapman", "ptbxl_100hz", "ptbxl_500hz", "prosim_simulator"):
        csv_path = os.path.join(
            cfg.SENSOR_RECORD_DIR,
            dataset_type,
            f"{record_id}.csv"
        )
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"File CSV sensor record tidak ditemukan: {csv_path}")

        df = pd.read_csv(csv_path)
        required = ["lead_i_mV", "lead_ii_mV", "lead_iii_mV"]
        for col in required:
            if col not in df.columns:
                raise ValueError(
                    f"Kolom '{col}' tidak ditemukan di CSV."
                )

        signal = df[required].to_numpy(dtype=np.float32)

        # Load sampling frequency dynamically from metadata json
        json_path = os.path.join(
            cfg.SENSOR_RECORD_DIR,
            dataset_type,
            f"{record_id}.json"
        )
        fs = 250.0
        if os.path.exists(json_path):
            try:
                import json
                with open(json_path, "r") as f:
                    meta = json.load(f)
                fs = float(meta.get("sample_rate_hz", 250.0))
            except Exception as e:
                logger.warning(f"Gagal membaca sampling rate dari JSON: {e}")

    else:
        raise ValueError(
            f"Dataset '{dataset_type}' tidak dikenali."
        )

    # =============================================================
    # VALIDATION
    # =============================================================
    if not isinstance(signal, np.ndarray):
        raise TypeError(f"Sinyal harus berupa numpy.ndarray, tetapi didapat {type(signal)}")

    if signal.ndim != 2:
        raise ValueError(f"Sinyal harus berdimensi 2 [samples, channels], tetapi didapat shape {signal.shape}")

    if not isinstance(fs, float) or fs <= 0.0:
        raise ValueError(f"Sampling frequency (fs) harus berupa float positif, tetapi didapat {fs}")

    logger.info(f"Successfully loaded '{record_id}' | Shape: {signal.shape} | fs: {fs} Hz")
    return signal, fs
