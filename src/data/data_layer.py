# =====================================================================
# FILE: src/data/data_layer.py
# PURPOSE: DATA ACCESS LAYER WITH PROSIM SIMULATOR INTEGRATION
# =====================================================================

import os
import scipy.io
import wfdb
import numpy as np
from functools import lru_cache
import src.config as cfg # Import konfigurasi global
import pandas as pd

@lru_cache(maxsize=1)
def get_available_records():
    """Memindai direktori secara dinamis berbasis path dari config.py"""
    records = {
        "chapman": [],
        "ptbxl_100hz": [],
        "ptbxl_500hz": [],
        "prosim_simulator": []
    }
    
    # 1. Scan Chapman
    if os.path.exists(cfg.CHAPMAN_DIR):
        records["chapman"] = sorted(list(set([
            os.path.splitext(f)[0] for f in os.listdir(cfg.CHAPMAN_DIR) if f.endswith('.mat')
        ])))
        
    # 2. Scan PTB-XL 100Hz
    if os.path.exists(cfg.PTBXL_100HZ_DIR):
        records["ptbxl_100hz"] = sorted(list(set([
            os.path.splitext(f)[0] for f in os.listdir(cfg.PTBXL_100HZ_DIR) if f.endswith('.hea')
        ])))
        
    # 3. Scan PTB-XL 500Hz
    if os.path.exists(cfg.PTBXL_500HZ_DIR):
        records["ptbxl_500hz"] = sorted(list(set([
            os.path.splitext(f)[0] for f in os.listdir(cfg.PTBXL_500HZ_DIR) if f.endswith('.hea')
        ])))

    # 4. Scan Folder Kalibrasi ProSim Berbasis BASE_DIR (Proteksi Typo & Eksklusi)
    calibrated_record_path = os.path.join(cfg.BASE_DIR, "Kalibrasi Prosim")
    if os.path.exists(calibrated_record_path):
        records["prosim_simulator"] = sorted([
            d for d in os.listdir(calibrated_record_path)
            if os.path.isdir(os.path.join(calibrated_record_path, d)) and
            ("bpm" in d or "Arr" in d or "Sinus" in d or "Afib" in d or "Afb" in d or "Missed" in d or "PAC" in d)
        ])
        
    return records


def load_raw_signal(dataset_type, record_id):
    """Memuat sinyal EKG mentah menggunakan konstanta dari config.py"""
    if not record_id:
        raise ValueError("Record ID tidak boleh kosong.")
        
    # Jalur Khusus A: Mengambil Data Rekaman Fisik Simulator ProSim
    if dataset_type == "prosim_simulator":
        path = os.path.join(cfg.BASE_DIR, "Kalibrasi Prosim", record_id, "data", "raw_ecg.csv")
        df_sim = pd.read_csv(path)
        
        # Ekstrak kolom ch1, ch2, ch3 menjadi numpy array
        data = df_sim[['ch1', 'ch2', 'ch3']].values
        fs = 250.0  # Frekuensi sampling tetap alat ADS1293 Anda
        return data, fs
        
    # Jalur B: Mengambil Dataset Publik Kategori Chapman
    elif dataset_type == "chapman":
        path = os.path.join(cfg.CHAPMAN_DIR, f"{record_id}.mat")
        mat_data = scipy.io.loadmat(path)
        data = mat_data['val'].T
        fs = 500.0
        
    # Jalur C: Mengambil Dataset Publik Kategori PTB-XL (100Hz / 500Hz)
    else:
        sub_folder = cfg.PTBXL_100HZ_DIR if dataset_type == "ptbxl_100hz" else cfg.PTBXL_500HZ_DIR
        path = os.path.join(sub_folder, record_id)
        
        record, meta = wfdb.rdsamp(path)
        data = record
        fs = float(meta['fs'])
        
    # Slicing otomatis menggunakan settingan DEFAULT_LEADS untuk data Chapman & PTB-XL
    return data[:, cfg.DEFAULT_LEADS], fs