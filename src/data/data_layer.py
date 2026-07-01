# src/data/data_layer.py
import os
import scipy.io
import wfdb
import numpy as np
from functools import lru_cache
import src.config as cfg # Import konfigurasi global

@lru_cache(maxsize=1)
def get_available_records():
    """Memindai direktori secara dinamis berbasis path dari config.py"""
    records = {
        "chapman": [],
        "ptbxl_100hz": [],
        "ptbxl_500hz": []
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
        
    return records


def load_raw_signal(dataset_type, record_id):
    """Memuat sinyal EKG mentah menggunakan konstanta dari config.py"""
    if not record_id:
        raise ValueError("Record ID tidak boleh kosong.")
        
    if dataset_type == "chapman":
        path = os.path.join(cfg.CHAPMAN_DIR, f"{record_id}.mat")
        mat_data = scipy.io.loadmat(path)
        data = mat_data['val'].T
        fs = 500.0
    else:
        sub_folder = cfg.PTBXL_100HZ_DIR if dataset_type == "ptbxl_100hz" else cfg.PTBXL_500HZ_DIR
        path = os.path.join(sub_folder, record_id)
        
        record, meta = wfdb.rdsamp(path)
        data = record
        fs = float(meta['fs'])
        
    # Slicing otomatis menggunakan settingan DEFAULT_LEADS dari config
    return data[:, cfg.DEFAULT_LEADS], fs