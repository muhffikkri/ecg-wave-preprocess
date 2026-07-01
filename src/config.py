# src/app/config.py
import os

# =====================================================================
# GLOBAL PATH CONFIGURATION
# =====================================================================
# Cukup ubah bagian ini saat deploy ke Raspberry Pi (misal: "/home/pi/dataset")
BASE_DIR = r"C:\Users\muhffikkri\Desktop\ecg-wave-preproccess\dataset"

# Sub-direktori Dataset
CHAPMAN_DIR = os.path.join(BASE_DIR, "chapman", "sample")
PTBXL_100HZ_DIR = os.path.join(BASE_DIR, "ptbxl", "sample_100hz")
PTBXL_500HZ_DIR = os.path.join(BASE_DIR, "ptbxl", "sample_500hz")

# =====================================================================
# SYSTEM & HARDWARE SIMULATION CONFIG
# =====================================================================
TARGET_FS = 250.0  # Laju sampel target untuk model inferensi
DEFAULT_LEADS = [0, 1, 2]  # Lead I, II, dan III untuk visualisasi UI

# Metadata untuk validasi internal
PIPELINE_VERSION = "v5.0_research_grade"