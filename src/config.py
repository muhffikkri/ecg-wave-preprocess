# src/app/config.py
import os

# =====================================================================
# GLOBAL PATH CONFIGURATION
# =====================================================================
BASE_DIR = r"D:\Project\ecg-wave-preproccess\dataset"

# Sub-direktori Dataset
CHAPMAN_DIR = os.path.join(BASE_DIR, "chapman", "sample")
PTBXL_100HZ_DIR = os.path.join(BASE_DIR, "ptbxl", "sample_100hz")
PTBXL_500HZ_DIR = os.path.join(BASE_DIR, "ptbxl", "sample_500hz")
PROSIM_SIMULATOR = os.path.join(BASE_DIR, "Kalibrasi Prosim")

# =====================================================================
# SYSTEM & HARDWARE SIMULATION CONFIG
# =====================================================================
TARGET_FS = 250.0  # Laju sampel target untuk model inferensi
DEFAULT_LEADS = [0, 1, 2]  # Lead I, II, dan III untuk visualisasi UI

# Metadata untuk validasi internal
PIPELINE_VERSION = "v5.0_research_grade"

# Model CNN
MODEL_PATH = r"D:\Project\ecg-wave-preproccess\model\Pure CNN Multi Label\best_model.keras"

# Kelas Target
TARGET_CLASSES = ["Normal", "AF", "Takikardia", "Bradikardia"]

# Ambang batas optimal multi-label dari hasil grid search 
OPTIMIZED_THRESHOLDS = [0.34, 0.37, 0.57, 0.57]